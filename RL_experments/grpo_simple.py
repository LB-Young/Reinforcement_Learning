#!/usr/bin/env python3
"""
简化版GRPO训练脚本 - 最小化实现
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm

# ==================== 配置 ====================
POLICY_MODEL = r"E:\models\Qwen\Qwen3-0___6B"
REWARD_MODEL = r"E:\models\reward-model-deberta-v3-large-v2"

BATCH_SIZE = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
MAX_LENGTH = 512
GROUP_SIZE = 4
GRPO_EPOCHS = 4
CLIP_RANGE = 0.2
OUTPUT_DIR = "./grpo_output"

# ==================== 数据集 ====================
class SimpleDataset(Dataset):
    def __init__(self, prompts: List[str], tokenizer, max_length: int):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "prompt": prompt
        }

# ==================== GRPO训练器 ====================
class GRPOTrainer:
    def __init__(self):
        # 检查GPU数量
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 张GPU")
        
        if num_gpus < 2:
            print("警告: 只检测到1张GPU，将使用单卡模式")
            self.device_policy = torch.device("cuda:0")
            self.device_ref = torch.device("cuda:0")
            self.device_reward = torch.device("cuda:0")
        else:
            # 双卡配置：策略模型在GPU0，参考模型和奖励模型在GPU1
            self.device_policy = torch.device("cuda:0")
            self.device_ref = torch.device("cuda:1")
            self.device_reward = torch.device("cuda:1")
            print(f"策略模型 -> GPU:0")
            print(f"参考模型 -> GPU:1")
            print(f"奖励模型 -> GPU:1")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型到指定GPU
        print("加载策略模型...")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            POLICY_MODEL,
            torch_dtype=torch.float16,
            device_map={"": self.device_policy},
            trust_remote_code=True
        )
        
        print("加载参考模型...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            POLICY_MODEL,
            torch_dtype=torch.float16,
            device_map={"": self.device_ref},
            trust_remote_code=True
        )
        self.ref_model.eval()
        
        print("加载奖励模型...")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL,
            torch_dtype=torch.float16,
            device_map={"": self.device_reward},
            trust_remote_code=True
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        
        # 优化器
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=LEARNING_RATE)
        
        print("模型加载完成！")
    
    def generate_responses(self, prompts: List[str]) -> Tuple[List[str], List[str]]:
        """为每个prompt生成GROUP_SIZE个回复"""
        self.policy_model.eval()
        all_responses = []
        all_prompts = []
        
        for prompt in prompts:
            for _ in range(GROUP_SIZE):
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device_policy) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                all_responses.append(response)
                all_prompts.append(prompt)
        
        return all_responses, all_prompts
    
    def compute_log_probs(self, prompts: List[str], responses: List[str], use_ref: bool = False) -> torch.Tensor:
        """计算log概率"""
        model = self.ref_model if use_ref else self.policy_model
        device = self.device_ref if use_ref else self.device_policy
        all_log_probs = []
        
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
            full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
            
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            response_start = prompt_inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = model(**full_inputs)
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
                response_log_probs = token_log_probs[0, response_start-1:-1]
                all_log_probs.append(response_log_probs.sum())
        
        # 将结果移到策略模型所在的设备
        return torch.stack(all_log_probs).to(self.device_policy)
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算奖励"""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt} {response}"
            inputs = self.reward_tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                reward = outputs.logits[0, 0]
                rewards.append(reward)
        
        return torch.stack(rewards)
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """计算GRPO优势函数（组内相对奖励）"""
        batch_size = rewards.shape[0]
        num_groups = batch_size // GROUP_SIZE
        rewards = rewards[:num_groups * GROUP_SIZE]
        
        # 重塑为组
        rewards_grouped = rewards.view(num_groups, GROUP_SIZE)
        
        # 计算组内均值作为基线
        baselines = rewards_grouped.mean(dim=1, keepdim=True)
        
        # 相对奖励
        advantages = rewards_grouped - baselines
        advantages = advantages.view(-1)
        
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """训练一步"""
        # 生成回复
        responses, prompts_expanded = self.generate_responses(batch_prompts)
        
        # 计算奖励
        rewards = self.compute_rewards(prompts_expanded, responses)
        
        # 计算优势
        advantages = self.compute_advantages(rewards)
        
        # 截断数据
        prompts_truncated = prompts_expanded[:len(advantages)]
        responses_truncated = responses[:len(advantages)]
        
        # 计算初始log概率
        old_log_probs = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref=False)
        ref_log_probs = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref=True)
        kl_penalty = old_log_probs - ref_log_probs
        
        old_log_probs = old_log_probs.detach()
        
        # GRPO更新
        total_loss = 0
        self.policy_model.train()
        
        for _ in range(GRPO_EPOCHS):
            new_log_probs = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref=False)
            
            # 计算损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            kl_loss = 0.2 * kl_penalty.mean()
            loss = policy_loss + kl_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return {
            "loss": total_loss / GRPO_EPOCHS,
            "reward_mean": rewards.mean().item(),
            "advantage_mean": advantages.mean().item(),
        }
    
    def train(self, dataset):
        """训练循环"""
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                batch_prompts = batch["prompt"]
                metrics = self.train_step(batch_prompts)
                
                if batch_idx % 10 == 0:
                    print(f"Step {batch_idx}: {metrics}")
            
            # 保存模型
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"模型已保存到 {save_path}")
        
        print("\n训练完成！")

# ==================== 主函数 ====================
def main():
    # 加载数据 - 使用本地示例数据
    print("加载训练数据...")
    print("使用本地示例数据（避免从HuggingFace下载）")
    prompts = [
        "请解释什么是机器学习？",
        "如何学习Python编程？",
        "什么是深度学习？",
        "请介绍一下人工智能的发展历史。",
        "什么是神经网络？",
        "如何优化深度学习模型？",
        "请解释什么是强化学习？",
        "Python和Java有什么区别？",
        "什么是自然语言处理？",
        "如何开始学习数据科学？",
    ] * 10
    
    print(f"加载了 {len(prompts)} 个训练样本")
    
    # 创建数据集
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = SimpleDataset(prompts, tokenizer, MAX_LENGTH)
    
    # 创建训练器并训练
    trainer = GRPOTrainer()
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()
