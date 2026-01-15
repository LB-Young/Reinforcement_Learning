#!/usr/bin/env python3
"""
修复版GRPO训练脚本 - 解决多GPU张量调度问题
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
POLICY_MODEL = "/home/bayon/models/Qwen/Qwen3-0___6B"
REWARD_MODEL = "/home/bayon/models/reward-model-deberta-v3-large-v2"

BATCH_SIZE = 4
LEARNING_RATE = 1e-6
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
        return {"prompt": prompt}

# ==================== GRPO训练器 ====================
class GRPOTrainer:
    def __init__(self):
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 张GPU")
        
        if num_gpus < 2:
            print("警告: 只检测到1张GPU，将使用单卡模式")
            self.device_policy = torch.device("cuda:0")
            self.device_ref = torch.device("cuda:0")
            self.device_reward = torch.device("cuda:0")
        else:
            # 双卡配置
            self.device_policy = torch.device("cuda:0")
            self.device_ref = torch.device("cuda:1")
            self.device_reward = torch.device("cuda:1")
            print(f"策略模型 -> GPU:0 | 参考模型 -> GPU:1 | 奖励模型 -> GPU:1")
        
        self.tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("加载策略模型...")
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            POLICY_MODEL,
            torch_dtype=DTYPE,
            device_map={"": self.device_policy}
        )
        
        print("加载参考模型...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            POLICY_MODEL,
            torch_dtype=DTYPE,
            device_map={"": self.device_ref}
        )
        self.ref_model.eval()
        
        print("加载奖励模型...")
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL,
            torch_dtype=DTYPE,
            device_map={"": self.device_reward}
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=LEARNING_RATE)
        print("模型加载完成！")
    
    def generate_responses(self, prompts: List[str]) -> Tuple[List[str], List[str]]:
        self.policy_model.eval()
        all_responses = []
        all_prompts = []
        
        # 优化：为了减少GPU切换开销，这里建议不要分次循环
        for prompt in prompts:
            # 修改点：构建输入并显式移动到 policy 设备
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device_policy)
            
            with torch.no_grad():
                # 修改点：对于 GRPO，我们一次性为每个 Prompt 生成 GROUP_SIZE 个样本
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=GROUP_SIZE, # 效率更高
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 解码生成的文本
            generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            all_responses.extend(responses)
            all_prompts.extend([prompt] * GROUP_SIZE)
        
        return all_responses, all_prompts
    
    def compute_log_probs(self, prompts: List[str], responses: List[str], use_ref: bool = False) -> torch.Tensor:
        model = self.ref_model if use_ref else self.policy_model
        device = self.device_ref if use_ref else self.device_policy
        all_log_probs = []
        
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            # 修改点：将输入移动到对应的 model 所在的设备 (device)
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            
            response_start = prompt_inputs["input_ids"].shape[1]
            
            # 修改点：根据是否是 ref 模型决定是否开启 grad
            context_manager = torch.no_grad() if use_ref else torch.enable_grad()
            
            with context_manager:
                outputs = model(**full_inputs)
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                
                # 提取对应 token 的 log_prob
                token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
                # 对齐 response 部分
                response_log_probs = token_log_probs[0, response_start-1:-1]
                all_log_probs.append(response_log_probs.sum())
        
        # 修改点：最后统一放回到 policy 设备进行 Loss 计算
        return torch.stack(all_log_probs).to(self.device_policy)
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        rewards = []
        for prompt, response in zip(prompts, responses):
            full_text = f"{prompt} {response}"
            # 修改点：显式移动到 reward 模型设备
            inputs = self.reward_tokenizer(
                full_text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device_reward)
            
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                reward = outputs.logits[0, 0]
                rewards.append(reward)
        
        # 修改点：将计算出的 Reward 移回到 policy 设备用于后续 Advantages 计算
        return torch.stack(rewards).to(self.device_policy)
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        # 此时 rewards 已经在 device_policy 了
        batch_size = rewards.shape[0]
        num_groups = batch_size // GROUP_SIZE
        
        rewards_grouped = rewards.view(num_groups, GROUP_SIZE)
        baselines = rewards_grouped.mean(dim=1, keepdim=True)
        advantages = rewards_grouped - baselines
        advantages = advantages.view(-1)
        
        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        responses, prompts_expanded = self.generate_responses(batch_prompts)
        
        # 奖励计算 (内部已处理设备转换)
        rewards = self.compute_rewards(prompts_expanded, responses)
        advantages = self.compute_advantages(rewards)
        
        # 参考概率 (内部已处理设备转换)
        ref_log_probs = self.compute_log_probs(prompts_expanded, responses, use_ref=True).detach()
        # 初始概率用于 PPO 重要性采样
        old_log_probs = self.compute_log_probs(prompts_expanded, responses, use_ref=False).detach()
        
        total_step_loss = 0
        self.policy_model.train()
        
        for _ in range(GRPO_EPOCHS):
            # 重新计算当前 policy 的 log_probs
            new_log_probs = self.compute_log_probs(prompts_expanded, responses, use_ref=False)
        
            # 增加数值稳定性保护
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(torch.clamp(log_ratio, -10, 10)) # 防止 exp(inf)
            
            # PPO 损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL 散度约束 (相对于 Ref 模型)
            # 修改点：KL 计算必须在同一设备
            kl_penalty = new_log_probs - ref_log_probs
            kl_loss = 0.1 * kl_penalty.mean() # 这里的系数可以根据需要调整
            
            loss = policy_loss + kl_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            total_step_loss += loss.item()
        
        return {
            "loss": total_step_loss / GRPO_EPOCHS,
            "reward_mean": rewards.mean().item(),
        }
    
    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            pbar = tqdm(dataloader)
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch["prompt"])
                pbar.set_description(f"Loss: {metrics['loss']:.4f} | Rew: {metrics['reward_mean']:.4f}")
            
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            self.policy_model.save_pretrained(save_path)
            print(f"模型保存至: {save_path}")

def main():
    prompts = ["请解释什么是机器学习？", "如何学习Python编程？", "什么是神经网络？"] * 5
    tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = SimpleDataset(prompts, tokenizer, MAX_LENGTH)
    trainer = GRPOTrainer()
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()