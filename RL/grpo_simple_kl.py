#!/usr/bin/env python3
"""
标准版 GRPO 训练脚本 - 实现 Token-level Ratio 与 KL 计算
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import List, Tuple, Dict
from tqdm import tqdm

# ==================== 配置 ====================
POLICY_MODEL = "/home/bayon/models/Qwen/Qwen3-0___6B"
REWARD_MODEL = "/home/bayon/models/reward-model-deberta-v3-large-v2"

BATCH_SIZE = 2 # 增加 Token-level 计算后显存占用会升高，适当调小
LEARNING_RATE = 1e-6
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
NUM_EPOCHS = 1
GROUP_SIZE = 4
GRPO_EPOCHS = 4
CLIP_RANGE = 0.2
KL_COEF = 0.01  # KL 惩罚系数
OUTPUT_DIR = "./grpo_output/grpo_simple_kl_test1"

# ==================== 数据集 ====================
class SimpleDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

# ==================== GRPO训练器 ====================
class GRPOTrainer:
    def __init__(self):
        num_gpus = torch.cuda.device_count()
        self.device_policy = torch.device("cuda:0")
        self.device_ref = torch.device("cuda:1") if num_gpus > 1 else torch.device("cuda:0")
        self.device_reward = torch.device("cuda:1") if num_gpus > 1 else torch.device("cuda:0")
        
        self.tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL, torch_dtype=DTYPE, device_map={"": self.device_policy})
        self.ref_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL, torch_dtype=DTYPE, device_map={"": self.device_ref})
        self.ref_model.eval()
        
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL, torch_dtype=DTYPE, device_map={"": self.device_reward})
        self.reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL)
        
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=LEARNING_RATE)

    def generate_responses(self, prompts: List[str]) -> Tuple[List[str], List[str]]:
        self.policy_model.eval()
        all_responses, all_prompts = [], []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_policy)
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=GROUP_SIZE,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 仅提取生成的 Response 部分
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            responses = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_responses.extend(responses)
            all_prompts.extend([prompt] * GROUP_SIZE)
        
        return all_responses, all_prompts

    def get_token_log_probs(self, model, prompts: List[str], responses: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
        """核心修改：获取 Token 级别的 log_probs 并返回 Mask"""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # 为了计算 Response 部分，需要知道 Prompt 的长度
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        prompt_lens = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :] # Shift 对齐
        labels = inputs["input_ids"][:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        # 制作 Mask: 1 仅在 Response 区域且非 Padding 处
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lens):
            mask[i, p_len:] = (labels[i, p_len:] != self.tokenizer.pad_token_id)
            
        return token_log_probs, mask

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        rewards = []
        for p, r in zip(prompts, responses):
            inputs = self.reward_tokenizer(p + r, return_tensors="pt", truncation=True).to(self.device_reward)
            with torch.no_grad():
                reward = self.reward_model(**inputs).logits[0, 0]
                rewards.append(reward)
        return torch.stack(rewards).to(self.device_policy)

    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        responses, prompts_expanded = self.generate_responses(batch_prompts)
        
        # 1. 计算 Advantages (Group-level)
        rewards = self.compute_rewards(prompts_expanded, responses)
        num_groups = len(batch_prompts)
        rewards_grouped = rewards.view(num_groups, GROUP_SIZE)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_grouped - mean_r) / std_r).view(-1) # [B*G]

        # 2. 获取参考 log_probs 和初始旧分布 log_probs (Token-level)
        with torch.no_grad():
            ref_log_probs, _ = self.get_token_log_probs(self.ref_model, prompts_expanded, responses, self.device_ref)
            ref_log_probs = ref_log_probs.to(self.device_policy)
            old_log_probs, mask = self.get_token_log_probs(self.policy_model, prompts_expanded, responses, self.device_policy)
        
        # 3. 策略优化循环
        self.policy_model.train()
        total_loss = 0
        for _ in range(GRPO_EPOCHS):
            new_log_probs, _ = self.get_token_log_probs(self.policy_model, prompts_expanded, responses, self.device_policy)
            
            # --- 标准计算逻辑 ---
            # Importance Sampling Ratio
            log_ratio = (new_log_probs - old_log_probs) * mask
            ratio = torch.exp(log_ratio)
            
            # PPO Clip Loss
            adv_t = advantages.unsqueeze(1) # 广播优势到每个 Token
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv_t
            policy_loss = -torch.min(surr1, surr2)
            
            # KL Penalty (Token-level) - 使用稳定版近似: exp(ref-new) - (ref-new) - 1
            # 或者更通用的: new_log_probs - ref_log_probs
            kl_div = (new_log_probs - ref_log_probs)
            
            # 组合 Loss 并对 Mask 求均值
            loss_map = (policy_loss + KL_COEF * kl_div) * mask
            step_loss = loss_map.sum() / mask.sum()
            
            self.optimizer.zero_grad()
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += step_loss.item()
            
        return {"loss": total_loss / GRPO_EPOCHS, "reward": rewards.mean().item()}

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(NUM_EPOCHS):
            pbar = tqdm(dataloader)
            for batch in pbar:
                metrics = self.train_step(batch["prompt"])
                pbar.set_description(f"L:{metrics['loss']:.4f} R:{metrics['reward']:.2f}")
                
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            self.policy_model.save_pretrained(save_path)
            print(f"模型保存至: {save_path}")

def main():
    prompts = ["如何制作一杯好咖啡？", "解释量子纠缠。", "写一段冒泡排序代码。"] * 10
    dataset = SimpleDataset(prompts)
    trainer = GRPOTrainer()
    trainer.train(dataset)

if __name__ == "__main__":
    main()