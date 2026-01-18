#!/usr/bin/env python3
"""
GSPO (Group Sequence Policy Optimization) 训练脚本 - 基于Qwen2-0.5B
GSPO是一种结合了组采样和序列级优化的策略优化算法，主要特点：
1. Group Sampling: 为每个prompt生成多个回复进行组内比较
2. Sequence-Level Rewards: 序列级别的奖励计算
3. Relative Advantage: 使用组内相对优势
4. 无需Critic模型: 类似GRPO，使用组内均值作为基线
5. Token-Level Optimization: 支持token级别的策略优化
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import wandb
from tqdm import tqdm
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GSPOConfig:
    """GSPO训练配置"""
    # 模型配置
    policy_model_name: str = "Qwen/Qwen2-0.5B"
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    
    # 训练配置
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 3
    max_length: int = 512
    
    # GSPO特有超参数
    gspo_epochs: int = 4  # 每个批次数据的GSPO更新次数
    clip_range: float = 0.2  # PPO裁剪范围
    entropy_coef: float = 0.01  # 熵正则化系数
    kl_coef: float = 0.2  # KL散度惩罚系数
    target_kl: float = 0.01  # 目标KL散度
    adaptive_kl: bool = True  # 是否启用自适应KL系数调整
    
    # 🔥 GSPO Group Sampling参数
    group_size: int = 4  # 每组的样本数量
    use_group_normalization: bool = True  # 是否使用组内标准化
    
    # 🔥 GSPO Sequence-Level优化参数
    use_sequence_level_reward: bool = True  # 是否使用序列级别奖励
    use_token_level_loss: bool = False  # 是否使用token级别损失（可选）
    
    # 🔥 GSPO优势计算参数
    advantage_type: str = "relative"  # 优势类型: "relative"(相对), "normalized"(标准化)
    use_reward_shaping: bool = True  # 是否使用奖励塑形
    
    # 其他配置
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./gspo_output"
    use_wandb: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class GSPODataset(Dataset):
    """GSPO训练数据集"""
    
    def __init__(self, prompts: List[str], tokenizer, max_length: int = 512):
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

class GSPOTrainer:
    """GSPO训练器"""
    
    def __init__(self, config: GSPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化模型
        self._init_models()
        
        # 初始化优化器
        self._init_optimizers()
        
        # 初始化KL系数
        self.kl_coef = config.kl_coef
        
        # 初始化wandb
        if config.use_wandb:
            wandb.init(project="gspo-qwen", config=config.__dict__)
    
    def _init_models(self):
        """初始化策略模型和奖励模型（GSPO不需要critic模型）"""
        logger.info("正在加载模型...")
        
        # 策略模型
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.policy_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # 奖励模型
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.reward_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.config.reward_model_name)
        
        # 参考策略模型
        self.ref_policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.policy_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        self.ref_policy_model.eval()
        
        logger.info("模型加载完成（GSPO无需critic模型）")
    
    def _init_optimizers(self):
        """初始化优化器"""
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate
        )
    
    def generate_responses_with_group_sampling(self, prompts: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """
        🔥 GSPO Group Sampling: 为每个prompt生成group_size个回复
        返回：(所有回复, 对应的prompt, 回复长度)
        """
        self.policy_model.eval()
        
        all_responses = []
        all_prompts_expanded = []
        all_response_lengths = []
        
        for prompt in prompts:
            for _ in range(self.config.group_size):
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                response_length = len(generated_ids)
                
                all_responses.append(response)
                all_prompts_expanded.append(prompt)
                all_response_lengths.append(response_length)
        
        return all_responses, all_prompts_expanded, all_response_lengths
    
    def compute_log_probs(self, prompts: List[str], responses: List[str], 
                         use_ref_model: bool = False,
                         return_per_token: bool = False) -> torch.Tensor:
        """
        计算log概率
        
        Args:
            return_per_token: 是否返回token级别的log概率
        """
        all_log_probs = []
        all_token_log_probs = []
        
        model = self.ref_policy_model if use_ref_model else self.policy_model
        
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
            
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            response_start = prompt_inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                policy_outputs = model(**full_inputs)
                logits = policy_outputs.logits
                
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
                
                response_log_probs = token_log_probs[0, response_start-1:-1]
                
                if return_per_token:
                    all_token_log_probs.append(response_log_probs)
                
                all_log_probs.append(response_log_probs.sum())
        
        if return_per_token:
            return all_token_log_probs
        else:
            return torch.stack(all_log_probs)
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """🔥 GSPO Sequence-Level Rewards: 计算序列级别奖励"""
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
                reward_outputs = self.reward_model(**inputs)
                reward = reward_outputs.logits[0, 0]
                rewards.append(reward)
        
        return torch.stack(rewards)
    
    def compute_relative_advantages(self, rewards: torch.Tensor, group_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        🔥 GSPO Relative Advantage: 计算组内相对优势
        这是GSPO的核心创新之一
        """
        if group_size is None:
            group_size = self.config.group_size
        
        batch_size = rewards.shape[0]
        if batch_size % group_size != 0:
            num_complete_groups = batch_size // group_size
            rewards = rewards[:num_complete_groups * group_size]
            batch_size = rewards.shape[0]
        
        # 重塑为组的形状
        rewards_grouped = rewards.view(-1, group_size)
        
        # 🔥 计算组内均值作为基线
        group_baselines = rewards_grouped.mean(dim=1, keepdim=True)
        
        # 🔥 计算相对优势
        if self.config.advantage_type == "relative":
            # 相对优势：reward - baseline
            relative_advantages = rewards_grouped - group_baselines
        elif self.config.advantage_type == "normalized":
            # 标准化优势：(reward - baseline) / std
            relative_advantages = rewards_grouped - group_baselines
            if self.config.use_group_normalization:
                group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
                relative_advantages = relative_advantages / group_std
        else:
            raise ValueError(f"Unknown advantage type: {self.config.advantage_type}")
        
        # 重新展平
        relative_advantages = relative_advantages.view(-1)
        group_baselines = group_baselines.repeat(1, group_size).view(-1)
        
        return relative_advantages, group_baselines
    
    def compute_kl_penalty(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算KL散度惩罚"""
        current_log_probs = self.compute_log_probs(prompts, responses, use_ref_model=False)
        ref_log_probs = self.compute_log_probs(prompts, responses, use_ref_model=True)
        kl_divergence = current_log_probs - ref_log_probs
        return kl_divergence
    
    def compute_policy_loss_sequence_level(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                                          advantages: torch.Tensor) -> torch.Tensor:
        """
        🔥 GSPO Sequence-Level Policy Loss: 序列级别的策略损失
        """
        ratio = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss
    
    def compute_policy_loss_token_level(self, token_log_probs_list: List[torch.Tensor],
                                       old_token_log_probs_list: List[torch.Tensor],
                                       advantages: torch.Tensor) -> torch.Tensor:
        """
        🔥 GSPO Token-Level Policy Loss: token级别的策略损失（可选）
        """
        total_token_loss = 0.0
        total_tokens = 0
        
        for i, (token_log_probs, old_token_log_probs) in enumerate(zip(token_log_probs_list, old_token_log_probs_list)):
            advantage = advantages[i]
            
            token_ratios = torch.exp(token_log_probs - old_token_log_probs)
            
            surr1 = token_ratios * advantage
            surr2 = torch.clamp(token_ratios,
                               1 - self.config.clip_range,
                               1 + self.config.clip_range) * advantage
            
            token_loss = -torch.min(surr1, surr2).sum()
            
            total_token_loss += token_loss
            total_tokens += len(token_log_probs)
        
        return total_token_loss / total_tokens
    
    def compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                          advantages: torch.Tensor, kl_penalty: torch.Tensor,
                          token_log_probs_list: List[torch.Tensor] = None,
                          old_token_log_probs_list: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算GSPO策略损失"""
        # 选择序列级别或token级别损失
        if self.config.use_token_level_loss and token_log_probs_list is not None:
            policy_loss = self.compute_policy_loss_token_level(
                token_log_probs_list, old_token_log_probs_list, advantages
            )
        else:
            policy_loss = self.compute_policy_loss_sequence_level(
                log_probs, old_log_probs, advantages
            )
        
        # 熵损失
        entropy = -log_probs.mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        # KL损失
        kl_loss = self.kl_coef * kl_penalty.mean()
        
        return policy_loss, entropy_loss, kl_loss
    
    def update_kl_coef(self, kl_divergence: torch.Tensor):
        """自适应调整KL散度系数"""
        if not self.config.adaptive_kl:
            return
        
        mean_kl = kl_divergence.mean().item()
        
        if mean_kl > 2.0 * self.config.target_kl:
            self.kl_coef *= 1.5
        elif mean_kl < 0.5 * self.config.target_kl:
            self.kl_coef *= 0.5
        
        self.kl_coef = max(0.01, min(self.kl_coef, 1.0))
    
    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """
        执行一步GSPO训练
        
        GSPO训练流程：
        1. Group Sampling: 为每个prompt生成group_size个回复
        2. Sequence-Level Rewards: 计算序列级别奖励
        3. Relative Advantage: 计算组内相对优势
        4. Policy Optimization: 使用PPO-style裁剪更新策略
        """
        # 🔥 1. Group Sampling
        responses, prompts_expanded, response_lengths = self.generate_responses_with_group_sampling(batch_prompts)
        
        # 🔥 2. Sequence-Level Rewards
        raw_rewards = self.compute_rewards(prompts_expanded, responses)
        
        # 🔥 3. Relative Advantage
        relative_advantages, group_baselines = self.compute_relative_advantages(raw_rewards)
        
        # 截断数据
        prompts_truncated = prompts_expanded[:len(relative_advantages)]
        responses_truncated = responses[:len(relative_advantages)]
        
        # 计算log概率
        log_probs = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref_model=False)
        
        # 如果使用token-level loss，获取token级别的log概率
        old_token_log_probs_list = None
        if self.config.use_token_level_loss:
            old_token_log_probs_list = self.compute_log_probs(
                prompts_truncated, responses_truncated, use_ref_model=False, return_per_token=True
            )
            old_token_log_probs_list = [t.detach() for t in old_token_log_probs_list]
        
        # 计算KL散度惩罚
        kl_penalty = self.compute_kl_penalty(prompts_truncated, responses_truncated)
        
        # 标准化优势
        advantages = (relative_advantages - relative_advantages.mean()) / (relative_advantages.std() + 1e-8)
        
        # 保存旧的log概率
        old_log_probs = log_probs.detach()
        
        # 🔥 4. GSPO更新循环
        total_policy_loss = 0
        total_entropy_loss = 0
        total_kl_loss = 0
        
        for gspo_step in range(self.config.gspo_epochs):
            # 重新计算当前策略的log概率
            new_log_probs = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref_model=False)
            
            # 如果使用token-level loss
            new_token_log_probs_list = None
            if self.config.use_token_level_loss:
                new_token_log_probs_list = self.compute_log_probs(
                    prompts_truncated, responses_truncated, use_ref_model=False, return_per_token=True
                )
            
            # 计算损失
            policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(
                new_log_probs, old_log_probs, advantages, kl_penalty,
                token_log_probs_list=new_token_log_probs_list,
                old_token_log_probs_list=old_token_log_probs_list
            )
            
            # 总损失
            total_loss = policy_loss + entropy_loss + kl_loss
            
            # 更新策略模型
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.policy_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_kl_loss += kl_loss.item()
        
        # 自适应调整KL系数
        self.update_kl_coef(kl_penalty)
        
        return {
            "policy_loss": total_policy_loss / self.config.gspo_epochs,
            "entropy_loss": total_entropy_loss / self.config.gspo_epochs,
            "kl_loss": total_kl_loss / self.config.gspo_epochs,
            "raw_reward_mean": raw_rewards.mean().item(),
            "raw_reward_std": raw_rewards.std().item(),
            "relative_advantage_mean": relative_advantages.mean().item(),
            "relative_advantage_std": relative_advantages.std().item(),
            "group_baseline_mean": group_baselines.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "kl_divergence": kl_penalty.mean().item(),
            "kl_coef": self.kl_coef,
            "avg_response_length": np.mean(response_lengths[:len(relative_advantages)])
        }
    
    def train(self, train_dataset: GSPODataset):
        """主训练循环"""
        logger.info("开始GSPO训练...")
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_metrics = []
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
                batch_prompts = batch["prompt"]
                
                metrics = self.train_step(batch_prompts)
                epoch_metrics.append(metrics)
                
                if self.config.use_wandb:
                    wandb.log({
                        "step": global_step,
                        "epoch": epoch,
                        **metrics
                    })
                
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(global_step)
                
                global_step += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Step {global_step}: {metrics}")
            
            # 计算epoch平均指标
            avg_metrics = {}
            for key in epoch_metrics[0].keys():
                avg_metrics[f"epoch_{key}"] = np.mean([m[key] for m in epoch_metrics])
            
            logger.info(f"Epoch {epoch + 1} 平均指标: {avg_metrics}")
            
            if self.config.use_wandb:
                wandb.log(avg_metrics)
        
        logger.info("GSPO训练完成!")
        self.save_checkpoint("final")
    
    def save_checkpoint(self, step):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存策略模型
        self.policy_model.save_pretrained(os.path.join(checkpoint_dir, "policy"))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"检查点已保存到 {checkpoint_dir}")

def load_training_data() -> List[str]:
    """加载训练数据"""
    logger.info("正在加载训练数据...")
    
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
        
        prompts = []
        for item in dataset:
            conversation = item["chosen"]
            if conversation.startswith("Human:"):
                human_part = conversation.split("Assistant:")[0].replace("Human:", "").strip()
                if human_part:
                    prompts.append(human_part)
        
        logger.info(f"加载了 {len(prompts)} 个训练样本")
        return prompts
    
    except Exception as e:
        logger.warning(f"无法加载HH数据集: {e}")
        logger.info("使用示例数据进行训练")
        return [
            "请解释什么是机器学习？",
            "如何学习Python编程？",
            "什么是深度学习？",
            "请介绍一下人工智能的发展历史。",
            "如何提高编程技能？",
            "什么是自然语言处理？",
            "请解释神经网络的工作原理。",
            "如何选择合适的机器学习算法？"
        ] * 50

def main():
    """主函数"""
    config = GSPOConfig()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    prompts = load_training_data()
    
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = GSPODataset(prompts, tokenizer, config.max_length)
    
    trainer = GSPOTrainer(config)
    
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()
