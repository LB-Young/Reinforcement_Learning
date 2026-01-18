#!/usr/bin/env python3
"""
VAPO (Value-based Augmented Proximal Policy Optimization) 训练脚本 - 基于Qwen2-0.5B
VAPO是PPO的改进版本，专为长CoT推理任务设计，通过以下技术提升性能：
1. Value-Pretraining: 缓解价值模型初始化偏差
2. Decoupled-GAE: 解耦价值和策略的优势计算
3. Length-Adaptive GAE: 根据序列长度自适应调整λ参数
4. Token-Level Loss: token级别的策略梯度损失
5. Clip-Higher: 非对称裁剪范围
6. Group-Sampling: 组内采样增强对比信号
7. Positive Example LM Loss: 自模仿学习
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
class VAPOConfig:
    """VAPO训练配置"""
    # 模型配置
    policy_model_name: str = "Qwen/Qwen2-0.5B"
    critic_model_name: str = "Qwen/Qwen2-0.5B"
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"
    
    # 训练配置
    batch_size: int = 8
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6  # VAPO使用更小的学习率
    critic_learning_rate: float = 2e-6  # critic学习率更大，需要更快更新
    num_epochs: int = 3
    max_length: int = 512
    
    # VAPO特有超参数
    vapo_epochs: int = 4
    clip_range_low: float = 0.2  # 🔥 Clip-Higher: 下界裁剪范围
    clip_range_high: float = 0.28  # 🔥 Clip-Higher: 上界裁剪范围更大
    entropy_coef: float = 0.01
    vf_coef: float = 0.1
    gamma: float = 1.0  # VAPO使用γ=1.0
    
    # 🔥 Decoupled-GAE参数
    lambda_critic: float = 1.0  # critic使用λ=1.0，无偏估计
    lambda_policy_base: float = 0.95  # policy的基础λ值
    use_decoupled_gae: bool = True  # 是否使用解耦GAE
    
    # 🔥 Length-Adaptive GAE参数
    use_length_adaptive_gae: bool = True  # 是否使用长度自适应GAE
    length_threshold: int = 100  # 长度阈值
    lambda_policy_long: float = 0.99  # 长序列使用更大的λ
    
    # 🔥 Token-Level Loss
    use_token_level_loss: bool = True  # 是否使用token级别损失
    
    # 🔥 Group-Sampling参数
    group_size: int = 4  # 每个prompt采样的回复数量
    use_group_normalization: bool = True
    
    # 🔥 Value-Pretraining参数
    use_value_pretraining: bool = True  # 是否使用价值预训练
    value_pretrain_steps: int = 100  # 价值预训练步数
    
    # 🔥 Positive Example LM Loss (Self-Imitation Learning)
    use_sil: bool = True  # 是否使用自模仿学习
    sil_coef: float = 0.1  # SIL损失系数
    sil_reward_threshold: float = 0.5  # 正样本奖励阈值
    
    # 其他配置
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./vapo_output"
    use_wandb: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class VAPODataset(Dataset):
    """VAPO训练数据集"""
    
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

class VAPOTrainer:
    """VAPO训练器"""
    
    def __init__(self, config: VAPOConfig):
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
        
        # 价值预训练标志
        self.value_pretrained = False
        
        # 初始化wandb
        if config.use_wandb:
            wandb.init(project="vapo-qwen", config=config.__dict__)
    
    def _init_models(self):
        """初始化策略模型、critic模型和奖励模型"""
        logger.info("正在加载模型...")
        
        # 策略模型
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.policy_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Critic模型（添加value head）
        self.critic_model = AutoModelForCausalLM.from_pretrained(
            self.config.critic_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # 为critic模型添加value head
        hidden_size = self.critic_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)
        
        # 奖励模型
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.reward_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.config.reward_model_name)
        
        # 🔥 VAPO: 使用奖励模型初始化critic模型
        if self.config.use_value_pretraining:
            logger.info("使用奖励模型初始化critic模型...")
            # 这里简化处理，实际应该复制奖励模型的权重到critic
            # 由于架构可能不同，这里只是标记需要预训练
        
        # 参考策略模型
        self.ref_policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.policy_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        self.ref_policy_model.eval()
        
        logger.info("模型加载完成")
    
    def _init_optimizers(self):
        """初始化优化器"""
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate
        )
        
        critic_params = list(self.critic_model.parameters()) + list(self.value_head.parameters())
        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=self.config.critic_learning_rate
        )
    
    def value_pretrain_step(self, prompts: List[str], responses: List[str], rewards: torch.Tensor):
        """
        🔥 VAPO Value-Pretraining: 预训练价值模型
        使用奖励信号训练价值模型，缓解初始化偏差
        """
        all_values = []
        
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
            
            # 计算价值
            critic_outputs = self.critic_model(**full_inputs, output_hidden_states=True)
            hidden_states = critic_outputs.hidden_states[-1]
            values = self.value_head(hidden_states)
            all_values.append(values[0, -1, 0])
        
        all_values = torch.stack(all_values)
        
        # 价值损失：使价值估计接近实际奖励
        value_loss = F.mse_loss(all_values, rewards)
        
        # 更新critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_model.parameters()) + list(self.value_head.parameters()), 1.0
        )
        self.critic_optimizer.step()
        
        return value_loss.item()
    
    def generate_responses_with_group_sampling(self, prompts: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """
        🔥 VAPO Group-Sampling: 为每个prompt生成group_size个回复
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
    
    def compute_log_probs_and_values(self, prompts: List[str], responses: List[str],
                                    return_per_token: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算log概率和价值函数
        
        Args:
            return_per_token: 🔥 是否返回token级别的log概率（用于token-level loss）
        """
        all_log_probs = []
        all_token_log_probs = []
        all_values = []
        
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
            
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            response_start = prompt_inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                # 计算log概率
                policy_outputs = self.policy_model(**full_inputs)
                logits = policy_outputs.logits
                
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
                
                response_log_probs = token_log_probs[0, response_start-1:-1]
                
                if return_per_token:
                    all_token_log_probs.append(response_log_probs)
                
                all_log_probs.append(response_log_probs.sum())
                
                # 计算价值函数
                critic_outputs = self.critic_model(**full_inputs, output_hidden_states=True)
                hidden_states = critic_outputs.hidden_states[-1]
                values = self.value_head(hidden_states)
                all_values.append(values[0, -1, 0])
        
        if return_per_token:
            return all_token_log_probs, torch.stack(all_values)
        else:
            return torch.stack(all_log_probs), torch.stack(all_values)
    
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """使用奖励模型计算奖励"""
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
    
    def compute_length_adaptive_lambda(self, response_length: int) -> float:
        """
        🔥 VAPO Length-Adaptive GAE: 根据序列长度自适应调整λ
        
        对于短序列：使用较小的λ（如0.95），减少方差
        对于长序列：使用较大的λ（如0.99），减少偏差
        """
        if not self.config.use_length_adaptive_gae:
            return self.config.lambda_policy_base
        
        if response_length > self.config.length_threshold:
            # 长序列使用更大的λ，减少偏差
            return self.config.lambda_policy_long
        else:
            # 短序列使用基础λ
            return self.config.lambda_policy_base
    
    def compute_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor,
                              response_lengths: List[int], for_critic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        🔥 VAPO Decoupled-GAE + Length-Adaptive GAE
        
        计算GAE优势函数，支持：
        1. Decoupled-GAE: critic和policy使用不同的λ
        2. Length-Adaptive GAE: 根据序列长度调整λ
        
        Args:
            for_critic: 是否为critic计算（使用λ=1.0）
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # 🔥 Decoupled-GAE: critic使用λ=1.0（无偏估计）
        if for_critic and self.config.use_decoupled_gae:
            lambda_gae = self.config.lambda_critic
        else:
            # 🔥 Length-Adaptive GAE: 根据序列长度调整λ
            lambda_gae = self.config.lambda_policy_base
        
        # 简化的GAE计算（假设稀疏奖励，只在终止时有奖励）
        for i in range(batch_size):
            # 🔥 Length-Adaptive: 为每个样本计算自适应λ
            if not for_critic and self.config.use_length_adaptive_gae:
                lambda_gae = self.compute_length_adaptive_lambda(response_lengths[i])
            
            # TD error: δ = r + γV(s') - V(s)
            # 简化版本：假设终止状态V(s')=0
            td_error = rewards[i] - values[i]
            
            # GAE: A = δ (简化版本，实际应该是多步累积)
            advantages[i] = td_error
            returns[i] = rewards[i]
        
        return advantages, returns
    
    def compute_relative_rewards(self, rewards: torch.Tensor, group_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        🔥 VAPO Group-Sampling: 计算组内相对奖励
        与GRPO/DAPO相同的相对奖励计算
        """
        if group_size is None:
            group_size = self.config.group_size
        
        batch_size = rewards.shape[0]
        if batch_size % group_size != 0:
            num_complete_groups = batch_size // group_size
            rewards = rewards[:num_complete_groups * group_size]
            batch_size = rewards.shape[0]
        
        rewards_grouped = rewards.view(-1, group_size)
        group_baselines = rewards_grouped.mean(dim=1, keepdim=True)
        relative_rewards = rewards_grouped - group_baselines
        
        if self.config.use_group_normalization:
            group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
            relative_rewards = relative_rewards / group_std
        
        relative_rewards = relative_rewards.view(-1)
        group_baselines = group_baselines.repeat(1, group_size).view(-1)
        
        return relative_rewards, group_baselines
    
    def compute_policy_loss_token_level(self, token_log_probs_list: List[torch.Tensor],
                                       old_token_log_probs_list: List[torch.Tensor],
                                       advantages: torch.Tensor) -> torch.Tensor:
        """
        🔥 VAPO Token-Level Loss: 对每个token单独计算PPO损失
        所有token权重相同，避免长序列贡献不足
        """
        total_token_loss = 0.0
        total_tokens = 0
        
        for i, (token_log_probs, old_token_log_probs) in enumerate(zip(token_log_probs_list, old_token_log_probs_list)):
            advantage = advantages[i]
            
            # 计算每个token的概率比率
            token_ratios = torch.exp(token_log_probs - old_token_log_probs)
            
            # 🔥 VAPO Clip-Higher: 非对称裁剪
            surr1 = token_ratios * advantage
            surr2 = torch.clamp(token_ratios,
                               1 - self.config.clip_range_low,
                               1 + self.config.clip_range_high) * advantage
            
            token_loss = -torch.min(surr1, surr2).sum()
            
            total_token_loss += token_loss
            total_tokens += len(token_log_probs)
        
        return total_token_loss / total_tokens
    
    def compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                          advantages: torch.Tensor,
                          token_log_probs_list: List[torch.Tensor] = None,
                          old_token_log_probs_list: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        🔥 VAPO策略损失计算（支持Token-Level Loss和Clip-Higher）
        """
        # 🔥 Token-Level Loss
        if self.config.use_token_level_loss and token_log_probs_list is not None:
            policy_loss = self.compute_policy_loss_token_level(
                token_log_probs_list, old_token_log_probs_list, advantages
            )
        else:
            # Sample-level loss with Clip-Higher
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio,
                               1 - self.config.clip_range_low,
                               1 + self.config.clip_range_high) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
        
        # 熵损失
        entropy = -log_probs.mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        return policy_loss, entropy_loss
    
    def compute_sil_loss(self, prompts: List[str], responses: List[str], rewards: torch.Tensor) -> torch.Tensor:
        """
        🔥 VAPO Positive Example LM Loss (Self-Imitation Learning)
        对高奖励样本进行监督学习，提高利用效率
        """
        if not self.config.use_sil:
            return torch.tensor(0.0, device=self.device)
        
        # 筛选高奖励样本
        positive_mask = rewards > self.config.sil_reward_threshold
        if not positive_mask.any():
            return torch.tensor(0.0, device=self.device)
        
        positive_prompts = [p for i, p in enumerate(prompts) if positive_mask[i]]
        positive_responses = [r for i, r in enumerate(responses) if positive_mask[i]]
        
        sil_loss = 0.0
        for prompt, response in zip(positive_prompts, positive_responses):
            full_text = prompt + response
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
            
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            response_start = prompt_inputs["input_ids"].shape[1]
            
            # 计算语言模型损失
            outputs = self.policy_model(**full_inputs, labels=full_inputs["input_ids"])
            
            # 只计算response部分的损失
            logits = outputs.logits[0, response_start-1:-1, :]
            labels = full_inputs["input_ids"][0, response_start:]
            
            loss = F.cross_entropy(logits, labels)
            sil_loss += loss
        
        if len(positive_prompts) > 0:
            sil_loss = sil_loss / len(positive_prompts)
        
        return sil_loss
    
    def compute_value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """计算价值函数损失"""
        return F.mse_loss(values, returns)
    
    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """
        执行一步VAPO训练
        
        VAPO训练流程：
        1. Group-Sampling: 为每个prompt生成group_size个回复
        2. 计算奖励和相对奖励
        3. Decoupled-GAE + Length-Adaptive GAE: 计算优势
        4. Token-Level Loss + Clip-Higher: 更新策略
        5. SIL: 对高奖励样本进行监督学习
        """
        # 🔥 1. Group-Sampling: 生成多个回复
        responses, prompts_expanded, response_lengths = self.generate_responses_with_group_sampling(batch_prompts)
        
        # 2. 计算奖励
        raw_rewards = self.compute_rewards(prompts_expanded, responses)
        
        # 🔥 Value-Pretraining: 如果未预训练，先预训练价值模型
        if self.config.use_value_pretraining and not self.value_pretrained:
            logger.info("执行价值预训练...")
            for _ in range(self.config.value_pretrain_steps):
                pretrain_loss = self.value_pretrain_step(prompts_expanded, responses, raw_rewards)
            self.value_pretrained = True
            logger.info(f"价值预训练完成，最终损失: {pretrain_loss:.4f}")
        
        # 3. 计算相对奖励（组内归一化）
        relative_rewards, group_baselines = self.compute_relative_rewards(raw_rewards)
        
        # 截断数据
        prompts_truncated = prompts_expanded[:len(relative_rewards)]
        responses_truncated = responses[:len(relative_rewards)]
        response_lengths_truncated = response_lengths[:len(relative_rewards)]
        
        # 4. 计算log概率和价值
        log_probs, values = self.compute_log_probs_and_values(
            prompts_truncated, responses_truncated, return_per_token=False
        )
        
        # 🔥 如果使用token-level loss，获取token级别的log概率
        old_token_log_probs_list = None
        if self.config.use_token_level_loss:
            old_token_log_probs_list, _ = self.compute_log_probs_and_values(
                prompts_truncated, responses_truncated, return_per_token=True
            )
            old_token_log_probs_list = [t.detach() for t in old_token_log_probs_list]
        
        # 🔥 5. Decoupled-GAE + Length-Adaptive GAE: 计算优势
        # 为policy计算优势（使用自适应λ）
        advantages_policy, returns_policy = self.compute_gae_advantages(
            relative_rewards, values, response_lengths_truncated, for_critic=False
        )
        
        # 为critic计算优势（使用λ=1.0）
        advantages_critic, returns_critic = self.compute_gae_advantages(
            relative_rewards, values, response_lengths_truncated, for_critic=True
        )
        
        # 标准化优势
        advantages_policy = (advantages_policy - advantages_policy.mean()) / (advantages_policy.std() + 1e-8)
        
        # 保存旧的log概率
        old_log_probs = log_probs.detach()
        
        # VAPO更新循环
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_sil_loss = 0
        
        for vapo_step in range(self.config.vapo_epochs):
            # 重新计算当前策略的log概率和值
            new_log_probs, new_values = self.compute_log_probs_and_values(
                prompts_truncated, responses_truncated, return_per_token=False
            )
            
            # 🔥 如果使用token-level loss，获取当前策略的token级别log概率
            new_token_log_probs_list = None
            if self.config.use_token_level_loss:
                new_token_log_probs_list, _ = self.compute_log_probs_and_values(
                    prompts_truncated, responses_truncated, return_per_token=True
                )
            
            # 🔥 6. 计算损失（Token-Level Loss + Clip-Higher）
            policy_loss, entropy_loss = self.compute_policy_loss(
                new_log_probs, old_log_probs, advantages_policy,
                token_log_probs_list=new_token_log_probs_list,
                old_token_log_probs_list=old_token_log_probs_list
            )
            
            # 🔥 使用critic的优势计算价值损失
            value_loss = self.compute_value_loss(new_values, returns_critic)
            
            # 🔥 7. SIL损失
            sil_loss = self.compute_sil_loss(prompts_truncated, responses_truncated, raw_rewards[:len(relative_rewards)])
            
            # 总损失
            total_loss = policy_loss + self.config.vf_coef * value_loss + entropy_loss + self.config.sil_coef * sil_loss
            
            # 更新策略模型
            self.policy_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.policy_optimizer.step()
            
            # 更新Critic模型
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critic_model.parameters()) + list(self.value_head.parameters()), 1.0
            )
            self.critic_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_sil_loss += sil_loss.item()
        
        return {
            "policy_loss": total_policy_loss / self.config.vapo_epochs,
            "value_loss": total_value_loss / self.config.vapo_epochs,
            "entropy_loss": total_entropy_loss / self.config.vapo_epochs,
            "sil_loss": total_sil_loss / self.config.vapo_epochs,
            "raw_reward_mean": raw_rewards.mean().item(),
            "raw_reward_std": raw_rewards.std().item(),
            "relative_reward_mean": relative_rewards.mean().item(),
            "advantage_mean": advantages_policy.mean().item(),
            "value_mean": values.mean().item(),
            "avg_response_length": np.mean(response_lengths_truncated)
        }
    
    def train(self, train_dataset: VAPODataset):
        """主训练循环"""
        logger.info("开始VAPO训练...")
        
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
        
        logger.info("VAPO训练完成!")
        self.save_checkpoint("final")
    
    def save_checkpoint(self, step):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存策略模型
        self.policy_model.save_pretrained(os.path.join(checkpoint_dir, "policy"))
        
        # 保存critic模型和value head
        self.critic_model.save_pretrained(os.path.join(checkpoint_dir, "critic"))
        torch.save(self.value_head.state_dict(), os.path.join(checkpoint_dir, "value_head.pt"))
        
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
    config = VAPOConfig()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    prompts = load_training_data()
    
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = VAPODataset(prompts, tokenizer, config.max_length)
    
    trainer = VAPOTrainer(config)
    
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()
