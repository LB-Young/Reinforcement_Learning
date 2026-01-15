#!/usr/bin/env python3
"""
DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 训练脚本 - 基于Qwen2-0.5B
DAPO是GRPO的改进版本，通过Clip-Higher、Token-Level Loss、Dynamic Sampling等技术提升性能
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
class DAPOConfig:
    """DAPO训练配置"""
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
    
    # DAPO特有超参数
    dapo_epochs: int = 4
    clip_range_low: float = 0.2  # 下界裁剪范围
    clip_range_high: float = 0.28  # 🔥 Clip-Higher: 上界裁剪范围更大
    entropy_coef: float = 0.01
    kl_coef: float = 0.0  # 🔥 DAPO移除KL惩罚
    use_kl_penalty: bool = False  # 🔥 是否使用KL惩罚
    
    # DAPO特有参数
    group_size: int = 4
    use_group_normalization: bool = True
    use_dynamic_sampling: bool = True  # 🔥 是否启用动态采样
    max_dynamic_samples: int = 8  # 动态采样最大样本数
    use_token_level_loss: bool = True  # 🔥 是否使用token级别损失
    
    # Overlong response处理
    max_response_length: int = 256
    use_overlong_filtering: bool = True  # 🔥 过滤过长回复
    use_soft_overlong_punishment: bool = False  # 软惩罚过长回复
    overlong_threshold: float = 0.8  # 超过max_length的比例开始惩罚
    
    # 其他配置
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = "./dapo_output"
    use_wandb: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class DAPODataset(Dataset):
    """DAPO训练数据集"""
    
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

class DAPOTrainer:
    """DAPO训练器"""
    
    def __init__(self, config: DAPOConfig):
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
        
        # 统计信息
        self.dynamic_sampling_stats = {
            "total_questions": 0,
            "resampled_questions": 0,
            "avg_extra_samples": 0.0
        }
        
        # 初始化wandb
        if config.use_wandb:
            wandb.init(project="dapo-qwen", config=config.__dict__)
    
    def _init_models(self):
        """初始化策略模型和奖励模型"""
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
        
        logger.info("模型加载完成")
    
    def _init_optimizers(self):
        """初始化优化器"""
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config.learning_rate
        )
    
    def apply_soft_overlong_punishment(self, rewards: torch.Tensor, response_lengths: List[int]) -> torch.Tensor:
        """
        🔥 DAPO特性：软惩罚过长回复
        对超过阈值的回复进行渐进式惩罚
        """
        if not self.config.use_soft_overlong_punishment:
            return rewards
        
        threshold_length = int(self.config.max_response_length * self.config.overlong_threshold)
        punished_rewards = []
        
        for reward, length in zip(rewards, response_lengths):
            if length > threshold_length:
                # 渐进式惩罚：超出部分越多，惩罚越大
                excess_ratio = (length - threshold_length) / threshold_length
                punishment = -0.5 * excess_ratio  # 惩罚系数可调
                punished_reward = reward + punishment
            else:
                punished_reward = reward
            punished_rewards.append(punished_reward)
        
        return torch.stack(punished_rewards)
    
    def compute_log_probs(self, prompts: List[str], responses: List[str], 
                         use_ref_model: bool = False, return_per_token: bool = False) -> torch.Tensor:
        """
        批量计算log概率
        
        Args:
            prompts: 提示列表
            responses: 回复列表
            use_ref_model: 是否使用参考模型
            return_per_token: 🔥 是否返回每个token的log概率（用于token-level loss）
        
        Returns:
            如果return_per_token=False: 返回每个样本的总log概率 [batch_size]
            如果return_per_token=True: 返回每个token的log概率列表 List[Tensor]
        """
        all_log_probs = []
        all_token_log_probs = []  # 🔥 存储每个样本的token级别log概率
        
        model = self.ref_policy_model if use_ref_model else self.policy_model
        
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
            
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            response_start = prompt_inputs["input_ids"].shape[1]
            
            with torch.no_grad() if use_ref_model else torch.enable_grad():
                policy_outputs = model(**full_inputs)
                logits = policy_outputs.logits
                
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
                
                # 只考虑生成部分的log概率
                response_log_probs = token_log_probs[0, response_start-1:-1]
                
                if return_per_token:
                    all_token_log_probs.append(response_log_probs)  # 🔥 保存每个token的log概率
                
                all_log_probs.append(response_log_probs.sum())
        
        if return_per_token:
            return all_token_log_probs  # 🔥 返回token级别的log概率列表
        else:
            return torch.stack(all_log_probs)  # 返回样本级别的总log概率
    
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
    
    def compute_relative_rewards(self, rewards: torch.Tensor, group_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算DAPO的相对奖励（与GRPO相同）
        返回：(相对奖励, 组内均值基线)
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
        
        if self.config.use_group_normalization:     # 同一个问题的不同答案之间做归一化
            group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
            relative_rewards = relative_rewards / group_std
        
        relative_rewards = relative_rewards.view(-1)
        group_baselines = group_baselines.repeat(1, group_size).view(-1)
        
        return relative_rewards, group_baselines
    
    def compute_kl_penalty_simple(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算KL散度惩罚（DAPO默认不使用）"""
        if not self.config.use_kl_penalty:
            return torch.zeros(len(prompts), device=self.device)
        
        current_log_probs = self.compute_log_probs(prompts, responses, use_ref_model=False)
        ref_log_probs = self.compute_log_probs(prompts, responses, use_ref_model=True)
        kl_divergence = current_log_probs - ref_log_probs
        
        return kl_divergence
    
    def compute_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """标准化优势"""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def compute_policy_loss_token_level(self, token_log_probs_list: List[torch.Tensor], 
                                       old_token_log_probs_list: List[torch.Tensor],
                                       advantages: torch.Tensor) -> torch.Tensor:
        """
        🔥 DAPO Token-Level Loss: 对每个token单独计算PPO损失
        
        Args:
            token_log_probs_list: 当前策略每个样本的token级别log概率列表
            old_token_log_probs_list: 旧策略每个样本的token级别log概率列表
            advantages: 每个样本的优势值 [batch_size]
        
        Returns:
            token级别的策略损失
        """
        total_token_loss = 0.0
        total_tokens = 0
        
        for i, (token_log_probs, old_token_log_probs) in enumerate(zip(token_log_probs_list, old_token_log_probs_list)):
            # 对每个token计算ratio和clipped surrogate loss
            advantage = advantages[i]  # 该样本的优势值
            
            # 计算每个token的概率比率
            token_ratios = torch.exp(token_log_probs - old_token_log_probs)  # [num_tokens]
            
            # 🔥 DAPO Clip-Higher: 非对称裁剪，对每个token应用
            surr1 = token_ratios * advantage
            surr2 = torch.clamp(token_ratios, 
                               1 - self.config.clip_range_low, 
                               1 + self.config.clip_range_high) * advantage
            
            # 对每个token取最小值，然后求和
            token_loss = -torch.min(surr1, surr2).sum()  # 对该样本的所有token求和
            
            total_token_loss += token_loss
            total_tokens += len(token_log_probs)
        
        # 返回平均token损失
        return total_token_loss / total_tokens
    
    def compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                          advantages: torch.Tensor, kl_penalty: torch.Tensor,
                          token_log_probs_list: List[torch.Tensor] = None,
                          old_token_log_probs_list: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        🔥 计算DAPO策略损失（使用Clip-Higher和Token-Level Loss）
        
        Args:
            log_probs: 样本级别的log概率（用于熵计算）
            old_log_probs: 旧策略的样本级别log概率
            advantages: 优势值
            kl_penalty: KL惩罚
            token_log_probs_list: 🔥 token级别的log概率列表（用于token-level loss）
            old_token_log_probs_list: 🔥 旧策略的token级别log概率列表
        """
        # 🔥 Token-Level Loss: 对每个token单独计算损失
        if self.config.use_token_level_loss and token_log_probs_list is not None:
            policy_loss = self.compute_policy_loss_token_level(
                token_log_probs_list, old_token_log_probs_list, advantages
            )
        else:
            # Sample-level loss (GRPO方式)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 
                               1 - self.config.clip_range_low, 
                               1 + self.config.clip_range_high) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
        
        # 熵损失（使用样本级别的log概率）
        entropy = -log_probs.mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        # KL损失（DAPO默认不使用）
        kl_loss = self.kl_coef * kl_penalty.mean() if self.config.use_kl_penalty else torch.tensor(0.0, device=self.device)
        
        return policy_loss, entropy_loss, kl_loss
    
    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """
        执行一步DAPO训练
        🔥 关键：每个prompt生成group_size个回复（可能通过动态采样增加）
        """
        self.dynamic_sampling_stats["total_questions"] += len(batch_prompts)
        
        all_prompts = []
        all_responses = []
        all_response_lengths = []
        all_raw_rewards = []
        
        # 🔥 为每个prompt生成group_size个回复并应用动态采样
        for prompt in batch_prompts:
            # 生成初始回复组（同一个prompt生成group_size次）
            responses = []
            response_lengths = []
            
            for _ in range(self.config.group_size):
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_response_length,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                responses.append(response)
                response_lengths.append(len(generated_ids))
            
            # 为这组回复创建对应的prompt列表
            prompts_repeated = [prompt] * len(responses)
            
            # 计算奖励，同一个prompt的不同answer计算奖励
            raw_rewards = self.compute_rewards(prompts_repeated, responses)
            
            # 🔥 应用软惩罚（如果启用）
            raw_rewards = self.apply_soft_overlong_punishment(raw_rewards, response_lengths)
            
            # 🔥 动态采样：如果所有奖励相同，继续采样
            if self.config.use_dynamic_sampling:
                reward_std = raw_rewards.std().item()
                extra_samples = 0
                
                while reward_std < 1e-6 and len(responses) < self.config.max_dynamic_samples:
                    # 采样额外的回复
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.policy_model.generate(
                            **inputs,
                            max_new_tokens=self.config.max_response_length,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                    
                    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                    extra_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    extra_length = len(generated_ids)
                    
                    responses.append(extra_response)
                    response_lengths.append(extra_length)
                    prompts_repeated.append(prompt)
                    
                    # 重新计算奖励
                    extra_reward = self.compute_rewards([prompt], [extra_response])
                    extra_reward = self.apply_soft_overlong_punishment(extra_reward, [extra_length])
                    raw_rewards = torch.cat([raw_rewards, extra_reward])
                    
                    reward_std = raw_rewards.std().item()
                    extra_samples += 1
                
                if extra_samples > 0:
                    self.dynamic_sampling_stats["resampled_questions"] += 1
                    self.dynamic_sampling_stats["avg_extra_samples"] += extra_samples
            
            # 🔥 过滤过长回复（如果启用）
            if self.config.use_overlong_filtering:
                valid_indices = [i for i, length in enumerate(response_lengths) 
                               if length < self.config.max_response_length]
                if len(valid_indices) > 0:
                    responses = [responses[i] for i in valid_indices]
                    prompts_repeated = [prompts_repeated[i] for i in valid_indices]
                    raw_rewards = raw_rewards[valid_indices]
                    response_lengths = [response_lengths[i] for i in valid_indices]
            
            all_prompts.extend(prompts_repeated)
            all_responses.extend(responses)
            all_response_lengths.extend(response_lengths)
            all_raw_rewards.append(raw_rewards)
            """
            all_prompts:          List[str],    长度=8  ['q1','q1','q1','q1','q2','q2','q2','q2']
            all_responses:        List[str],    长度=8  ['a1','a2','a3','a4','b1','b2','b3','b4']
            all_response_lengths: List[int],    长度=8  [10, 15, 12, 20, 8, 18, 14, 11]
            all_raw_rewards:      List[Tensor], 长度=2  [Tensor([...]), Tensor([...])]每个Tensor shape=[4]
            """
        
        if len(all_responses) == 0:
            logger.warning("所有回复都被过滤，跳过此步")
            return {}
        
        # 合并所有奖励 - 将每个prompt组的奖励合并成一个张量，用于后续计算相对奖励和优势函数
        all_raw_rewards = torch.cat(all_raw_rewards)
        """
        # 输出: Tensor, shape=[8]
        """
        
        # 计算相对奖励
        relative_rewards, group_baselines = self.compute_relative_rewards(
            all_raw_rewards, 
            group_size=len(all_responses) // len(batch_prompts)
        )
        
        # 截断数据
        all_prompts = all_prompts[:len(relative_rewards)]
        all_responses = all_responses[:len(relative_rewards)]
        all_response_lengths = all_response_lengths[:len(relative_rewards)]
        
        # 🔥 计算log概率（同时获取样本级别和token级别）
        log_probs = self.compute_log_probs(all_prompts, all_responses, return_per_token=False)
        old_token_log_probs_list = None
        
        # 🔥 如果使用token-level loss，获取token级别的log概率
        if self.config.use_token_level_loss:
            old_token_log_probs_list = self.compute_log_probs(
                all_prompts, all_responses, use_ref_model=False, return_per_token=True
            )
            # detach以避免梯度传播
            old_token_log_probs_list = [t.detach() for t in old_token_log_probs_list]
        
        # 计算KL散度
        kl_penalty = self.compute_kl_penalty_simple(all_prompts, all_responses)
        
        # 计算优势
        advantages = self.compute_advantages(relative_rewards)
        
        # 保存旧的log概率
        old_log_probs = log_probs.detach()
        
        # DAPO更新循环
        total_policy_loss = 0
        total_entropy_loss = 0
        total_kl_loss = 0
        
        for dapo_step in range(self.config.dapo_epochs):
            # 重新计算当前策略的log概率
            new_log_probs = self.compute_log_probs(all_prompts, all_responses, use_ref_model=False, return_per_token=False)
            
            # 🔥 如果使用token-level loss，获取当前策略的token级别log概率
            new_token_log_probs_list = None
            if self.config.use_token_level_loss:
                new_token_log_probs_list = self.compute_log_probs(
                    all_prompts, all_responses, use_ref_model=False, return_per_token=True
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
        
        # 计算动态采样统计
        resample_rate = (self.dynamic_sampling_stats["resampled_questions"] / 
                        max(1, self.dynamic_sampling_stats["total_questions"]))
        avg_extra = (self.dynamic_sampling_stats["avg_extra_samples"] / 
                    max(1, self.dynamic_sampling_stats["resampled_questions"]))
        
        return {
            "policy_loss": total_policy_loss / self.config.dapo_epochs,
            "entropy_loss": total_entropy_loss / self.config.dapo_epochs,
            "kl_loss": total_kl_loss / self.config.dapo_epochs,
            "raw_reward_mean": all_raw_rewards.mean().item(),
            "raw_reward_std": all_raw_rewards.std().item(),
            "relative_reward_mean": relative_rewards.mean().item(),
            "relative_reward_std": relative_rewards.std().item(),
            "advantage_mean": advantages.mean().item(),
            "avg_response_length": np.mean(all_response_lengths),
            "kl_divergence": kl_penalty.mean().item() if self.config.use_kl_penalty else 0.0,
            "dynamic_resample_rate": resample_rate,
            "avg_extra_samples": avg_extra
        }
    
    def train(self, train_dataset: DAPODataset):
        """主训练循环"""
        logger.info("开始DAPO训练...")
        
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
                
                if metrics:  # 如果不为空
                    epoch_metrics.append(metrics)
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "step": global_step,
                            "epoch": epoch,
                            **metrics
                        })
                    
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(global_step)
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Step {global_step}: {metrics}")
                
                global_step += 1
            
            # 计算epoch平均指标
            if epoch_metrics:
                avg_metrics = {}
                for key in epoch_metrics[0].keys():
                    avg_metrics[f"epoch_{key}"] = np.mean([m[key] for m in epoch_metrics])
                
                logger.info(f"Epoch {epoch + 1} 平均指标: {avg_metrics}")
                
                if self.config.use_wandb:
                    wandb.log(avg_metrics)
        
        logger.info("DAPO训练完成!")
        self.save_checkpoint("final")
    
    def save_checkpoint(self, step):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.policy_model.save_pretrained(os.path.join(checkpoint_dir, "policy"))
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
    config = DAPOConfig()
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    prompts = load_training_data()
    
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = DAPODataset(prompts, tokenizer, config.max_length)
    
    trainer = DAPOTrainer(config)
    
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()
