#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) 训练脚本 - 基于Qwen2-0.5B
GRPO是PPO的变种，使用相对奖励和组内比较来优化策略
"""

import os  # 操作系统接口，用于文件路径操作
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数库
from torch.utils.data import DataLoader, Dataset  # 数据加载器和数据集基类
from transformers import (  # Hugging Face transformers库
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,  # 自动模型和分词器
    TrainingArguments, Trainer, pipeline  # 训练参数、训练器、管道
)
from datasets import load_dataset  # 数据集加载工具
import numpy as np  # 数值计算库
from typing import Dict, List, Optional, Tuple  # 类型提示
import logging  # 日志记录
from dataclasses import dataclass  # 数据类装饰器
import wandb  # 实验跟踪工具
from tqdm import tqdm  # 进度条显示
import json  # JSON数据处理
from accelerate import Accelerator  # 多GPU训练加速器

# 设置日志
logging.basicConfig(level=logging.INFO)  # 配置日志级别为INFO
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

@dataclass
class GRPOConfig:
    """GRPO训练配置"""
    # 模型配置
    policy_model_name: str = r"E:\models\Qwen\Qwen3-0___6B"  # 策略模型名称，用于生成回复
    reward_model_name: str = r"E:\models\reward-model-deberta-v3-large-v2"  # 奖励模型名称，用于评估回复质量
    # 注意：GRPO不需要critic模型！
    
    # 训练配置
    batch_size: int = 8  # 每个训练批次的样本数量
    mini_batch_size: int = 2  # GRPO更新时的小批次大小，用于内存优化
    gradient_accumulation_steps: int = 4  # 梯度累积步数，模拟更大的批次大小
    learning_rate: float = 1e-5  # 策略模型的学习率
    num_epochs: int = 3  # 总训练轮数
    max_length: int = 512  # 输入序列的最大长度
    
    # GRPO特有超参数
    grpo_epochs: int = 4  # 每个批次数据的GRPO更新次数
    clip_range: float = 0.2  # GRPO裁剪范围，防止策略更新过大
    entropy_coef: float = 0.01  # 熵正则化系数，鼓励探索
    kl_coef: float = 0.2  # KL散度惩罚系数，防止策略偏离reference model太远
    target_kl: float = 0.01  # 目标KL散度，用于自适应调整kl_coef
    adaptive_kl: bool = True  # 是否启用自适应KL系数调整
    
    # GRPO特有参数
    group_size: int = 4  # 每组的样本数量，用于相对比较
    use_group_normalization: bool = True  # 是否使用组内标准化
    
    # 多GPU配置
    use_multi_gpu: bool = True  # 是否使用多GPU训练
    mixed_precision: str = "fp16"  # 混合精度训练：fp16, bf16, no
    
    # 其他配置
    save_steps: int = 500  # 每隔多少步保存一次模型检查点
    eval_steps: int = 100  # 每隔多少步进行一次评估
    output_dir: str = "./grpo_output"  # 模型输出和检查点保存目录
    use_wandb: bool = True  # 是否使用wandb进行实验跟踪
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备，优先使用GPU

class GRPODataset(Dataset):
    """GRPO训练数据集"""
    
    def __init__(self, prompts: List[str], tokenizer, max_length: int = 512):
        self.prompts = prompts  # 存储所有的提示文本
        self.tokenizer = tokenizer  # 分词器，用于文本编码
        self.max_length = max_length  # 序列最大长度，超出部分会被截断
    
    def __len__(self):
        return len(self.prompts)  # 返回数据集大小
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]  # 获取指定索引的提示文本
        encoding = self.tokenizer(  # 对文本进行编码
            prompt,
            truncation=True,  # 启用截断，超出max_length的部分会被删除
            padding="max_length",  # 填充到最大长度，短序列用pad_token填充
            max_length=self.max_length,  # 设置最大序列长度
            return_tensors="pt"  # 返回PyTorch张量格式
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),  # 输入token的ID序列，去除批次维度
            "attention_mask": encoding["attention_mask"].squeeze(),  # 注意力掩码，标识哪些位置是真实token
            "prompt": prompt  # 原始提示文本，用于后续处理
        }

class GRPOTrainer:
    """GRPO训练器"""
    
    def __init__(self, config: GRPOConfig):
        self.config = config  # 保存训练配置
        
        # 🔥 初始化Accelerator用于多GPU训练
        if config.use_multi_gpu:
            self.accelerator = Accelerator(
                mixed_precision=config.mixed_precision,  # 混合精度训练
                gradient_accumulation_steps=config.gradient_accumulation_steps  # 梯度累积
            )
            self.device = self.accelerator.device  # 使用accelerator管理的设备
            logger.info(f"使用多GPU训练，设备数量: {self.accelerator.num_processes}")
        else:
            self.accelerator = None
            self.device = torch.device(config.device)  # 设置计算设备(CPU/GPU)
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)  # 加载预训练分词器
        if self.tokenizer.pad_token is None:  # 如果没有填充token
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用结束token作为填充token
        
        # 初始化模型
        self._init_models()  # 调用模型初始化方法
        
        # 初始化优化器
        self._init_optimizers()  # 调用优化器初始化方法
        
        # 🔥 使用Accelerator准备模型和优化器
        if self.accelerator:
            self.policy_model, self.policy_optimizer = self.accelerator.prepare(
                self.policy_model, self.policy_optimizer
            )
            # 奖励模型和参考模型不需要训练，只需要移到设备上
            self.reward_model = self.accelerator.prepare(self.reward_model)
            self.ref_policy_model = self.accelerator.prepare(self.ref_policy_model)
        
        # 初始化KL系数（用于自适应调整）
        self.kl_coef = config.kl_coef  # 当前KL散度惩罚系数
        
        # 初始化wandb（只在主进程）
        if config.use_wandb and (not self.accelerator or self.accelerator.is_main_process):
            wandb.init(project="grpo-qwen", config=config.__dict__)  # 初始化wandb项目
    
    def _init_models(self):
        """初始化策略模型和奖励模型（GRPO不需要critic模型）"""
        logger.info("正在加载模型...")
        
        # 🔥 多GPU训练时不使用device_map="auto"，让Accelerator管理设备分配
        if self.config.use_multi_gpu:
            # 策略模型 (Qwen2-0.5B)
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model_name,
                torch_dtype=torch.float16,  # 使用半精度
            )
            
            # 奖励模型
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reward_model_name,
                torch_dtype=torch.float16,
            )
            
            # 参考策略模型
            self.ref_policy_model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model_name,
                torch_dtype=torch.float16,
            )
        else:
            # 单GPU或CPU训练
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reward_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            self.ref_policy_model = AutoModelForCausalLM.from_pretrained(
                self.config.policy_model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
        
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.config.reward_model_name)
        # 为reward tokenizer设置pad_token
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token
        
        self.ref_policy_model.eval()  # 设置为评估模式，不更新参数
        
        logger.info("模型加载完成（GRPO无需critic模型）")
    
    def _init_optimizers(self):
        """初始化优化器（GRPO只需要策略优化器）"""
        self.policy_optimizer = torch.optim.AdamW(  # 策略模型优化器，使用AdamW算法
            self.policy_model.parameters(),  # 策略模型的所有可训练参数
            lr=self.config.learning_rate  # 设置学习率
        )
    
    def generate_responses(self, prompts: List[str]) -> Tuple[List[str], torch.Tensor, List[str]]:
        """
        🔥 GRPO核心：为每个prompt生成group_size个回复
        返回：(回复列表, log概率, 对应的prompt列表)
        """
        self.policy_model.eval()
        
        all_responses = []
        all_prompts_expanded = []
        
        # 🔥 关键：为每个prompt生成多个回复
        for prompt in prompts:
            for _ in range(self.config.group_size):
                # 编码输入
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 生成回复
                with torch.no_grad():
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,  # 必须启用采样才能生成不同的回复
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # 解码生成的文本
                generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                all_responses.append(response)
                all_prompts_expanded.append(prompt)
        
        # 批量计算log概率
        log_probs = self.compute_log_probs(all_prompts_expanded, all_responses)
        
        return all_responses, log_probs, all_prompts_expanded
    def compute_log_probs(self, prompts: List[str], responses: List[str], 
                         use_ref_model: bool = False) -> torch.Tensor:
        """批量计算log概率（GRPO不需要values）"""
        all_log_probs = []  # 存储所有log概率
        
        # 选择使用的模型
        model = self.ref_policy_model if use_ref_model else self.policy_model  # 根据参数选择参考模型或当前策略模型
        
        for prompt, response in zip(prompts, responses):  # 遍历提示和回复对
            # 拼接完整对话
            full_text = prompt + response  # 组合完整文本
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)  # 编码完整文本
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}  # 移到指定设备
            
            # 编码prompt以确定回复开始位置
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)  # 编码提示部分
            response_start = prompt_inputs["input_ids"].shape[1]  # 计算回复开始的token位置
            
            with torch.no_grad():  # 禁用梯度计算
                # 计算log概率
                policy_outputs = model(**full_inputs)  # 获取模型输出
                logits = policy_outputs.logits  # 提取logits
                
                # 计算token级别的log概率
                log_probs = F.log_softmax(logits, dim=-1)  # 应用log softmax
                token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)  # 收集实际token的log概率
                
                # 只考虑生成部分的log概率
                response_log_probs = token_log_probs[0, response_start-1:-1]  # 提取回复部分，排除最后一个token
                all_log_probs.append(response_log_probs.sum())  # 使用sum而不是mean，保持与token数量的关系
        
        return torch.stack(all_log_probs)  # 返回堆叠的张量

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """使用奖励模型计算奖励"""
        rewards = []  # 存储计算得到的奖励值
        
        for prompt, response in zip(prompts, responses):  # 遍历提示和回复对
            # 组合prompt和response
            full_text = f"{prompt} {response}"  # 拼接完整对话文本
            
            # 使用奖励模型tokenizer编码
            inputs = self.reward_tokenizer(  # 使用奖励模型专用分词器
                full_text,
                return_tensors="pt",  # 返回PyTorch张量
                padding=True,  # 启用填充
                truncation=True,  # 启用截断
                max_length=512  # 设置最大长度为512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 将输入移到指定设备
            
            # 计算奖励
            with torch.no_grad():  # 禁用梯度计算
                reward_outputs = self.reward_model(**inputs)  # 通过奖励模型获取输出
                reward = reward_outputs.logits[0, 0]  # 假设是二分类，取第一个类别的logit作为奖励
                rewards.append(reward)  # 添加到奖励列表
        
        return torch.stack(rewards)  # 将奖励列表转换为张量并返回
    
    def compute_relative_rewards(self, rewards: torch.Tensor, group_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GRPO的相对奖励 - GRPO的核心创新
        返回：(相对奖励, 组内均值基线)
        """
        if group_size is None:
            group_size = self.config.group_size
        
        batch_size = rewards.shape[0]
        if batch_size % group_size != 0:
            # 如果批次大小不能被组大小整除，截断到最大的完整组数
            num_complete_groups = batch_size // group_size
            rewards = rewards[:num_complete_groups * group_size]
            batch_size = rewards.shape[0]
        
        # 将奖励重塑为组的形状 [num_groups, group_size]
        rewards_grouped = rewards.view(-1, group_size)
        
        # 🔥 GRPO核心：计算每组的平均奖励作为基线（替代critic的value）
        group_baselines = rewards_grouped.mean(dim=1, keepdim=True)  # [num_groups, 1]
        
        # 🔥 计算相对奖励：每个样本的奖励减去组内平均值
        # 这就是优势函数：advantage = reward - baseline
        relative_rewards = rewards_grouped - group_baselines  # [num_groups, group_size]
        
        # 可选：组内标准化
        if self.config.use_group_normalization:
            group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
            relative_rewards = relative_rewards / group_std
        
        # 重新展平为原始形状
        relative_rewards = relative_rewards.view(-1)
        group_baselines = group_baselines.repeat(1, group_size).view(-1)
        
        return relative_rewards, group_baselines
    
    def compute_kl_penalty_simple(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算KL散度惩罚的简化版本（常用于实际实现）"""
        # 计算当前策略的log概率
        current_log_probs = self.compute_log_probs(prompts, responses, use_ref_model=False)  # 当前策略模型的log概率
        
        # 计算参考模型的log概率
        ref_log_probs = self.compute_log_probs(prompts, responses, use_ref_model=True)  # 参考模型的log概率
        
        # 简化的KL散度估计：对于已生成的序列，这是一个合理的近似
        # 因为我们已经从当前策略采样了动作，所以 E_{a~π_θ}[log π_θ - log π_ref] ≈ log π_θ(a) - log π_ref(a)
        kl_divergence = current_log_probs - ref_log_probs  # 简化的KL散度估计
        
        return kl_divergence  # 返回KL散度估计
    
    def compute_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        GRPO的优势函数计算（已经在compute_relative_rewards中完成）
        这里只需要标准化
        """
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化优势，减均值除标准差，加小常数防止除零
        
        return advantages  # 返回标准化的优势
    
    def compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                          advantages: torch.Tensor, kl_penalty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算GRPO策略损失、熵损失和KL损失（无value loss）"""
        # 计算概率比率
        ratio = torch.exp(log_probs - old_log_probs)  # 新策略概率 / 旧策略概率
        
        # GRPO clip损失 (与PPO相同的裁剪机制)
        surr1 = ratio * advantages  # 未裁剪的策略梯度目标？
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages  # 裁剪后的目标，限制比率在[1-ε, 1+ε]范围内
        policy_loss = -torch.min(surr1, surr2).mean()  # 取两者最小值的负数作为损失（因为要最大化目标）
        
        # 计算熵损失（鼓励探索）
        entropy = -log_probs.mean()  # 简化的熵计算
        entropy_loss = -self.config.entropy_coef * entropy  # 熵损失，负号因为要最大化熵
        
        # 计算KL损失
        kl_loss = self.kl_coef * kl_penalty.mean()  # KL散度惩罚损失
        
        return policy_loss, entropy_loss, kl_loss  # 返回策略损失、熵损失和KL损失（无value loss）
    
    def update_kl_coef(self, kl_divergence: torch.Tensor):
        """自适应调整KL散度系数"""
        if not self.config.adaptive_kl:  # 如果未启用自适应调整
            return
        
        mean_kl = kl_divergence.mean().item()  # 计算平均KL散度
        
        if mean_kl > 2.0 * self.config.target_kl:  # 如果KL散度过大
            self.kl_coef *= 1.5  # 增加KL惩罚系数
        elif mean_kl < 0.5 * self.config.target_kl:  # 如果KL散度过小
            self.kl_coef *= 0.5  # 减少KL惩罚系数
        
        # 限制KL系数的范围
        self.kl_coef = max(0.01, min(self.kl_coef, 1.0))  # 将KL系数限制在[0.01, 1.0]范围内
    
    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """
        执行一步GRPO训练
        🔥 关键：每个prompt会生成group_size个回复，然后计算组内相对奖励
        """
        # 🔥 生成回复：每个prompt生成group_size个回复
        # 例如：batch_prompts=['q1', 'q2'], group_size=4
        # 返回：responses=['a1_1', 'a1_2', 'a1_3', 'a1_4', 'a2_1', 'a2_2', 'a2_3', 'a2_4']
        responses, log_probs, prompts_expanded = self.generate_responses(batch_prompts)
        
        # 计算原始奖励
        raw_rewards = self.compute_rewards(prompts_expanded, responses)
        
        # 🔥 GRPO核心：计算相对奖励（优势）和基线
        # 将rewards按group_size分组，计算组内相对奖励
        # 例如：[r1_1, r1_2, r1_3, r1_4] -> 减去组内均值 -> [adv1_1, adv1_2, adv1_3, adv1_4]
        relative_rewards, group_baselines = self.compute_relative_rewards(raw_rewards)
        
        # 截断数据以匹配相对奖励的长度
        prompts_truncated = prompts_expanded[:len(relative_rewards)]
        responses_truncated = responses[:len(relative_rewards)]
        log_probs_truncated = log_probs[:len(relative_rewards)]
        
        # 计算KL散度惩罚
        kl_penalty = self.compute_kl_penalty_simple(prompts_truncated, responses_truncated)
        
        # 🔥 GRPO的优势函数就是相对奖励（已经减去了组内均值基线）
        advantages = self.compute_advantages(relative_rewards)
        
        # 保存旧的log概率用于GRPO
        old_log_probs = log_probs_truncated.detach()
        
        # GRPO更新循环
        total_policy_loss = 0
        total_entropy_loss = 0
        total_kl_loss = 0
        
        for grpo_step in range(self.config.grpo_epochs):
            # 重新计算当前策略的log概率
            new_log_probs = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref_model=False)
            
            # 计算重要性采样比率（用于调试）
            ratio = torch.exp(new_log_probs - old_log_probs)  # π_new / π_old
            ratio_mean = ratio.mean().item()  # 平均比率
            
            # 计算损失（GRPO无value loss）
            policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(new_log_probs, old_log_probs, advantages, kl_penalty)
            
            # 🔥 总损失：GRPO损失 + 熵损失 + KL损失（无value loss）
            total_loss = policy_loss + entropy_loss + kl_loss
            
            # 策略模型更新
            self.policy_optimizer.zero_grad()  # 清零策略模型梯度
            
            # 🔥 使用Accelerator的backward或普通backward
            if self.accelerator:
                self.accelerator.backward(total_loss)  # Accelerator管理的反向传播
            else:
                total_loss.backward()  # 普通反向传播
            
            # 梯度裁剪
            if self.accelerator:
                self.accelerator.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            
            self.policy_optimizer.step()  # 更新策略模型参数
            
            total_policy_loss += policy_loss.item()  # 累加策略损失值
            total_entropy_loss += entropy_loss.item()  # 累加熵损失值
            total_kl_loss += kl_loss.item()  # 累加KL损失值
            
            # 记录每步的比率变化（调试信息）
            if grpo_step == 0:
                first_ratio = ratio_mean  # 第一步的比率应该接近1.0
        
        # 自适应调整KL系数
        self.update_kl_coef(kl_penalty)  # 根据当前KL散度调整惩罚系数
        
        return {  # 返回训练指标字典
            "policy_loss": total_policy_loss / self.config.grpo_epochs,  # 平均策略损失
            "entropy_loss": total_entropy_loss / self.config.grpo_epochs,  # 平均熵损失
            "kl_loss": total_kl_loss / self.config.grpo_epochs,  # 平均KL损失
            "raw_reward_mean": raw_rewards.mean().item(),  # 原始奖励均值
            "raw_reward_std": raw_rewards.std().item(),  # 原始奖励标准差
            "relative_reward_mean": relative_rewards.mean().item(),  # 相对奖励均值（应接近0）
            "relative_reward_std": relative_rewards.std().item(),  # 相对奖励标准差
            "group_baseline_mean": group_baselines.mean().item(),  # 组内基线均值
            "advantage_mean": advantages.mean().item(),  # 优势均值（标准化后应接近0）
            "kl_divergence": kl_penalty.mean().item(),  # 平均KL散度
            "kl_coef": self.kl_coef,  # 当前KL系数
            "first_step_ratio": first_ratio if 'first_ratio' in locals() else 1.0  # 第一步的重要性采样比率
        }
    
    def train(self, train_dataset: GRPODataset):
        """主训练循环"""
        logger.info("开始GRPO训练...")
        
        dataloader = DataLoader(  # 创建数据加载器
            train_dataset,  # 训练数据集
            batch_size=self.config.batch_size,  # 批次大小
            shuffle=True  # 每个epoch随机打乱数据顺序
        )
        
        # 🔥 使用Accelerator准备dataloader
        if self.accelerator:
            dataloader = self.accelerator.prepare(dataloader)
        
        global_step = 0  # 全局训练步数计数器
        
        for epoch in range(self.config.num_epochs):  # 遍历每个训练轮次
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_metrics = []  # 存储当前epoch的所有指标
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):  # 遍历每个批次，显示进度条
                # 提取prompts
                batch_prompts = batch["prompt"]  # 从批次中提取提示文本列表
                
                # 执行训练步骤
                metrics = self.train_step(batch_prompts)  # 执行一步GRPO训练并获取指标
                epoch_metrics.append(metrics)  # 将指标添加到epoch指标列表
                
                # 记录指标（只在主进程）
                if self.config.use_wandb and (not self.accelerator or self.accelerator.is_main_process):
                    wandb.log({  # 记录训练指标到wandb
                        "step": global_step,  # 当前步数
                        "epoch": epoch,  # 当前epoch
                        **metrics  # 展开所有训练指标
                    })
                
                # 保存检查点（只在主进程）
                if global_step % self.config.save_steps == 0:
                    if not self.accelerator or self.accelerator.is_main_process:
                        self.save_checkpoint(global_step)
                
                global_step += 1  # 增加全局步数计数
                
                # 打印进度（只在主进程）
                if batch_idx % 10 == 0:
                    if not self.accelerator or self.accelerator.is_main_process:
                        logger.info(f"Step {global_step}: {metrics}")
            
            # 计算epoch平均指标
            avg_metrics = {}  # 存储平均指标的字典
            for key in epoch_metrics[0].keys():  # 遍历指标的所有键
                avg_metrics[f"epoch_{key}"] = np.mean([m[key] for m in epoch_metrics])  # 计算每个指标在整个epoch的平均值
            
            if not self.accelerator or self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1} 平均指标: {avg_metrics}")
            
            if self.config.use_wandb and (not self.accelerator or self.accelerator.is_main_process):
                wandb.log(avg_metrics)  # 记录epoch平均指标
        
        if not self.accelerator or self.accelerator.is_main_process:
            logger.info("GRPO训练完成!")
            self.save_checkpoint("final")  # 保存最终模型检查点
    
    def save_checkpoint(self, step):
        """保存模型检查点（GRPO只需保存策略模型）"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 🔥 使用Accelerator的unwrap_model获取原始模型
        if self.accelerator:
            unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
            unwrapped_model.save_pretrained(
                os.path.join(checkpoint_dir, "policy"),
                save_function=self.accelerator.save  # 使用accelerator的保存函数
            )
        else:
            self.policy_model.save_pretrained(os.path.join(checkpoint_dir, "policy"))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"检查点已保存到 {checkpoint_dir}")

def load_training_data() -> List[str]:
    """加载训练数据"""
    logger.info("正在加载训练数据...")
    
    try:
        # 使用Anthropic HH数据集作为示例
        dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")  # 取前1000条用于演示
        
        prompts = []
        for item in dataset:
            # 提取human的问题作为prompt
            conversation = item["chosen"]
            if conversation.startswith("Human:"):
                # 提取Human的部分作为prompt
                human_part = conversation.split("Assistant:")[0].replace("Human:", "").strip()
                if human_part:
                    prompts.append(human_part)
        
        logger.info(f"加载了 {len(prompts)} 个训练样本")
        return prompts
    
    except Exception as e:
        logger.warning(f"无法加载HH数据集: {e}")
        # 使用示例数据
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
        ] * 50  # 重复以增加数据量

def main():
    """主函数"""
    # 创建配置
    config = GRPOConfig()
    
    # 验证模型路径是否存在
    if not os.path.exists(config.policy_model_name):
        raise FileNotFoundError(f"策略模型路径不存在: {config.policy_model_name}")
    if not os.path.exists(config.reward_model_name):
        raise FileNotFoundError(f"奖励模型路径不存在: {config.reward_model_name}")
    
    logger.info(f"策略模型路径: {config.policy_model_name}")
    logger.info(f"奖励模型路径: {config.reward_model_name}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载训练数据
    prompts = load_training_data()
    
    # 创建数据集
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = GRPODataset(prompts, tokenizer, config.max_length)
    
    # 创建训练器
    trainer = GRPOTrainer(config)
    
    # 开始训练
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()