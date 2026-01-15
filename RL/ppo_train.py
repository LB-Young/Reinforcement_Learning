#!/usr/bin/env python3
"""
on-policy PPO训练脚本 - 使用Qwen2-0.5B作为策略模型和critic模型
支持RLHF训练流程
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

# 设置日志
logging.basicConfig(level=logging.INFO)  # 配置日志级别为INFO
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

@dataclass
class PPOConfig:
    """PPO训练配置"""
    # 模型配置
    policy_model_name: str = "E:\models\Qwen\Qwen3-0___6B"  # 策略模型名称，用于生成回复
    critic_model_name: str = "E:\models\Qwen\Qwen3-0___6BB"  # 价值函数模型名称，用于估计状态价值
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"  # 奖励模型名称，用于评估回复质量
    
    # 训练配置
    batch_size: int = 8  # 每个训练批次的样本数量
    mini_batch_size: int = 2  # PPO更新时的小批次大小，用于内存优化
    gradient_accumulation_steps: int = 4  # 梯度累积步数，模拟更大的批次大小
    learning_rate: float = 1e-5  # 策略模型的学习率
    critic_learning_rate: float = 5e-6  # 价值函数模型的学习率，通常比策略学习率小
    num_epochs: int = 3  # 总训练轮数
    max_length: int = 512  # 输入序列的最大长度
    
    # PPO超参数
    ppo_epochs: int = 4  # 每个批次数据的PPO更新次数
    clip_range: float = 0.2  # PPO裁剪范围，防止策略更新过大
    vf_coef: float = 0.1  # 价值函数损失的权重系数
    entropy_coef: float = 0.01  # 熵正则化系数，鼓励探索
    gamma: float = 0.99  # 折扣因子，用于计算未来奖励的现值
    lam: float = 0.95  # GAE(广义优势估计)的lambda参数
    kl_coef: float = 0.2  # KL散度惩罚系数，防止策略偏离reference model太远
    target_kl: float = 0.01  # 目标KL散度，用于自适应调整kl_coef
    adaptive_kl: bool = True  # 是否启用自适应KL系数调整
    use_exact_kl: bool = False  # 是否使用精确的KL散度计算（True=完整计算，False=简化估计）
    
    # 其他配置
    save_steps: int = 500  # 每隔多少步保存一次模型检查点
    eval_steps: int = 100  # 每隔多少步进行一次评估
    output_dir: str = "./ppo_output"  # 模型输出和检查点保存目录
    use_wandb: bool = True  # 是否使用wandb进行实验跟踪
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备，优先使用GPU

class PPODataset(Dataset):
    """PPO训练数据集"""
    
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

class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, config: PPOConfig):
        self.config = config  # 保存训练配置
        self.device = torch.device(config.device)  # 设置计算设备(CPU/GPU)
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)  # 加载预训练分词器
        if self.tokenizer.pad_token is None:  # 如果没有填充token
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用结束token作为填充token
        
        # 初始化模型
        self._init_models()  # 调用模型初始化方法
        
        # 初始化优化器
        self._init_optimizers()  # 调用优化器初始化方法
        
        # 初始化KL系数（用于自适应调整）
        self.kl_coef = config.kl_coef  # 当前KL散度惩罚系数
        
        # 初始化wandb
        if config.use_wandb:  # 如果启用wandb实验跟踪
            wandb.init(project="ppo-qwen", config=config.__dict__)  # 初始化wandb项目
    
    def _init_models(self):
        """初始化策略模型、critic模型和奖励模型"""
        logger.info("正在加载模型...")
        
        # 策略模型 (Qwen2-0.5B)
        self.policy_model = AutoModelForCausalLM.from_pretrained(  # 加载因果语言模型用作策略
            self.config.policy_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,  # GPU使用半精度，CPU使用单精度
            device_map="auto" if self.device.type == "cuda" else None  # GPU自动分配设备，CPU不分配
        )
        
        # Critic模型 (基于Qwen2-0.5B，添加value head)
        self.critic_model = AutoModelForCausalLM.from_pretrained(  # 加载critic模型基础架构
            self.config.critic_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,  # 数据类型设置
            device_map="auto" if self.device.type == "cuda" else None  # 设备映射设置
        )
        
        # 为critic模型添加value head
        hidden_size = self.critic_model.config.hidden_size  # 获取模型隐藏层大小
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)  # 创建线性层输出标量价值，并移到指定设备
        
        # 奖励模型
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(  # 加载序列分类模型用作奖励模型
            self.config.reward_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,  # 数据类型配置
            device_map="auto" if self.device.type == "cuda" else None  # 设备分配配置
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(self.config.reward_model_name)  # 奖励模型专用分词器
        
        # 保存参考策略模型
        self.ref_policy_model = AutoModelForCausalLM.from_pretrained(  # 加载参考策略模型，用于计算KL散度
            self.config.policy_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,  # 数据类型设置
            device_map="auto" if self.device.type == "cuda" else None  # 设备映射
        )
        self.ref_policy_model.eval()  # 设置为评估模式，不更新参数
        
        logger.info("模型加载完成")
    
    def _init_optimizers(self):
        """初始化优化器"""
        self.policy_optimizer = torch.optim.AdamW(  # 策略模型优化器，使用AdamW算法
            self.policy_model.parameters(),  # 策略模型的所有可训练参数
            lr=self.config.learning_rate  # 设置学习率
        )
        
        critic_params = list(self.critic_model.parameters()) + list(self.value_head.parameters())  # 合并critic模型和value head的参数
        self.critic_optimizer = torch.optim.AdamW(  # critic模型优化器
            critic_params,  # critic相关的所有参数
            lr=self.config.critic_learning_rate  # 使用专门的critic学习率
        )
    
    def generate_responses(self, prompts: List[str]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """生成回复并计算log概率"""
        self.policy_model.eval()  # 设置策略模型为评估模式
        
        responses = []  # 存储生成的回复
        all_log_probs = []  # 存储所有回复的log概率
        all_values = []  # 存储所有状态的价值估计
        
        for prompt in prompts:  # 遍历每个提示
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)  # 将提示编码为token
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  # 将输入移到指定设备
            
            # 生成回复
            with torch.no_grad():  # 禁用梯度计算以节省内存
                outputs = self.policy_model.generate(  # 使用策略模型生成文本
                    **inputs,
                    max_new_tokens=128,  # 最多生成128个新token
                    do_sample=True,  # 启用采样而非贪心解码
                    temperature=0.7,  # 控制生成随机性，值越小越确定
                    pad_token_id=self.tokenizer.pad_token_id,  # 设置填充token ID
                    return_dict_in_generate=True,  # 返回字典格式结果
                    output_scores=True  # 输出每个token的分数
                )
            
            # 解码生成的文本
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]  # 提取新生成的token ID
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)  # 解码为文本，跳过特殊token
            responses.append(response)  # 添加到回复列表
        
        # 批量计算log概率和价值
        log_probs, values = self.compute_log_probs_and_values(prompts, responses)  # 使用新的批量计算方法
        
        return responses, log_probs, values  # 返回回复、log概率和价值估计
    def compute_log_probs_and_values(self, prompts: List[str], responses: List[str], 
                                   use_ref_model: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量计算log概率和价值函数"""
        all_log_probs = []  # 存储所有log概率
        all_values = []  # 存储所有价值估计
        
        # 选择使用的模型
        model = self.ref_policy_model if use_ref_model else self.policy_model  # 根据参数选择参考模型或当前策略模型
        
        for prompt, response in zip(prompts, responses):  # 遍历提示和回复对
            # 拼接完整对话 - 用于价值函数评估完整对话质量
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
                
                # 计算价值函数（只有非参考模型时才计算）
                # 关键设计：使用完整对话(query+answer)作为critic输入
                # 原因：价值函数需要评估"给定prompt，生成这个response"的整体价值
                if not use_ref_model:  # 如果不是使用参考模型
                    critic_outputs = self.critic_model(**full_inputs, output_hidden_states=True)  # 获取critic输出
                    hidden_states = critic_outputs.hidden_states[-1]  # 取最后一层隐藏状态
                    values = self.value_head(hidden_states)  # 通过value head计算价值
                    # 使用最后一个token的表示来估计整个对话的价值
                    # 这相当于V(prompt, response) - 状态-动作价值函数
                    all_values.append(values[0, -1, 0])  # 取最后一个token的价值
                else:
                    all_values.append(torch.tensor(0.0, device=self.device))  # 参考模型时返回0
        
        return torch.stack(all_log_probs), torch.stack(all_values)  # 返回堆叠的张量
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
    
    def compute_kl_penalty(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算与参考模型的KL散度惩罚"""
        kl_divergences = []  # 存储每个样本的KL散度
        
        for prompt, response in zip(prompts, responses):  # 遍历每个样本
            # 拼接完整文本
            full_text = prompt + response  # 组合完整对话
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True)  # 编码
            full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}  # 移到设备
            
            # 计算prompt长度，确定response开始位置
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            response_start = prompt_inputs["input_ids"].shape[1]  # response开始的token位置
            response_end = full_inputs["input_ids"].shape[1]  # response结束位置
            
            with torch.no_grad():  # 禁用梯度计算
                # 当前策略的输出
                current_outputs = self.policy_model(**full_inputs)  # 当前策略模型
                current_logits = current_outputs.logits  # 获取logits
                
                # 参考模型的输出
                ref_outputs = self.ref_policy_model(**full_inputs)  # 参考模型
                ref_logits = ref_outputs.logits  # 获取参考logits
                
                # 计算概率分布（在词汇表维度上）
                current_probs = F.softmax(current_logits, dim=-1)  # 当前策略的概率分布 [seq_len, vocab_size]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)  # 参考模型的log概率分布
                
                # 只计算response部分的KL散度
                response_current_probs = current_probs[0, response_start-1:response_end-1, :]  # response部分的当前概率
                response_ref_log_probs = ref_log_probs[0, response_start-1:response_end-1, :]  # response部分的参考log概率
                
                # 计算KL散度：KL(current||ref) = Σ p_current * log(p_current / p_ref)
                # = Σ p_current * (log p_current - log p_ref)
                current_log_probs = torch.log(response_current_probs + 1e-10)  # 加小常数防止log(0)
                
                # 逐token计算KL散度，然后求和
                token_kl = response_current_probs * (current_log_probs - response_ref_log_probs)  # [seq_len, vocab_size]
                token_kl = token_kl.sum(dim=-1)  # 在词汇表维度求和 [seq_len]
                sequence_kl = token_kl.sum()  # 在序列维度求和，得到整个response的KL散度
                
                kl_divergences.append(sequence_kl)  # 添加到列表
        
        return torch.stack(kl_divergences)  # 返回所有样本的KL散度
    
    def compute_kl_penalty_simple(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算KL散度惩罚的简化版本（常用于实际实现）"""
        # 计算当前策略的log概率
        current_log_probs, _ = self.compute_log_probs_and_values(prompts, responses, use_ref_model=False)  # 当前策略模型的log概率
        
        # 计算参考模型的log概率
        ref_log_probs, _ = self.compute_log_probs_and_values(prompts, responses, use_ref_model=True)  # 参考模型的log概率
        
        # 简化的KL散度估计：对于已生成的序列，这是一个合理的近似
        # 因为我们已经从当前策略采样了动作，所以 E_{a~π_θ}[log π_θ - log π_ref] ≈ log π_θ(a) - log π_ref(a)
        kl_divergence = current_log_probs - ref_log_probs  # 简化的KL散度估计
        
        return kl_divergence  # 返回KL散度估计
       
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算优势函数和目标值"""
        # 简化版GAE计算
        advantages = rewards - values  # 计算优势 = 奖励 - 价值估计
        returns = rewards  # 目标回报等于奖励（简化版，实际应该考虑折扣）
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 标准化优势，减均值除标准差，加小常数防止除零
        
        return advantages, returns  # 返回标准化的优势和目标回报
    
    def compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                          advantages: torch.Tensor, kl_penalty: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算策略损失、熵损失和KL损失"""
        # 计算概率比率
        ratio = torch.exp(log_probs - old_log_probs)  # 新策略概率 / 旧策略概率
        
        # PPO clip损失
        surr1 = ratio * advantages  # 未裁剪的策略梯度目标
        surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages  # 裁剪后的目标，限制比率在[1-ε, 1+ε]范围内
        policy_loss = -torch.min(surr1, surr2).mean()  # 取两者最小值的负数作为损失（因为要最大化目标）
        
        # 计算熵损失（鼓励探索）
        entropy = -log_probs.mean()  # 简化的熵计算
        entropy_loss = -self.config.entropy_coef * entropy  # 熵损失，负号因为要最大化熵
        
        # 计算KL损失（正确位置：在损失函数中）
        kl_loss = self.kl_coef * kl_penalty.mean()  # KL散度惩罚损失
        
        return policy_loss, entropy_loss, kl_loss  # 返回策略损失、熵损失和KL损失
    
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
    def compute_value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """计算价值函数损失"""
        return F.mse_loss(values, returns)  # 使用均方误差损失，衡量价值估计与实际回报的差距
    
    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """执行一步PPO训练"""
        # 生成回复
        responses, log_probs, values = self.generate_responses(batch_prompts)  # 使用当前策略生成回复并计算相关值
        
        # 计算奖励
        rewards = self.compute_rewards(batch_prompts, responses)  # 使用奖励模型评估生成回复的质量
        
        # 计算KL散度惩罚
        # 提供两种计算方式：完整KL散度 vs 简化估计
        if hasattr(self.config, 'use_exact_kl') and self.config.use_exact_kl:  # 如果配置使用精确KL计算
            kl_penalty = self.compute_kl_penalty(batch_prompts, responses)  # 使用完整的KL散度计算
        else:
            kl_penalty = self.compute_kl_penalty_simple(batch_prompts, responses)  # 使用简化的KL估计（默认）
        
        # 计算优势和回报
        advantages, returns = self.compute_advantages(rewards, values)  # 计算优势函数和目标回报值（不包含KL惩罚）
        
        # 保存旧的log概率用于PPO
        old_log_probs = log_probs.detach()  # 分离梯度，作为PPO算法中的参考概率
        
        # PPO更新循环
        total_policy_loss = 0  # 累计策略损失
        total_value_loss = 0  # 累计价值损失
        total_entropy_loss = 0  # 累计熵损失
        total_kl_loss = 0  # 累计KL损失
        
        for ppo_step in range(self.config.ppo_epochs):  # 对同一批数据进行多次PPO更新
            # 重新计算当前策略的log概率和值
            new_log_probs, new_values = self.compute_log_probs_and_values(batch_prompts, responses, use_ref_model=False)  # 用更新后的策略重新计算
            
            # 计算重要性采样比率（用于调试）
            ratio = torch.exp(new_log_probs - old_log_probs)  # π_new / π_old
            ratio_mean = ratio.mean().item()  # 平均比率
            
            # 计算损失
            policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(new_log_probs, old_log_probs, advantages, kl_penalty)  # 计算PPO策略损失、熵损失和KL损失
            value_loss = self.compute_value_loss(new_values, returns)  # 计算价值函数损失
            
            # 总损失：PPO损失 + 价值损失 + 熵损失 + KL损失
            total_loss = policy_loss + self.config.vf_coef * value_loss + entropy_loss + kl_loss  # 组合所有损失项
            
            # 策略模型更新
            self.policy_optimizer.zero_grad()  # 清零策略模型梯度
            total_loss.backward(retain_graph=True)  # 反向传播总损失，保留计算图
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
            self.policy_optimizer.step()  # 更新策略模型参数 - 这里策略才开始改变！
            
            # 策略更新后，下一轮循环的new_log_probs才会与old_log_probs不同
            
            # Critic模型更新
            self.critic_optimizer.zero_grad()  # 清零critic模型梯度
            value_loss.backward()  # 反向传播价值损失
            torch.nn.utils.clip_grad_norm_(  # 对critic相关参数进行梯度裁剪
                list(self.critic_model.parameters()) + list(self.value_head.parameters()), 1.0
            )
            self.critic_optimizer.step()  # 更新critic模型参数
            
            total_policy_loss += policy_loss.item()  # 累加策略损失值
            total_value_loss += value_loss.item()  # 累加价值损失值
            total_entropy_loss += entropy_loss.item()  # 累加熵损失值
            total_kl_loss += kl_loss.item()  # 累加KL损失值
            
            # 记录每步的比率变化（调试信息）
            if ppo_step == 0:
                first_ratio = ratio_mean  # 第一步的比率应该接近1.0
        
        # 自适应调整KL系数
        self.update_kl_coef(kl_penalty)  # 根据当前KL散度调整惩罚系数
        
        return {  # 返回训练指标字典
            "policy_loss": total_policy_loss / self.config.ppo_epochs,  # 平均策略损失
            "value_loss": total_value_loss / self.config.ppo_epochs,  # 平均价值损失
            "entropy_loss": total_entropy_loss / self.config.ppo_epochs,  # 平均熵损失
            "kl_loss": total_kl_loss / self.config.ppo_epochs,  # 平均KL损失
            "reward_mean": rewards.mean().item(),  # 奖励均值
            "reward_std": rewards.std().item(),  # 奖励标准差
            "advantage_mean": advantages.mean().item(),  # 优势均值
            "kl_divergence": kl_penalty.mean().item(),  # 平均KL散度
            "kl_coef": self.kl_coef,  # 当前KL系数
            "first_step_ratio": first_ratio if 'first_ratio' in locals() else 1.0  # 第一步的重要性采样比率，应该接近1.0
        }
    
    def train(self, train_dataset: PPODataset):
        """主训练循环"""
        logger.info("开始PPO训练...")
        
        dataloader = DataLoader(  # 创建数据加载器
            train_dataset,  # 训练数据集
            batch_size=self.config.batch_size,  # 批次大小
            shuffle=True  # 每个epoch随机打乱数据顺序
        )
        
        global_step = 0  # 全局训练步数计数器
        
        for epoch in range(self.config.num_epochs):  # 遍历每个训练轮次
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_metrics = []  # 存储当前epoch的所有指标
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):  # 遍历每个批次，显示进度条
                # 提取prompts
                batch_prompts = batch["prompt"]  # 从批次中提取提示文本列表
                
                # 执行训练步骤
                metrics = self.train_step(batch_prompts)  # 执行一步PPO训练并获取指标
                epoch_metrics.append(metrics)  # 将指标添加到epoch指标列表
                
                # 记录指标
                if self.config.use_wandb:  # 如果启用wandb日志记录
                    wandb.log({  # 记录训练指标到wandb
                        "step": global_step,  # 当前步数
                        "epoch": epoch,  # 当前epoch
                        **metrics  # 展开所有训练指标
                    })
                
                # 保存检查点
                if global_step % self.config.save_steps == 0:  # 每隔指定步数保存检查点
                    self.save_checkpoint(global_step)  # 保存当前模型状态
                
                global_step += 1  # 增加全局步数计数
                
                # 打印进度
                if batch_idx % 10 == 0:  # 每10个批次打印一次进度
                    logger.info(f"Step {global_step}: {metrics}")
            
            # 计算epoch平均指标
            avg_metrics = {}  # 存储平均指标的字典
            for key in epoch_metrics[0].keys():  # 遍历指标的所有键
                avg_metrics[f"epoch_{key}"] = np.mean([m[key] for m in epoch_metrics])  # 计算每个指标在整个epoch的平均值
            
            logger.info(f"Epoch {epoch + 1} 平均指标: {avg_metrics}")
            
            if self.config.use_wandb:  # 如果启用wandb
                wandb.log(avg_metrics)  # 记录epoch平均指标
        
        logger.info("训练完成!")
        self.save_checkpoint("final")  # 保存最终模型检查点
    
    def save_checkpoint(self, step):
        """保存模型检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")  # 构建检查点目录路径
        os.makedirs(checkpoint_dir, exist_ok=True)  # 创建检查点目录，如果已存在则不报错
        
        # 保存策略模型
        self.policy_model.save_pretrained(os.path.join(checkpoint_dir, "policy"))  # 保存策略模型到policy子目录
        
        # 保存critic模型和value head
        self.critic_model.save_pretrained(os.path.join(checkpoint_dir, "critic"))  # 保存critic模型到critic子目录
        torch.save(self.value_head.state_dict(), os.path.join(checkpoint_dir, "value_head.pt"))  # 保存value head的状态字典
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)  # 保存分词器配置和词汇表
        
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
    config = PPOConfig()
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 加载训练数据
    prompts = load_training_data()
    
    # 创建数据集
    tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = PPODataset(prompts, tokenizer, config.max_length)
    
    # 创建训练器
    trainer = PPOTrainer(config)
    
    # 开始训练
    trainer.train(train_dataset)

if __name__ == "__main__":
    main()