#!/usr/bin/env python3
# author: YoungL
# date: 2026/01/20
# email: lby15356@gmail.com

"""
PPO with Experience Replay (v2)
新增功能（基于 v1）：
1. 经验回放缓冲区 (Experience Replay Buffer)
2. 优先级采样 (Priority Sampling) - 可选
3. 重要性权重修正 (Importance Sampling Correction)
4. 缓冲区管理策略
5. 样本效率统计

注：
1. 经验回放可以提高样本效率，但需要注意 off-policy 修正
2. PPO 本质是 on-policy 算法，使用经验回放时需要谨慎处理重要性权重
"""

import os
import sys
import json
import gc
import random
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Deque

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import shutil
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import plot_ppo_metrics

# ===================== 1. 配置 ==========================
# Model paths
ACTOR_MODEL = r"E:\models\Qwen\Qwen3-0___6B"
CRITIC_MODEL = r"E:\models\Qwen\Qwen3-0___6B"
REWARD_MODEL = r"E:\models\reward-model-deberta-v3-large-v2"

train_datasets = [
    {
        "path":r"E:\datasets\gsm8k\main\train-00000-of-00001.parquet",
        "type":"parquet",
        "input":"question",
        "output":"answer"
    }
]

# Training hyperparameters
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LEARNING_RATE = 1e-6
NUM_EPOCHES = 1
BATCH_SIZE = 4
GROUP_SIZE = 1
GROUP_EPOCHES = 4
CLIP_RANGE = 0.2
ENTROPY_COEF = 0.01
KL_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 1.0

# v1 features
GRADIENT_ACCUMULATION_STEPS = 1
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 100
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_THRESHOLD = 0.001
SAVE_EVERY_N_STEPS = 100
EVAL_EVERY_N_STEPS = 50
USE_WANDB = False
WANDB_PROJECT = "ppo-training"

# 🔥 v2 新增：经验回放配置
USE_REPLAY_BUFFER = True              # 是否使用经验回放
REPLAY_BUFFER_SIZE = 1000             # 缓冲区大小
MIN_REPLAY_SIZE = 100                 # 开始训练的最小样本数
REPLAY_SAMPLE_SIZE = 32               # 每次从缓冲区采样的数量
USE_PRIORITY_SAMPLING = False         # 是否使用优先级采样
PRIORITY_ALPHA = 0.6                  # 优先级指数
PRIORITY_BETA = 0.4                   # 重要性采样修正系数
PRIORITY_BETA_INCREMENT = 0.001       # Beta 增长率
MAX_REPLAY_REUSE = 4                  # 每个样本最多重用次数
STALENESS_THRESHOLD = 100             # 样本过期阈值（步数）

OUTPUT_DIR = r"E:\projects\train_related\trained_model\rl_exprement\grpo_output\ppo_gsm8k_v2"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ===================== 2. 经验数据结构 =====================
@dataclass
class Experience:
    """单个经验样本"""
    prompt: str
    response: str
    reward: float
    value: float
    advantage: float
    old_log_prob: torch.Tensor  # Token-level log probs
    mask: torch.Tensor          # Response mask
    step: int                   # 生成时的训练步数
    priority: float = 1.0       # 优先级（用于优先级采样）
    reuse_count: int = 0        # 重用次数

# ===================== 3. 经验回放缓冲区 =====================
class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int, use_priority: bool = False):
        self.capacity = capacity
        self.use_priority = use_priority
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self.priorities: Deque[float] = deque(maxlen=capacity) if use_priority else None
        
    def add(self, experience: Experience):
        """添加经验"""
        self.buffer.append(experience)
        if self.use_priority:
            self.priorities.append(experience.priority)
    
    def add_batch(self, experiences: List[Experience]):
        """批量添加经验"""
        for exp in experiences:
            self.add(exp)
    
    def sample(self, batch_size: int, current_step: int, beta: float = 0.4) -> Tuple[List[Experience], Optional[np.ndarray]]:
        """采样经验"""
        # 过滤过期和过度重用的样本
        valid_experiences = [
            exp for exp in self.buffer
            if (current_step - exp.step) < STALENESS_THRESHOLD
            and exp.reuse_count < MAX_REPLAY_REUSE
        ]
        
        if len(valid_experiences) == 0:
            return [], None
        
        batch_size = min(batch_size, len(valid_experiences))
        
        if self.use_priority:
            # 优先级采样
            priorities = np.array([exp.priority for exp in valid_experiences])
            probs = priorities ** PRIORITY_ALPHA
            probs /= probs.sum()
            
            indices = np.random.choice(len(valid_experiences), batch_size, p=probs, replace=False)
            sampled = [valid_experiences[i] for i in indices]
            
            # 计算重要性采样权重
            weights = (len(valid_experiences) * probs[indices]) ** (-beta)
            weights /= weights.max()  # 归一化
            
            return sampled, weights
        else:
            # 均匀采样
            sampled = random.sample(valid_experiences, batch_size)
            return sampled, None
    
    def update_priority(self, experience: Experience, new_priority: float):
        """更新优先级"""
        experience.priority = new_priority
    
    def __len__(self):
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, float]:
        """获取缓冲区统计信息"""
        if len(self.buffer) == 0:
            return {}
        
        reuse_counts = [exp.reuse_count for exp in self.buffer]
        ages = [exp.step for exp in self.buffer]
        
        return {
            "buffer_size": len(self.buffer),
            "avg_reuse_count": np.mean(reuse_counts),
            "max_reuse_count": np.max(reuse_counts),
            "avg_age": np.mean(ages),
            "oldest_sample": np.min(ages)
        }

# ===================== 4. 数据集处理 =====================
class PPODataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

# ===================== 5. 日志记录器 =====================
class TrainingLogger:
    """训练日志记录器"""
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.metrics_file = os.path.join(log_dir, "metrics.jsonl")
        
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"
        print(log_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """记录指标到 JSONL 文件"""
        metrics_entry = {"step": step, **metrics}
        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_entry) + "\n")

# ===================== 6. PPOTrainer with Replay ===================
class PPOTrainerWithReplay:
    def __init__(self, resume_from_checkpoint: Optional[str] = None):
        # 设备配置
        self.device_policy = torch.device("cuda:0")
        self.device_ref = torch.device("cuda:0")
        self.device_critic = torch.device("cuda:1")
        self.device_reward = torch.device("cuda:1")
        
        # 初始化日志记录器
        self.logger = TrainingLogger(LOG_DIR)
        self.logger.log("初始化 PPO Trainer with Replay Buffer...")
        
        # 🔥 初始化经验回放缓冲区
        if USE_REPLAY_BUFFER:
            self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, USE_PRIORITY_SAMPLING)
            self.priority_beta = PRIORITY_BETA
            self.logger.log(f"经验回放缓冲区已初始化: capacity={REPLAY_BUFFER_SIZE}, priority={USE_PRIORITY_SAMPLING}")
        else:
            self.replay_buffer = None
        
        # 初始化指标记录
        self.metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'reward': [],
            'advantage': [],
            'entropy': [],
            'kl_divergence': [],
            'learning_rate': [],
            # 🔥 新增：回放缓冲区指标
            'buffer_size': [],
            'replay_ratio': [],  # 回放样本占比
            'avg_sample_reuse': [],
            'sample_efficiency': []  # 样本效率
        }
        
        # 训练状态
        self.global_step = 0
        self.best_reward = float('-inf')
        self.patience_counter = 0
        self.total_samples_generated = 0  # 总生成样本数
        self.total_training_samples = 0   # 总训练样本数（包括回放）
        
        # 初始化 Wandb（可选）
        if USE_WANDB:
            try:
                import wandb
                wandb.init(project=WANDB_PROJECT, config={
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "use_replay_buffer": USE_REPLAY_BUFFER,
                    "replay_buffer_size": REPLAY_BUFFER_SIZE,
                    "use_priority_sampling": USE_PRIORITY_SAMPLING
                })
                self.wandb = wandb
                self.logger.log("Wandb 初始化成功")
            except ImportError:
                self.logger.log("Wandb 未安装，跳过", "WARNING")
                self.wandb = None
        else:
            self.wandb = None
        
        # 加载模型
        self._load_models()
        
        # 初始化优化器
        self.policy_optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.AdamW(self.critic_model.parameters(), lr=LEARNING_RATE)
        
        # 初始化学习率调度器
        if USE_LR_SCHEDULER:
            if LR_SCHEDULER_TYPE == "cosine":
                self.policy_scheduler = CosineAnnealingLR(self.policy_optimizer, T_max=1000, eta_min=1e-7)
                self.critic_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=1000, eta_min=1e-7)
            elif LR_SCHEDULER_TYPE == "plateau":
                self.policy_scheduler = ReduceLROnPlateau(self.policy_optimizer, mode='max', patience=3, factor=0.5)
                self.critic_scheduler = ReduceLROnPlateau(self.critic_optimizer, mode='max', patience=3, factor=0.5)
            self.logger.log(f"使用学习率调度器: {LR_SCHEDULER_TYPE}")
        else:
            self.policy_scheduler = None
            self.critic_scheduler = None
        
        # 从检查点恢复（如果提供）
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
        
        self.logger.log("PPO Trainer with Replay 初始化完成")
    
    def _load_models(self):
        """加载所有模型"""
        self.logger.log("加载模型...")
        
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            ACTOR_MODEL, torch_dtype=DTYPE, device_map={"": self.device_policy}
        )
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            ACTOR_MODEL, torch_dtype=DTYPE, device_map={"": self.device_ref}
        )
        self.reference_model.eval()
        self.policy_tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL, trust_remote_code=True)
        if self.policy_tokenizer.pad_token_id is None:
            self.policy_tokenizer.pad_token_id = self.policy_tokenizer.eos_token_id

        self.critic_model = AutoModelForCausalLM.from_pretrained(
            CRITIC_MODEL, torch_dtype=DTYPE, device_map={"": self.device_critic}
        )
        self.critic_tokenizer = AutoTokenizer.from_pretrained(CRITIC_MODEL, trust_remote_code=True)
        if self.critic_tokenizer.pad_token_id is None:
            self.critic_tokenizer.pad_token_id = self.critic_tokenizer.eos_token_id

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL, torch_dtype=DTYPE, device_map={"": self.device_reward}
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL, trust_remote_code=True)
        
        self.logger.log("模型加载完成")

    def generate_response(self, prompts: List[str]) -> Tuple[List[str], List[str]]:
        """生成响应"""
        self.policy_model.eval()
        all_responses, all_prompts = [], []

        for prompt in prompts:
            inputs = self.policy_tokenizer(prompt, return_tensors='pt').to(self.device_policy)
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=GROUP_SIZE,
                    pad_token_id=self.policy_tokenizer.pad_token_id
                )
            gen_ids = outputs[:, inputs['input_ids'].shape[1]:]
            responses = self.policy_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_responses.extend(responses)
            all_prompts.extend([prompt]*GROUP_SIZE)
            
            del inputs, outputs, gen_ids
        
        torch.cuda.empty_cache()
        return all_responses, all_prompts

    def get_token_log_probs(self, model, prompts: List[str], responses: List[str], device, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 Token 级别的 log_probs 并返回 Mask"""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        
        with torch.no_grad() if model.training == False else torch.enable_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lens):
            mask[i, p_len:] = (labels[i, p_len:] != tokenizer.pad_token_id)
        
        del inputs, outputs, logits, labels
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return token_log_probs, mask

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """使用 reward_model 计算奖励"""
        rewards = []
        for p, r in zip(prompts, responses):
            inputs = self.reward_tokenizer(p + r, return_tensors="pt", truncation=True).to(self.device_reward)
            with torch.no_grad():
                reward = self.reward_model(**inputs).logits[0, 0]
                rewards.append(reward)
            del inputs
        
        result = torch.stack(rewards).to(self.device_policy)
        torch.cuda.empty_cache()
        return result

    def compute_values(self, prompts: List[str], responses: List[str], requires_grad: bool = False) -> torch.Tensor:
        """使用 critic_model 计算状态价值"""
        values = []
        for p, r in zip(prompts, responses):
            full_text = p + r
            inputs = self.critic_tokenizer(full_text, return_tensors="pt", truncation=True).to(self.device_critic)
            if requires_grad:
                outputs = self.critic_model(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.critic_model(**inputs)
            value = outputs.logits[0, -1, :].mean()
            values.append(value)
            del inputs, outputs
        
        result = torch.stack(values).to(self.device_policy)
        torch.cuda.empty_cache()
        return result
    
    def create_experiences(self, prompts: List[str], responses: List[str]) -> List[Experience]:
        """🔥 创建经验样本"""
        # 计算奖励和价值
        rewards = self.compute_rewards(prompts, responses)
        values = self.compute_values(prompts, responses, requires_grad=False)
        advantages = rewards - values
        
        # 获取 log_probs 和 mask
        with torch.no_grad():
            old_log_probs, mask = self.get_token_log_probs(
                self.policy_model, prompts, responses, self.device_policy, self.policy_tokenizer
            )
        
        # 创建经验对象
        experiences = []
        for i, (p, r) in enumerate(zip(prompts, responses)):
            exp = Experience(
                prompt=p,
                response=r,
                reward=rewards[i].item(),
                value=values[i].item(),
                advantage=advantages[i].item(),
                old_log_prob=old_log_probs[i].detach().cpu(),
                mask=mask[i].detach().cpu(),
                step=self.global_step,
                priority=abs(advantages[i].item()) + 1e-6  # 使用优势的绝对值作为优先级
            )
            experiences.append(exp)
        
        return experiences

    def train_on_experiences(self, experiences: List[Experience], is_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """🔥 使用经验样本进行训练"""
        if len(experiences) == 0:
            return {}
        
        # 提取数据
        prompts = [exp.prompt for exp in experiences]
        responses = [exp.response for exp in experiences]
        old_log_probs = torch.stack([exp.old_log_prob for exp in experiences]).to(self.device_policy)
        masks = torch.stack([exp.mask for exp in experiences]).to(self.device_policy)
        advantages = torch.tensor([exp.advantage for exp in experiences], device=self.device_policy)
        rewards = torch.tensor([exp.reward for exp in experiences], device=self.device_policy)
        
        # 重要性采样权重
        if is_weights is not None:
            is_weights = torch.tensor(is_weights, device=self.device_policy, dtype=torch.float32)
        else:
            is_weights = torch.ones(len(experiences), device=self.device_policy)
        
        # 获取参考 log_probs
        with torch.no_grad():
            ref_log_probs, _ = self.get_token_log_probs(
                self.reference_model, prompts, responses, self.device_ref, self.policy_tokenizer
            )
            ref_log_probs = ref_log_probs.to(self.device_policy)
        
        # PPO更新循环
        self.policy_model.train()
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_div = 0
        
        for epoch_idx in range(GROUP_EPOCHES):
            # 获取新的 log_probs
            new_log_probs, _ = self.get_token_log_probs(
                self.policy_model, prompts, responses, self.device_policy, self.policy_tokenizer
            )
            
            # 计算策略损失 (PPO Clip)
            log_ratio = (new_log_probs - old_log_probs) * masks
            ratio = torch.exp(log_ratio)
            adv_t = advantages.unsqueeze(1)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv_t
            policy_loss = -torch.min(surr1, surr2)
            
            # KL 惩罚
            kl_div = (new_log_probs - ref_log_probs)
            
            # 应用重要性采样权重
            loss_map = (policy_loss + KL_COEF * kl_div) * masks
            weighted_loss = loss_map * is_weights.unsqueeze(1)
            step_policy_loss = weighted_loss.sum() / masks.sum()
            
            # 梯度累积
            step_policy_loss = step_policy_loss / GRADIENT_ACCUMULATION_STEPS
            step_policy_loss.backward()
            
            if (epoch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), MAX_GRAD_NORM)
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad()
            
            # 计算价值损失
            new_values = self.compute_values(prompts, responses, requires_grad=True)
            value_loss = F.mse_loss(new_values, rewards, reduction='none')
            weighted_value_loss = (value_loss * is_weights).mean() * VALUE_LOSS_COEF
            weighted_value_loss = weighted_value_loss / GRADIENT_ACCUMULATION_STEPS
            weighted_value_loss.backward()
            
            if (epoch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()
            
            total_policy_loss += step_policy_loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_value_loss += weighted_value_loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_kl_div += kl_div.mean().item()
            
            # 🔥 更新经验的优先级和重用次数
            if USE_PRIORITY_SAMPLING:
                td_errors = (new_values - rewards).abs().detach().cpu().numpy()
                for i, exp in enumerate(experiences):
                    new_priority = td_errors[i] + 1e-6
                    self.replay_buffer.update_priority(exp, new_priority)
            
            # 清理显存
            del new_log_probs, log_ratio, ratio, surr1, surr2, policy_loss, kl_div
            del new_values, value_loss, weighted_value_loss
            torch.cuda.empty_cache()
            gc.collect()
        
        # 更新重用次数
        for exp in experiences:
            exp.reuse_count += 1
        
        metrics = {
            "policy_loss": total_policy_loss / GROUP_EPOCHES,
            "value_loss": total_value_loss / GROUP_EPOCHES,
            "kl_divergence": total_kl_div / GROUP_EPOCHES,
        }
        
        return metrics

    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """🔥 执行一步训练（带经验回放）"""
        # 1. 生成新经验
        responses, prompts = self.generate_response(batch_prompts)
        new_experiences = self.create_experiences(prompts, responses)
        self.total_samples_generated += len(new_experiences)
        
        # 2. 添加到回放缓冲区
        if USE_REPLAY_BUFFER and self.replay_buffer:
            self.replay_buffer.add_batch(new_experiences)
        
        # 3. 训练
        all_metrics = []
        
        # 3.1 使用新生成的经验训练
        new_metrics = self.train_on_experiences(new_experiences)
        if new_metrics:
            all_metrics.append(new_metrics)
            self.total_training_samples += len(new_experiences)
        
        # 3.2 从回放缓冲区采样并训练
        if USE_REPLAY_BUFFER and self.replay_buffer and len(self.replay_buffer) >= MIN_REPLAY_SIZE:
            # 更新 beta（重要性采样修正系数）
            if USE_PRIORITY_SAMPLING:
                self.priority_beta = min(1.0, self.priority_beta + PRIORITY_BETA_INCREMENT)
            
            # 从缓冲区采样
            replay_experiences, is_weights = self.replay_buffer.sample(
                REPLAY_SAMPLE_SIZE, 
                self.global_step,
                self.priority_beta if USE_PRIORITY_SAMPLING else 0.4
            )
            
            if len(replay_experiences) > 0:
                replay_metrics = self.train_on_experiences(replay_experiences, is_weights)
                if replay_metrics:
                    all_metrics.append(replay_metrics)
                    self.total_training_samples += len(replay_experiences)
        
        # 4. 聚合指标
        if len(all_metrics) == 0:
            return {}
        
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            aggregated_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # 5. 添加额外指标
        aggregated_metrics.update({
            "reward": np.mean([exp.reward for exp in new_experiences]),
            "advantage": np.mean([exp.advantage for exp in new_experiences]),
            "entropy": 0.0,  # 简化，可以计算
            "learning_rate": self.policy_optimizer.param_groups[0]['lr'],
            "total_loss": aggregated_metrics["policy_loss"] + aggregated_metrics["value_loss"]
        })
        
        # 🔥 回放缓冲区指标
        if USE_REPLAY_BUFFER and self.replay_buffer:
            buffer_stats = self.replay_buffer.get_stats()
            aggregated_metrics.update({
                "buffer_size": buffer_stats.get("buffer_size", 0),
                "replay_ratio": len(all_metrics) - 1,  # 回放训练次数
                "avg_sample_reuse": buffer_stats.get("avg_reuse_count", 0),
                "sample_efficiency": self.total_training_samples / max(1, self.total_samples_generated)
            })
        
        # 更新学习率
        if self.policy_scheduler and LR_SCHEDULER_TYPE == "cosine":
            self.policy_scheduler.step()
            self.critic_scheduler.step()
        
        # 记录指标
        for key, value in aggregated_metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        self.global_step += 1
        
        return aggregated_metrics
    
    def train(self, dataset, eval_dataset=None):
        """主训练循环"""
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 保存配置
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        config_path = os.path.join(OUTPUT_DIR, "training_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "actor_model": ACTOR_MODEL,
                "use_replay_buffer": USE_REPLAY_BUFFER,
                "replay_buffer_size": REPLAY_BUFFER_SIZE,
                "use_priority_sampling": USE_PRIORITY_SAMPLING,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
            }, f, indent=2)
        self.logger.log(f"训练配置已保存至: {config_path}")

        for epoch in range(NUM_EPOCHES):
            self.logger.log(f"开始 Epoch {epoch + 1}/{NUM_EPOCHES}")
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHES}")
            
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch["prompt"])
                
                if not metrics:
                    continue
                
                # 更新进度条
                desc = f"PL:{metrics['policy_loss']:.4f} VL:{metrics['value_loss']:.4f} R:{metrics['reward']:.2f}"
                if USE_REPLAY_BUFFER:
                    desc += f" BUF:{metrics.get('buffer_size', 0)} EFF:{metrics.get('sample_efficiency', 1.0):.2f}x"
                pbar.set_description(desc)
                
                # 记录到日志
                self.logger.log_metrics(self.global_step, metrics)
                
                # Wandb 记录
                if self.wandb:
                    self.wandb.log(metrics, step=self.global_step)

            # Epoch 结束，保存模型
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            self.policy_model.save_pretrained(save_path)
            self.policy_tokenizer.save_pretrained(save_path)
            
            self.logger.log(f"模型保存至: {save_path}")
        
        # 训练结束后绘制指标图表
        self.logger.log("正在生成训练指标图表...")
        
        from utils.plot_metrics import plot_training_metrics
        plot_training_metrics(
            metrics_history=self.metrics_history,
            save_path=os.path.join(OUTPUT_DIR, "training_metrics_v2.png"),
            title="PPO with Replay Buffer Training Metrics"
        )
        
        self.logger.log(f"训练完成！")
        
        if self.wandb:
            self.wandb.finish()

# ===================== 7. 入口函数 ======================

def train_main():
    """主训练函数"""
    prompts = []
    
    for datasets in train_datasets:
        if datasets['type'] == "jsonl":
            import json
            with open(datasets['path'], "r", encoding='utf-8') as f:
                for item in json.load(f):
                    prompts.append(item[datasets['input']])
        if datasets['type'] == 'parquet':
            import pyarrow.parquet as pq
            table = pq.read_table(datasets['path'])
            df = table.to_pandas()
            for index, row in df.iterrows():
                prompts.append(row['question'])

    train_prompts = prompts[:20]
    
    train_dataset = PPODataset(train_prompts)
    
    trainer = PPOTrainerWithReplay()
    trainer.train(train_dataset)

if __name__ == "__main__":
    train_main()
