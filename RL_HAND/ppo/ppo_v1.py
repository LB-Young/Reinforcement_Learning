#!/usr/bin/env python3
# author: YoungL
# date: 2026/01/20
# email: lby15356@gmail.com

"""
改进版 PPO 训练脚本 (v1)
新增功能：
1. 学习率调度器 (Cosine Annealing)
2. 梯度累积支持
3. 检查点恢复机制
4. 详细日志记录
5. 早停机制
6. 显存优化
7. 配置文件保存
8. 验证集评估
9. 可选的 Wandb 集成

注：
1. 由于项目着重于强化学习算法的实现和各种强化学习算法异同的对比，故只涉及整体流程，不包含工程层面的调度优化；
2. 由于本项目实验环境仅包含5060ti-16G * 2，所以将policy、reference两个模型放在0号gpu，critic、reward两个模型放在1号gpu；
"""

import os
import sys
import json
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

import shutil
from typing import List, Tuple, Dict, Optional
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
ENTROPY_COEF = 0.01  # 熵系数
KL_COEF = 0.01  # KL散度系数
VALUE_LOSS_COEF = 0.5  # 价值损失系数
MAX_GRAD_NORM = 1.0  # 梯度裁剪

# 新增配置
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数
USE_LR_SCHEDULER = True  # 是否使用学习率调度器
LR_SCHEDULER_TYPE = "cosine"  # cosine 或 plateau
WARMUP_STEPS = 100  # 预热步数
EARLY_STOPPING_PATIENCE = 5  # 早停耐心值
EARLY_STOPPING_THRESHOLD = 0.001  # 早停阈值
SAVE_EVERY_N_STEPS = 100  # 每N步保存一次检查点
EVAL_EVERY_N_STEPS = 50  # 每N步评估一次
USE_WANDB = False  # 是否使用 Wandb
WANDB_PROJECT = "ppo-training"  # Wandb 项目名

OUTPUT_DIR = r"E:\projects\train_related\trained_model\rl_exprement\grpo_output\ppo_gsm8k_v1"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# ===================== 2. 数据集处理 =====================
class PPODataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

# ===================== 3. 日志记录器 =====================
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

# ===================== 4. PPOTrainer ===================
class PPOTrainer:
    def __init__(self, resume_from_checkpoint: Optional[str] = None):
        # 设备配置
        self.device_policy = torch.device("cuda:0")
        self.device_ref = torch.device("cuda:0")
        self.device_critic = torch.device("cuda:1")
        self.device_reward = torch.device("cuda:1")
        
        # 初始化日志记录器
        self.logger = TrainingLogger(LOG_DIR)
        self.logger.log("初始化 PPO Trainer...")
        
        # 初始化指标记录
        self.metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'reward': [],
            'advantage': [],
            'entropy': [],
            'kl_divergence': [],
            'learning_rate': []
        }
        
        # 训练状态
        self.global_step = 0
        self.best_reward = float('-inf')
        self.patience_counter = 0
        
        # 初始化 Wandb（可选）
        if USE_WANDB:
            try:
                import wandb
                wandb.init(project=WANDB_PROJECT, config={
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "group_size": GROUP_SIZE,
                    "clip_range": CLIP_RANGE,
                    "entropy_coef": ENTROPY_COEF,
                    "kl_coef": KL_COEF
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
        
        self.logger.log("PPO Trainer 初始化完成")
    
    def _load_models(self):
        """加载所有模型"""
        self.logger.log("加载模型...")
        
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            ACTOR_MODEL,
            torch_dtype=DTYPE,
            device_map={"": self.device_policy}
        )
        self.reference_model = AutoModelForCausalLM.from_pretrained(
            ACTOR_MODEL,
            torch_dtype=DTYPE,
            device_map={"": self.device_ref}
        )
        self.reference_model.eval()
        self.policy_tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL, trust_remote_code=True)
        if self.policy_tokenizer.pad_token_id is None:
            self.policy_tokenizer.pad_token_id = self.policy_tokenizer.eos_token_id

        self.critic_model = AutoModelForCausalLM.from_pretrained(
            CRITIC_MODEL,
            torch_dtype=DTYPE,
            device_map={"": self.device_critic}
        )
        self.critic_tokenizer = AutoTokenizer.from_pretrained(CRITIC_MODEL, trust_remote_code=True)
        if self.critic_tokenizer.pad_token_id is None:
            self.critic_tokenizer.pad_token_id = self.critic_tokenizer.eos_token_id

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL,
            torch_dtype=DTYPE,
            device_map={"": self.device_reward}
        )
        self.reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL, trust_remote_code=True)
        
        self.logger.log("模型加载完成")

    def _save_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]):
        """保存训练检查点"""
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型
        self.policy_model.save_pretrained(os.path.join(checkpoint_path, "policy"))
        self.policy_tokenizer.save_pretrained(os.path.join(checkpoint_path, "policy"))
        self.critic_model.save_pretrained(os.path.join(checkpoint_path, "critic"))
        self.critic_tokenizer.save_pretrained(os.path.join(checkpoint_path, "critic"))
        
        # 保存训练状态
        state = {
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "patience_counter": self.patience_counter,
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "metrics": metrics
        }
        
        if self.policy_scheduler:
            state["policy_scheduler"] = self.policy_scheduler.state_dict()
            state["critic_scheduler"] = self.critic_scheduler.state_dict()
        
        torch.save(state, os.path.join(checkpoint_path, "training_state.pt"))
        
        # 保存配置
        config = {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "group_size": GROUP_SIZE,
            "clip_range": CLIP_RANGE,
            "entropy_coef": ENTROPY_COEF,
            "kl_coef": KL_COEF,
            "value_loss_coef": VALUE_LOSS_COEF,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS
        }
        with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        self.logger.log(f"检查点已保存至: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练"""
        self.logger.log(f"从检查点恢复: {checkpoint_path}")
        
        # 加载训练状态
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.global_step = state["global_step"]
            self.best_reward = state["best_reward"]
            self.patience_counter = state["patience_counter"]
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.metrics_history = state["metrics_history"]
            
            if self.policy_scheduler and "policy_scheduler" in state:
                self.policy_scheduler.load_state_dict(state["policy_scheduler"])
                self.critic_scheduler.load_state_dict(state["critic_scheduler"])
            
            self.logger.log(f"恢复训练状态: step={self.global_step}, best_reward={self.best_reward:.4f}")

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
            
            # 清理显存
            del inputs, outputs, gen_ids
        
        torch.cuda.empty_cache()
        return all_responses, all_prompts

    def get_token_log_probs(self, model, prompts: List[str], responses: List[str], device, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 Token 级别的 log_probs 并返回 Mask"""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # 计算 Prompt 的长度
        prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        
        with torch.no_grad() if model.training == False else torch.enable_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        # 制作 Mask
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lens):
            mask[i, p_len:] = (labels[i, p_len:] != tokenizer.pad_token_id)
        
        # 清理显存
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
    
    def compute_entropy(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算策略的熵"""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.policy_tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device_policy)
        
        prompt_lens = [len(self.policy_tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        
        with torch.no_grad():
            outputs = self.policy_model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lens):
            mask[i, p_len:] = (labels[i, p_len:] != self.policy_tokenizer.pad_token_id)
        
        masked_entropy = entropy * mask
        avg_entropy = masked_entropy.sum() / mask.sum()
        
        # 清理显存
        del inputs, outputs, logits, probs, log_probs, labels
        torch.cuda.empty_cache()
        
        return avg_entropy

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

    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """执行一步训练"""
        # 1. 生成响应
        responses, prompts = self.generate_response(batch_prompts)

        # 2. 计算奖励和价值
        rewards = self.compute_rewards(prompts, responses)
        values = self.compute_values(prompts, responses, requires_grad=False)
        advantages = rewards - values
        
        # 3. 计算熵
        entropy = self.compute_entropy(prompts, responses)

        # 4. 获取参考 log_probs 和初始旧分布 log_probs
        with torch.no_grad():
            ref_log_probs, _ = self.get_token_log_probs(
                self.reference_model, prompts, responses, self.device_ref, self.policy_tokenizer
            )
            ref_log_probs = ref_log_probs.to(self.device_policy)
            old_log_probs, mask = self.get_token_log_probs(
                self.policy_model, prompts, responses, self.device_policy, self.policy_tokenizer
            )

        # 5. PPO更新循环
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
            log_ratio = (new_log_probs - old_log_probs) * mask
            ratio = torch.exp(log_ratio)
            adv_t = advantages.unsqueeze(1)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv_t
            policy_loss = -torch.min(surr1, surr2)
            
            # KL 惩罚
            kl_div = (new_log_probs - ref_log_probs)
            
            # 组合损失
            loss_map = (policy_loss + KL_COEF * kl_div - ENTROPY_COEF * entropy) * mask
            step_policy_loss = loss_map.sum() / mask.sum()

            # 梯度累积
            step_policy_loss = step_policy_loss / GRADIENT_ACCUMULATION_STEPS
            step_policy_loss.backward()
            
            if (epoch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), MAX_GRAD_NORM)
                self.policy_optimizer.step()
                self.policy_optimizer.zero_grad()

            # 计算价值损失
            new_values = self.compute_values(prompts, responses, requires_grad=True)
            value_loss = F.mse_loss(new_values, rewards) * VALUE_LOSS_COEF
            value_loss = value_loss / GRADIENT_ACCUMULATION_STEPS
            value_loss.backward()
            
            if (epoch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), MAX_GRAD_NORM)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

            total_policy_loss += step_policy_loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_value_loss += value_loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_kl_div += kl_div.mean().item()
            
            # 清理显存
            del new_log_probs, log_ratio, ratio, surr1, surr2, policy_loss, kl_div, loss_map
            del new_values, value_loss
            torch.cuda.empty_cache()
            gc.collect()

        # 更新学习率
        if self.policy_scheduler and LR_SCHEDULER_TYPE == "cosine":
            self.policy_scheduler.step()
            self.critic_scheduler.step()

        metrics = {
            "policy_loss": total_policy_loss / GROUP_EPOCHES,
            "value_loss": total_value_loss / GROUP_EPOCHES,
            "total_loss": (total_policy_loss + total_value_loss) / GROUP_EPOCHES,
            "reward": rewards.mean().item(),
            "advantage": advantages.mean().item(),
            "entropy": entropy.item(),
            "kl_divergence": total_kl_div / GROUP_EPOCHES,
            "learning_rate": self.policy_optimizer.param_groups[0]['lr']
        }
        
        # 记录指标
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        self.global_step += 1
        
        return metrics

    def evaluate(self, eval_prompts: List[str]) -> Dict[str, float]:
        """在验证集上评估"""
        self.logger.log("开始评估...")
        self.policy_model.eval()
        
        eval_rewards = []
        eval_entropies = []
        
        with torch.no_grad():
            for prompt in eval_prompts:
                responses, prompts_expanded = self.generate_response([prompt])
                rewards = self.compute_rewards(prompts_expanded, responses)
                entropy = self.compute_entropy(prompts_expanded, responses)
                
                eval_rewards.append(rewards.mean().item())
                eval_entropies.append(entropy.item())
        
        eval_metrics = {
            "eval_reward": sum(eval_rewards) / len(eval_rewards),
            "eval_entropy": sum(eval_entropies) / len(eval_entropies)
        }
        
        self.logger.log(f"评估结果: Reward={eval_metrics['eval_reward']:.4f}, Entropy={eval_metrics['eval_entropy']:.4f}")
        return eval_metrics
    
    def check_early_stopping(self, current_reward: float) -> bool:
        """检查是否应该早停"""
        if current_reward > self.best_reward + EARLY_STOPPING_THRESHOLD:
            self.best_reward = current_reward
            self.patience_counter = 0
            self.logger.log(f"新的最佳奖励: {self.best_reward:.4f}")
            return False
        else:
            self.patience_counter += 1
            self.logger.log(f"早停计数器: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                self.logger.log("触发早停机制", "WARNING")
                return True
        return False

    def train(self, dataset, eval_dataset=None):
        """主训练循环"""
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 保存配置
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        config_path = os.path.join(OUTPUT_DIR, "training_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "actor_model": ACTOR_MODEL,
                "critic_model": CRITIC_MODEL,
                "reward_model": REWARD_MODEL,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "group_size": GROUP_SIZE,
                "clip_range": CLIP_RANGE,
                "entropy_coef": ENTROPY_COEF,
                "kl_coef": KL_COEF,
                "num_epochs": NUM_EPOCHES,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "use_lr_scheduler": USE_LR_SCHEDULER,
                "lr_scheduler_type": LR_SCHEDULER_TYPE
            }, f, indent=2)
        self.logger.log(f"训练配置已保存至: {config_path}")

        for epoch in range(NUM_EPOCHES):
            self.logger.log(f"开始 Epoch {epoch + 1}/{NUM_EPOCHES}")
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHES}")
            
            for batch_idx, batch in enumerate(pbar):
                metrics = self.train_step(batch["prompt"])
                
                # 更新进度条
                pbar.set_description(
                    f"PL:{metrics['policy_loss']:.4f} VL:{metrics['value_loss']:.4f} "
                    f"R:{metrics['reward']:.2f} A:{metrics['advantage']:.2f} E:{metrics['entropy']:.3f} "
                    f"LR:{metrics['learning_rate']:.2e}"
                )
                
                # 记录到日志
                self.logger.log_metrics(self.global_step, metrics)
                
                # Wandb 记录
                if self.wandb:
                    self.wandb.log(metrics, step=self.global_step)
                
                # 定期保存检查点
                if self.global_step % SAVE_EVERY_N_STEPS == 0:
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"step_{self.global_step}")
                    self._save_checkpoint(checkpoint_path, metrics)
                
                # 定期评估
                if eval_dataset and self.global_step % EVAL_EVERY_N_STEPS == 0:
                    eval_metrics = self.evaluate(eval_dataset.prompts[:10])  # 评估前10个样本
                    
                    if self.wandb:
                        self.wandb.log(eval_metrics, step=self.global_step)
                    
                    # 检查早停
                    if self.check_early_stopping(eval_metrics['eval_reward']):
                        self.logger.log("早停触发，结束训练")
                        return
                    
                    # 更新学习率（如果使用 plateau 调度器）
                    if self.policy_scheduler and LR_SCHEDULER_TYPE == "plateau":
                        self.policy_scheduler.step(eval_metrics['eval_reward'])
                        self.critic_scheduler.step(eval_metrics['eval_reward'])

            # Epoch 结束，保存模型
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            self.policy_model.save_pretrained(save_path)
            self.policy_tokenizer.save_pretrained(save_path)
            
            critic_save_path = os.path.join(save_path, "critic")
            self.critic_model.save_pretrained(critic_save_path)
            self.critic_tokenizer.save_pretrained(critic_save_path)
            
            # 备份训练脚本
            current_script = os.path.abspath(__file__)
            target_script = os.path.join(save_path, "train_script.py")
            
            try:
                shutil.copy2(current_script, target_script)
                self.logger.log(f"脚本已备份至: {target_script}")
            except Exception as e:
                self.logger.log(f"脚本备份失败: {e}", "ERROR")
            
            self.logger.log(f"模型保存至: {save_path}")
        
        # 训练结束后绘制指标图表
        self.logger.log("正在生成训练指标图表...")
        
        from utils.plot_metrics import plot_training_metrics, plot_ppo_metrics_with_entropy
        plot_training_metrics(
            metrics_history=self.metrics_history,
            save_path=os.path.join(OUTPUT_DIR, "training_metrics_detailed.png"),
            title="PPO Training Metrics (v1)"
        )
        
        plot_ppo_metrics_with_entropy(
            policy_losses=self.metrics_history['policy_loss'],
            value_losses=self.metrics_history['value_loss'],
            rewards=self.metrics_history['reward'],
            advantages=self.metrics_history['advantage'],
            entropies=self.metrics_history['entropy'],
            save_path=os.path.join(OUTPUT_DIR, "ppo_metrics_with_entropy.png")
        )
        
        self.logger.log(f"训练完成！指标图表已保存至: {OUTPUT_DIR}")
        
        if self.wandb:
            self.wandb.finish()

# ===================== 5. 入口函数 ======================

def main():
    """测试函数"""
    prompts = ["如何制作一杯好咖啡？", "解释量子纠缠。", "写一段冒泡排序代码。"] * 10
    dataset = PPODataset(prompts)
    trainer = PPOTrainer()
    trainer.train(dataset)

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

    # 划分训练集和验证集
    train_size = int(0.9 * len(prompts))
    train_prompts = prompts[:train_size]
    eval_prompts = prompts[train_size:]
    
    print(f"训练集大小: {len(train_prompts)}")
    print(f"验证集大小: {len(eval_prompts)}")
    
    # 限制数据量（用于快速测试）
    train_prompts = train_prompts[:20]
    eval_prompts = eval_prompts[:5] if eval_prompts else None
    
    train_dataset = PPODataset(train_prompts)
    eval_dataset = PPODataset(eval_prompts) if eval_prompts else None
    
    # 检查是否从检查点恢复
    resume_checkpoint = None
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("step_")]
        if checkpoints:
            # 获取最新的检查点
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1]))
            resume_checkpoint = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
            print(f"发现检查点: {resume_checkpoint}")
            user_input = input("是否从检查点恢复训练？(y/n): ")
            if user_input.lower() != 'y':
                resume_checkpoint = None
    
    trainer = PPOTrainer(resume_from_checkpoint=resume_checkpoint)
    trainer.train(train_dataset, eval_dataset)

if __name__ == "__main__":
    # main()
    train_main()
