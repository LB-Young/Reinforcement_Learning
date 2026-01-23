#!/usr/bin/env python3
# author: YoungL
# date: 2026/01/19
# email: lby15356@gmail.com

"""
标准版 GSPO 训练脚本
GSPO (Group Sequence Policy Optimization) 是一种结合了组采样和序列级优化的策略优化算法
主要特性：
1. Group Sampling: 为每个prompt生成多个回复进行组内比较
2. Sequence-Level Rewards: 序列级别的奖励计算
3. Relative Advantage: 使用组内相对优势替代critic模型
4. 灵活优化: 支持序列级和token级优化策略

注：
1. 由于项目着重于强化学习算法的实现和各种强化学习算法异同的对比，故只涉及整体流程，不包含工程层面的调度优化；
2. 由于本项目实验环境仅包含5060ti-16G * 2，所以将policy模型放在0号gpu，reference、reward两个模型放在1号gpu；
"""

import os
import shutil
import sys
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import List, Tuple, Dict
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入绘图函数
from utils.plot_metrics import plot_gspo_metrics

# ==================== 配置 ====================
POLICY_MODEL = r"E:\models\Qwen\Qwen3-0___6B"
REWARD_MODEL = r"E:\models\reward-model-deberta-v3-large-v2"
train_datasets = [
    {
        "path":r"E:\datasets\gsm8k\main\train-00000-of-00001.parquet",
        "type":"parquet",
        "input":"question",
        "output":"answer"
    }
]

BATCH_SIZE = 2              # GSPO需要生成多个回复，适当调小
LEARNING_RATE = 1e-6
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
NUM_EPOCHS = 1
GROUP_SIZE = 4              # 🔥 每个prompt生成的回复数量
GSPO_EPOCHS = 4             # GSPO更新轮数

# GSPO特有参数
CLIP_RANGE = 0.2            # PPO裁剪范围
ENTROPY_COEF = 0.01         # 熵正则化系数
KL_COEF = 0.2               # KL散度惩罚系数
TARGET_KL = 0.01            # 目标KL散度
ADAPTIVE_KL = True          # 自适应KL系数调整

# 优势计算参数
ADVANTAGE_TYPE = "relative"     # "relative" 或 "normalized"
USE_GROUP_NORMALIZATION = True  # 组内标准化
USE_SEQUENCE_LEVEL_REWARD = True    # 序列级别奖励
USE_TOKEN_LEVEL_LOSS = False        # Token级别损失（可选）

OUTPUT_DIR = r"E:\projects\train_related\trained_model\rl_exprement\grpo_output\gspo_gsm8k_v1"

# ==================== 数据集 ====================
class GSPODataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

# ==================== GSPO训练器 ====================
class GSPOTrainer:
    def __init__(self):
        num_gpus = torch.cuda.device_count()
        self.device_policy = torch.device("cuda:0")
        self.device_ref = torch.device("cuda:1") if num_gpus > 1 else torch.device("cuda:0")
        self.device_reward = torch.device("cuda:1") if num_gpus > 1 else torch.device("cuda:0")
        
        # 初始化指标记录
        self.metrics_history = {
            'policy_loss': [],
            'entropy_loss': [],
            'kl_loss': [],
            'reward': [],
            'relative_advantage': [],
            'kl_divergence': [],
            'kl_coef': [],
            'avg_response_length': []
        }
        
        # 初始化KL系数
        self.kl_coef = KL_COEF
        
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

    def generate_responses_with_group_sampling(self, prompts: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """🔥 GSPO Group Sampling: 为每个prompt生成GROUP_SIZE个回复"""
        self.policy_model.eval()
        all_responses, all_prompts, all_lengths = [], [], []
        
        for prompt in prompts:
            for _ in range(GROUP_SIZE):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_policy)
                with torch.no_grad():
                    outputs = self.policy_model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                # 仅提取生成的 Response 部分
                gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                response = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
                length = len(gen_ids[0])
                
                all_responses.append(response)
                all_prompts.append(prompt)
                all_lengths.append(length)
                
                # 释放当前循环的张量
                del inputs, outputs, gen_ids
        
        # 清理显存
        torch.cuda.empty_cache()
        return all_responses, all_prompts, all_lengths

    def compute_log_probs(self, prompts: List[str], responses: List[str], 
                         use_ref_model: bool = False, return_per_token: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """计算log概率，支持序列级和token级"""
        all_log_probs = []
        all_token_log_probs = []
        
        model = self.ref_model if use_ref_model else self.policy_model
        device = self.device_ref if use_ref_model else self.device_policy
        
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            full_inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True).to(device)
            
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            response_start = prompt_inputs["input_ids"].shape[1]
            
            with torch.no_grad() if use_ref_model else torch.enable_grad():
                outputs = model(**full_inputs)
                logits = outputs.logits
                
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
                
                # 只考虑生成部分的log概率
                response_log_probs = token_log_probs[0, response_start-1:-1]
                
                if return_per_token:
                    all_token_log_probs.append(response_log_probs)
                
                # 序列级log概率（所有token的和）
                all_log_probs.append(response_log_probs.sum())
                
                # 释放张量
                del full_inputs, outputs, logits, log_probs, token_log_probs
        
        # 清理显存
        torch.cuda.empty_cache()
        
        if return_per_token:
            return torch.stack(all_log_probs).to(self.device_policy), all_token_log_probs
        else:
            return torch.stack(all_log_probs).to(self.device_policy), []

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """🔥 GSPO Sequence-Level Rewards: 计算序列级别奖励"""
        rewards = []
        for p, r in zip(prompts, responses):
            full_text = f"{p} {r}"
            inputs = self.reward_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(self.device_reward)
            with torch.no_grad():
                reward = self.reward_model(**inputs).logits[0, 0]
                rewards.append(reward)
            # 释放当前循环的张量
            del inputs
        
        result = torch.stack(rewards).to(self.device_policy)
        # 清理显存
        torch.cuda.empty_cache()
        return result

    def compute_relative_advantages(self, rewards: torch.Tensor, group_size: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """🔥 GSPO Relative Advantage: 计算组内相对优势"""
        if group_size is None:
            group_size = GROUP_SIZE
        
        batch_size = rewards.shape[0]
        if batch_size % group_size != 0:
            num_complete_groups = batch_size // group_size
            rewards = rewards[:num_complete_groups * group_size]
            batch_size = rewards.shape[0]
        
        # 重塑为组的形状 [num_groups, group_size]
        rewards_grouped = rewards.view(-1, group_size)
        
        # 🔥 计算组内均值作为基线
        group_baselines = rewards_grouped.mean(dim=1, keepdim=True)
        
        # 🔥 计算相对优势
        if ADVANTAGE_TYPE == "relative":
            # 相对优势：reward - baseline
            relative_advantages = rewards_grouped - group_baselines
        elif ADVANTAGE_TYPE == "normalized":
            # 标准化优势：(reward - baseline) / std
            relative_advantages = rewards_grouped - group_baselines
            if USE_GROUP_NORMALIZATION:
                group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
                relative_advantages = relative_advantages / group_std
        else:
            raise ValueError(f"Unknown advantage type: {ADVANTAGE_TYPE}")
        
        # 重新展平
        relative_advantages = relative_advantages.view(-1)
        group_baselines = group_baselines.repeat(1, group_size).view(-1)
        
        return relative_advantages, group_baselines

    def compute_policy_loss_sequence_level(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                                          advantages: torch.Tensor) -> torch.Tensor:
        """🔥 GSPO Sequence-Level Policy Loss: 序列级别的策略损失"""
        ratio = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        return policy_loss

    def compute_policy_loss_token_level(self, token_log_probs_list: List[torch.Tensor],
                                       old_token_log_probs_list: List[torch.Tensor],
                                       advantages: torch.Tensor) -> torch.Tensor:
        """🔥 GSPO Token-Level Policy Loss: token级别的策略损失（可选）"""
        total_token_loss = 0.0
        total_tokens = 0
        
        for i, (token_log_probs, old_token_log_probs) in enumerate(zip(token_log_probs_list, old_token_log_probs_list)):
            if len(token_log_probs) == 0:  # 跳过空的token序列
                continue
                
            advantage = advantages[i]  # 序列级优势
            
            # 对每个token计算比率
            token_ratios = torch.exp(token_log_probs - old_token_log_probs)
            
            # PPO裁剪
            surr1 = token_ratios * advantage
            surr2 = torch.clamp(token_ratios, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantage
            
            token_loss = -torch.min(surr1, surr2).sum()
            
            total_token_loss += token_loss
            total_tokens += len(token_log_probs)
        
        return total_token_loss / max(total_tokens, 1)

    def compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                          advantages: torch.Tensor, kl_penalty: torch.Tensor,
                          token_log_probs_list: List[torch.Tensor] = None,
                          old_token_log_probs_list: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算GSPO策略损失"""
        # 选择序列级别或token级别损失
        if USE_TOKEN_LEVEL_LOSS and token_log_probs_list is not None:
            policy_loss = self.compute_policy_loss_token_level(
                token_log_probs_list, old_token_log_probs_list, advantages
            )
        else:
            policy_loss = self.compute_policy_loss_sequence_level(
                log_probs, old_log_probs, advantages
            )
        
        # 熵损失
        entropy = -log_probs.mean()
        entropy_loss = -ENTROPY_COEF * entropy
        
        # KL损失
        kl_loss = self.kl_coef * kl_penalty.mean()
        
        return policy_loss, entropy_loss, kl_loss

    def update_kl_coef(self, kl_divergence: torch.Tensor):
        """自适应调整KL散度系数"""
        if not ADAPTIVE_KL:
            return
        
        mean_kl = kl_divergence.mean().item()
        
        if mean_kl > 2.0 * TARGET_KL:
            self.kl_coef *= 1.5
        elif mean_kl < 0.5 * TARGET_KL:
            self.kl_coef *= 0.5
        
        self.kl_coef = max(0.01, min(self.kl_coef, 1.0))

    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """执行一步GSPO训练"""
        # 🔥 1. Group Sampling: 为每个prompt生成GROUP_SIZE个回复
        responses, prompts_expanded, response_lengths = self.generate_responses_with_group_sampling(batch_prompts)
        
        # 🔥 2. Sequence-Level Rewards: 计算序列级别奖励
        raw_rewards = self.compute_rewards(prompts_expanded, responses)
        
        # 🔥 3. Relative Advantage: 计算组内相对优势
        relative_advantages, group_baselines = self.compute_relative_advantages(raw_rewards)
        
        # 截断数据以匹配相对优势的长度
        prompts_truncated = prompts_expanded[:len(relative_advantages)]
        responses_truncated = responses[:len(relative_advantages)]
        response_lengths_truncated = response_lengths[:len(relative_advantages)]
        
        # 计算KL散度惩罚（使用参考模型）
        with torch.no_grad():
            ref_log_probs, _ = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref_model=True)
        
        # 计算初始log概率（用于后续对比）
        with torch.no_grad():
            old_log_probs, old_token_log_probs_list = self.compute_log_probs(
                prompts_truncated, responses_truncated, use_ref_model=False, return_per_token=USE_TOKEN_LEVEL_LOSS
            )
            # 计算初始KL散度
            initial_kl_penalty = old_log_probs - ref_log_probs
        
        # 标准化优势
        advantages = (relative_advantages - relative_advantages.mean()) / (relative_advantages.std() + 1e-8)
        
        # 🔥 4. GSPO更新循环
        self.policy_model.train()
        total_policy_loss = 0
        total_entropy_loss = 0
        total_kl_loss = 0
        
        for _ in range(GSPO_EPOCHS):
            # 重新计算当前策略的log概率
            new_log_probs, new_token_log_probs_list = self.compute_log_probs(
                prompts_truncated, responses_truncated, use_ref_model=False, return_per_token=USE_TOKEN_LEVEL_LOSS
            )
            
            # 重新计算KL散度（使用当前策略）
            current_kl_penalty = new_log_probs - ref_log_probs.detach()
            
            # 计算损失
            policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(
                new_log_probs, old_log_probs, advantages, current_kl_penalty,
                token_log_probs_list=new_token_log_probs_list,
                old_token_log_probs_list=old_token_log_probs_list
            )
            
            # 总损失
            total_loss = policy_loss + entropy_loss + kl_loss
            
            # 更新策略模型
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_kl_loss += kl_loss.item()
            
            # 显式清理显存
            del new_log_probs, current_kl_penalty, policy_loss, entropy_loss, kl_loss, total_loss
            if new_token_log_probs_list:
                del new_token_log_probs_list
            
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
        
        # 自适应调整KL系数（使用初始KL散度）
        self.update_kl_coef(initial_kl_penalty)
        
        metrics = {
            "policy_loss": total_policy_loss / GSPO_EPOCHS,
            "entropy_loss": total_entropy_loss / GSPO_EPOCHS,
            "kl_loss": total_kl_loss / GSPO_EPOCHS,
            "reward": raw_rewards.mean().item(),
            "relative_advantage": relative_advantages.mean().item(),
            "kl_divergence": initial_kl_penalty.mean().item(),
            "kl_coef": self.kl_coef,
            "avg_response_length": sum(response_lengths_truncated) / len(response_lengths_truncated)
        }
        
        # 记录指标
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        return metrics

    def train(self, dataset):
        """主训练循环"""
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(NUM_EPOCHS):
            pbar = tqdm(dataloader)
            for batch in pbar:
                metrics = self.train_step(batch["prompt"])
                if metrics:  # 如果不为空
                    pbar.set_description(
                        f"PL:{metrics['policy_loss']:.4f} R:{metrics['reward']:.2f} "
                        f"RA:{metrics['relative_advantage']:.3f} KL:{metrics['kl_divergence']:.4f}"
                    )

            # 保存模型
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # 备份训练脚本
            current_script = os.path.abspath(__file__)
            target_script = os.path.join(save_path, "train_script.py")
            
            try:
                shutil.copy2(current_script, target_script)
                print(f"脚本已备份至: {target_script}")
            except Exception as e:
                print(f"脚本备份失败: {e}")
            
            print(f"模型保存至: {save_path}")
        
        # 训练结束后绘制指标图表
        print("\n正在生成训练指标图表...")
        plot_gspo_metrics(
            policy_losses=self.metrics_history['policy_loss'],
            entropy_losses=self.metrics_history['entropy_loss'],
            kl_losses=self.metrics_history['kl_loss'],
            rewards=self.metrics_history['reward'],
            relative_advantages=self.metrics_history['relative_advantage'],
            kl_divergences=self.metrics_history['kl_divergence'],
            kl_coefs=self.metrics_history['kl_coef'],
            avg_response_lengths=self.metrics_history['avg_response_length'],
            save_path=os.path.join(OUTPUT_DIR, "training_metrics.png")
        )
        print(f"训练指标图表已保存至: {os.path.join(OUTPUT_DIR, 'training_metrics.png')}")

def main():
    """测试函数"""
    prompts = ["如何制作一杯好咖啡？", "解释量子纠缠。", "写一段冒泡排序代码。"] * 10
    dataset = GSPODataset(prompts)
    trainer = GSPOTrainer()
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

    prompts = prompts[:20]
    dataset = GSPODataset(prompts)
    trainer = GSPOTrainer()
    trainer.train(dataset)

if __name__ == "__main__":
    # main()
    train_main()