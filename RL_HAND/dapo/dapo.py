#!/usr/bin/env python3
# author: YoungL
# date: 2026/01/19
# email: lby15356@gmail.com

"""
标准版 DAPO 训练脚本
DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 是GRPO的改进版本
主要改进：
1. Clip-Higher: 非对称裁剪，防止熵崩溃
2. Token-Level Loss: 按token级别计算损失，避免短回复偏好
3. Dynamic Sampling: 动态采样确保训练信号
4. 移除KL惩罚: 允许策略更自由探索

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
from utils.plot_metrics import plot_dapo_metrics

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

BATCH_SIZE = 2  # DAPO需要更多显存，适当调小
LEARNING_RATE = 1e-6
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
NUM_EPOCHS = 1
GROUP_SIZE = 4
DAPO_EPOCHS = 4

# DAPO特有参数
CLIP_RANGE_LOW = 0.2    # 下界裁剪范围
CLIP_RANGE_HIGH = 0.28  # 🔥 Clip-Higher: 上界裁剪范围更大
KL_COEF = 0.0           # 🔥 DAPO移除KL惩罚
ENTROPY_COEF = 0.01
USE_DYNAMIC_SAMPLING = True     # 🔥 动态采样
MAX_DYNAMIC_SAMPLES = 8         # 动态采样最大样本数
USE_TOKEN_LEVEL_LOSS = True     # 🔥 Token级别损失
USE_OVERLONG_FILTERING = True   # 🔥 过滤过长回复
MAX_RESPONSE_LENGTH = 256       # 最大回复长度

OUTPUT_DIR = r"E:\projects\train_related\trained_model\rl_exprement\grpo_output\dapo_gsm8k_v1"

# ==================== 数据集 ====================
class DAPODataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

# ==================== DAPO训练器 ====================
class DAPOTrainer:
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
            'entropy': [],
            'dynamic_resample_rate': [],
            'avg_response_length': []
        }
        
        # 动态采样统计
        self.dynamic_sampling_stats = {
            "total_questions": 0,
            "resampled_questions": 0,
            "avg_extra_samples": 0.0
        }
        
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

    def generate_responses(self, prompts: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """生成回复，返回回复、对应的prompt和回复长度"""
        self.policy_model.eval()
        all_responses, all_prompts, all_lengths = [], [], []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_policy)
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=MAX_RESPONSE_LENGTH,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=GROUP_SIZE,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 仅提取生成的 Response 部分
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            responses = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            lengths = [len(gen_id) for gen_id in gen_ids]
            
            all_responses.extend(responses)
            all_prompts.extend([prompt] * GROUP_SIZE)
            all_lengths.extend(lengths)
            
            # 释放当前循环的张量
            del inputs, outputs, gen_ids
        
        # 清理显存
        torch.cuda.empty_cache()
        return all_responses, all_prompts, all_lengths

    def apply_dynamic_sampling(self, prompt: str, initial_responses: List[str], 
                             initial_rewards: torch.Tensor, initial_lengths: List[int]) -> Tuple[List[str], torch.Tensor, List[int]]:
        """🔥 DAPO特性：动态采样，如果所有奖励相同则继续采样"""
        if not USE_DYNAMIC_SAMPLING:
            return initial_responses, initial_rewards, initial_lengths
        
        responses = initial_responses.copy()
        rewards = initial_rewards.clone()
        lengths = initial_lengths.copy()
        
        reward_std = rewards.std().item()
        extra_samples = 0
        
        # 如果标准差太小（奖励都相同），继续采样
        while reward_std < 1e-6 and len(responses) < MAX_DYNAMIC_SAMPLES:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_policy)
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=MAX_RESPONSE_LENGTH,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            extra_response = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            extra_length = len(gen_ids[0])
            
            # 计算额外样本的奖励
            extra_reward = self.compute_rewards([prompt], [extra_response])
            
            responses.append(extra_response)
            rewards = torch.cat([rewards, extra_reward])
            lengths.append(extra_length)
            
            reward_std = rewards.std().item()
            extra_samples += 1
            
            # 释放张量
            del inputs, outputs, gen_ids
        
        if extra_samples > 0:
            self.dynamic_sampling_stats["resampled_questions"] += 1
            self.dynamic_sampling_stats["avg_extra_samples"] += extra_samples
        
        torch.cuda.empty_cache()
        return responses, rewards, lengths

    def apply_overlong_filtering(self, prompts: List[str], responses: List[str], 
                               rewards: torch.Tensor, lengths: List[int]) -> Tuple[List[str], List[str], torch.Tensor, List[int]]:
        """🔥 DAPO特性：过滤过长回复"""
        if not USE_OVERLONG_FILTERING:
            return prompts, responses, rewards, lengths
        
        valid_indices = [i for i, length in enumerate(lengths) if length < MAX_RESPONSE_LENGTH]
        
        if len(valid_indices) == 0:
            return prompts, responses, rewards, lengths
        
        filtered_prompts = [prompts[i] for i in valid_indices]
        filtered_responses = [responses[i] for i in valid_indices]
        filtered_rewards = rewards[valid_indices]
        filtered_lengths = [lengths[i] for i in valid_indices]
        
        return filtered_prompts, filtered_responses, filtered_rewards, filtered_lengths

    def get_token_log_probs(self, model, prompts: List[str], responses: List[str], device, 
                          return_per_token: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """获取 Token 级别的 log_probs 并返回 Mask、Entropy 和每个样本的token级别log_probs"""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # 计算 Prompt 的长度
        prompt_lens = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :] # Shift 对齐
        labels = inputs["input_ids"][:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        # 计算熵值 (仅在response区域)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch_size, seq_len]
        
        # 制作 Mask: 1 仅在 Response 区域且非 Padding 处
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lens):
            mask[i, p_len:] = (labels[i, p_len:] != self.tokenizer.pad_token_id)
        
        # 🔥 如果需要返回每个样本的token级别log_probs（用于token-level loss）
        per_token_log_probs = []
        if return_per_token:
            for i, p_len in enumerate(prompt_lens):
                # 提取该样本response部分的token log_probs
                response_mask = mask[i, p_len:]
                if response_mask.sum() > 0:
                    sample_token_log_probs = token_log_probs[i, p_len:][response_mask]
                    per_token_log_probs.append(sample_token_log_probs)
                else:
                    per_token_log_probs.append(torch.tensor([], device=device))
        
        # 释放不需要的张量
        del inputs, outputs, logits, probs, labels
        
        # 强制清理显存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return token_log_probs, mask, entropy, per_token_log_probs

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算奖励"""
        rewards = []
        for p, r in zip(prompts, responses):
            inputs = self.reward_tokenizer(p + r, return_tensors="pt", truncation=True).to(self.device_reward)
            with torch.no_grad():
                reward = self.reward_model(**inputs).logits[0, 0]
                rewards.append(reward)
            # 释放当前循环的张量
            del inputs
        
        result = torch.stack(rewards).to(self.device_policy)
        # 清理显存
        torch.cuda.empty_cache()
        return result

    def compute_relative_rewards(self, rewards: torch.Tensor, group_sizes: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算相对奖励（支持不同组大小）"""
        relative_rewards = []
        group_baselines = []
        
        start_idx = 0
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            group_rewards = rewards[start_idx:end_idx]
            
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8
            
            # 组内标准化
            group_relative = (group_rewards - group_mean) / group_std
            
            relative_rewards.append(group_relative)
            group_baselines.extend([group_mean] * group_size)
            
            start_idx = end_idx
        
        relative_rewards = torch.cat(relative_rewards)
        group_baselines = torch.tensor(group_baselines, device=rewards.device)
        
        return relative_rewards, group_baselines

    def compute_policy_loss_token_level(self, token_log_probs_list: List[torch.Tensor], 
                                       old_token_log_probs_list: List[torch.Tensor],
                                       advantages: torch.Tensor) -> torch.Tensor:
        """🔥 DAPO Token-Level Loss: 对每个token单独计算PPO损失"""
        total_token_loss = 0.0
        total_tokens = 0
        
        for i, (token_log_probs, old_token_log_probs) in enumerate(zip(token_log_probs_list, old_token_log_probs_list)):
            if len(token_log_probs) == 0:  # 跳过空的token序列
                continue
                
            advantage = advantages[i]  # 该样本的优势值
            
            # 计算每个token的概率比率
            token_ratios = torch.exp(token_log_probs - old_token_log_probs)  # [num_tokens]
            
            # 🔥 DAPO Clip-Higher: 非对称裁剪
            surr1 = token_ratios * advantage
            surr2 = torch.clamp(token_ratios, 
                               1 - CLIP_RANGE_LOW, 
                               1 + CLIP_RANGE_HIGH) * advantage
            
            # 对每个token取最小值，然后求和
            token_loss = -torch.min(surr1, surr2).sum()  # 对该样本的所有token求和
            
            total_token_loss += token_loss
            total_tokens += len(token_log_probs)
        
        # 返回平均token损失
        return total_token_loss / max(total_tokens, 1)

    def compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                          advantages: torch.Tensor, mask: torch.Tensor,
                          token_log_probs_list: List[torch.Tensor] = None,
                          old_token_log_probs_list: List[torch.Tensor] = None) -> torch.Tensor:
        """🔥 计算DAPO策略损失（使用Clip-Higher和Token-Level Loss）"""
        
        # 🔥 Token-Level Loss: 对每个token单独计算损失
        if USE_TOKEN_LEVEL_LOSS and token_log_probs_list is not None:
            policy_loss = self.compute_policy_loss_token_level(
                token_log_probs_list, old_token_log_probs_list, advantages
            )
        else:
            # Sample-level loss (类似GRPO)
            log_ratio = (log_probs - old_log_probs) * mask
            ratio = torch.exp(log_ratio)
            
            # 🔥 DAPO Clip-Higher: 非对称裁剪
            adv_t = advantages.unsqueeze(1) # 广播优势到每个 Token
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE_LOW, 1 + CLIP_RANGE_HIGH) * adv_t
            policy_loss = -torch.min(surr1, surr2)
            
            # 对 Mask 求均值
            loss_map = policy_loss * mask
            policy_loss = loss_map.sum() / mask.sum()
        
        return policy_loss

    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        """执行一步DAPO训练"""
        self.dynamic_sampling_stats["total_questions"] += len(batch_prompts)
        
        all_prompts = []
        all_responses = []
        all_response_lengths = []
        all_raw_rewards = []
        group_sizes = []
        
        # 🔥 为每个prompt生成回复并应用动态采样
        for prompt in batch_prompts:
            # 生成初始回复组
            responses, prompts_repeated, lengths = self.generate_responses([prompt])
            
            # 计算奖励
            raw_rewards = self.compute_rewards(prompts_repeated, responses)
            
            # 🔥 应用动态采样
            responses, raw_rewards, lengths = self.apply_dynamic_sampling(
                prompt, responses, raw_rewards, lengths
            )
            
            # 更新对应的prompt列表
            prompts_repeated = [prompt] * len(responses)
            
            # 🔥 应用过长回复过滤
            prompts_repeated, responses, raw_rewards, lengths = self.apply_overlong_filtering(
                prompts_repeated, responses, raw_rewards, lengths
            )
            
            if len(responses) > 0:  # 确保过滤后还有有效回复
                all_prompts.extend(prompts_repeated)
                all_responses.extend(responses)
                all_response_lengths.extend(lengths)
                all_raw_rewards.append(raw_rewards)
                group_sizes.append(len(responses))
        
        if len(all_responses) == 0:
            print("所有回复都被过滤，跳过此步")
            return {}
        
        # 合并所有奖励
        all_raw_rewards = torch.cat(all_raw_rewards)
        
        # 计算相对奖励
        relative_rewards, group_baselines = self.compute_relative_rewards(all_raw_rewards, group_sizes)
        
        # 标准化优势
        advantages = (relative_rewards - relative_rewards.mean()) / (relative_rewards.std() + 1e-8)
        
        # 🔥 获取token级别的log概率（用于token-level loss）
        old_token_log_probs, mask, _, old_token_log_probs_list = self.get_token_log_probs(
            self.policy_model, all_prompts, all_responses, self.device_policy, return_per_token=USE_TOKEN_LEVEL_LOSS
        )
        
        # detach以避免梯度传播
        old_token_log_probs = old_token_log_probs.detach()
        mask = mask.detach()
        if old_token_log_probs_list:
            old_token_log_probs_list = [t.detach() for t in old_token_log_probs_list]
        
        # DAPO更新循环
        self.policy_model.train()
        total_policy_loss = 0
        total_entropy_loss = 0
        total_kl_loss = 0
        total_entropy = 0
        
        for _ in range(DAPO_EPOCHS):
            # 重新计算当前策略的log概率
            new_token_log_probs, _, entropy, new_token_log_probs_list = self.get_token_log_probs(
                self.policy_model, all_prompts, all_responses, self.device_policy, return_per_token=USE_TOKEN_LEVEL_LOSS
            )
            
            # 计算平均熵值 (仅在response区域)
            masked_entropy = entropy * mask.float()
            avg_entropy = masked_entropy.sum() / mask.sum()
            total_entropy += avg_entropy.item()
            
            # 计算策略损失
            policy_loss = self.compute_policy_loss(
                new_token_log_probs, old_token_log_probs, advantages, mask,
                token_log_probs_list=new_token_log_probs_list,
                old_token_log_probs_list=old_token_log_probs_list
            )
            
            # 熵损失
            entropy_loss = -ENTROPY_COEF * avg_entropy
            
            # KL损失（DAPO默认为0）
            kl_loss = torch.tensor(0.0, device=self.device_policy)
            
            # 总损失
            total_loss = policy_loss + entropy_loss + kl_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_kl_loss += kl_loss.item()
            
            # 显式清理显存
            del new_token_log_probs, entropy, masked_entropy, policy_loss, entropy_loss, kl_loss, total_loss
            if new_token_log_probs_list:
                del new_token_log_probs_list
            
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
        
        # 计算动态采样统计
        resample_rate = (self.dynamic_sampling_stats["resampled_questions"] / 
                        max(1, self.dynamic_sampling_stats["total_questions"]))
        avg_extra = (self.dynamic_sampling_stats["avg_extra_samples"] / 
                    max(1, self.dynamic_sampling_stats["resampled_questions"]))
        
        metrics = {
            "policy_loss": total_policy_loss / DAPO_EPOCHS,
            "entropy_loss": total_entropy_loss / DAPO_EPOCHS,
            "kl_loss": total_kl_loss / DAPO_EPOCHS,
            "reward": all_raw_rewards.mean().item(),
            "entropy": total_entropy / DAPO_EPOCHS,
            "dynamic_resample_rate": resample_rate,
            "avg_response_length": sum(all_response_lengths) / len(all_response_lengths)
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
                        f"E:{metrics['entropy']:.3f} DSR:{metrics['dynamic_resample_rate']:.2f}"
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
        plot_dapo_metrics(
            policy_losses=self.metrics_history['policy_loss'],
            entropy_losses=self.metrics_history['entropy_loss'],
            rewards=self.metrics_history['reward'],
            entropies=self.metrics_history['entropy'],
            dynamic_resample_rates=self.metrics_history['dynamic_resample_rate'],
            avg_response_lengths=self.metrics_history['avg_response_length'],
            save_path=os.path.join(OUTPUT_DIR, "training_metrics.png")
        )
        print(f"训练指标图表已保存至: {os.path.join(OUTPUT_DIR, 'training_metrics.png')}")

def main():
    """测试函数"""
    prompts = ["如何制作一杯好咖啡？", "解释量子纠缠。", "写一段冒泡排序代码。"] * 10
    dataset = DAPODataset(prompts)
    trainer = DAPOTrainer()
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
    dataset = DAPODataset(prompts)
    trainer = DAPOTrainer()
    trainer.train(dataset)

if __name__ == "__main__":
    # main()
    train_main()