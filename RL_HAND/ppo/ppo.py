#!/usr/bin/env python3
# author: YoungL
# date: 2026/01/18
# email: lby15356@gmail.com

"""
标准版 PPO 训练脚本
注：
1. 由于项目着重于强化学习算法的实现和各种强化学习算法异同的对比，故只涉及整体流程，不包含工程层面的调度优化；
2. 由于本项目实验环境仅包含5060ti-16G * 2，所以将policy、reference两个模型放在0号gpu，critic、reward两个模型放在1号gpu；
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

import shutil
from typing import List, Tuple, Dict
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import plot_ppo_metrics

# ===================== 1. 配置 ==========================
# model
ACTOR_MODEL = r"E:\models\Qwen\Qwen3-0___6B"
CRITIC_MODEL = r"E:\models\Qwen\Qwen3-0___6B"
REWARD_MODEL = r"E:\models\reward-model-deberta-v3-large-v2"

train_datasets = [
    # {
    #     "path":"/home/bayon/datas/MATH-Hard/train/algebra.jsonl",
    #     "type":"jsonl",
    #     "input":"problem",
    #     "output":"solution"
    # },
    {
        "path":r"E:\datasets\gsm8k\main\train-00000-of-00001.parquet",
        "type":"parquet",
        "input":"question",
        "output":"answer"
    }
]

DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

LEARNING_RATE = 1e-6
NUM_EPOCHES = 1
BATCH_SIZE = 4
GROUP_SIZE = 1  # 为了与后续grpo对齐
GROUP_EPOCHES = 4
CLIP_RANGE = 0.2
OUTPUT_DIR = r"E:\projects\train_related\trained_model\rl_exprement\grpo_output\ppo_gsm8k_v1"

# ===================== 2. 数据集处理 =====================
class PPODataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

# ===================== 3. PPOTrainer ===================
class PPOTrainer:
    def __init__(self):

        # num_gpus = torch.cuda.device_count()
        # 
        self.device_policy = torch.device("cuda:0")
        self.device_ref = torch.device("cuda:0")
        self.device_critic = torch.device("cuda:1")
        self.device_reward = torch.device("cuda:1")
        
        # 初始化指标记录
        self.metrics_history = {
            'policy_loss': [],
            'value_loss': [],
            'reward': [],
            'advantage': [],
            'entropy': []
        }

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

        self.policy_optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = torch.optim.AdamW(self.critic_model.parameters(), lr=LEARNING_RATE)

    def generate_response(self, prompts: List[str]) -> Tuple[List[str], List[str]]:
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
        
        return all_responses, all_prompts

    def get_token_log_probs(self, model, prompts: List[str], responses: List[str], device, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 Token 级别的 log_probs 并返回 Mask"""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # 计算 Prompt 的长度
        prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        
        with torch.no_grad() if model.training == False else torch.enable_grad():
            outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]  # Shift 对齐
        labels = inputs["input_ids"][:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        
        # 制作 Mask: 1 仅在 Response 区域且非 Padding 处
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lens):
            mask[i, p_len:] = (labels[i, p_len:] != tokenizer.pad_token_id)
            
        return token_log_probs, mask

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """使用 reward_model 计算奖励"""
        rewards = []
        for p, r in zip(prompts, responses):
            inputs = self.reward_tokenizer(p + r, return_tensors="pt", truncation=True).to(self.device_reward)
            with torch.no_grad():
                reward = self.reward_model(**inputs).logits[0, 0]
                rewards.append(reward)
        return torch.stack(rewards).to(self.device_policy)
    
    def compute_entropy(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """计算策略的熵"""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.policy_tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device_policy)
        
        # 计算 Prompt 的长度
        prompt_lens = [len(self.policy_tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
        
        with torch.no_grad():
            outputs = self.policy_model(**inputs)
        logits = outputs.logits[:, :-1, :]  # Shift 对齐
        labels = inputs["input_ids"][:, 1:]
        
        # 计算概率分布和熵
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, seq_len]
        
        # 制作 Mask: 1 仅在 Response 区域且非 Padding 处
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for i, p_len in enumerate(prompt_lens):
            mask[i, p_len:] = (labels[i, p_len:] != self.policy_tokenizer.pad_token_id)
        
        # 计算平均熵（仅在response区域）
        masked_entropy = entropy * mask
        avg_entropy = masked_entropy.sum() / mask.sum()
        
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
            # 使用最后一个 token 的 logits 作为 value（简化实现）
            # 更标准的做法是添加一个 value head
            value = outputs.logits[0, -1, :].mean()  # 简化：取最后一层的平均
            values.append(value)
        return torch.stack(values).to(self.device_policy)

    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        # 1. 生成响应
        responses, prompts = self.generate_response(batch_prompts)

        # 2. 计算奖励和价值
        rewards = self.compute_rewards(prompts, responses)
        values = self.compute_values(prompts, responses, requires_grad=False)
        advantages = rewards - values  # [B]
        
        # 3. 计算熵
        entropy = self.compute_entropy(prompts, responses)

        # 4. 获取参考 log_probs 和初始旧分布 log_probs (Token-level)
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

        for _ in range(GROUP_EPOCHES):
            # 获取新的 log_probs
            new_log_probs, _ = self.get_token_log_probs(
                self.policy_model, prompts, responses, self.device_policy, self.policy_tokenizer
            )

            # 计算策略损失 (PPO Clip)
            log_ratio = (new_log_probs - old_log_probs) * mask
            ratio = torch.exp(log_ratio)
            adv_t = advantages.unsqueeze(1)  # 广播优势到每个 Token [B, 1]
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv_t
            policy_loss = -torch.min(surr1, surr2)
            
            # KL 惩罚
            kl_div = (new_log_probs - ref_log_probs)
            
            # 组合损失并对 Mask 求均值
            loss_map = (policy_loss + 0.01 * kl_div) * mask
            step_policy_loss = loss_map.sum() / mask.sum()

            # 更新策略网络
            self.policy_optimizer.zero_grad()
            step_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.policy_optimizer.step()

            # 计算价值损失（简化版：直接用 rewards 作为目标）
            new_values = self.compute_values(prompts, responses, requires_grad=True)
            value_loss = F.mse_loss(new_values, rewards)

            # 更新价值网络
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 1.0)
            self.critic_optimizer.step()

            total_policy_loss += step_policy_loss.item()
            total_value_loss += value_loss.item()

        metrics = {
            "policy_loss": total_policy_loss / GROUP_EPOCHES,
            "value_loss": total_value_loss / GROUP_EPOCHES,
            "reward": rewards.mean().item(),
            "advantage": advantages.mean().item(),
            "entropy": entropy.item()
        }
        
        # 记录指标
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        return metrics

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(NUM_EPOCHES):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHES}")
            for batch in pbar:
                metrics = self.train_step(batch["prompt"])
                pbar.set_description(
                    f"PL:{metrics['policy_loss']:.4f} VL:{metrics['value_loss']:.4f} "
                    f"R:{metrics['reward']:.2f} A:{metrics['advantage']:.2f} E:{metrics['entropy']:.3f}"
                )

            # 保存模型
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            self.policy_model.save_pretrained(save_path)
            self.policy_tokenizer.save_pretrained(save_path)
            
            # 保存 critic 模型
            critic_save_path = os.path.join(save_path, "critic")
            self.critic_model.save_pretrained(critic_save_path)
            self.critic_tokenizer.save_pretrained(critic_save_path)
            
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
        
        # 使用通用的训练指标绘制函数
        from utils.plot_metrics import plot_training_metrics, plot_ppo_metrics_with_entropy
        plot_training_metrics(
            metrics_history=self.metrics_history,
            save_path=os.path.join(OUTPUT_DIR, "training_metrics_detailed.png"),
            title="PPO Training Metrics with Entropy"
        )
        
        # 使用包含熵的PPO专用图表
        plot_ppo_metrics_with_entropy(
            policy_losses=self.metrics_history['policy_loss'],
            value_losses=self.metrics_history['value_loss'],
            rewards=self.metrics_history['reward'],
            advantages=self.metrics_history['advantage'],
            entropies=self.metrics_history['entropy'],
            save_path=os.path.join(OUTPUT_DIR, "ppo_metrics_with_entropy.png")
        )
        
        print(f"详细训练指标图表已保存至: {os.path.join(OUTPUT_DIR, 'training_metrics_detailed.png')}")
        print(f"PPO专用指标图表（含熵）已保存至: {os.path.join(OUTPUT_DIR, 'ppo_metrics_with_entropy.png')}")



# ===================== 4. 入口函数 ======================

def main():
    prompts = ["如何制作一杯好咖啡？", "解释量子纠缠。", "写一段冒泡排序代码。"] * 10
    datasets = PPODataset(prompts)
    trainer = PPOTrainer()
    trainer.train(datasets)



def train_main():
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
    # breakpoint()
    dataset = PPODataset(prompts)
    trainer = PPOTrainer()
    trainer.train(dataset)



if __name__ == "__main__":
    # main()
    train_main()