#!/usr/bin/env python3
# author: YoungL
# date: 2026/01/18
# email: lby15356@gmail.com

"""
标准版 GRPO 训练脚本
注：
1. 由于项目着重于强化学习算法的实现和各种强化学习算法异同的对比，故只涉及整体流程，不包含工程层面的调度优化；
2. 由于本项目实验环境仅包含5060ti-16G * 2，所以将policy模型放在0号gpu，reference、reward两个模型放在1号gpu；
"""

import os
import shutil  # 在脚本顶部添加导入
import sys
import gc  # 添加垃圾回收模块
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import List, Tuple, Dict
from tqdm import tqdm

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入绘图函数
from utils.plot_metrics import plot_grpo_metrics

# ==================== 配置 ====================
POLICY_MODEL = r"E:\models\Qwen\Qwen3-0___6B"
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

BATCH_SIZE = 2 # 增加 Token-level 计算后显存占用会升高，适当调小
LEARNING_RATE = 1e-6
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
NUM_EPOCHS = 1
GROUP_SIZE = 4
GRPO_EPOCHS = 4
CLIP_RANGE = 0.2
KL_COEF = 0.01  # KL 惩罚系数
OUTPUT_DIR = r"E:\projects\train_related\trained_model\rl_exprement\grpo_output\grpo_gsm8k_v1"

# ==================== 数据集 ====================
class GRPODataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

# ==================== GRPO训练器 ====================
class GRPOTrainer:
    def __init__(self):
        num_gpus = torch.cuda.device_count()
        self.device_policy = torch.device("cuda:0")
        self.device_ref = torch.device("cuda:1") if num_gpus > 1 else torch.device("cuda:0")
        self.device_reward = torch.device("cuda:1") if num_gpus > 1 else torch.device("cuda:0")
        
        # 初始化指标记录
        self.metrics_history = {
            'loss': [],
            'reward': [],
            'entropy': []
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

    def generate_responses(self, prompts: List[str]) -> Tuple[List[str], List[str]]:
        self.policy_model.eval()
        all_responses, all_prompts = [], []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_policy)
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=GROUP_SIZE,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # 仅提取生成的 Response 部分
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            responses = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            all_responses.extend(responses)
            all_prompts.extend([prompt] * GROUP_SIZE)
            
            # 释放当前循环的张量
            del inputs, outputs, gen_ids
        
        # 清理显存
        torch.cuda.empty_cache()
        return all_responses, all_prompts

    def get_token_log_probs(self, model, prompts: List[str], responses: List[str], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """核心修改：获取 Token 级别的 log_probs 并返回 Mask 和 Entropy"""
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # 为了计算 Response 部分，需要知道 Prompt 的长度
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
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
        
        # 释放labels，不再需要
        del prompt_inputs, inputs, outputs, logits, probs, labels
        
        # 强制清理显存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return token_log_probs, mask, entropy

    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
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

    def train_step(self, batch_prompts: List[str]) -> Dict[str, float]:
        responses, prompts_expanded = self.generate_responses(batch_prompts)
        
        # 1. 计算 Advantages (Group-level)
        rewards = self.compute_rewards(prompts_expanded, responses)
        num_groups = len(batch_prompts)
        rewards_grouped = rewards.view(num_groups, GROUP_SIZE)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = ((rewards_grouped - mean_r) / std_r).view(-1) # [B*G]

        # 2. 获取参考 log_probs 和初始旧分布 log_probs (Token-level)
        with torch.no_grad():
            ref_log_probs, _, _ = self.get_token_log_probs(self.ref_model, prompts_expanded, responses, self.device_ref)
            ref_log_probs = ref_log_probs.to(self.device_policy)
            old_log_probs, mask, _ = self.get_token_log_probs(self.policy_model, prompts_expanded, responses, self.device_policy)
            # 确保old_log_probs不需要梯度，避免累积计算图
            old_log_probs = old_log_probs.detach()
            mask = mask.detach()
        
        # 3. 策略优化循环
        self.policy_model.train()
        total_loss = 0
        total_entropy = 0
        for _ in range(GRPO_EPOCHS):
            new_log_probs, _, entropy = self.get_token_log_probs(self.policy_model, prompts_expanded, responses, self.device_policy)
            
            # 计算平均熵值 (仅在response区域)
            masked_entropy = entropy * mask.float()
            avg_entropy = masked_entropy.sum() / mask.sum()
            total_entropy += avg_entropy.item()
            
            # --- 标准计算逻辑 ---
            # Importance Sampling Ratio
            log_ratio = (new_log_probs - old_log_probs) * mask
            ratio = torch.exp(log_ratio)
            
            # PPO Clip Loss
            adv_t = advantages.unsqueeze(1) # 广播优势到每个 Token
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv_t
            policy_loss = -torch.min(surr1, surr2)
            
            # KL Penalty (Token-level) - 使用稳定版近似: exp(ref-new) - (ref-new) - 1
            # 或者更通用的: new_log_probs - ref_log_probs
            kl_div = (new_log_probs - ref_log_probs)
            
            # 组合 Loss 并对 Mask 求均值
            loss_map = (policy_loss + KL_COEF * kl_div) * mask
            step_loss = loss_map.sum() / mask.sum()
            
            self.optimizer.zero_grad()
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += step_loss.item()
            
            # 显式清理显存，避免OOM
            # 1. 删除张量变量
            del new_log_probs, entropy, masked_entropy, log_ratio, ratio, surr1, surr2, policy_loss, kl_div, loss_map, step_loss
            
            # 2. 清理梯度缓存（重要：清理模型的中间状态）
            self.optimizer.zero_grad(set_to_none=True)  # 更彻底地清理梯度
            
            # 3. 清理PyTorch的计算图缓存
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # 4. 强制垃圾回收（可选，但在显存紧张时有帮助）
            gc.collect()
            
        metrics = {
            "loss": total_loss / GRPO_EPOCHS, 
            "reward": rewards.mean().item(),
            "entropy": total_entropy / GRPO_EPOCHS
        }
        
        # 记录指标
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        return metrics

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(NUM_EPOCHS):
            pbar = tqdm(dataloader)
            for batch in pbar:
                metrics = self.train_step(batch["prompt"])
                pbar.set_description(f"L:{metrics['loss']:.4f} R:{metrics['reward']:.2f} E:{metrics['entropy']:.3f}")

            # --- 修改部分开始 ---
            save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
            self.policy_model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path) # 建议同时保存 tokenizer
            
            # 自动获取当前正在运行的脚本绝对路径
            current_script = os.path.abspath(__file__)
            # 目标路径，例如：.../epoch_1/train_script.py
            target_script = os.path.join(save_path, "train_script.py")
            
            try:
                shutil.copy2(current_script, target_script)
                print(f"脚本已备份至: {target_script}")
            except Exception as e:
                print(f"脚本备份失败: {e}")
            # --- 修改部分结束 ---
            
            print(f"模型保存至: {save_path}")
        
        # 训练结束后绘制指标图表
        print("\n正在生成训练指标图表...")
        plot_grpo_metrics(
            losses=self.metrics_history['loss'],
            rewards=self.metrics_history['reward'],
            entropies=self.metrics_history['entropy'],
            save_path=os.path.join(OUTPUT_DIR, "training_metrics.png")
        )
        print(f"训练指标图表已保存至: {os.path.join(OUTPUT_DIR, 'training_metrics.png')}")

def main():
    prompts = ["如何制作一杯好咖啡？", "解释量子纠缠。", "写一段冒泡排序代码。"] * 10
    dataset = GRPODataset(prompts)
    trainer = GRPOTrainer()
    trainer.train(dataset)




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
    dataset = GRPODataset(prompts)
    trainer = GRPOTrainer()
    trainer.train(dataset)


if __name__ == "__main__":
    # main()
    train_main()
