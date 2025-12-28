"""
PPO vs GRPO 算法对比演示脚本
展示两种算法的核心差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPOExample:
    """PPO算法核心逻辑示例"""
    
    def __init__(self):
        self.name = "PPO (Proximal Policy Optimization)"
        
    def compute_advantages(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """
        PPO使用GAE计算优势
        需要价值函数的输出
        """
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i+1]
                
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def policy_loss(self, old_log_probs, new_log_probs, advantages, clip_ratio=0.2):
        """
        PPO策略损失计算
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        return -torch.min(surrogate1, surrogate2).mean()
    
    def value_loss(self, predicted_values, target_returns):
        """
        PPO价值函数损失
        """
        return F.mse_loss(predicted_values, target_returns)
    
    def total_loss(self, policy_loss, value_loss, entropy_loss, value_coef=0.5, entropy_coef=0.01):
        """
        PPO总损失 = 策略损失 + 价值损失 + 熵损失
        """
        return policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

class GRPOExample:
    """GRPO算法核心逻辑示例"""
    
    def __init__(self):
        self.name = "GRPO (Group Relative Policy Optimization)"
        
    def compute_group_advantages(self, group_rewards):
        """
        GRPO使用组内归一化计算优势
        不需要价值函数
        """
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards) + 1e-8
        
        # 归一化就是优势估计
        advantages = [(r - mean_reward) / std_reward for r in group_rewards]
        return advantages
    
    def policy_loss(self, old_log_probs, new_log_probs, advantages, clip_ratio=0.2):
        """
        GRPO策略损失计算（与PPO类似，但优势计算不同）
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        return -torch.min(surrogate1, surrogate2).mean()
    
    def kl_divergence_loss(self, new_log_probs, ref_log_probs):
        """
        GRPO使用KL散度相对于参考策略的惩罚
        """
        return (new_log_probs - ref_log_probs).mean()
    
    def total_loss(self, policy_loss, kl_loss, entropy_loss, beta_kl=0.1, entropy_coef=0.01):
        """
        GRPO总损失 = 策略损失 + KL散度损失 + 熵损失
        注意：没有价值函数损失
        """
        return policy_loss + beta_kl * kl_loss + entropy_coef * entropy_loss

def demonstrate_difference():
    """
    演示PPO和GRPO的核心差异
    """
    print("=" * 60)
    print("PPO vs GRPO 算法对比演示")
    print("=" * 60)
    
    # 创建示例数据
    group_size = 8
    sequence_length = 10
    
    # 模拟一组响应的奖励
    group_rewards = [2.1, 1.8, 3.2, 0.9, 2.5, 1.2, 2.8, 1.6]
    
    # 模拟PPO需要的数据
    ppo_values = [1.5, 1.3, 2.1, 0.8, 1.9, 1.0, 2.2, 1.4]  # 价值函数输出
    ppo_dones = [False] * 7 + [True]  # 最后一个完成
    
    # 模拟对数概率
    old_log_probs = torch.tensor([-2.1, -1.8, -2.5, -3.2, -2.0, -2.8, -1.9, -2.3])
    new_log_probs = torch.tensor([-2.0, -1.9, -2.3, -3.0, -2.1, -2.6, -2.0, -2.2])
    ref_log_probs = torch.tensor([-2.1, -1.8, -2.5, -3.2, -2.0, -2.8, -1.9, -2.3])
    
    # 初始化算法
    ppo = PPOExample()
    grpo = GRPOExample()
    
    print(f"\n{ppo.name}:")
    print("-" * 40)
    
    # PPO优势计算
    ppo_advantages = ppo.compute_advantages(group_rewards, ppo_values, ppo_dones)
    print(f"原始奖励: {group_rewards}")
    print(f"价值函数输出: {ppo_values}")
    print(f"PPO优势估计: {[f'{a:.3f}' for a in ppo_advantages]}")
    
    # PPO损失计算
    ppo_advantages_tensor = torch.tensor(ppo_advantages)
    ppo_policy_loss = ppo.policy_loss(old_log_probs, new_log_probs, ppo_advantages_tensor)
    ppo_value_loss = ppo.value_loss(torch.tensor(ppo_values), torch.tensor(group_rewards))
    entropy_loss = torch.tensor(0.1)  # 模拟熵损失
    
    ppo_total_loss = ppo.total_loss(ppo_policy_loss, ppo_value_loss, entropy_loss)
    
    print(f"策略损失: {ppo_policy_loss:.4f}")
    print(f"价值损失: {ppo_value_loss:.4f}")
    print(f"总损失: {ppo_total_loss:.4f}")
    
    print(f"\n{grpo.name}:")
    print("-" * 40)
    
    # GRPO优势计算
    grpo_advantages = grpo.compute_group_advantages(group_rewards)
    print(f"原始奖励: {group_rewards}")
    print(f"组内平均: {np.mean(group_rewards):.3f}")
    print(f"组内标准差: {np.std(group_rewards):.3f}")
    print(f"GRPO优势估计: {[f'{a:.3f}' for a in grpo_advantages]}")
    
    # GRPO损失计算
    grpo_advantages_tensor = torch.tensor(grpo_advantages)
    grpo_policy_loss = grpo.policy_loss(old_log_probs, new_log_probs, grpo_advantages_tensor)
    grpo_kl_loss = grpo.kl_divergence_loss(new_log_probs, ref_log_probs)
    
    grpo_total_loss = grpo.total_loss(grpo_policy_loss, grpo_kl_loss, entropy_loss)
    
    print(f"策略损失: {grpo_policy_loss:.4f}")
    print(f"KL散度损失: {grpo_kl_loss:.4f}")
    print(f"总损失: {grpo_total_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("关键差异总结:")
    print("=" * 60)
    
    differences = [
        ("网络结构", "策略网络 + 价值网络", "仅策略网络"),
        ("优势估计", "GAE (需要价值函数)", "组内归一化"),
        ("数据需求", "需要标注数据", "只需奖励函数"),
        ("内存使用", "高 (两个网络)", "低 (一个网络)"),
        ("训练稳定性", "依赖价值函数准确性", "组内比较保证稳定性"),
        ("损失组成", "策略 + 价值 + 熵", "策略 + KL + 熵"),
    ]
    
    print(f"{'特性':<12} {'PPO':<25} {'GRPO':<25}")
    print("-" * 62)
    for feature, ppo_desc, grpo_desc in differences:
        print(f"{feature:<12} {ppo_desc:<25} {grpo_desc:<25}")
    
    print("\n" + "=" * 60)
    print("适用场景:")
    print("=" * 60)
    print("PPO适用于:")
    print("- 需要精确价值估计的任务")
    print("- 有充足标注数据的场景")
    print("- 传统的强化学习环境")
    
    print("\nGRPO适用于:")
    print("- 大语言模型微调")
    print("- 有可验证输出的任务 (数学、编程)")
    print("- 资源受限的训练环境")
    print("- 推理密集型任务")

if __name__ == "__main__":
    demonstrate_difference()