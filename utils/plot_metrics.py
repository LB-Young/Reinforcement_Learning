#!/usr/bin/env python3
# author: YoungL
# date: 2026/01/18
# email: lby15356@gmail.com

"""
训练指标可视化工具
用于绘制训练过程中的损失、奖励、熵等指标
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os


def plot_training_metrics(
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10),
    title: str = "Training Metrics"
):
    """
    绘制训练过程中的各项指标
    
    Args:
        metrics_history: 指标历史字典，例如:
            {
                'policy_loss': [0.5, 0.4, ...],
                'value_loss': [0.3, 0.2, ...],
                'reward': [1.2, 1.5, ...],
                'entropy': [0.8, 0.7, ...],
                ...
            }
        save_path: 保存图片的路径，如果为 None 则显示图片
        figsize: 图片大小
        title: 图片标题
    """
    num_metrics = len(metrics_history)
    if num_metrics == 0:
        print("没有指标数据可绘制")
        return
    
    # 计算子图布局
    cols = 2
    rows = (num_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 如果只有一个子图，axes 不是数组
    if num_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 绘制每个指标
    for idx, (metric_name, values) in enumerate(metrics_history.items()):
        ax = axes[idx]
        steps = range(1, len(values) + 1)
        
        ax.plot(steps, values, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
        ax.set_title(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        if len(values) > 1:
            z = np.polyfit(steps, values, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), "--", alpha=0.5, linewidth=1.5, label='Trend')
            ax.legend()
    
    # 隐藏多余的子图
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_and_entropy(
    losses: List[float],
    entropies: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5)
):
    """
    专门绘制损失和熵的变化
    
    Args:
        losses: 损失值列表
        entropies: 熵值列表
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    steps = range(1, len(losses) + 1)
    
    # 绘制损失
    ax1.plot(steps, losses, linewidth=2, marker='o', markersize=4, color='#e74c3c')
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 绘制熵
    ax2.plot(steps, entropies, linewidth=2, marker='s', markersize=4, color='#3498db')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Entropy', fontsize=12)
    ax2.set_title('Policy Entropy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ppo_metrics_with_entropy(
    policy_losses: List[float],
    value_losses: List[float],
    rewards: List[float],
    advantages: List[float],
    entropies: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 12)
):
    """
    专门为 PPO 算法绘制指标（包含熵）
    
    Args:
        policy_losses: 策略损失列表
        value_losses: 价值损失列表
        rewards: 奖励列表
        advantages: 优势值列表
        entropies: 熵列表
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('PPO Training Metrics (with Entropy)', fontsize=16, fontweight='bold')
    
    steps = range(1, len(policy_losses) + 1)
    
    # 策略损失
    axes[0, 0].plot(steps, policy_losses, linewidth=2, marker='o', markersize=4, color='#e74c3c')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Policy Loss')
    axes[0, 0].set_title('Policy Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 价值损失
    axes[0, 1].plot(steps, value_losses, linewidth=2, marker='s', markersize=4, color='#9b59b6')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Value Loss')
    axes[0, 1].set_title('Value Loss', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 奖励
    axes[1, 0].plot(steps, rewards, linewidth=2, marker='^', markersize=4, color='#2ecc71')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Average Reward', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 优势值
    axes[1, 1].plot(steps, advantages, linewidth=2, marker='d', markersize=4, color='#3498db')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Advantage')
    axes[1, 1].set_title('Average Advantage', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 熵
    axes[2, 0].plot(steps, entropies, linewidth=2, marker='*', markersize=6, color='#f39c12')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Entropy')
    axes[2, 0].set_title('Policy Entropy', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 隐藏最后一个子图
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ppo_metrics(
    policy_losses: List[float],
    value_losses: List[float],
    rewards: List[float],
    advantages: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
):
    """
    专门为 PPO 算法绘制指标
    
    Args:
        policy_losses: 策略损失列表
        value_losses: 价值损失列表
        rewards: 奖励列表
        advantages: 优势值列表
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('PPO Training Metrics', fontsize=16, fontweight='bold')
    
    steps = range(1, len(policy_losses) + 1)
    
    # 策略损失
    axes[0, 0].plot(steps, policy_losses, linewidth=2, marker='o', markersize=4, color='#e74c3c')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Policy Loss')
    axes[0, 0].set_title('Policy Loss', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 价值损失
    axes[0, 1].plot(steps, value_losses, linewidth=2, marker='s', markersize=4, color='#9b59b6')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Value Loss')
    axes[0, 1].set_title('Value Loss', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 奖励
    axes[1, 0].plot(steps, rewards, linewidth=2, marker='^', markersize=4, color='#2ecc71')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Average Reward', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 优势值
    axes[1, 1].plot(steps, advantages, linewidth=2, marker='d', markersize=4, color='#3498db')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Advantage')
    axes[1, 1].set_title('Average Advantage', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_grpo_metrics(losses, rewards, entropies, save_path):
    """绘制GRPO训练指标"""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Loss曲线
        ax1.plot(losses, 'b-', linewidth=2)
        ax1.set_title('Training Loss', fontsize=14)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Reward曲线
        ax2.plot(rewards, 'g-', linewidth=2)
        ax2.set_title('Average Reward', fontsize=14)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        
        # Entropy曲线
        ax3.plot(entropies, 'r-', linewidth=2)
        ax3.set_title('Policy Entropy', fontsize=14)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Entropy')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        print("matplotlib未安装，跳过绘图")


def plot_grpo_metrics_advanced(
    losses: List[float],
    rewards: List[float],
    kl_divs: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """
    专门为 GRPO 算法绘制指标（高级版本）
    
    Args:
        losses: 损失列表
        rewards: 奖励列表
        kl_divs: KL 散度列表（可选）
        save_path: 保存路径
        figsize: 图片大小
    """
    num_plots = 3 if kl_divs else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    fig.suptitle('GRPO Training Metrics', fontsize=16, fontweight='bold')
    
    steps = range(1, len(losses) + 1)
    
    # 损失
    axes[0].plot(steps, losses, linewidth=2, marker='o', markersize=4, color='#e74c3c')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # 奖励
    axes[1].plot(steps, rewards, linewidth=2, marker='s', markersize=4, color='#2ecc71')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Average Reward', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # KL 散度
    if kl_divs:
        axes[2].plot(steps, kl_divs, linewidth=2, marker='^', markersize=4, color='#f39c12')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title('KL Divergence', fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


# 使用示例
if __name__ == "__main__":
    # 示例数据
    example_metrics = {
        'policy_loss': [0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.31, 0.30],
        'value_loss': [0.3, 0.28, 0.25, 0.23, 0.21, 0.20, 0.19, 0.18],
        'reward': [1.0, 1.2, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
        'entropy': [0.8, 0.75, 0.7, 0.68, 0.65, 0.63, 0.61, 0.60]
    }
    
    # 绘制所有指标
    plot_training_metrics(example_metrics, save_path="example_metrics.png")
    
    # 绘制 PPO 指标
    plot_ppo_metrics(
        policy_losses=example_metrics['policy_loss'],
        value_losses=example_metrics['value_loss'],
        rewards=example_metrics['reward'],
        advantages=[0.1, 0.15, 0.2, 0.18, 0.16, 0.14, 0.12, 0.10],
        save_path="ppo_metrics.png"
    )

def plot_dapo_metrics(
    policy_losses: List[float],
    entropy_losses: List[float],
    rewards: List[float],
    entropies: List[float],
    dynamic_resample_rates: List[float],
    avg_response_lengths: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (18, 12)
):
    """
    专门为 DAPO 算法绘制指标
    
    Args:
        policy_losses: 策略损失列表
        entropy_losses: 熵损失列表
        rewards: 奖励列表
        entropies: 熵列表
        dynamic_resample_rates: 动态重采样率列表
        avg_response_lengths: 平均回复长度列表
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle('DAPO Training Metrics', fontsize=16, fontweight='bold')
    
    steps = range(1, len(policy_losses) + 1)
    
    # 策略损失
    axes[0, 0].plot(steps, policy_losses, linewidth=2, marker='o', markersize=4, color='#e74c3c')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Policy Loss')
    axes[0, 0].set_title('Policy Loss (Token-Level)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 熵损失
    axes[0, 1].plot(steps, entropy_losses, linewidth=2, marker='s', markersize=4, color='#9b59b6')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Entropy Loss')
    axes[0, 1].set_title('Entropy Loss', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 奖励
    axes[1, 0].plot(steps, rewards, linewidth=2, marker='^', markersize=4, color='#2ecc71')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Average Reward', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 熵
    axes[1, 1].plot(steps, entropies, linewidth=2, marker='d', markersize=4, color='#f39c12')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 🔥 动态重采样率 (DAPO特有)
    axes[2, 0].plot(steps, dynamic_resample_rates, linewidth=2, marker='*', markersize=6, color='#3498db')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Dynamic Resample Rate')
    axes[2, 0].set_title('Dynamic Resample Rate', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(0, 1)
    
    # 🔥 平均回复长度 (DAPO特有)
    axes[2, 1].plot(steps, avg_response_lengths, linewidth=2, marker='h', markersize=4, color='#e67e22')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('Avg Response Length')
    axes[2, 1].set_title('Average Response Length', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dapo_vs_grpo_comparison(
    dapo_losses: List[float],
    grpo_losses: List[float],
    dapo_rewards: List[float],
    grpo_rewards: List[float],
    dapo_entropies: List[float],
    grpo_entropies: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
):
    """
    绘制 DAPO 与 GRPO 的对比图表
    
    Args:
        dapo_losses: DAPO损失列表
        grpo_losses: GRPO损失列表
        dapo_rewards: DAPO奖励列表
        grpo_rewards: GRPO奖励列表
        dapo_entropies: DAPO熵列表
        grpo_entropies: GRPO熵列表
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('DAPO vs GRPO Comparison', fontsize=16, fontweight='bold')
    
    dapo_steps = range(1, len(dapo_losses) + 1)
    grpo_steps = range(1, len(grpo_losses) + 1)
    
    # 损失对比
    axes[0, 0].plot(dapo_steps, dapo_losses, linewidth=2, marker='o', markersize=4, 
                    color='#e74c3c', label='DAPO (Token-Level)')
    axes[0, 0].plot(grpo_steps, grpo_losses, linewidth=2, marker='s', markersize=4, 
                    color='#3498db', label='GRPO (Sample-Level)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 奖励对比
    axes[0, 1].plot(dapo_steps, dapo_rewards, linewidth=2, marker='o', markersize=4, 
                    color='#e74c3c', label='DAPO')
    axes[0, 1].plot(grpo_steps, grpo_rewards, linewidth=2, marker='s', markersize=4, 
                    color='#3498db', label='GRPO')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Average Reward Comparison', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 熵对比
    axes[1, 0].plot(dapo_steps, dapo_entropies, linewidth=2, marker='o', markersize=4, 
                    color='#e74c3c', label='DAPO (Clip-Higher)')
    axes[1, 0].plot(grpo_steps, grpo_entropies, linewidth=2, marker='s', markersize=4, 
                    color='#3498db', label='GRPO (Symmetric Clip)')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].set_title('Policy Entropy Comparison', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 算法特性对比（文本说明）
    axes[1, 1].axis('off')
    comparison_text = """
DAPO vs GRPO Key Differences:

🔥 DAPO Improvements:
• Clip-Higher: [0.8, 1.28] vs [0.8, 1.2]
• Token-Level Loss vs Sample-Level
• Dynamic Sampling for training signal
• No KL Penalty (KL_COEF = 0.0)
• Overlong Response Filtering

📊 Expected Benefits:
• Prevents entropy collapse
• Better long-chain reasoning
• Faster convergence (50% steps)
• Higher final performance
    """
    axes[1, 1].text(0.05, 0.95, comparison_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()
def plot_gspo_metrics(
    policy_losses: List[float],
    entropy_losses: List[float],
    kl_losses: List[float],
    rewards: List[float],
    relative_advantages: List[float],
    kl_divergences: List[float],
    kl_coefs: List[float],
    avg_response_lengths: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (20, 15)
):
    """
    专门为 GSPO 算法绘制指标
    
    Args:
        policy_losses: 策略损失列表
        entropy_losses: 熵损失列表
        kl_losses: KL损失列表
        rewards: 奖励列表
        relative_advantages: 相对优势列表
        kl_divergences: KL散度列表
        kl_coefs: KL系数列表
        avg_response_lengths: 平均回复长度列表
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle('GSPO Training Metrics', fontsize=16, fontweight='bold')
    
    steps = range(1, len(policy_losses) + 1)
    
    # 策略损失
    axes[0, 0].plot(steps, policy_losses, linewidth=2, marker='o', markersize=4, color='#e74c3c')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Policy Loss')
    axes[0, 0].set_title('Policy Loss (Sequence-Level)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 熵损失
    axes[0, 1].plot(steps, entropy_losses, linewidth=2, marker='s', markersize=4, color='#9b59b6')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Entropy Loss')
    axes[0, 1].set_title('Entropy Loss', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL损失
    axes[1, 0].plot(steps, kl_losses, linewidth=2, marker='^', markersize=4, color='#f39c12')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].set_title('KL Divergence Loss', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 奖励
    axes[1, 1].plot(steps, rewards, linewidth=2, marker='d', markersize=4, color='#2ecc71')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Average Reward', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 🔥 相对优势 (GSPO特有)
    axes[2, 0].plot(steps, relative_advantages, linewidth=2, marker='*', markersize=6, color='#3498db')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Relative Advantage')
    axes[2, 0].set_title('Relative Advantage (Group-based)', fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 🔥 KL散度 (GSPO特有)
    axes[2, 1].plot(steps, kl_divergences, linewidth=2, marker='h', markersize=4, color='#e67e22')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('KL Divergence')
    axes[2, 1].set_title('KL Divergence from Reference', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 🔥 自适应KL系数 (GSPO特有)
    axes[3, 0].plot(steps, kl_coefs, linewidth=2, marker='v', markersize=4, color='#8e44ad')
    axes[3, 0].set_xlabel('Step')
    axes[3, 0].set_ylabel('KL Coefficient')
    axes[3, 0].set_title('Adaptive KL Coefficient', fontweight='bold')
    axes[3, 0].grid(True, alpha=0.3)
    
    # 平均回复长度
    axes[3, 1].plot(steps, avg_response_lengths, linewidth=2, marker='p', markersize=4, color='#16a085')
    axes[3, 1].set_xlabel('Step')
    axes[3, 1].set_ylabel('Avg Response Length')
    axes[3, 1].set_title('Average Response Length', fontweight='bold')
    axes[3, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gspo_vs_grpo_comparison(
    gspo_losses: List[float],
    grpo_losses: List[float],
    gspo_rewards: List[float],
    grpo_rewards: List[float],
    gspo_relative_advantages: List[float],
    grpo_relative_rewards: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
):
    """
    绘制 GSPO 与 GRPO 的对比图表
    
    Args:
        gspo_losses: GSPO损失列表
        grpo_losses: GRPO损失列表
        gspo_rewards: GSPO奖励列表
        grpo_rewards: GRPO奖励列表
        gspo_relative_advantages: GSPO相对优势列表
        grpo_relative_rewards: GRPO相对奖励列表
        save_path: 保存路径
        figsize: 图片大小
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('GSPO vs GRPO Comparison', fontsize=16, fontweight='bold')
    
    gspo_steps = range(1, len(gspo_losses) + 1)
    grpo_steps = range(1, len(grpo_losses) + 1)
    
    # 损失对比
    axes[0, 0].plot(gspo_steps, gspo_losses, linewidth=2, marker='o', markersize=4, 
                    color='#e74c3c', label='GSPO (Group Sequence)')
    axes[0, 0].plot(grpo_steps, grpo_losses, linewidth=2, marker='s', markersize=4, 
                    color='#3498db', label='GRPO (Group Relative)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 奖励对比
    axes[0, 1].plot(gspo_steps, gspo_rewards, linewidth=2, marker='o', markersize=4, 
                    color='#e74c3c', label='GSPO')
    axes[0, 1].plot(grpo_steps, grpo_rewards, linewidth=2, marker='s', markersize=4, 
                    color='#3498db', label='GRPO')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].set_title('Average Reward Comparison', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 相对优势/奖励对比
    axes[1, 0].plot(gspo_steps, gspo_relative_advantages, linewidth=2, marker='o', markersize=4, 
                    color='#e74c3c', label='GSPO (Relative Advantage)')
    axes[1, 0].plot(grpo_steps, grpo_relative_rewards, linewidth=2, marker='s', markersize=4, 
                    color='#3498db', label='GRPO (Relative Reward)')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Relative Value')
    axes[1, 0].set_title('Relative Advantage/Reward Comparison', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 算法特性对比（文本说明）
    axes[1, 1].axis('off')
    comparison_text = """
GSPO vs GRPO Key Differences:

🔥 GSPO Features:
• Group Sampling: Multi-response per prompt
• Sequence-Level Rewards: Full sequence evaluation
• Relative Advantage: Group-based baseline
• Adaptive KL: Dynamic KL coefficient adjustment
• Flexible Optimization: Sequence/Token level

📊 GRPO Features:
• Group Relative Policy: Relative rewards
• Token-Level Loss: Fine-grained optimization
• Fixed KL: Static KL coefficient
• Simpler Architecture: Fewer hyperparameters

🎯 Use Cases:
• GSPO: Complex reasoning, diverse generation
• GRPO: Long text generation, efficiency focus
    """
    axes[1, 1].text(0.05, 0.95, comparison_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图表已保存至: {save_path}")
    else:
        plt.show()
    
    plt.close()