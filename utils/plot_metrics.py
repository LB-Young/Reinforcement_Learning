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
