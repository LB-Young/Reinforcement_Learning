#!/usr/bin/env python3
# author: YoungL
# date: 2026/01/18
# email: lby15356@gmail.com

"""
测试绘图功能
"""

from plot_metrics import plot_ppo_metrics, plot_grpo_metrics
import os

def test_ppo_plot():
    """测试 PPO 绘图"""
    print("测试 PPO 绘图功能...")
    
    # 模拟训练数据
    policy_losses = [0.5, 0.48, 0.45, 0.42, 0.40, 0.38, 0.36, 0.35, 0.33, 0.32]
    value_losses = [0.3, 0.29, 0.27, 0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19]
    rewards = [1.0, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7]
    advantages = [0.05, 0.08, 0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
    
    plot_ppo_metrics(
        policy_losses=policy_losses,
        value_losses=value_losses,
        rewards=rewards,
        advantages=advantages,
        save_path="./test_outputs/ppo_test.png"
    )
    print("PPO 测试图表已生成")

def test_grpo_plot():
    """测试 GRPO 绘图"""
    print("\n测试 GRPO 绘图功能...")
    
    # 模拟训练数据
    losses = [0.6, 0.55, 0.50, 0.47, 0.44, 0.41, 0.39, 0.37, 0.35, 0.34]
    rewards = [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    
    plot_grpo_metrics(
        losses=losses,
        rewards=rewards,
        save_path="./test_outputs/grpo_test.png"
    )
    print("GRPO 测试图表已生成")

if __name__ == "__main__":
    # 创建输出目录
    os.makedirs("./test_outputs", exist_ok=True)
    
    test_ppo_plot()
    test_grpo_plot()
    
    print("\n✅ 所有测试完成！")
    print("请查看 ./test_outputs/ 目录下的图表")
