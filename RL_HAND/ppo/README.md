# PPO (Proximal Policy Optimization)

## 概述

PPO是OpenAI提出的策略优化算法，通过裁剪机制限制策略更新幅度，在保证训练稳定性的同时实现高样本效率。本实现采用Actor-Critic架构，适用于各种文本生成任务。

## 核心特性

### 🔥 Actor-Critic架构
- **Actor**: 策略网络，负责生成文本
- **Critic**: 价值网络，负责评估状态价值
- **Reference**: 参考网络，提供KL散度基线

### 🔥 PPO-Clip机制
- 对称裁剪范围：`[1-0.2, 1+0.2] = [0.8, 1.2]`
- 防止策略更新过大，保证训练稳定性
- 平衡探索和利用

### 🔥 优势函数
- 使用 `A = R - V` 计算优势
- Critic网络提供价值估计
- 减少方差，提高学习效率

## 算法流程

1. **生成阶段**: 使用当前策略生成回复
2. **评估阶段**: 计算奖励和价值估计
3. **优势计算**: 计算优势函数 A = R - V
4. **策略更新**: 使用PPO-Clip损失更新Actor
5. **价值更新**: 使用MSE损失更新Critic
6. **多轮迭代**: 重复更新多个epoch

## 配置参数

```python
# 核心超参数
LEARNING_RATE = 1e-6        # 学习率
BATCH_SIZE = 4              # 批次大小
GROUP_SIZE = 1              # 每个prompt的回复数
GROUP_EPOCHES = 4           # PPO更新轮数
CLIP_RANGE = 0.2            # 裁剪范围

# 模型配置
ACTOR_MODEL = "Qwen3-0.6B"  # 策略模型
CRITIC_MODEL = "Qwen3-0.6B" # 价值模型
REWARD_MODEL = "reward-model-deberta-v3-large-v2"
```

## 使用方法

### 基本训练
```bash
python RL_HAND/ppo/ppo.py
```

### 自定义数据集
修改 `train_datasets` 配置：
```python
train_datasets = [
    {
        "path": "your_dataset.parquet",
        "type": "parquet", 
        "input": "question",
        "output": "answer"
    }
]
```

## 训练指标

- **Policy Loss**: 策略网络损失
- **Value Loss**: 价值网络损失  
- **Reward**: 平均奖励
- **Advantage**: 平均优势值
- **Entropy**: 策略熵

## 适用场景

### PPO更适合：
- 通用对话生成
- 文本摘要任务
- 需要稳定训练的场景
- 平衡质量和多样性的任务

### 优势：
- 训练稳定，不易崩溃
- 样本效率高
- 理论基础扎实
- 通用性强

### 局限性：
- 需要训练两个网络
- 计算开销较大
- 超参数较多
- 显存需求高

## 文件结构

```
RL_HAND/ppo/
├── ppo.py                    # 主训练脚本
├── README.md                 # 说明文档
├── IMPLEMENTATION_SUMMARY.md # 详细实现总结
└── (训练输出)
    ├── epoch_1/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   ├── critic/           # 价值网络
    │   └── train_script.py
    └── training_metrics.png
```

## 参考文献

1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
3. [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)