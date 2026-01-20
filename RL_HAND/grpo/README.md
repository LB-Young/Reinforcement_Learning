# GRPO (Group Relative Policy Optimization)

## 概述

GRPO是一种基于相对奖励的策略优化算法，通过组内比较学习来避免绝对奖励的偏差问题。采用Policy-Only架构，简化了训练过程，特别适合长文本生成任务。

## 核心特性

### 🔥 相对奖励机制
- 使用组内相对奖励而非绝对奖励
- 消除奖励模型的系统性偏差
- 提高训练稳定性

### 🔥 Policy-Only架构
- 无需价值网络，简化模型结构
- 使用组内基线替代价值估计
- 减少计算复杂度和显存需求

### 🔥 组内比较学习
- 每个prompt生成多个回复
- 在同一组内进行相对比较
- 学习相对质量而非绝对质量

### 🔥 Token级别计算
- 在token级别计算损失和梯度
- 精确控制生成质量
- 仅在response区域计算损失

## 算法流程

1. **批量生成**: 为每个prompt生成GROUP_SIZE个回复
2. **奖励计算**: 使用奖励模型计算绝对奖励
3. **相对化**: 计算组内相对奖励 R_rel = R - R_mean
4. **标准化**: 全局标准化优势函数
5. **策略更新**: 使用PPO-Clip损失更新策略
6. **多轮迭代**: 重复更新多个epoch

## 数学原理

### 相对奖励计算
```
R_rel(x_i, y_i) = R(x_i, y_i) - (1/K) * Σ R(x_i, y_j)
```

### 优势函数
```
A_i = (R_rel_i - μ) / σ
```

### GRPO目标函数
```
L^GRPO(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)] + β*KL(π_θ||π_ref)
```

## 配置参数

```python
# 核心超参数
BATCH_SIZE = 2              # 批次大小（显存优化）
LEARNING_RATE = 1e-6        # 学习率
GROUP_SIZE = 4              # 每个prompt的回复数量
GRPO_EPOCHS = 4             # GRPO更新轮数
CLIP_RANGE = 0.2            # PPO裁剪范围
KL_COEF = 0.01              # KL散度系数

# 模型配置
POLICY_MODEL = "Qwen3-0.6B"  # 策略模型
REWARD_MODEL = "reward-model-deberta-v3-large-v2"
```

## 使用方法

### 基本训练
```bash
python RL_HAND/grpo/grpo.py
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

- **Loss**: GRPO策略损失（包含PPO损失和KL惩罚）
- **Reward**: 平均奖励
- **Entropy**: 策略熵

## 与PPO对比

| 特性 | PPO | GRPO |
|------|-----|------|
| 架构 | Actor-Critic | Policy-Only |
| 奖励类型 | 绝对奖励 | 相对奖励 |
| 网络数量 | 2个 | 1个 |
| 显存需求 | 高 | 中等 |
| 训练复杂度 | 高 | 中等 |
| 适用场景 | 通用 | 长文本 |

## 适用场景

### GRPO更适合：
- 长文本生成任务
- 创意写作
- 对话系统
- 文本改写
- 显存受限的环境

### 优势：
- 架构简单，易于实现
- 显存需求较低
- 相对奖励更稳定
- 训练效率高

### 局限性：
- 需要生成多个回复
- 依赖GROUP_SIZE设置
- 计算开销较大
- 严重依赖奖励模型质量

## 显存优化

### 双GPU架构
```python
# GPU设备分配
device_policy = torch.device("cuda:0")  # 策略模型
device_ref = torch.device("cuda:1")     # 参考模型  
device_reward = torch.device("cuda:1")  # 奖励模型
```

### 显存管理
- 及时释放中间张量
- 使用`torch.cuda.empty_cache()`
- 梯度清理：`zero_grad(set_to_none=True)`
- 强制垃圾回收：`gc.collect()`

## 文件结构

```
RL_HAND/grpo/
├── grpo.py                   # 主训练脚本
├── README.md                 # 说明文档
├── IMPLEMENTATION_SUMMARY.md # 详细实现总结
└── (训练输出)
    ├── epoch_1/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── train_script.py
    └── training_metrics.png
```

## 与DAPO的关系

GRPO是DAPO的基础版本：

| 特性 | GRPO | DAPO |
|------|------|------|
| 裁剪方式 | 对称 [0.8, 1.2] | 非对称 [0.8, 1.28] |
| KL惩罚 | 使用 (0.01) | 移除 (0.0) |
| 动态采样 | ❌ | ✅ |
| 过长处理 | ❌ | ✅ |
| 适用场景 | 长文本生成 | 长链推理 |

## 参考文献

1. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
2. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2512.07611)
3. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)