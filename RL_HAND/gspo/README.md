# GSPO (Group Sequence Policy Optimization)

## 概述

GSPO是一种结合了组采样和序列级优化的策略优化算法，通过为每个prompt生成多个回复进行组内比较，使用相对优势替代critic模型，特别适合需要多样性和复杂推理的文本生成任务。

## 核心特性

### 🔥 Group Sampling（组采样）
- 为每个prompt生成GROUP_SIZE个不同回复
- 通过采样确保回复多样性
- 提供丰富的对比信号

### 🔥 Sequence-Level Rewards（序列级奖励）
- 在完整序列级别计算奖励
- 使用奖励模型评估整个prompt+response
- 得到标量奖励值

### 🔥 Relative Advantage（相对优势）
- 使用组内相对优势而非绝对奖励
- 组内均值作为基线，替代critic模型
- 减少奖励尺度影响，关注相对好坏

### 🔥 灵活优化策略
- 支持序列级和token级优化
- 自适应KL系数调整
- 可配置的优势计算方式

## 算法流程

1. **Group Sampling**: 为每个prompt生成GROUP_SIZE个回复
2. **Sequence-Level Rewards**: 计算序列级别奖励
3. **Relative Advantage**: 计算组内相对优势 A = R - R_mean
4. **Policy Optimization**: 使用PPO-style裁剪更新策略
5. **Adaptive KL**: 动态调整KL散度系数
6. **多轮迭代**: 重复更新多个epoch

## 数学原理

### 相对优势计算
```
A_ij = R_ij - mean(R_i)
```
其中：
- `R_ij`: 第i组第j个样本的奖励
- `mean(R_i)`: 第i组所有样本的平均奖励

### 标准化版本
```
A_ij = (R_ij - mean(R_i)) / std(R_i)
```

### GSPO目标函数
```
L^GSPO(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)] + β*KL(π_θ||π_ref) - α*H(π_θ)
```

## 配置参数

```python
# 核心超参数
BATCH_SIZE = 2              # 批次大小
LEARNING_RATE = 1e-6        # 学习率
GROUP_SIZE = 4              # 每个prompt的回复数量
GSPO_EPOCHS = 4             # GSPO更新轮数
CLIP_RANGE = 0.2            # PPO裁剪范围

# GSPO特有参数
ENTROPY_COEF = 0.01         # 熵正则化系数
KL_COEF = 0.2               # KL散度惩罚系数
TARGET_KL = 0.01            # 目标KL散度
ADAPTIVE_KL = True          # 自适应KL系数调整

# 优势计算参数
ADVANTAGE_TYPE = "relative"     # "relative" 或 "normalized"
USE_GROUP_NORMALIZATION = True  # 组内标准化
USE_SEQUENCE_LEVEL_REWARD = True    # 序列级别奖励
USE_TOKEN_LEVEL_LOSS = False        # Token级别损失（可选）

# 模型配置
POLICY_MODEL = "Qwen3-0.6B"  # 策略模型
REWARD_MODEL = "reward-model-deberta-v3-large-v2"
```

## 使用方法

### 基本训练
```bash
python RL_HAND/gspo/gspo.py
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

- **Policy Loss**: 策略损失（序列级或token级）
- **Entropy Loss**: 熵损失
- **KL Loss**: KL散度损失
- **Reward**: 平均奖励
- **Relative Advantage**: 相对优势
- **KL Divergence**: KL散度值
- **KL Coefficient**: 自适应KL系数
- **Average Response Length**: 平均回复长度

## 与其他算法对比

### GSPO vs PPO
| 特性 | PPO | GSPO |
|------|-----|------|
| Critic模型 | 需要 | 不需要 |
| 基线估计 | Value function V(s) | 组内均值 |
| 采样策略 | 单个回复 | 组采样（多个回复） |
| 优势计算 | GAE | 相对优势 |
| 训练复杂度 | 高（需训练critic） | 中（只训练policy） |

### GSPO vs GRPO
| 特性 | GRPO | GSPO |
|------|------|------|
| 组采样 | ✅ | ✅ |
| 相对奖励 | ✅ | ✅ |
| 序列级优化 | ✅ | ✅ |
| Token级优化 | ❌ | ✅（可选） |
| 自适应KL | ❌ | ✅ |
| 奖励塑形 | 基础 | 增强 |

### GSPO vs DAPO
| 特性 | DAPO | GSPO |
|------|------|------|
| 裁剪方式 | 非对称 [0.8, 1.28] | 对称 [0.8, 1.2] |
| KL惩罚 | 移除 (0.0) | 自适应 (0.01-1.0) |
| 动态采样 | ✅ | ❌ |
| 组采样 | ✅ | ✅ |
| 适用场景 | 长链推理 | 复杂推理+多样性 |

## 适用场景

### GSPO更适合：
- 需要多样性的生成任务
- 复杂推理任务
- 创意写作
- 对话系统
- 需要探索不同解决方案的任务

### 优势：
- 无需训练critic模型
- 相对优势更稳定
- 支持灵活的优化策略
- 自适应KL系数调整
- 丰富的对比信号

### 局限性：
- 需要生成多个回复（计算开销大）
- 内存需求较高
- 超参数较多
- 依赖GROUP_SIZE设置

## 自适应KL机制

GSPO的一个重要特性是自适应KL系数调整：

```python
def update_kl_coef(self, kl_divergence):
    mean_kl = kl_divergence.mean().item()
    
    if mean_kl > 2.0 * TARGET_KL:
        self.kl_coef *= 1.5  # 增大KL系数，更保守
    elif mean_kl < 0.5 * TARGET_KL:
        self.kl_coef *= 0.5  # 减小KL系数，更激进
    
    self.kl_coef = max(0.01, min(self.kl_coef, 1.0))
```

这确保了策略更新既不会过于保守也不会过于激进。

## 超参数调优建议

### 基础配置
```python
# 适合大多数任务的默认配置
GROUP_SIZE = 4
GSPO_EPOCHS = 4
CLIP_RANGE = 0.2
ADVANTAGE_TYPE = "relative"
USE_GROUP_NORMALIZATION = True
```

### 针对不同任务的调整

#### 长序列任务
```python
GROUP_SIZE = 6  # 更大的组，更稳定的基线
GSPO_EPOCHS = 5  # 更多更新
ADVANTAGE_TYPE = "normalized"  # 标准化优势
```

#### 短序列任务
```python
GROUP_SIZE = 3  # 更小的组，减少计算
GSPO_EPOCHS = 3  # 更少更新
ADVANTAGE_TYPE = "relative"  # 相对优势
```

#### 创意任务
```python
ENTROPY_COEF = 0.02  # 更大的熵系数，鼓励探索
KL_COEF = 0.1  # 更小的KL系数，允许更多变化
```

## 文件结构

```
RL_HAND/gspo/
├── gspo.py                   # 主训练脚本
├── README.md                 # 说明文档
├── IMPLEMENTATION_SUMMARY.md # 详细实现总结
└── (训练输出)
    ├── epoch_1/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── train_script.py
    └── training_metrics.png
```

## 监控指标

### 关键指标
- **Relative Advantage**: 应该围绕0波动，标准差稳定
- **KL Divergence**: 应该在TARGET_KL附近
- **KL Coefficient**: 应该自适应调整
- **Reward**: 应该逐步上升

### 异常情况处理
- 如果相对优势标准差过大：增大GROUP_SIZE
- 如果KL散度过大：检查KL系数是否正常调整
- 如果奖励不增长：检查奖励模型和学习率

## 参考文献

1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
3. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2512.07611)
4. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)