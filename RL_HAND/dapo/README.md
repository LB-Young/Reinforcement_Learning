# DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)

## 概述

DAPO是GRPO的改进版本，专门针对长链推理任务进行优化。本实现基于PPO和GRPO的代码结构，保持了一致的编程风格。

## 核心改进

### 1. 🔥 Clip-Higher（非对称裁剪）
- **GRPO**: 对称裁剪 `[1-0.2, 1+0.2] = [0.8, 1.2]`
- **DAPO**: 非对称裁剪 `[1-0.2, 1+0.28] = [0.8, 1.28]`
- **作用**: 防止熵崩溃，保持探索能力

### 2. 🔥 Token-Level Loss（token级别损失）
- **GRPO**: Sample-Level，短回复权重更大
- **DAPO**: Token-Level，所有token权重相同
- **作用**: 鼓励详细推理，减少奖励黑客

### 3. 🔥 Dynamic Sampling（动态采样）
- 当所有回复奖励相同时，继续采样直到有差异
- 确保每个问题都有有效的训练信号
- 防止训练后期信号消失

### 4. 🔥 移除KL惩罚
- **GRPO**: `KL_COEF = 0.01`
- **DAPO**: `KL_COEF = 0.0`
- **作用**: 允许策略更自由探索，适合长链推理

### 5. 🔥 过长回复处理
- 过滤超长回复，避免不公平惩罚
- 可选的软惩罚机制

## 配置参数

```python
# DAPO特有参数
CLIP_RANGE_LOW = 0.2        # 下界裁剪范围
CLIP_RANGE_HIGH = 0.28      # 上界裁剪范围（Clip-Higher）
KL_COEF = 0.0               # 移除KL惩罚
USE_DYNAMIC_SAMPLING = True # 动态采样
USE_TOKEN_LEVEL_LOSS = True # Token级别损失
USE_OVERLONG_FILTERING = True # 过长回复过滤
MAX_RESPONSE_LENGTH = 256   # 最大回复长度
```

## 使用方法

### 基本训练
```python
python RL_HAND/dapo/dapo.py
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

## 性能对比

根据论文实验结果（AIME 2024，Qwen2.5-32B）：

| 模型 | 分数 | 训练步数 | 效率提升 |
|------|------|----------|----------|
| GRPO | 47 | 100% | - |
| DAPO | 50 | 50% | 2x |

## 训练指标

DAPO会自动记录和可视化以下指标：
- Policy Loss（策略损失）
- Entropy Loss（熵损失）
- Reward（奖励）
- Entropy（熵值）
- Dynamic Resample Rate（动态重采样率）
- Average Response Length（平均回复长度）

## 文件结构

```
RL_HAND/dapo/
├── dapo.py          # 主训练脚本
├── README.md        # 说明文档
└── (训练输出)
    ├── epoch_1/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   └── train_script.py
    └── training_metrics.png
```

## 适用场景

### DAPO更适合：
- 数学推理任务
- 代码生成任务
- 需要详细解释的复杂问答
- 长链推理任务
- 有充足计算资源的场景

### GRPO更适合：
- 短文本生成
- 简单问答
- 计算资源有限的场景
- 需要严格控制策略偏离的任务

## 注意事项

1. **显存需求**: DAPO比GRPO需要更多显存（Token-Level计算）
2. **计算开销**: 动态采样会增加约25%的计算时间
3. **超参数**: 建议根据具体任务调整裁剪范围和采样参数
4. **模型兼容**: 支持所有HuggingFace格式的语言模型

## 参考文献

1. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2512.07611)
2. DeepSeek-R1 Technical Report
3. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models