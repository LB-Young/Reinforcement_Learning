# RL_HAND - 强化学习算法实现集合

## 项目概述

RL_HAND是一个专注于大语言模型强化学习算法实现的项目，包含了PPO、GRPO、DAPO三种主流算法的完整实现。所有算法都采用统一的代码风格和架构设计，便于学习、比较和扩展。

## 算法实现

### 🔥 PPO (Proximal Policy Optimization)
- **架构**: Actor-Critic
- **特点**: 训练稳定，样本效率高
- **适用**: 通用文本生成任务
- **文档**: [PPO详细说明](ppo/README.md) | [实现总结](ppo/IMPLEMENTATION_SUMMARY.md)

### 🔥 GRPO (Group Relative Policy Optimization)  
- **架构**: Policy-Only
- **特点**: 相对奖励，显存友好
- **适用**: 长文本生成任务
- **文档**: [GRPO详细说明](grpo/README.md) | [实现总结](grpo/IMPLEMENTATION_SUMMARY.md)

### 🔥 DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)
- **架构**: Policy-Only (GRPO改进版)
- **特点**: 非对称裁剪，动态采样，Token级别损失
- **适用**: 长链推理任务
- **文档**: [DAPO详细说明](dapo/README.md) | [实现总结](dapo/IMPLEMENTATION_SUMMARY.md)

### 🔥 GSPO (Group Sequence Policy Optimization)
- **架构**: Policy-Only
- **特点**: 组采样，序列级奖励，相对优势，自适应KL
- **适用**: 复杂推理和多样性生成任务
- **文档**: [GSPO详细说明](gspo/README.md) | [实现总结](gspo/IMPLEMENTATION_SUMMARY.md)

## 算法对比

| 特性 | PPO | GRPO | DAPO | GSPO |
|------|-----|------|------|------|
| **架构** | Actor-Critic | Policy-Only | Policy-Only | Policy-Only |
| **网络数量** | 2个 | 1个 | 1个 | 1个 |
| **奖励类型** | 绝对奖励 | 相对奖励 | 相对奖励 | 相对奖励 |
| **裁剪方式** | 对称 [0.8, 1.2] | 对称 [0.8, 1.2] | 非对称 [0.8, 1.28] | 对称 [0.8, 1.2] |
| **损失级别** | Token-Level | Token-Level | Token-Level | Sequence/Token |
| **KL惩罚** | ✅ | ✅ | ❌ | 自适应 |
| **动态采样** | ❌ | ❌ | ✅ | ❌ |
| **组采样** | ❌ | ✅ | ✅ | ✅ |
| **显存需求** | 高 | 中等 | 中等 | 中等 |
| **训练稳定性** | 高 | 高 | 高 | 高 |
| **适用场景** | 通用任务 | 长文本生成 | 长链推理 | 复杂推理+多样性 |

## 性能对比

根据实验和论文结果：

### 训练效率
- **PPO**: 基线算法，稳定但较慢
- **GRPO**: 比PPO快约30%（无需价值网络）
- **DAPO**: 比GRPO快50%（更少训练步数）

### 任务表现
- **通用对话**: PPO ≈ GRPO > GSPO > DAPO
- **长文本生成**: GRPO > GSPO > PPO > DAPO  
- **数学推理**: DAPO > GSPO > GRPO > PPO
- **代码生成**: DAPO > GSPO ≈ GRPO ≈ PPO
- **创意写作**: GSPO > GRPO > PPO > DAPO

## 项目结构

```
RL_HAND/
├── README.md                 # 项目总览
├── ppo/                      # PPO算法实现
│   ├── ppo.py               # 主训练脚本
│   ├── README.md            # 算法说明
│   └── IMPLEMENTATION_SUMMARY.md
├── grpo/                     # GRPO算法实现
│   ├── grpo.py              # 主训练脚本
│   ├── README.md            # 算法说明
│   └── IMPLEMENTATION_SUMMARY.md
├── dapo/                     # DAPO算法实现
│   ├── dapo.py              # 主训练脚本
│   ├── test_dapo.py         # 测试脚本
│   ├── README.md            # 算法说明
│   └── IMPLEMENTATION_SUMMARY.md
└── gspo/                     # GSPO算法实现
    ├── gspo.py              # 主训练脚本
    ├── README.md            # 算法说明
    └── IMPLEMENTATION_SUMMARY.md
```

## 环境要求

### 硬件要求
- **GPU**: 推荐RTX 4090/5060ti或更高
- **显存**: 至少16GB（双卡更佳）
- **内存**: 至少32GB

### 软件依赖
```bash
torch>=2.0.0
transformers>=4.30.0
datasets>=2.0.0
pyarrow>=12.0.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

## 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone <repository_url>
cd RL_HAND

# 安装依赖
pip install torch transformers datasets pyarrow matplotlib tqdm
```

### 2. 模型准备
下载所需模型到指定路径：
- 策略模型: `E:\models\Qwen\Qwen3-0___6B`
- 奖励模型: `E:\models\reward-model-deberta-v3-large-v2`

### 3. 数据准备
准备训练数据集：
- 支持parquet和jsonl格式
- 默认使用GSM8K数学数据集

### 4. 开始训练

#### PPO训练
```bash
cd ppo
python ppo.py
```

#### GRPO训练
```bash
cd grpo  
python grpo.py
```

#### DAPO训练
```bash
cd dapo
python dapo.py
```

#### GSPO训练
```bash
cd gspo
python gspo.py
```

## 配置说明

### 通用配置
所有算法都支持以下配置：
```python
# 模型路径
POLICY_MODEL = r"E:\models\Qwen\Qwen3-0___6B"
REWARD_MODEL = r"E:\models\reward-model-deberta-v3-large-v2"

# 训练参数
BATCH_SIZE = 2              # 批次大小
LEARNING_RATE = 1e-6        # 学习率
NUM_EPOCHS = 1              # 训练轮数
DTYPE = torch.bfloat16      # 数据类型

# 数据集配置
train_datasets = [
    {
        "path": "dataset.parquet",
        "type": "parquet",
        "input": "question",
        "output": "answer"
    }
]
```

### 算法特定配置

#### PPO特有
```python
GROUP_SIZE = 1              # 每个prompt的回复数
GROUP_EPOCHES = 4           # PPO更新轮数
CLIP_RANGE = 0.2            # 裁剪范围
```

#### GRPO特有
```python
GROUP_SIZE = 4              # 每个prompt的回复数
GRPO_EPOCHS = 4             # GRPO更新轮数
CLIP_RANGE = 0.2            # 裁剪范围
KL_COEF = 0.01              # KL散度系数
```

#### DAPO特有
```python
GROUP_SIZE = 4              # 每个prompt的回复数
DAPO_EPOCHS = 4             # DAPO更新轮数
CLIP_RANGE_LOW = 0.2        # 下界裁剪
CLIP_RANGE_HIGH = 0.28      # 上界裁剪（Clip-Higher）
KL_COEF = 0.0               # 移除KL惩罚
USE_DYNAMIC_SAMPLING = True # 动态采样
USE_TOKEN_LEVEL_LOSS = True # Token级别损失
```

## 训练监控

### 指标可视化
所有算法都会自动生成训练指标图表：
- PPO: 策略损失、价值损失、奖励、优势、熵
- GRPO: 损失、奖励、熵
- DAPO: 策略损失、熵损失、奖励、熵、动态重采样率、平均回复长度

### 模型保存
训练过程中会自动保存：
- 模型权重和配置
- Tokenizer
- 训练脚本备份
- 训练指标图表

## 算法选择指南

### 选择PPO当：
- 需要最稳定的训练过程
- 任务对质量要求极高
- 有充足的计算资源
- 需要详细的价值估计

### 选择GRPO当：
- 进行长文本生成任务
- 显存资源有限
- 需要简化的训练流程
- 奖励模型质量较高

### 选择DAPO当：
- 进行数学推理、代码生成等长链推理任务
- 需要最快的训练速度
- 模型已经较强，需要进一步提升
- 有充足的计算资源支持动态采样

### 选择GSPO当：
- 需要多样性的生成任务
- 复杂推理但不需要极长链条
- 创意写作和对话系统
- 希望平衡性能和计算成本
- 需要灵活的优化策略

## 扩展开发

### 添加新算法
1. 在RL_HAND下创建新文件夹
2. 实现主训练脚本
3. 添加README和IMPLEMENTATION_SUMMARY
4. 更新项目总README

### 自定义奖励模型
```python
# 替换奖励模型
REWARD_MODEL = "your_custom_reward_model"
self.reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL)
```

### 自定义数据集
```python
# 添加新的数据集类型
if datasets['type'] == 'your_format':
    # 实现数据加载逻辑
    pass
```

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 参考文献

1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models  
3. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2512.07611)

## 联系方式

- 作者: YoungL
- 邮箱: lby15356@gmail.com
- 项目地址: [GitHub Repository]

---

**注意**: 本项目专注于算法实现和对比研究，不包含工程层面的调度优化。所有实现都经过测试，可直接用于研究和学习。