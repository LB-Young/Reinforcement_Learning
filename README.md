# 🚀 Reinforcement Learning 实战项目

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

**从基础到前沿的强化学习算法实现集合**

[快速开始](#-快速开始) • [算法实现](#-算法实现) • [项目结构](#-项目结构) • [文档](#-文档)

</div>

---

## 📖 项目简介

本项目是一个**完整的强化学习算法实现集合**，涵盖从经典算法到最新的大语言模型强化学习方法。项目分为两大部分：

- **RL_basic**: 经典强化学习算法（Q-Learning, SARSA, DQN, Policy Gradient）
- **RL_HAND**: 大语言模型强化学习算法（PPO, GRPO, DAPO, GSPO等）

所有代码都经过实际测试，配有详细文档，适合学习、研究和实际应用。

---

## 🎓 算法实现

### 📘 经典强化学习算法 (RL_basic)

基于迷宫环境的经典算法实现，适合入门学习：

| 算法 | 类型 | 特点 | 文件 |
|------|------|------|------|
| **Q-Learning** | Value-Based | Off-policy, 表格法 | `maze_value_iteration_q_learning.py` |
| **SARSA** | Value-Based | On-policy, 表格法 | `maze_value_iteration_TD_sarsa.py` |
| **Expected SARSA** | Value-Based | On-policy, 期望更新 | `maze_value_iteration_TD_expected_sarsa.py` |
| **DQN** | Value-Based | Deep Q-Network | `DQN_cartpole.py` |
| **Policy Gradient** | Policy-Based | 策略梯度 | `maze_policy_gradient.py` |
| **Deep PG** | Policy-Based | 深度策略梯度 | `deep_policy_gradient_cartpole.py` |

**适用场景**: 学习强化学习基础概念，理解算法原理

---

### 🔥 大语言模型强化学习 (RL_HAND)


---

## 🚀 快速开始

### 1. 环境准备

#### 系统要求
- **Python**: 3.8+
- **CUDA**: 11.8+ (推荐)
- **GPU**: 16GB+ 显存（推荐 RTX 4090/5060Ti）
- **内存**: 32GB+

#### 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/Reinforcement_Learning.git
cd Reinforcement_Learning

# 安装依赖
pip install -r requirements.txt

# 或手动安装
pip install torch>=2.0.0 transformers>=4.30.0 datasets>=2.0.0 \
    pyarrow>=12.0.0 matplotlib>=3.5.0 tqdm>=4.64.0
```

---

### 2. 经典算法快速体验

```bash
# 进入基础算法目录
cd RL_basic

# 运行 Q-Learning
python maze_value_iteration_q_learning.py

# 运行 DQN
python DQN_cartpole.py

# 运行策略梯度
python maze_policy_gradient.py
```

---

### 3. 大语言模型 RLHF 训练

#### 准备模型

下载所需模型（或修改代码中的路径）：
```python
# 策略模型（如 Qwen, LLaMA 等）
POLICY_MODEL = "path/to/your/model"

# 奖励模型
REWARD_MODEL = "path/to/reward/model"
```

#### 准备数据

支持两种格式：

**Parquet 格式**:
```python
train_datasets = [{
    "path": "data.parquet",
    "type": "parquet",
    "input": "question",
    "output": "answer"
}]
```

**JSONL 格式**:
```python
train_datasets = [{
    "path": "data.jsonl",
    "type": "jsonl",
    "input": "problem",
    "output": "solution"
}]
```

#### 开始训练

```bash
# PPO 训练（学习版）
cd RL_HAND/ppo
python ppo.py

# PPO 训练（生产版，推荐）
python ppo_v1.py

# PPO 训练（高效版，带经验回放）
python ppo_v2.py

# GRPO 训练
cd ../grpo
python grpo.py

# DAPO 训练
cd ../dapo
python dapo.py

# GSPO 训练
cd ../gspo
python gspo.py
```

---

## ⚙️ 配置说明

### 基础配置

所有 RL_HAND 算法都支持以下配置：

```python
# 模型路径
POLICY_MODEL = "path/to/policy/model"
REWARD_MODEL = "path/to/reward/model"

# 训练参数
BATCH_SIZE = 4              # 批次大小
LEARNING_RATE = 1e-6        # 学习率
NUM_EPOCHS = 1              # 训练轮数
DTYPE = torch.bfloat16      # 数据类型

# 输出目录
OUTPUT_DIR = "output/path"
```

### 算法特定配置

详见各算法的文档：
- PPO: [ppo_v1]_使用指南.md
- GRPO: grpo/README.md
- DAPO: dapo/README.md
- GSPO: gspo/README.md

---

## 📈 训练监控

### 自动生成的内容

训练过程中会自动生成：
- ✅ 训练指标图表（PNG）
- ✅ 模型检查点
- ✅ 训练日志（JSONL）
- ✅ 配置文件备份

### 可视化工具

使用 `utils/plot_metrics.py` 绘制自定义图表：

```python
from utils.plot_metrics import plot_training_metrics

plot_training_metrics(
    metrics_history=your_metrics,
    save_path="metrics.png"
)
```

---

## 🔧 高级功能

### PPO v1 生产级功能
- ✅ 学习率调度（Cosine/Plateau）
- ✅ 梯度累积
- ✅ 检查点恢复
- ✅ 早停机制
- ✅ 验证集评估
- ✅ Wandb 集成
- ✅ 详细日志

### PPO v2 经验回放
- ✅ 经验回放缓冲区
- ✅ 优先级采样
- ✅ 重要性采样修正
- ✅ 样本管理策略
- ✅ 样本效率提升 2-4x

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 贡献方向
- 🎯 添加新的算法实现
- 📚 改进文档和教程
- 🐛 修复 Bug
- ✨ 添加新功能
- 🧪 添加测试用例

---

## 📝 更新日志

### v2.0.0 (2026-01-20)
- ✨ 新增 PPO v2（经验回放版本）
- ✨ 新增 GSPO 算法实现
- 📚 重组 PPO 文档结构
- 📚 添加 12 个详细文档
- 🔧 优化代码结构

### v1.0.0 (2026-01-18)
- ✨ 新增 PPO v1（生产级版本）
- ✨ 新增 GRPO 算法实现
- ✨ 新增 DAPO 算法实现
- 📚 完善文档系统

---

## 📖 参考文献

### 经典算法
1. Sutton & Barto - Reinforcement Learning: An Introduction
2. Mnih et al. - Playing Atari with Deep Reinforcement Learning (DQN)
3. Schulman et al. - Proximal Policy Optimization Algorithms (PPO)

### 大语言模型 RL
1. [PPO 原论文](https://arxiv.org/abs/1707.06347)
2. DeepSeekMath: Pushing the Limits of Mathematical Reasoning
3. [DAPO 论文](https://arxiv.org/abs/2512.07611)
4. GRPO: Group Relative Policy Optimization

---

## 📧 联系方式

- **作者**: YoungL
- **邮箱**: lby15356@gmail.com
- **项目**: [GitHub Repository](https://github.com/yourusername/Reinforcement_Learning)

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- 感谢 [chunhuizhang/bilibili_vlogs](https://github.com/chunhuizhang/bilibili_vlogs) 提供的基础算法参考
- 感谢 OpenAI、Qwen 等团队的开源贡献

---

## ⭐ Star History

如果这个项目对你有帮助，请给个 Star ⭐️

---

<div align="center">

**[⬆ 回到顶部](#-reinforcement-learning-实战项目)**

Made with ❤️ by YoungL

</div>
