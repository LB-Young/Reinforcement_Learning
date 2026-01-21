# PPO 版本对比

## 三个版本概览

| 特性 | ppo.py | ppo_v1.py | ppo_v2.py |
|------|--------|-----------|-----------|
| **定位** | 学习版 | 生产版 | 高效版 |
| **代码行数** | ~350 | ~450 | ~550 |
| **样本效率** | 1.0x | 1.0x | 2-4x |
| **学习率调度** | ❌ | ✅ | ✅ |
| **梯度累积** | ❌ | ✅ | ✅ |
| **检查点恢复** | ❌ | ✅ | ✅ |
| **详细日志** | ❌ | ✅ | ✅ |
| **早停机制** | ❌ | ✅ | ✅ |
| **验证集评估** | ❌ | ✅ | ✅ |
| **Wandb 集成** | ❌ | ✅ | ✅ |
| **经验回放** | ❌ | ❌ | ✅ |
| **优先级采样** | ❌ | ❌ | ✅ (可选) |
| **样本管理** | ❌ | ❌ | ✅ |

## 详细对比

### ppo.py - 标准实现

**特点**:
- ✅ 代码简洁，易于理解
- ✅ 完整的 PPO 算法实现
- ✅ 适合学习和快速原型
- ❌ 功能较少
- ❌ 样本效率低

**适用场景**:
- 学习 PPO 算法
- 快速验证想法
- 简单实验

**示例代码**:
```python
from ppo import PPOTrainer

dataset = PPODataset(prompts)
trainer = PPOTrainer()
trainer.train(dataset)
```

### ppo_v1.py - 生产级实现

**特点**:
- ✅ 生产级功能完善
- ✅ 支持长时间训练
- ✅ 完善的日志和监控
- ✅ 检查点恢复
- ✅ 早停机制
- ❌ 样本效率仍为 1.0x

**适用场景**:
- 生产环境训练
- 长时间训练任务
- 团队协作项目
- 需要实验管理

**示例代码**:
```python
from ppo_v1 import PPOTrainer

train_dataset = PPODataset(train_prompts)
eval_dataset = PPODataset(eval_prompts)

trainer = PPOTrainer(resume_from_checkpoint=None)
trainer.train(train_dataset, eval_dataset)
```

### ppo_v2.py - 高效实现

**特点**:
- ✅ 包含 v1 所有功能
- ✅ 经验回放机制
- ✅ 样本效率 2-4x
- ✅ 优先级采样（可选）
- ✅ 智能样本管理
- ⚠️ 配置稍复杂
- ⚠️ 需要更多显存

**适用场景**:
- 数据生成成本高
- 需要最大化样本利用率
- 有足够显存
- 对样本效率有要求

**示例代码**:
```python
from ppo_v2 import PPOTrainerWithReplay

train_dataset = PPODataset(train_prompts)

trainer = PPOTrainerWithReplay()
trainer.train(train_dataset)
```

## 性能对比

### 训练时间（相同数据量）

假设训练 1000 个样本：

| 版本 | 数据生成 | 训练时间 | 总时间 | 相对速度 |
|------|---------|---------|--------|---------|
| ppo.py | 100% | 100% | 100% | 1.0x |
| ppo_v1.py | 100% | 105% | 102.5% | 0.98x |
| ppo_v2.py | 40% | 120% | 68% | 1.47x |

**说明**:
- v1 略慢是因为额外的日志和检查点
- v2 快是因为减少了数据生成（样本重用）

### 样本效率

| 版本 | 生成样本 | 训练样本 | 样本效率 |
|------|---------|---------|---------|
| ppo.py | 1000 | 1000 | 1.0x |
| ppo_v1.py | 1000 | 1000 | 1.0x |
| ppo_v2.py | 1000 | 2500 | 2.5x |

### 显存占用

| 版本 | 模型显存 | 缓冲区 | 总显存 |
|------|---------|--------|--------|
| ppo.py | 100% | 0% | 100% |
| ppo_v1.py | 100% | 0% | 100% |
| ppo_v2.py | 100% | 5-10% | 105-110% |

## 功能矩阵

### 核心功能

| 功能 | ppo.py | ppo_v1.py | ppo_v2.py |
|------|--------|-----------|-----------|
| PPO 算法 | ✅ | ✅ | ✅ |
| Policy 网络 | ✅ | ✅ | ✅ |
| Critic 网络 | ✅ | ✅ | ✅ |
| Reward 模型 | ✅ | ✅ | ✅ |
| Token-level Loss | ✅ | ✅ | ✅ |
| KL 惩罚 | ✅ | ✅ | ✅ |
| 熵正则化 | ✅ | ✅ | ✅ |

### 训练功能

| 功能 | ppo.py | ppo_v1.py | ppo_v2.py |
|------|--------|-----------|-----------|
| 学习率调度 | ❌ | ✅ | ✅ |
| 梯度累积 | ❌ | ✅ | ✅ |
| 梯度裁剪 | ✅ | ✅ | ✅ |
| 混合精度 | ❌ | ❌ | ❌ |

### 数据管理

| 功能 | ppo.py | ppo_v1.py | ppo_v2.py |
|------|--------|-----------|-----------|
| 经验回放 | ❌ | ❌ | ✅ |
| 优先级采样 | ❌ | ❌ | ✅ (可选) |
| 样本过期 | ❌ | ❌ | ✅ |
| 重用限制 | ❌ | ❌ | ✅ |

### 监控和日志

| 功能 | ppo.py | ppo_v1.py | ppo_v2.py |
|------|--------|-----------|-----------|
| 进度条 | ✅ | ✅ | ✅ |
| 文件日志 | ❌ | ✅ | ✅ |
| JSONL 指标 | ❌ | ✅ | ✅ |
| Wandb | ❌ | ✅ | ✅ |
| 缓冲区统计 | ❌ | ❌ | ✅ |
| 样本效率 | ❌ | ❌ | ✅ |

### 模型管理

| 功能 | ppo.py | ppo_v1.py | ppo_v2.py |
|------|--------|-----------|-----------|
| 模型保存 | ✅ | ✅ | ✅ |
| 检查点 | ❌ | ✅ | ✅ |
| 恢复训练 | ❌ | ✅ | ✅ |
| 早停 | ❌ | ✅ | ✅ |
| 验证集评估 | ❌ | ✅ | ✅ |

## 使用建议

### 选择流程图

```
开始
  ↓
是否需要学习 PPO？
  ├─ 是 → ppo.py
  └─ 否 ↓
是否需要长时间训练？
  ├─ 否 → ppo.py
  └─ 是 ↓
数据生成是否昂贵？
  ├─ 否 → ppo_v1.py
  └─ 是 ↓
是否有足够显存？
  ├─ 否 → ppo_v1.py
  └─ 是 → ppo_v2.py
```

### 具体场景

#### 场景 1: 学习和研究
**推荐**: ppo.py
**原因**: 代码简洁，易于理解和修改

#### 场景 2: 快速实验
**推荐**: ppo.py 或 ppo_v1.py
**原因**: 启动快，配置简单

#### 场景 3: 生产训练（数据充足）
**推荐**: ppo_v1.py
**原因**: 功能完善，稳定可靠

#### 场景 4: 生产训练（数据受限）
**推荐**: ppo_v2.py
**原因**: 样本效率高，节省数据生成成本

#### 场景 5: 团队协作
**推荐**: ppo_v1.py 或 ppo_v2.py
**原因**: 日志完善，支持 Wandb

#### 场景 6: 显存受限
**推荐**: ppo_v1.py
**原因**: v2 需要额外显存存储缓冲区

## 迁移指南

### 从 ppo.py 到 ppo_v1.py

**改动**: 最小
```python
# 原来
from ppo import PPOTrainer
trainer = PPOTrainer()
trainer.train(dataset)

# 现在
from ppo_v1 import PPOTrainer
trainer = PPOTrainer()
trainer.train(dataset, eval_dataset)  # 可选验证集
```

### 从 ppo_v1.py 到 ppo_v2.py

**改动**: 中等
```python
# 原来
from ppo_v1 import PPOTrainer
trainer = PPOTrainer()

# 现在
from ppo_v2 import PPOTrainerWithReplay
trainer = PPOTrainerWithReplay()

# 添加配置
USE_REPLAY_BUFFER = True
REPLAY_BUFFER_SIZE = 1000
```

### 从 ppo.py 直接到 ppo_v2.py

**改动**: 较大
1. 修改导入
2. 添加 v1 的配置
3. 添加 v2 的配置
4. 准备验证集（可选）

## 配置对比

### 基础配置（三个版本都有）

```python
LEARNING_RATE = 1e-6
BATCH_SIZE = 4
GROUP_SIZE = 1
GROUP_EPOCHES = 4
CLIP_RANGE = 0.2
```

### v1 新增配置

```python
GRADIENT_ACCUMULATION_STEPS = 1
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "cosine"
EARLY_STOPPING_PATIENCE = 5
SAVE_EVERY_N_STEPS = 100
EVAL_EVERY_N_STEPS = 50
USE_WANDB = False
```

### v2 新增配置

```python
USE_REPLAY_BUFFER = True
REPLAY_BUFFER_SIZE = 1000
MIN_REPLAY_SIZE = 100
REPLAY_SAMPLE_SIZE = 32
USE_PRIORITY_SAMPLING = False
MAX_REPLAY_REUSE = 4
STALENESS_THRESHOLD = 100
```

## 实验对比

### 实验设置
- 数据集: GSM8K (20 samples)
- 模型: Qwen3-0.6B
- 硬件: 2x RTX 5060Ti 16GB

### 结果（模拟）

| 指标 | ppo.py | ppo_v1.py | ppo_v2.py |
|------|--------|-----------|-----------|
| 训练时间 | 60 min | 62 min | 42 min |
| 最终奖励 | 1.45 | 1.52 | 1.58 |
| 样本效率 | 1.0x | 1.0x | 2.8x |
| 峰值显存 | 14GB | 14GB | 15GB |
| 日志大小 | 1MB | 50MB | 60MB |

## 总结

### ppo.py
- 🎯 **最适合**: 学习、快速实验
- ⭐ **优势**: 简洁、易懂
- ⚠️ **限制**: 功能少

### ppo_v1.py
- 🎯 **最适合**: 生产训练、团队协作
- ⭐ **优势**: 功能完善、稳定
- ⚠️ **限制**: 样本效率 1.0x

### ppo_v2.py
- 🎯 **最适合**: 数据受限、高效训练
- ⭐ **优势**: 样本效率 2-4x
- ⚠️ **限制**: 配置复杂、显存稍高

## 相关文档

- [ppo.py](ppo.py) - 标准实现
- [ppo_v1.py](ppo_v1.py) - 生产实现
- [ppo_v2.py](ppo_v2.py) - 高效实现
- [PPO_V1_IMPROVEMENTS.md](PPO_V1_IMPROVEMENTS.md) - v1 改进说明
- [REPLAY_BUFFER_GUIDE.md](REPLAY_BUFFER_GUIDE.md) - 经验回放指南
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - 使用指南
