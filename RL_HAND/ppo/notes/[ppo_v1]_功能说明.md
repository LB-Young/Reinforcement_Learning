# PPO v1 改进说明

## 概述
`ppo_v1.py` 是在原始 `ppo.py` 基础上的改进版本，添加了多个生产级功能，使训练更加稳定、可控和可追溯。

## 新增功能

### 1. 学习率调度器 (Learning Rate Scheduler)
- **Cosine Annealing**: 余弦退火调度，学习率平滑下降
- **ReduceLROnPlateau**: 基于验证集性能自适应调整学习率
- 配置参数:
  - `USE_LR_SCHEDULER`: 是否启用
  - `LR_SCHEDULER_TYPE`: "cosine" 或 "plateau"
  - `WARMUP_STEPS`: 预热步数

**优势**: 防止训练后期震荡，提高收敛稳定性

### 2. 梯度累积 (Gradient Accumulation)
- 支持在显存受限时模拟更大的批次大小
- 配置参数: `GRADIENT_ACCUMULATION_STEPS`
- 实现: 累积多个小批次的梯度后再更新参数

**优势**: 在有限显存下实现大批次训练效果

### 3. 检查点恢复机制 (Checkpoint Resume)
- 自动保存训练状态（优化器、调度器、指标历史）
- 支持从中断处继续训练
- 保存内容:
  - 模型权重（policy + critic）
  - 优化器状态
  - 学习率调度器状态
  - 训练步数和最佳指标
  - 完整的指标历史

**优势**: 训练中断后无需重新开始，节省时间和资源

### 4. 详细日志记录 (Detailed Logging)
- `TrainingLogger` 类: 统一的日志管理
- 日志文件: 带时间戳的训练日志
- 指标文件: JSONL 格式的指标记录
- 日志级别: INFO, WARNING, ERROR

**优势**: 便于调试和分析训练过程

### 5. 早停机制 (Early Stopping)
- 监控验证集奖励，防止过拟合
- 配置参数:
  - `EARLY_STOPPING_PATIENCE`: 耐心值（默认5）
  - `EARLY_STOPPING_THRESHOLD`: 改进阈值（默认0.001）
- 自动保存最佳模型

**优势**: 自动停止无效训练，节省计算资源

### 6. 显存优化 (Memory Optimization)
- 参考 GRPO/DAPO 的显存管理策略
- 及时释放中间张量
- 定期调用 `torch.cuda.empty_cache()`
- 使用 `gc.collect()` 强制垃圾回收

**优势**: 减少 OOM 错误，支持更大的模型和批次

### 7. 配置文件保存 (Config Saving)
- 自动保存训练配置到 JSON 文件
- 包含所有超参数
- 便于实验复现

**优势**: 确保实验可复现性

### 8. 验证集评估 (Validation Evaluation)
- 定期在验证集上评估模型性能
- 配置参数: `EVAL_EVERY_N_STEPS`
- 评估指标: 奖励、熵

**优势**: 及时发现过拟合，指导训练调整

### 9. Wandb 集成 (可选)
- 支持 Weights & Biases 实验跟踪
- 配置参数:
  - `USE_WANDB`: 是否启用
  - `WANDB_PROJECT`: 项目名称
- 自动记录所有训练指标

**优势**: 可视化训练过程，便于团队协作

### 10. 增强的指标记录
新增指标:
- `total_loss`: 总损失
- `kl_divergence`: KL 散度
- `learning_rate`: 当前学习率

**优势**: 更全面地监控训练状态

## 配置参数对比

### 原始版本 (ppo.py)
```python
LEARNING_RATE = 1e-6
NUM_EPOCHES = 1
BATCH_SIZE = 4
GROUP_SIZE = 1
GROUP_EPOCHES = 4
CLIP_RANGE = 0.2
```

### 改进版本 (ppo_v1.py)
```python
# 基础参数（保持不变）
LEARNING_RATE = 1e-6
NUM_EPOCHES = 1
BATCH_SIZE = 4
GROUP_SIZE = 1
GROUP_EPOCHES = 4
CLIP_RANGE = 0.2

# 新增参数
ENTROPY_COEF = 0.01              # 熵系数
KL_COEF = 0.01                   # KL散度系数
VALUE_LOSS_COEF = 0.5            # 价值损失系数
MAX_GRAD_NORM = 1.0              # 梯度裁剪
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积
USE_LR_SCHEDULER = True          # 学习率调度
LR_SCHEDULER_TYPE = "cosine"     # 调度器类型
WARMUP_STEPS = 100               # 预热步数
EARLY_STOPPING_PATIENCE = 5      # 早停耐心
SAVE_EVERY_N_STEPS = 100         # 保存频率
EVAL_EVERY_N_STEPS = 50          # 评估频率
USE_WANDB = False                # Wandb集成
```

## 使用方法

### 基本训练
```bash
python ppo_v1.py
```

### 从检查点恢复
脚本会自动检测最新的检查点并询问是否恢复：
```
发现检查点: checkpoints/step_500
是否从检查点恢复训练？(y/n): y
```

### 启用 Wandb
```python
USE_WANDB = True
WANDB_PROJECT = "my-ppo-experiment"
```

## 文件结构

训练后会生成以下文件结构：
```
OUTPUT_DIR/
├── training_config.json              # 训练配置
├── training_metrics_detailed.png     # 详细指标图表
├── ppo_metrics_with_entropy.png      # PPO专用图表
├── checkpoints/                      # 检查点目录
│   ├── step_100/
│   │   ├── policy/                   # 策略模型
│   │   ├── critic/                   # 价值模型
│   │   ├── training_state.pt         # 训练状态
│   │   └── config.json               # 配置快照
│   └── step_200/
│       └── ...
├── logs/                             # 日志目录
│   ├── training_20260120_143022.log  # 训练日志
│   └── metrics.jsonl                 # 指标记录
└── epoch_1/                          # Epoch结束保存
    ├── policy/
    ├── critic/
    └── train_script.py
```

## 性能优化建议

### 显存受限时
```python
BATCH_SIZE = 2                      # 减小批次大小
GRADIENT_ACCUMULATION_STEPS = 4     # 增加梯度累积
```

### 加速训练
```python
SAVE_EVERY_N_STEPS = 500           # 减少保存频率
EVAL_EVERY_N_STEPS = 200           # 减少评估频率
USE_LR_SCHEDULER = True            # 启用学习率调度
LR_SCHEDULER_TYPE = "cosine"       # 使用余弦调度
```

### 提高稳定性
```python
MAX_GRAD_NORM = 0.5                # 更严格的梯度裁剪
EARLY_STOPPING_PATIENCE = 10       # 增加早停耐心
VALUE_LOSS_COEF = 1.0              # 增加价值损失权重
```

## 与原版的兼容性

- ✅ 完全兼容原版的数据集格式
- ✅ 保持相同的模型架构
- ✅ 保持相同的核心算法逻辑
- ✅ 可以使用原版训练的模型继续训练

## 注意事项

1. **首次运行**: 建议先用小数据集测试，确保所有功能正常
2. **Wandb**: 如果不需要，设置 `USE_WANDB = False`
3. **检查点**: 定期清理旧检查点以节省磁盘空间
4. **验证集**: 建议至少准备 5-10 个验证样本
5. **学习率调度**: 根据训练曲线选择合适的调度器类型

## 未来改进方向

- [ ] 支持分布式训练
- [ ] 添加混合精度训练 (AMP)
- [ ] 支持更多学习率调度策略
- [ ] 添加模型量化支持
- [ ] 集成 TensorBoard
- [ ] 支持自定义奖励函数
- [ ] 添加更多评估指标

## 参考

- 原始实现: `ppo.py`
- GRPO 实现: `../grpo/grpo.py`
- DAPO 实现: `../dapo/dapo.py`
- 绘图工具: `../../utils/plot_metrics.py`
