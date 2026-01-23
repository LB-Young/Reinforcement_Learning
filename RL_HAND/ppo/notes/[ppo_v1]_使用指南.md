# PPO v1 使用指南

## 快速开始

### 1. 基本训练

```bash
cd RL_HAND/ppo
python ppo.py
```

### 2. 自定义配置

编辑 `ppo.py` 中的配置部分：

```python
# 模型路径
ACTOR_MODEL = r"your/model/path"
CRITIC_MODEL = r"your/model/path"
REWARD_MODEL = r"your/reward/model/path"

# 训练参数
LEARNING_RATE = 1e-6
BATCH_SIZE = 4
NUM_EPOCHES = 3

# 输出目录
OUTPUT_DIR = r"your/output/path"
```

### 3. 从检查点恢复

脚本会自动检测并询问：

```
发现检查点: checkpoints/step_500
是否从检查点恢复训练？(y/n): y
```

或者手动指定：

```python
trainer = PPOTrainer(resume_from_checkpoint="path/to/checkpoint")
```

## 常见配置场景

### 场景 1: 显存受限（8GB GPU）

```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # 有效批次 = 1 * 8 = 8
GROUP_SIZE = 1
SAVE_EVERY_N_STEPS = 200
EVAL_EVERY_N_STEPS = 100
```

### 场景 2: 快速实验（16GB GPU）

```python
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
GROUP_SIZE = 2
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "cosine"
EARLY_STOPPING_PATIENCE = 3
```

### 场景 3: 生产训练（多GPU）

```python
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
GROUP_SIZE = 4
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "plateau"
EARLY_STOPPING_PATIENCE = 10
SAVE_EVERY_N_STEPS = 500
EVAL_EVERY_N_STEPS = 100
USE_WANDB = True
```

### 场景 4: 调试模式

```python
BATCH_SIZE = 2
NUM_EPOCHES = 1
SAVE_EVERY_N_STEPS = 10
EVAL_EVERY_N_STEPS = 5
USE_WANDB = False

# 在 train_main() 中
train_prompts = train_prompts[:5]  # 只用5个样本
eval_prompts = eval_prompts[:2]
```

## 监控训练

### 1. 实时监控（终端）

训练时会显示：
```
Epoch 1/3: PL:0.3245 VL:0.1234 R:1.56 A:0.23 E:0.678 LR:1.00e-06
```

- **PL**: Policy Loss（策略损失）
- **VL**: Value Loss（价值损失）
- **R**: Reward（奖励）
- **A**: Advantage（优势）
- **E**: Entropy（熵）
- **LR**: Learning Rate（学习率）

### 2. 日志文件

```bash
# 查看训练日志
tail -f OUTPUT_DIR/logs/training_*.log

# 查看指标记录
cat OUTPUT_DIR/logs/metrics.jsonl | jq
```

### 3. Wandb 监控

启用 Wandb：
```python
USE_WANDB = True
WANDB_PROJECT = "my-ppo-project"
```

访问: https://wandb.ai/your-username/my-ppo-project

### 4. 可视化图表

训练结束后自动生成：
- `training_metrics_detailed.png`: 所有指标
- `ppo_metrics_with_entropy.png`: PPO 专用指标

## 数据集格式

### JSONL 格式

```json
{"problem": "问题1", "solution": "答案1"}
{"problem": "问题2", "solution": "答案2"}
```

配置：
```python
{
    "path": "data.jsonl",
    "type": "jsonl",
    "input": "problem",
    "output": "solution"
}
```

### Parquet 格式

```python
{
    "path": "data.parquet",
    "type": "parquet",
    "input": "question",
    "output": "answer"
}
```

## 检查点管理

### 检查点结构

```
checkpoints/
├── step_100/
│   ├── policy/          # 策略模型
│   ├── critic/          # 价值模型
│   ├── training_state.pt  # 训练状态
│   └── config.json      # 配置快照
└── step_200/
    └── ...
```

### 手动加载检查点

```python
# 加载策略模型
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("checkpoints/step_500/policy")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/step_500/policy")

# 生成文本
inputs = tokenizer("你的问题", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 清理旧检查点

```bash
# 保留最新的 3 个检查点
cd OUTPUT_DIR/checkpoints
ls -t | tail -n +4 | xargs rm -rf
```

## 故障排除

### 问题 1: CUDA Out of Memory

**解决方案**:
```python
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
GROUP_SIZE = 1
```

### 问题 2: 训练不稳定

**解决方案**:
```python
MAX_GRAD_NORM = 0.5  # 更严格的梯度裁剪
LEARNING_RATE = 5e-7  # 降低学习率
VALUE_LOSS_COEF = 1.0  # 增加价值损失权重
```

### 问题 3: 奖励不增长

**解决方案**:
```python
ENTROPY_COEF = 0.02  # 增加探索
KL_COEF = 0.005      # 减少 KL 惩罚
CLIP_RANGE = 0.3     # 增大裁剪范围
```

### 问题 4: 熵崩溃

**解决方案**:
```python
ENTROPY_COEF = 0.05  # 大幅增加熵系数
CLIP_RANGE = 0.15    # 减小裁剪范围
```

### 问题 5: 检查点加载失败

**检查**:
1. 检查点路径是否正确
2. `training_state.pt` 是否存在
3. 模型架构是否匹配

## 性能优化

### 1. 加速训练

```python
# 减少保存和评估频率
SAVE_EVERY_N_STEPS = 1000
EVAL_EVERY_N_STEPS = 500

# 使用更大的批次
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
```

### 2. 节省显存

```python
# 使用梯度累积
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

# 减少生成长度
# 在 generate_response() 中
max_new_tokens=64  # 从 128 改为 64
```

### 3. 提高质量

```python
# 更多的训练轮次
NUM_EPOCHES = 5
GROUP_EPOCHES = 8

# 更细致的学习率调度
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "cosine"
WARMUP_STEPS = 200
```

## 最佳实践

1. **开始训练前**
   - 用小数据集测试（5-10 个样本）
   - 检查模型路径是否正确
   - 确认输出目录有足够空间

2. **训练过程中**
   - 监控奖励和熵的变化
   - 定期查看生成的样本质量
   - 注意显存使用情况

3. **训练结束后**
   - 查看训练曲线图表
   - 在测试集上评估
   - 清理不需要的检查点

4. **实验管理**
   - 使用有意义的 OUTPUT_DIR 名称
   - 记录重要的配置变更
   - 保存训练配置文件

## 进阶用法

### 自定义奖励函数

修改 `compute_rewards()` 方法：

```python
def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
    rewards = []
    for p, r in zip(prompts, responses):
        # 使用 reward model
        model_reward = self.reward_model_score(p, r)
        
        # 添加自定义奖励
        length_penalty = -0.01 * len(r)  # 惩罚过长回复
        diversity_bonus = self.compute_diversity(r)  # 奖励多样性
        
        total_reward = model_reward + length_penalty + diversity_bonus
        rewards.append(total_reward)
    
    return torch.stack(rewards)
```

### 多数据集训练

```python
train_datasets = [
    {
        "path": "dataset1.parquet",
        "type": "parquet",
        "input": "question",
        "output": "answer"
    },
    {
        "path": "dataset2.jsonl",
        "type": "jsonl",
        "input": "problem",
        "output": "solution"
    }
]
```

### 自定义评估指标

```python
def evaluate(self, eval_prompts: List[str]) -> Dict[str, float]:
    # ... 原有代码 ...
    
    # 添加自定义指标
    eval_metrics["custom_metric"] = self.compute_custom_metric(responses)
    
    return eval_metrics
```

## 相关文档

- [改进说明](PPO_V1_IMPROVEMENTS.md): 详细的功能说明
- [对比文档](COMPARISON.md): 与原版的对比
- [原始实现](ppo.py): 简化版实现

## 支持

如有问题，请查看：
1. 日志文件: `OUTPUT_DIR/logs/training_*.log`
2. 配置文件: `OUTPUT_DIR/training_config.json`
3. 项目 README: `../../README.md`
