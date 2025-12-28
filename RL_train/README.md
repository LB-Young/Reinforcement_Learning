# 完整PPO-RLHF训练脚本

这是一个使用Qwen2-0.5B模型进行完整PPO-RLHF (Proximal Policy Optimization with Reinforcement Learning from Human Feedback) 训练的实现。

## 特性

- **策略模型**: Qwen2-0.5B
- **Critic模型**: Qwen2-0.5B (带有value head)
- **参考模型**: Qwen2-0.5B (冻结参数，用于KL散度约束)
- **奖励模型**: OpenAssistant/reward-model-deberta-v3-large-v2
- **训练数据**: Anthropic HH-RLHF数据集
- **完整PPO-RLHF**: 包含KL散度惩罚、熵正则化、自适应KL系数调整

## 核心改进

### 1. KL散度约束
- 防止策略偏离原始预训练模型太远
- 使用参考模型计算KL散度: `KL(π_new || π_ref)`
- 自适应调整KL惩罚系数

### 2. 完整的损失函数
```python
total_loss = policy_loss + vf_coef * value_loss + entropy_loss
adjusted_reward = reward - kl_coef * kl_penalty
```

### 3. 熵正则化
- 鼓励策略探索，防止过早收敛
- 可配置的熵系数

### 4. 自适应KL调整
- 根据实际KL散度动态调整惩罚系数
- 目标KL散度可配置

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本训练

```bash
python ppo_train.py
```

### 关键配置参数

```python
@dataclass
class PPOConfig:
    # KL散度相关
    kl_coef: float = 0.2          # KL散度惩罚系数
    target_kl: float = 0.01       # 目标KL散度
    adaptive_kl: bool = True      # 自适应KL调整
    
    # 熵正则化
    entropy_coef: float = 0.01    # 熵正则化系数
    
    # PPO核心参数
    clip_range: float = 0.2       # PPO裁剪范围
    vf_coef: float = 0.1         # 价值函数损失权重
```

## 训练流程

### 完整PPO-RLHF流程：

1. **策略生成**: 使用当前策略模型生成回复
2. **奖励计算**: 奖励模型评估回复质量
3. **KL惩罚**: 计算与参考模型的KL散度
4. **调整奖励**: `adjusted_reward = reward - kl_coef * kl_penalty`
5. **优势计算**: 使用调整后奖励计算优势函数
6. **PPO更新**: 
   - 策略损失 (clip目标)
   - 价值损失 (MSE)
   - 熵损失 (探索鼓励)
7. **自适应调整**: 根据KL散度调整惩罚系数

### 关键算法组件：

**PPO Clip目标**:
```python
ratio = exp(log_π_new - log_π_old)
L_clip = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

**KL散度惩罚**:
```python
KL_penalty = log_π_new - log_π_ref
adjusted_reward = reward - β * KL_penalty
```

**自适应KL调整**:
```python
if KL > 2 * target_KL: β *= 1.5
elif KL < 0.5 * target_KL: β *= 0.5
```

## 训练指标

新增的关键指标：
- `kl_divergence`: 与参考模型的KL散度
- `kl_coef`: 当前KL惩罚系数
- `entropy_loss`: 熵正则化损失
- `policy_loss`: PPO策略损失
- `value_loss`: 价值函数损失

## 与标准PPO的区别

| 特性 | 标准PPO | PPO-RLHF |
|------|---------|-----------|
| 参考模型 | 无 | 冻结的预训练模型 |
| KL约束 | 无 | KL(π\|\|π_ref) |
| 奖励调整 | reward | reward - β*KL |
| 熵正则化 | 可选 | 标准配置 |
| 自适应调整 | 无 | 自适应KL系数 |

## 超参数调优建议

### KL散度相关：
- `kl_coef`: 0.1-0.5，控制与原模型的偏离程度
- `target_kl`: 0.01-0.05，目标KL散度
- `adaptive_kl`: 建议启用

### 训练稳定性：
- 较小的学习率 (1e-6 to 1e-5)
- 适中的clip_range (0.1-0.3)
- 合适的batch_size (4-16)

## 故障排除

### KL散度过大
- 减小学习率
- 增加kl_coef
- 减少ppo_epochs

### 训练不稳定
- 启用adaptive_kl
- 调整target_kl
- 检查梯度裁剪

### 性能优化
- 使用混合精度训练
- 批量计算log概率
- 优化内存使用