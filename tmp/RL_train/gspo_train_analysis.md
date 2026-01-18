# GSPO (Group Sequence Policy Optimization) 训练脚本分析

## 1. 算法概述

GSPO (Group Sequence Policy Optimization) 是一种结合了组采样和序列级优化的策略优化算法，主要用于大语言模型的强化学习训练。

### 核心特点

1. **Group Sampling（组采样）**：为每个prompt生成多个回复进行组内比较
2. **Sequence-Level Rewards（序列级奖励）**：在完整序列级别计算奖励
3. **Relative Advantage（相对优势）**：使用组内相对优势而非绝对奖励
4. **无需Critic模型**：类似GRPO，使用组内均值作为基线
5. **灵活的优化策略**：支持序列级和token级优化

## 2. 算法原理

### 2.1 Group Sampling（组采样）

```python
def generate_responses_with_group_sampling(self, prompts: List[str]):
    """为每个prompt生成group_size个回复"""
    for prompt in prompts:
        for _ in range(self.config.group_size):
            # 生成一个回复
            response = self.policy_model.generate(...)
            all_responses.append(response)
```

**原理**：
- 对每个prompt生成K个不同的回复（K=group_size）
- 通过采样（do_sample=True）确保回复的多样性
- 这些回复构成一个"组"，用于组内比较

**优势**：
- 提供更丰富的对比信号
- 减少单个样本的方差
- 更稳定的梯度估计

### 2.2 Sequence-Level Rewards（序列级奖励）

```python
def compute_rewards(self, prompts: List[str], responses: List[str]):
    """计算序列级别奖励"""
    for prompt, response in zip(prompts, responses):
        full_text = f"{prompt} {response}"
        reward = self.reward_model(full_text)
        rewards.append(reward)
```

**原理**：
- 将完整的prompt+response作为输入
- 使用奖励模型评估整个序列的质量
- 得到标量奖励值

**与token-level的区别**：
- Sequence-level：一个序列一个奖励
- Token-level：每个token一个奖励（可选）

### 2.3 Relative Advantage（相对优势）

```python
def compute_relative_advantages(self, rewards: torch.Tensor):
    """计算组内相对优势"""
    # 重塑为组的形状 [num_groups, group_size]
    rewards_grouped = rewards.view(-1, group_size)
    
    # 计算组内均值作为基线
    group_baselines = rewards_grouped.mean(dim=1, keepdim=True)
    
    # 计算相对优势
    relative_advantages = rewards_grouped - group_baselines
    
    # 可选：组内标准化
    if use_group_normalization:
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        relative_advantages = relative_advantages / group_std
```

**数学表达**：

对于第i组的第j个样本：
```
A_ij = R_ij - mean(R_i)
```

其中：
- `R_ij`：第i组第j个样本的奖励
- `mean(R_i)`：第i组所有样本的平均奖励
- `A_ij`：相对优势

**标准化版本**：
```
A_ij = (R_ij - mean(R_i)) / std(R_i)
```

**核心思想**：
- 不使用绝对奖励，而是使用相对于组内平均的奖励
- 组内均值作为基线（baseline），替代critic模型
- 减少奖励尺度的影响，关注相对好坏

### 2.4 Policy Optimization（策略优化）

#### 序列级损失（默认）

```python
def compute_policy_loss_sequence_level(self, log_probs, old_log_probs, advantages):
    """序列级别的策略损失"""
    # 计算重要性采样比率
    ratio = torch.exp(log_probs - old_log_probs)
    
    # PPO裁剪
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages
    
    # 取最小值
    policy_loss = -torch.min(surr1, surr2).mean()
```

**数学表达**：

```
L^CLIP(θ) = -E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]
```

其中：
- `r(θ) = π_θ(a|s) / π_old(a|s)`：重要性采样比率
- `A`：优势函数（相对优势）
- `ε`：裁剪范围（默认0.2）

#### Token级损失（可选）

```python
def compute_policy_loss_token_level(self, token_log_probs_list, old_token_log_probs_list, advantages):
    """Token级别的策略损失"""
    for i, (token_log_probs, old_token_log_probs) in enumerate(...):
        advantage = advantages[i]  # 序列级优势
        
        # 对每个token计算比率
        token_ratios = torch.exp(token_log_probs - old_token_log_probs)
        
        # PPO裁剪
        surr1 = token_ratios * advantage
        surr2 = torch.clamp(token_ratios, 1-ε, 1+ε) * advantage
        
        token_loss = -torch.min(surr1, surr2).sum()
```

**特点**：
- 使用序列级优势，但在token级别应用
- 每个token使用相同的优势值
- 所有token的损失求和后平均

## 3. 与其他算法的对比

### 3.1 GSPO vs PPO

| 特性 | PPO | GSPO |
|------|-----|------|
| Critic模型 | 需要 | 不需要 |
| 基线估计 | Value function V(s) | 组内均值 |
| 采样策略 | 单个回复 | 组采样（多个回复） |
| 优势计算 | GAE | 相对优势 |
| 训练复杂度 | 高（需训练critic） | 中（只训练policy） |

### 3.2 GSPO vs GRPO

| 特性 | GRPO | GSPO |
|------|------|------|
| 组采样 | ✓ | ✓ |
| 相对奖励 | ✓ | ✓ |
| 序列级优化 | ✓ | ✓ |
| Token级优化 | ✗ | ✓（可选） |
| 奖励塑形 | 基础 | 增强 |

**GSPO的改进**：
1. 支持token级优化（可选）
2. 更灵活的优势计算方式
3. 可配置的奖励塑形策略

### 3.3 GSPO vs VAPO

| 特性 | VAPO | GSPO |
|------|------|------|
| Critic模型 | 需要 | 不需要 |
| 组采样 | ✓ | ✓ |
| Value预训练 | ✓ | ✗ |
| 解耦GAE | ✓ | ✗ |
| 长度自适应 | ✓ | ✗ |
| SIL | ✓ | ✗ |

**GSPO的优势**：
- 更简单：无需critic模型和复杂的GAE计算
- 更高效：训练速度更快
- 更稳定：组内相对优势更稳定

## 4. 训练流程

### 4.1 完整训练步骤

```
for each epoch:
    for each batch of prompts:
        # 1. Group Sampling
        responses = []
        for prompt in batch:
            for _ in range(group_size):
                response = policy_model.generate(prompt)
                responses.append(response)
        
        # 2. Compute Rewards
        rewards = reward_model(prompts, responses)
        
        # 3. Compute Relative Advantages
        advantages = compute_relative_advantages(rewards)
        
        # 4. Compute Log Probabilities
        log_probs = policy_model.log_prob(prompts, responses)
        old_log_probs = log_probs.detach()
        
        # 5. GSPO Update Loop
        for _ in range(gspo_epochs):
            new_log_probs = policy_model.log_prob(prompts, responses)
            
            # Compute losses
            policy_loss = compute_policy_loss(new_log_probs, old_log_probs, advantages)
            entropy_loss = compute_entropy_loss(new_log_probs)
            kl_loss = compute_kl_loss(new_log_probs, ref_log_probs)
            
            # Update policy
            total_loss = policy_loss + entropy_loss + kl_loss
            total_loss.backward()
            optimizer.step()
```

### 4.2 数据流示例

假设：
- batch_size = 2（2个prompt）
- group_size = 4（每个prompt生成4个回复）

```
输入：
prompts = ["问题1", "问题2"]

Group Sampling后：
responses = [
    "回复1-1", "回复1-2", "回复1-3", "回复1-4",  # 问题1的4个回复
    "回复2-1", "回复2-2", "回复2-3", "回复2-4"   # 问题2的4个回复
]

Rewards：
rewards = [r1_1, r1_2, r1_3, r1_4, r2_1, r2_2, r2_3, r2_4]

Relative Advantages：
group1_mean = mean([r1_1, r1_2, r1_3, r1_4])
group2_mean = mean([r2_1, r2_2, r2_3, r2_4])

advantages = [
    r1_1 - group1_mean,
    r1_2 - group1_mean,
    r1_3 - group1_mean,
    r1_4 - group1_mean,
    r2_1 - group2_mean,
    r2_2 - group2_mean,
    r2_3 - group2_mean,
    r2_4 - group2_mean
]
```

## 5. 关键超参数

### 5.1 Group Sampling参数

```python
group_size: int = 4  # 每组的样本数量
```

**影响**：
- 更大的group_size：更稳定的基线估计，但计算成本更高
- 更小的group_size：计算更快，但基线估计可能不稳定

**推荐值**：4-8

### 5.2 优势计算参数

```python
advantage_type: str = "relative"  # "relative" 或 "normalized"
use_group_normalization: bool = True
```

**advantage_type**：
- "relative"：简单的相对优势（reward - baseline）
- "normalized"：标准化的相对优势（(reward - baseline) / std）

**use_group_normalization**：
- True：对组内优势进行标准化
- False：使用原始相对优势

### 5.3 优化参数

```python
gspo_epochs: int = 4  # GSPO更新次数
clip_range: float = 0.2  # PPO裁剪范围
learning_rate: float = 1e-5  # 学习率
```

**gspo_epochs**：
- 对同一批数据进行多次更新
- 更多的epochs：更充分的优化，但可能过拟合
- 推荐值：3-5

**clip_range**：
- 控制策略更新的幅度
- 更大的clip_range：更激进的更新
- 推荐值：0.1-0.3

### 5.4 正则化参数

```python
entropy_coef: float = 0.01  # 熵系数
kl_coef: float = 0.2  # KL散度系数
target_kl: float = 0.01  # 目标KL散度
adaptive_kl: bool = True  # 自适应KL系数
```

**entropy_coef**：
- 鼓励策略探索
- 更大的值：更多探索，更随机的输出
- 推荐值：0.001-0.01

**kl_coef**：
- 防止策略偏离参考模型太远
- 更大的值：更保守的更新
- 推荐值：0.1-0.5

## 6. 优势与局限

### 6.1 优势

1. **简单高效**
   - 无需训练critic模型
   - 训练速度快
   - 内存占用少

2. **稳定性好**
   - 组内相对优势减少方差
   - PPO裁剪防止过大更新
   - 自适应KL系数调整

3. **灵活性强**
   - 支持序列级和token级优化
   - 可配置的优势计算方式
   - 易于调整超参数

4. **效果优秀**
   - 在多个任务上表现良好
   - 特别适合长序列生成任务
   - 对奖励尺度不敏感

### 6.2 局限

1. **计算成本**
   - 需要生成多个回复（group_size倍）
   - 推理成本较高

2. **内存需求**
   - 需要同时存储多个回复
   - batch_size受限

3. **超参数敏感**
   - group_size的选择影响性能
   - 需要调整多个超参数

4. **基线估计**
   - 组内均值可能不够准确
   - 对于小group_size可能不稳定

## 7. 使用建议

### 7.1 适用场景

GSPO特别适合：
- 长序列生成任务（如推理、代码生成）
- 需要多样性的任务
- 计算资源充足的场景
- 奖励信号稀疏的任务

### 7.2 超参数调优建议

1. **从默认值开始**
   ```python
   group_size = 4
   gspo_epochs = 4
   clip_range = 0.2
   learning_rate = 1e-5
   ```

2. **根据任务调整**
   - 长序列任务：增大group_size（6-8）
   - 短序列任务：减小group_size（2-4）
   - 奖励稀疏：使用normalized优势
   - 奖励密集：使用relative优势

3. **监控指标**
   - raw_reward_mean：原始奖励均值（应该上升）
   - relative_advantage_std：相对优势标准差（应该稳定）
   - kl_divergence：KL散度（应该在target_kl附近）
   - policy_loss：策略损失（应该下降）

### 7.3 常见问题

**Q1: 训练不稳定怎么办？**
- 减小learning_rate
- 增大group_size
- 启用group_normalization
- 减小clip_range

**Q2: 奖励不增长怎么办？**
- 检查奖励模型是否合理
- 增大learning_rate
- 增大gspo_epochs
- 调整entropy_coef

**Q3: 内存不足怎么办？**
- 减小batch_size
- 减小group_size
- 使用梯度累积
- 使用混合精度训练

## 8. 代码示例

### 8.1 基础使用

```python
from gspo_train import GSPOConfig, GSPOTrainer, GSPODataset

# 创建配置
config = GSPOConfig(
    policy_model_name="Qwen/Qwen2-0.5B",
    batch_size=8,
    group_size=4,
    learning_rate=1e-5,
    num_epochs=3
)

# 加载数据
prompts = load_training_data()
tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
dataset = GSPODataset(prompts, tokenizer)

# 创建训练器
trainer = GSPOTrainer(config)

# 开始训练
trainer.train(dataset)
```

### 8.2 自定义配置

```python
# 长序列任务配置
config = GSPOConfig(
    group_size=6,  # 更大的组
    gspo_epochs=5,  # 更多更新
    clip_range=0.15,  # 更保守的裁剪
    use_group_normalization=True,  # 启用标准化
    advantage_type="normalized"  # 使用标准化优势
)

# 短序列任务配置
config = GSPOConfig(
    group_size=3,  # 更小的组
    gspo_epochs=3,  # 更少更新
    clip_range=0.25,  # 更激进的裁剪
    use_group_normalization=False,  # 不使用标准化
    advantage_type="relative"  # 使用相对优势
)
```

## 9. 总结

GSPO是一种简单而有效的策略优化算法，通过组采样和相对优势计算，在无需critic模型的情况下实现了稳定的策略优化。它特别适合长序列生成任务，并且易于实现和调优。

**核心创新点**：
1. 组采样提供丰富的对比信号
2. 相对优势替代critic模型
3. 序列级和token级优化的灵活支持
4. 简单高效的训练流程

**推荐使用场景**：
- 大语言模型的RLHF训练
- 长文本生成任务
- 代码生成任务
- 推理任务

**与其他算法的选择**：
- 需要最简单的实现：选择GRPO
- 需要最好的性能：选择VAPO
- 需要平衡简单性和性能：选择GSPO
