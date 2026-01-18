# DAPO vs GRPO 算法对比分析

## 概述

DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization) 是 GRPO 的改进版本，由研究人员在尝试复现 DeepSeek-R1 时开发。DAPO 在 AIME 2024 数学竞赛基准上达到 50 分，超过了 DeepSeek-R1 的 47 分，且仅使用了 50% 的训练步数。

## 核心差异对比

### 1. Clip-Higher（非对称裁剪）

**GRPO 实现：**
```python
# 对称裁剪
surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
# 例如：clip_range = 0.2，范围为 [0.8, 1.2]
```

**DAPO 实现：**
```python
# 非对称裁剪
surr2 = torch.clamp(ratio, 
                   1 - self.config.clip_range_low,   # 0.2 → [0.8, ...]
                   1 + self.config.clip_range_high)  # 0.28 → [..., 1.28]
                   * advantages
```

**改进原因：**
- GRPO 的对称裁剪会导致**熵崩溃**（entropy collapse）
- 模型过早变得过于确定，限制探索能力
- Clip-Higher 允许模型更积极地增加好回复的概率
- 保持下界不变，避免将低概率 token 完全压制

**效果：**
- 防止熵崩溃，保持探索能力
- 提高模型在长链推理任务中的表现
- 训练更稳定，收敛更快

---

### 2. Token-Level Loss（token 级别损失）

**GRPO 实现（Sample-Level）：**
```python
# grpo_train.py 中的目标函数
# 先对每个回复内的 token 平均，再对所有回复平均
loss = (1/G) * Σ[(1/|o_i|) * Σ token_loss]

# 在代码中体现为：
# 每个样本的贡献被其长度归一化
policy_loss = -torch.min(surr1, surr2).mean()  # 直接对所有样本平均
```

**问题：**
- 短回复的每个 token 权重更大
- 模型倾向于生成短回复来"作弊"获得高奖励
- 不利于需要详细推理的复杂任务

**DAPO 实现（Token-Level）：**
```python
# dapo_train.py 中的实现
# 对所有 token 一起平均，按回复长度加权
if self.config.use_token_level_loss and response_lengths is not None:
    weights = torch.tensor(response_lengths, dtype=torch.float32, device=self.device)
    weights = weights / weights.sum()  # 归一化
    policy_loss = -(torch.min(surr1, surr2) * weights).sum()
```

**改进效果：**
- 长回复和短回复的 token 权重相同
- 鼓励模型生成详细的推理链
- 减少"奖励黑客"（reward hacking）行为
- 更适合需要多步推理的任务

---

### 3. Dynamic Sampling（动态采样）

**GRPO 实现：**
```python
# 固定采样 group_size 个回复
responses, log_probs = self.generate_responses(batch_prompts)
raw_rewards = self.compute_rewards(batch_prompts, responses)
# 如果所有奖励相同，相对奖励为 0，没有训练信号
```

**问题：**
- 当模型变强后，同一问题的所有回复可能都正确
- 导致相对奖励全为 0（因为都一样）
- 没有有效的训练信号
- 实际 batch size 减小

**DAPO 实现：**
```python
def dynamic_sampling(self, prompt: str, initial_responses: List[str], 
                    initial_rewards: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
    """
    如果所有回复的奖励相同，继续采样直到有差异
    """
    responses = initial_responses.copy()
    rewards = initial_rewards.clone()
    
    reward_std = rewards.std().item()
    extra_samples = 0
    
    # 如果标准差太小（奖励都相同），继续采样
    while reward_std < 1e-6 and len(responses) < self.config.max_dynamic_samples:
        extra_response, _, _ = self.generate_responses([prompt], num_responses=1)
        extra_reward = self.compute_rewards([prompt], extra_response)
        
        responses.extend(extra_response)
        rewards = torch.cat([rewards, extra_reward])
        reward_std = rewards.std().item()
        extra_samples += 1
    
    return responses, rewards
```

**改进效果：**
- 确保每个问题都有有效的训练信号
- 防止训练后期信号消失
- 虽然增加计算成本，但可以用更少的训练步数达到相同效果
- 实验显示：动态采样只需 1/3 的训练步数就能达到相同性能

**注意事项：**
- 增加约 25% 的计算时间
- 可能在模型已经很强时引入次优样本
- 需要配合熵奖励使用以减少开销

---

### 4. KL Divergence Penalty（KL 散度惩罚）

**GRPO 实现：**
```python
# grpo_train.py 中显式包含 KL 惩罚
kl_penalty = self.compute_kl_penalty_simple(prompts, responses)
kl_loss = self.kl_coef * kl_penalty.mean()  # kl_coef = 0.2

# 在目标函数中
total_loss = policy_loss + entropy_loss + kl_loss
```

**DAPO 实现：**
```python
# dapo_train.py 中默认移除 KL 惩罚
kl_coef: float = 0.0
use_kl_penalty: bool = False

# 在损失计算中
kl_loss = self.kl_coef * kl_penalty.mean() if self.config.use_kl_penalty else torch.tensor(0.0)
```

**移除原因：**
- 在长链推理任务中，模型分布需要显著偏离预训练模型
- KL 约束会限制这种必要的偏离
- Clip-Higher 机制已经提供了足够的稳定性
- 移除 KL 惩罚可以促进更多探索

**权衡：**
- 优点：允许模型更自由地探索，适合需要大幅改变行为的任务
- 缺点：可能导致策略偏离过远，生成不连贯的输出
- 建议：根据任务特性选择是否使用

---

### 5. Overlong Response Handling（过长回复处理）

**GRPO 实现：**
```python
# 没有特殊处理
# 如果回复被截断，直接给负奖励
```

**问题：**
- 被截断的回复即使前面推理正确也会得到负奖励
- 混淆模型的学习信号
- 不公平地惩罚了高质量的长推理链

**DAPO 实现：**

**方案 1：Overlong Filtering（过滤过长回复）**
```python
if self.config.use_overlong_filtering:
    valid_indices = [i for i, length in enumerate(response_lengths) 
                   if length < self.config.max_response_length]
    if len(valid_indices) > 0:
        responses = [responses[i] for i in valid_indices]
        rewards = rewards[valid_indices]
```

**方案 2：Soft Overlong Punishment（软惩罚）**
```python
def apply_soft_overlong_punishment(self, rewards: torch.Tensor, 
                                  response_lengths: List[int]) -> torch.Tensor:
    threshold_length = int(self.config.max_response_length * self.config.overlong_threshold)
    
    for reward, length in zip(rewards, response_lengths):
        if length > threshold_length:
            # 渐进式惩罚：超出部分越多，惩罚越大
            excess_ratio = (length - threshold_length) / threshold_length
            punishment = -0.5 * excess_ratio
            punished_reward = reward + punishment
```

**改进效果：**
- 避免不公平地惩罚高质量长回复
- 渐进式惩罚更合理
- 提高训练稳定性

---

## 配置参数对比

| 参数 | GRPO | DAPO | 说明 |
|------|------|------|------|
| `clip_range` | 0.2 (对称) | `clip_range_low=0.2`<br>`clip_range_high=0.28` | DAPO 使用非对称裁剪 |
| `kl_coef` | 0.2 | 0.0 | DAPO 默认移除 KL 惩罚 |
| `use_dynamic_sampling` | ❌ | ✅ | DAPO 特有功能 |
| `use_token_level_loss` | ❌ (Sample-level) | ✅ | DAPO 使用 token 级别损失 |
| `use_overlong_filtering` | ❌ | ✅ | DAPO 特有功能 |
| 损失聚合方式 | Sample-level | Token-level | 核心差异 |

---

## 性能对比

### 实验结果（AIME 2024，Qwen2.5-32B）

| 模型 | 分数 | 训练步数 |
|------|------|----------|
| DeepSeek-R1 (GRPO) | 47 | 100% |
| DAPO | 50 | 50% |

### 关键指标改进

1. **准确率**：DAPO 提高 6.4%（47→50）
2. **训练效率**：DAPO 仅需 50% 的训练步数
3. **熵稳定性**：DAPO 避免了熵崩溃
4. **回复长度**：DAPO 生成更长、更详细的推理链

---

## 代码结构对比

### GRPO 核心流程
```python
# 1. 生成回复
responses, log_probs = generate_responses(prompts)

# 2. 计算奖励
rewards = compute_rewards(prompts, responses)

# 3. 计算相对奖励（组内标准化）
relative_rewards, baselines = compute_relative_rewards(rewards)

# 4. 标准化优势（全局标准化）
advantages = compute_advantages(relative_rewards)

# 5. 计算损失（对称裁剪 + KL 惩罚）
policy_loss = compute_policy_loss(log_probs, old_log_probs, advantages, kl_penalty)
```

### DAPO 核心流程
```python
# 1. 生成回复
responses, log_probs, lengths = generate_responses(prompts)

# 2. 计算奖励
rewards = compute_rewards(prompts, responses)

# 3. 🔥 应用软惩罚
rewards = apply_soft_overlong_punishment(rewards, lengths)

# 4. 🔥 动态采样
responses, rewards = dynamic_sampling(prompt, responses, rewards)

# 5. 🔥 过滤过长回复
responses, rewards = filter_overlong(responses, rewards, lengths)

# 6. 计算相对奖励
relative_rewards, baselines = compute_relative_rewards(rewards)

# 7. 标准化优势
advantages = compute_advantages(relative_rewards)

# 8. 🔥 计算损失（非对称裁剪 + Token-Level + 无 KL）
policy_loss = compute_policy_loss(log_probs, old_log_probs, advantages, 
                                 kl_penalty=None, response_lengths=lengths)
```

---

## 适用场景

### GRPO 更适合：
- 短文本生成任务
- 计算资源有限的场景
- 需要严格控制策略偏离的任务
- 简单的问答任务

### DAPO 更适合：
- 长链推理任务（数学、编程、复杂推理）
- 需要详细解释的任务
- 模型已经较强，需要进一步提升的场景
- 有充足计算资源的场景
- 需要快速收敛的场景

---

## 实现建议

### 从 GRPO 迁移到 DAPO

1. **最小改动（核心功能）**：
   - 修改裁剪范围为非对称
   - 移除或减小 KL 惩罚系数
   - 实现 token-level loss

2. **完整功能**：
   - 添加动态采样逻辑
   - 实现过长回复处理
   - 调整超参数

3. **超参数调优**：
   ```python
   # DAPO 推荐配置
   clip_range_low = 0.2
   clip_range_high = 0.28
   kl_coef = 0.0
   use_dynamic_sampling = True
   use_token_level_loss = True
   group_size = 4  # 可以更小，因为有动态采样
   ```

---

## 总结

DAPO 通过四个核心改进解决了 GRPO 在长链推理任务中的主要问题：

1. **Clip-Higher** → 解决熵崩溃
2. **Token-Level Loss** → 解决奖励黑客和短回复偏好
3. **Dynamic Sampling** → 解决训练信号消失
4. **移除 KL 惩罚** → 允许必要的策略偏离

这些改进使 DAPO 在数学推理等需要长链思考的任务上显著优于 GRPO，同时保持了训练稳定性并提高了样本效率。

---

## 参考文献

1. DAPO 论文：[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2512.07611)
2. DeepSeek-R1 论文：DeepSeek-R1 Technical Report
3. GRPO 原始论文：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

---

## 附录：关键代码片段对比

### A. 裁剪函数对比

**GRPO:**
```python
surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
# 范围：[0.8, 1.2]
```

**DAPO:**
```python
surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.28) * advantages
# 范围：[0.8, 1.28]
```

### B. 损失聚合对比

**GRPO (Sample-Level):**
```python
policy_loss = -torch.min(surr1, surr2).mean()
```

**DAPO (Token-Level):**
```python
weights = torch.tensor(response_lengths, dtype=torch.float32)
weights = weights / weights.sum()
policy_loss = -(torch.min(surr1, surr2) * weights).sum()
```

### C. 动态采样伪代码

```python
# DAPO 独有
while reward_std < threshold and samples < max_samples:
    new_sample = generate_one_more()
    samples.append(new_sample)
    reward_std = compute_std(samples)
```
