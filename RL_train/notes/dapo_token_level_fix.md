# DAPO Token-Level Loss 修复说明

## 问题描述

原始实现中的 token-level loss 实际上是 **sample-level 的加权**，而不是真正的 token-level 计算：

```python
# ❌ 原始实现（错误）
weights = torch.tensor(response_lengths, dtype=torch.float32, device=self.device)
weights = weights / weights.sum()
policy_loss = -(torch.min(surr1, surr2) * weights).sum()
```

这种方式：
- 只是给长回复更高的权重
- 仍然是在样本级别计算损失
- 会导致模型倾向于生成更长的回复

## 修复方案

### 1. 修改 `compute_log_probs` 方法

添加 `return_per_token` 参数，支持返回每个 token 的 log 概率：

```python
def compute_log_probs(self, prompts, responses, use_ref_model=False, return_per_token=False):
    if return_per_token:
        return all_token_log_probs  # List[Tensor], 每个元素是一个样本的token级别log概率
    else:
        return torch.stack(all_log_probs)  # Tensor [batch_size], 样本级别的总log概率
```

### 2. 新增 `compute_policy_loss_token_level` 方法

真正实现 token-level 的 PPO 损失计算：

```python
def compute_policy_loss_token_level(self, token_log_probs_list, old_token_log_probs_list, advantages):
    """对每个token单独计算PPO损失"""
    for i, (token_log_probs, old_token_log_probs) in enumerate(...):
        advantage = advantages[i]
        
        # 对每个token计算ratio
        token_ratios = torch.exp(token_log_probs - old_token_log_probs)
        
        # 对每个token应用Clip-Higher
        surr1 = token_ratios * advantage
        surr2 = torch.clamp(token_ratios, 1 - clip_low, 1 + clip_high) * advantage
        
        # 对该样本的所有token求和
        token_loss = -torch.min(surr1, surr2).sum()
        total_token_loss += token_loss
    
    return total_token_loss / total_tokens
```

### 3. 更新 `compute_policy_loss` 方法

支持 token-level 和 sample-level 两种模式：

```python
def compute_policy_loss(self, log_probs, old_log_probs, advantages, kl_penalty,
                       token_log_probs_list=None, old_token_log_probs_list=None):
    if self.config.use_token_level_loss and token_log_probs_list is not None:
        # 🔥 Token-Level Loss
        policy_loss = self.compute_policy_loss_token_level(...)
    else:
        # Sample-Level Loss (GRPO方式)
        ratio = torch.exp(log_probs - old_log_probs)
        ...
```

### 4. 更新 `train_step` 方法

在训练循环中获取 token 级别的 log 概率：

```python
# 获取旧策略的token级别log概率（用于计算ratio）
if self.config.use_token_level_loss:
    old_token_log_probs_list = self.compute_log_probs(
        all_prompts, all_responses, use_ref_model=False, return_per_token=True
    )
    old_token_log_probs_list = [t.detach() for t in old_token_log_probs_list]

# 在DAPO更新循环中
for dapo_step in range(self.config.dapo_epochs):
    # 获取当前策略的token级别log概率
    if self.config.use_token_level_loss:
        new_token_log_probs_list = self.compute_log_probs(
            all_prompts, all_responses, use_ref_model=False, return_per_token=True
        )
    
    # 计算损失
    policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(
        ..., 
        token_log_probs_list=new_token_log_probs_list,
        old_token_log_probs_list=old_token_log_probs_list
    )
```

## 关键差异对比

| 维度 | 原始实现（错误） | 修复后实现（正确） |
|------|-----------------|-------------------|
| 损失计算粒度 | Sample-level | Token-level |
| ratio 计算 | 整个序列一个 ratio | 每个 token 一个 ratio |
| clip 应用 | 对整个序列应用一次 | 对每个 token 分别应用 |
| 优势传播 | 样本级别的优势 | 每个 token 使用相同的样本优势 |
| 长度影响 | 长回复权重更高 | 长回复有更多 token 参与优化 |

## DAPO 的核心优势

通过 token-level loss，DAPO 能够：

1. **更精细的优化**：每个 token 都有自己的 ratio 和 clip 操作
2. **更好的信用分配**：长序列中的每个 token 都能得到独立的梯度信号
3. **避免长度偏差**：不会因为简单的加权而偏向长回复
4. **更稳定的训练**：token 级别的 clip 提供更细粒度的约束

## 使用方式

配置文件中设置：

```python
config = DAPOConfig(
    use_token_level_loss=True,  # 启用token-level loss
    clip_range_low=0.2,         # 下界裁剪
    clip_range_high=0.28,       # 上界裁剪（Clip-Higher）
)
```

如果设置 `use_token_level_loss=False`，则回退到 GRPO 的 sample-level loss 计算方式。
