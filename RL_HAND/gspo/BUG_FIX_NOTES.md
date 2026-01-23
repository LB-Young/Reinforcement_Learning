# GSPO Bug 修复说明

## 🐛 问题描述

### 错误信息
```
RuntimeError: Trying to backward through the graph a second time 
(or directly access saved tensors after they have already been freed). 
Saved intermediate values of the graph are freed when you call .backward() 
or autograd.grad(). Specify retain_graph=True if you need to backward 
through the graph a second time or if you need to access saved tensors 
after calling backward.
```

### 错误原因

在 `train_step` 方法中，GSPO 更新循环（`GSPO_EPOCHS` 次迭代）中重复使用了同一个计算图：

```python
# ❌ 错误的代码
# 在循环外计算 log_probs（带梯度）
log_probs, token_log_probs_list = self.compute_log_probs(...)

# 计算 KL 散度（依赖 log_probs 的计算图）
kl_penalty = log_probs - ref_log_probs

# 在循环内多次使用 kl_penalty
for _ in range(GSPO_EPOCHS):
    new_log_probs, _ = self.compute_log_probs(...)
    
    # 使用 kl_penalty（第一次 backward 后计算图已被释放）
    policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(
        new_log_probs, old_log_probs, advantages, kl_penalty, ...
    )
    
    total_loss = policy_loss + entropy_loss + kl_loss
    total_loss.backward()  # 第一次 backward 后，kl_penalty 的计算图被释放
    # 第二次循环时再使用 kl_penalty 就会报错
```

**核心问题**：
1. `kl_penalty` 是在循环外计算的，包含梯度信息
2. 第一次 `backward()` 后，PyTorch 释放了计算图
3. 第二次循环时尝试使用已释放的计算图，导致错误

---

## ✅ 解决方案

### 方案 1: 使用 `torch.no_grad()` 计算初始值（推荐）

```python
# ✅ 正确的代码
# 使用 no_grad 计算参考值和初始值
with torch.no_grad():
    ref_log_probs, _ = self.compute_log_probs(..., use_ref_model=True)
    old_log_probs, old_token_log_probs_list = self.compute_log_probs(
        ..., use_ref_model=False, return_per_token=USE_TOKEN_LEVEL_LOSS
    )
    # 计算初始KL散度（仅用于监控）
    initial_kl_penalty = old_log_probs - ref_log_probs

# 在循环内重新计算当前策略的 KL 散度
for _ in range(GSPO_EPOCHS):
    new_log_probs, new_token_log_probs_list = self.compute_log_probs(...)
    
    # 每次都重新计算 KL 散度（使用当前策略）
    current_kl_penalty = new_log_probs - ref_log_probs.detach()
    
    policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(
        new_log_probs, old_log_probs, advantages, current_kl_penalty, ...
    )
    
    total_loss = policy_loss + entropy_loss + kl_loss
    total_loss.backward()  # 每次都是新的计算图
```

**优点**：
- ✅ 避免计算图重用问题
- ✅ 每次迭代使用最新的策略计算 KL 散度
- ✅ 更符合 PPO 的设计理念

---

### 方案 2: 使用 `retain_graph=True`（不推荐）

```python
# ⚠️ 可行但不推荐
for _ in range(GSPO_EPOCHS):
    new_log_probs, _ = self.compute_log_probs(...)
    
    policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(
        new_log_probs, old_log_probs, advantages, kl_penalty, ...
    )
    
    total_loss = policy_loss + entropy_loss + kl_loss
    total_loss.backward(retain_graph=True)  # 保留计算图
```

**缺点**：
- ❌ 显存占用增加
- ❌ 计算效率降低
- ❌ 不符合算法设计（应该使用当前策略的 KL）

---

## 🔍 详细修改

### 修改 1: 初始化阶段

**之前**：
```python
# 计算log概率（带梯度）
log_probs, token_log_probs_list = self.compute_log_probs(
    prompts_truncated, responses_truncated, use_ref_model=False, return_per_token=USE_TOKEN_LEVEL_LOSS
)

# 计算KL散度（依赖log_probs的计算图）
ref_log_probs, _ = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref_model=True)
kl_penalty = log_probs - ref_log_probs

# 保存旧的log概率
old_log_probs = log_probs.detach()
```

**之后**：
```python
# 使用 no_grad 计算参考值和初始值
with torch.no_grad():
    ref_log_probs, _ = self.compute_log_probs(prompts_truncated, responses_truncated, use_ref_model=True)
    old_log_probs, old_token_log_probs_list = self.compute_log_probs(
        prompts_truncated, responses_truncated, use_ref_model=False, return_per_token=USE_TOKEN_LEVEL_LOSS
    )
    # 计算初始KL散度（仅用于监控）
    initial_kl_penalty = old_log_probs - ref_log_probs
```

### 修改 2: 更新循环

**之前**：
```python
for _ in range(GSPO_EPOCHS):
    new_log_probs, new_token_log_probs_list = self.compute_log_probs(...)
    
    # 使用循环外计算的 kl_penalty
    policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(
        new_log_probs, old_log_probs, advantages, kl_penalty, ...
    )
    
    total_loss = policy_loss + entropy_loss + kl_loss
    total_loss.backward()
```

**之后**：
```python
for _ in range(GSPO_EPOCHS):
    new_log_probs, new_token_log_probs_list = self.compute_log_probs(...)
    
    # 每次都重新计算 KL 散度
    current_kl_penalty = new_log_probs - ref_log_probs.detach()
    
    policy_loss, entropy_loss, kl_loss = self.compute_policy_loss(
        new_log_probs, old_log_probs, advantages, current_kl_penalty, ...
    )
    
    total_loss = policy_loss + entropy_loss + kl_loss
    total_loss.backward()
```

### 修改 3: 指标记录

**之前**：
```python
"kl_divergence": kl_penalty.mean().item(),
```

**之后**：
```python
"kl_divergence": initial_kl_penalty.mean().item(),
```

---

## 📚 相关知识

### PyTorch 计算图机制

1. **动态计算图**：PyTorch 使用动态计算图，每次前向传播都会构建新的计算图

2. **自动释放**：调用 `.backward()` 后，为了节省内存，PyTorch 会自动释放计算图

3. **保留计算图**：如果需要多次反向传播，可以使用 `retain_graph=True`，但会增加内存占用

4. **detach 操作**：`.detach()` 会创建一个新的张量，与原张量共享数据但不共享计算图

### PPO 算法中的最佳实践

1. **旧策略的 log_probs**：应该使用 `no_grad` 或 `detach` 计算，因为它们只是参考值

2. **参考模型的输出**：应该使用 `no_grad` 计算，因为参考模型不需要更新

3. **当前策略的输出**：需要保留梯度，因为需要更新策略网络

4. **KL 散度**：在每次迭代中重新计算，使用当前策略和参考策略

---

## 🧪 测试验证

### 测试代码

```python
# 测试修复后的代码
def test_gspo_training():
    prompts = ["测试问题1", "测试问题2"] * 5
    dataset = GSPODataset(prompts)
    trainer = GSPOTrainer()
    
    # 应该能正常运行多个 epoch
    trainer.train(dataset)
    
    print("✅ GSPO 训练成功完成！")

if __name__ == "__main__":
    test_gspo_training()
```

### 预期结果

- ✅ 不再出现 "backward through the graph a second time" 错误
- ✅ 训练正常进行
- ✅ 指标正常记录和显示

---

## 💡 经验总结

### 避免类似问题的建议

1. **明确区分**：
   - 需要梯度的张量（当前策略输出）
   - 不需要梯度的张量（旧策略、参考模型输出）

2. **使用 `no_grad`**：
   - 计算参考值时使用 `with torch.no_grad():`
   - 或使用 `.detach()` 断开计算图

3. **循环内重新计算**：
   - 如果需要在循环中多次使用某个值
   - 考虑在每次迭代中重新计算

4. **及时清理**：
   - 使用 `del` 删除不需要的张量
   - 调用 `torch.cuda.empty_cache()` 清理显存

5. **代码审查**：
   - 检查是否有在循环外计算、循环内使用的带梯度张量
   - 确保每次 `backward()` 使用的是新的计算图

---

## 📖 参考资料

1. [PyTorch Autograd 文档](https://pytorch.org/docs/stable/autograd.html)
2. [PPO 算法原论文](https://arxiv.org/abs/1707.06347)
3. [PyTorch 常见错误及解决方案](https://pytorch.org/docs/stable/notes/faq.html)

---

**修复日期**: 2026-01-20  
**修复者**: Kiro AI Assistant  
**状态**: ✅ 已修复并测试
