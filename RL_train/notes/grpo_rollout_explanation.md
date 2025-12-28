# GRPO/DAPO Rollout 机制详解

## 问题发现

原始实现中存在一个**关键缺陷**：`generate_responses` 函数对每个 prompt 只生成了**一个**回复，但 GRPO/DAPO 的核心思想是**同一个 query 生成多个 answer** 来计算组内相对奖励。

## GRPO 的核心思想

GRPO (Group Relative Policy Optimization) 的"Group"指的是：
- **同一个问题生成多个回复（group_size 个）**
- 在这个组内计算相对奖励
- 用组内均值作为 baseline（替代 PPO 的 critic）

### 错误实现示例

```python
# ❌ 错误：每个 prompt 只生成一个回复
def generate_responses(self, prompts: List[str]):
    responses = []
    for prompt in prompts:  # prompts = ['q1', 'q2', 'q3']
        response = self.model.generate(prompt)  # 只生成一次
        responses.append(response)
    return responses  # ['a1', 'a2', 'a3'] - 每个问题一个答案
```

这样的话：
- `prompts = ['q1', 'q2', 'q3']`
- `responses = ['a1', 'a2', 'a3']`
- 无法计算组内相对奖励，因为每个问题只有一个答案

### 正确实现

```python
# ✅ 正确：每个 prompt 生成 group_size 个回复
def generate_responses(self, prompts: List[str]):
    all_responses = []
    all_prompts_expanded = []
    
    for prompt in prompts:  # prompts = ['q1', 'q2']
        for _ in range(self.config.group_size):  # group_size = 4
            response = self.model.generate(prompt, do_sample=True)
            all_responses.append(response)
            all_prompts_expanded.append(prompt)
    
    return all_responses, all_prompts_expanded
    # responses = ['a1_1', 'a1_2', 'a1_3', 'a1_4', 'a2_1', 'a2_2', 'a2_3', 'a2_4']
    # prompts   = ['q1',   'q1',   'q1',   'q1',   'q2',   'q2',   'q2',   'q2']
```

## 完整流程示例

### 输入
```python
batch_prompts = ['如何学习Python?', '什么是机器学习?']
group_size = 4
```

### Step 1: 生成回复（Rollout）

```python
responses, prompts_expanded = generate_responses(batch_prompts)

# 结果：
prompts_expanded = [
    '如何学习Python?',  # q1 的第 1 个回复
    '如何学习Python?',  # q1 的第 2 个回复
    '如何学习Python?',  # q1 的第 3 个回复
    '如何学习Python?',  # q1 的第 4 个回复
    '什么是机器学习?',  # q2 的第 1 个回复
    '什么是机器学习?',  # q2 的第 2 个回复
    '什么是机器学习?',  # q2 的第 3 个回复
    '什么是机器学习?',  # q2 的第 4 个回复
]

responses = [
    '可以从基础语法开始...',      # a1_1
    '建议先学习数据结构...',      # a1_2
    '推荐阅读官方文档...',        # a1_3
    '参加在线课程是个好选择...',  # a1_4
    '机器学习是人工智能的分支...', # a2_1
    '它让计算机从数据中学习...', # a2_2
    '主要包括监督学习等...',      # a2_3
    '应用广泛，如图像识别...',    # a2_4
]
```

### Step 2: 计算奖励

```python
rewards = compute_rewards(prompts_expanded, responses)

# 结果：
rewards = [
    0.8,  # r1_1
    0.6,  # r1_2
    0.9,  # r1_3
    0.7,  # r1_4
    0.85, # r2_1
    0.75, # r2_2
    0.80, # r2_3
    0.90, # r2_4
]
```

### Step 3: 计算组内相对奖励（GRPO 核心）

```python
relative_rewards, baselines = compute_relative_rewards(rewards, group_size=4)

# 处理过程：
# Group 1 (q1): [0.8, 0.6, 0.9, 0.7]
#   - 组内均值 baseline = 0.75
#   - 相对奖励 = [0.8-0.75, 0.6-0.75, 0.9-0.75, 0.7-0.75]
#              = [0.05, -0.15, 0.15, -0.05]
#   - 标准化后 = [0.33, -1.0, 1.0, -0.33]

# Group 2 (q2): [0.85, 0.75, 0.80, 0.90]
#   - 组内均值 baseline = 0.825
#   - 相对奖励 = [0.025, -0.075, -0.025, 0.075]
#   - 标准化后 = [0.5, -1.5, -0.5, 1.5]

relative_rewards = [0.33, -1.0, 1.0, -0.33, 0.5, -1.5, -0.5, 1.5]
```

### Step 4: 策略更新

```python
# 使用相对奖励作为优势函数
advantages = compute_advantages(relative_rewards)

# 计算策略损失
for response, advantage in zip(responses, advantages):
    if advantage > 0:
        # 增加生成这个回复的概率
        increase_probability(response)
    else:
        # 减少生成这个回复的概率
        decrease_probability(response)
```

## 为什么必须同一个 query 生成多次？

### 1. 计算相对优势

GRPO 的核心创新是用**组内均值**替代 PPO 的 critic：

```python
# PPO: 需要训练一个 critic 网络
advantage = reward - critic(state)

# GRPO: 用组内均值作为 baseline
advantage = reward - mean(group_rewards)
```

如果每个 query 只生成一次，就无法计算组内均值。

### 2. 减少方差

多次采样可以：
- 更准确地估计该 query 的平均回复质量
- 减少单次采样的随机性
- 提供更稳定的训练信号

### 3. 对比学习

同一个问题的不同回复质量不同：
- 好的回复：advantage > 0，增加概率
- 差的回复：advantage < 0，减少概率
- 这是一种隐式的对比学习

## DAPO 的额外改进

DAPO 在 GRPO 基础上增加了**动态采样**：

```python
# 生成初始 group_size 个回复
responses = generate_group(prompt, group_size=4)
rewards = compute_rewards(responses)

# 🔥 如果所有奖励都相同（标准差太小）
if rewards.std() < threshold:
    # 继续采样直到有差异
    while rewards.std() < threshold and len(responses) < max_samples:
        extra_response = generate_one_more(prompt)
        responses.append(extra_response)
        rewards = compute_rewards(responses)
```

这确保了：
- 每个 query 都有有效的训练信号
- 避免"所有回复都正确"或"都错误"的情况
- 训练后期模型变强时仍能学习

## 代码对比

### 修复前（错误）

```python
def train_step(self, batch_prompts):
    # ❌ 每个 prompt 只生成一个回复
    responses, log_probs = self.generate_responses(batch_prompts)
    # batch_prompts = ['q1', 'q2']
    # responses = ['a1', 'a2']  # 只有 2 个回复
    
    rewards = self.compute_rewards(batch_prompts, responses)
    # rewards = [r1, r2]  # 无法分组
    
    # ❌ 无法计算组内相对奖励
    relative_rewards = self.compute_relative_rewards(rewards)
```

### 修复后（正确）

```python
def train_step(self, batch_prompts):
    # ✅ 每个 prompt 生成 group_size 个回复
    responses, log_probs, prompts_expanded = self.generate_responses(batch_prompts)
    # batch_prompts = ['q1', 'q2']
    # responses = ['a1_1', 'a1_2', 'a1_3', 'a1_4', 'a2_1', 'a2_2', 'a2_3', 'a2_4']
    # prompts_expanded = ['q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q2', 'q2']
    
    rewards = self.compute_rewards(prompts_expanded, responses)
    # rewards = [r1_1, r1_2, r1_3, r1_4, r2_1, r2_2, r2_3, r2_4]
    
    # ✅ 可以按 group_size=4 分组计算相对奖励
    relative_rewards = self.compute_relative_rewards(rewards, group_size=4)
    # Group 1: [r1_1, r1_2, r1_3, r1_4] -> 计算组内相对奖励
    # Group 2: [r2_1, r2_2, r2_3, r2_4] -> 计算组内相对奖励
```

## 关键要点总结

1. **GRPO/DAPO 必须对同一个 query 生成多个 answer**
2. **group_size 是指同一个问题生成几个回复**
3. **组内相对奖励是 GRPO 的核心创新**
4. **必须启用 `do_sample=True` 才能生成不同的回复**
5. **prompts 需要扩展以匹配 responses 的数量**

## 性能影响

### 计算成本

```python
# 原始 batch_size = 8
# 修复前：生成 8 个回复
# 修复后：生成 8 × 4 = 32 个回复（group_size=4）

# 计算成本增加 4 倍，但这是 GRPO 算法的必要开销
```

### 训练效果

- ✅ 正确实现 GRPO 算法
- ✅ 更稳定的训练信号
- ✅ 更好的性能表现
- ✅ 符合论文描述

## 参考

- DeepSeekMath 论文：GRPO 原始提出
- DAPO 论文：动态采样改进
- 本次修复：确保正确实现组内采样机制
