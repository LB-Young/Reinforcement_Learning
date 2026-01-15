# Shape 变换详细追踪

## 完整代码上下文

```python
# 在 compute_log_probs 方法中
policy_outputs = model(**full_inputs)
logits = policy_outputs.logits

log_probs = F.log_softmax(logits, dim=-1)
token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
```

## 假设场景

```python
# 输入文本: "Hello world"
# Tokenized: [101, 7592, 2088, 102]
# 序列长度: 4
# 词表大小: 50000 (Qwen2-0.5B 的实际词表大小)
# Batch size: 1
```

## 逐步 Shape 变换

### Step 0: 模型输入

```python
full_inputs["input_ids"]
# shape: [batch_size, seq_len]
# shape: [1, 4]
# 值: [[101, 7592, 2088, 102]]
```

### Step 1: 模型输出 logits

```python
policy_outputs = model(**full_inputs)
logits = policy_outputs.logits

# shape: [batch_size, seq_len, vocab_size]
# shape: [1, 4, 50000]

# 含义：
# logits[0, 0, :] → 位置0对所有50000个token的未归一化分数
# logits[0, 1, :] → 位置1对所有50000个token的未归一化分数
# logits[0, 2, :] → 位置2对所有50000个token的未归一化分数
# logits[0, 3, :] → 位置3对所有50000个token的未归一化分数
```

### Step 2: Log Softmax

```python
log_probs = F.log_softmax(logits, dim=-1)

# 输入 shape: [1, 4, 50000]
# 输出 shape: [1, 4, 50000]  ← shape 不变！
# dim=-1 表示在最后一个维度（vocab维度）上做 softmax

# 操作：
# 对于每个 [batch, seq_pos] 位置：
#   1. 对 50000 个 logits 做 softmax → 概率分布（和为1）
#   2. 取 log → log 概率

# 数值变化示例：
# logits[0, 0, :] = [0.5, 1.2, -0.3, 2.1, ...]  (50000个值)
#   ↓ softmax
# probs[0, 0, :] = [0.012, 0.024, 0.005, 0.059, ...]  (和为1)
#   ↓ log
# log_probs[0, 0, :] = [-4.42, -3.73, -5.30, -2.83, ...]  (负数)
```

### Step 3: unsqueeze(-1)

```python
full_inputs["input_ids"]
# shape: [1, 4]
# 值: [[101, 7592, 2088, 102]]

full_inputs["input_ids"].unsqueeze(-1)
# shape: [1, 4, 1]  ← 在最后增加一个维度
# 值: [[[101],
#       [7592],
#       [2088],
#       [102]]]

# 为什么？
# gather 需要 index 的维度数 = input 的维度数
# log_probs 是 3D [1, 4, 50000]
# 所以 index 也要是 3D [1, 4, 1]
```

### Step 4: gather(2, ...)

```python
log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1))

# 输入：
#   log_probs: [1, 4, 50000]
#   index:     [1, 4, 1]
#   dim: 2

# 输出 shape: [1, 4, 1]  ← 输出shape = index的shape

# 操作详解：
# dim=2 表示沿着第2维（vocab维度）进行索引
# 
# result[0, 0, 0] = log_probs[0, 0, index[0, 0, 0]]
#                 = log_probs[0, 0, 101]
#                 = -2.83  (假设值)
#
# result[0, 1, 0] = log_probs[0, 1, index[0, 1, 0]]
#                 = log_probs[0, 1, 7592]
#                 = -1.52  (假设值)
#
# result[0, 2, 0] = log_probs[0, 2, index[0, 2, 0]]
#                 = log_probs[0, 2, 2088]
#                 = -0.98  (假设值)
#
# result[0, 3, 0] = log_probs[0, 3, index[0, 3, 0]]
#                 = log_probs[0, 3, 102]
#                 = -1.25  (假设值)

# 结果：
# shape: [1, 4, 1]
# 值: [[[-2.83],
#       [-1.52],
#       [-0.98],
#       [-1.25]]]
```

### Step 5: squeeze(-1)

```python
token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)

# 输入 shape: [1, 4, 1]
# 输出 shape: [1, 4]  ← 去掉最后一个维度（大小为1）

# 值: [[-2.83, -1.52, -0.98, -1.25]]

# squeeze(-1) 的作用：
# 去掉最后一个大小为1的维度，让数据更干净
```

## 完整的 Shape 变换流程图

```
模型输入
full_inputs["input_ids"]
    ↓
[1, 4]  ← Token IDs
    ↓
模型前向传播
    ↓
logits
    ↓
[1, 4, 50000]  ← 每个位置对所有token的未归一化分数
    ↓
F.log_softmax(logits, dim=-1)
    ↓
log_probs
    ↓
[1, 4, 50000]  ← 每个位置对所有token的log概率（shape不变）
    ↓
准备索引
    ↓
full_inputs["input_ids"].unsqueeze(-1)
    ↓
[1, 4, 1]  ← 增加一个维度
    ↓
gather(2, index)
    ↓
[1, 4, 1]  ← 提取实际token的log概率
    ↓
squeeze(-1)
    ↓
token_log_probs
    ↓
[1, 4]  ← 最终结果：每个位置实际token的log概率
```

## 维度对应关系

```python
# 维度含义：
# 维度 0: batch_size (批次大小)
# 维度 1: seq_len (序列长度)
# 维度 2: vocab_size (词表大小) 或 1 (提取后)

# 变换过程：
[1, 4]           # input_ids
    ↓ model
[1, 4, 50000]    # logits
    ↓ log_softmax (dim=-1，在vocab维度操作)
[1, 4, 50000]    # log_probs (shape不变)
    ↓ unsqueeze(-1)
[1, 4, 1]        # index (为gather准备)
    ↓ gather(dim=2, 沿vocab维度索引)
[1, 4, 1]        # gathered (提取特定token)
    ↓ squeeze(-1)
[1, 4]           # token_log_probs (最终结果)
```

## 数值示例（简化版）

假设 vocab_size=5（简化演示）

```python
import torch
import torch.nn.functional as F

# ========== Step 1: logits ==========
logits = torch.tensor([
    [[0.5, 1.2, -0.3, 2.1, 0.8],   # 位置0对5个token的分数
     [0.3, -0.5, 1.8, 0.2, 1.5],   # 位置1
     [1.0, 0.4, 0.7, -0.2, 0.5]]   # 位置2
])
print("logits shape:", logits.shape)  # [1, 3, 5]

# ========== Step 2: log_softmax ==========
log_probs = F.log_softmax(logits, dim=-1)
print("log_probs shape:", log_probs.shape)  # [1, 3, 5] ← shape不变
print("log_probs:\n", log_probs)
# tensor([[[-1.8326, -1.1326, -2.6326, -0.2326, -1.5326],
#          [-1.8945, -2.6945, -0.3945, -1.9945, -0.6945],
#          [-1.3133, -1.9133, -1.6133, -2.5133, -1.8133]]])

# ========== Step 3: input_ids ==========
input_ids = torch.tensor([[3, 2, 4]])  # 实际token: 位置0选token3, 位置1选token2, 位置2选token4
print("input_ids shape:", input_ids.shape)  # [1, 3]

# ========== Step 4: unsqueeze ==========
index = input_ids.unsqueeze(-1)
print("index shape:", index.shape)  # [1, 3, 1]
print("index:\n", index)
# tensor([[[3],
#          [2],
#          [4]]])

# ========== Step 5: gather ==========
gathered = log_probs.gather(2, index)
print("gathered shape:", gathered.shape)  # [1, 3, 1]
print("gathered:\n", gathered)
# tensor([[[-0.2326],  # log_probs[0, 0, 3]
#          [-0.3945],  # log_probs[0, 1, 2]
#          [-1.8133]]])# log_probs[0, 2, 4]

# ========== Step 6: squeeze ==========
token_log_probs = gathered.squeeze(-1)
print("token_log_probs shape:", token_log_probs.shape)  # [1, 3]
print("token_log_probs:\n", token_log_probs)
# tensor([[-0.2326, -0.3945, -1.8133]])

# ========== 验证 ==========
print("\n验证：")
print("位置0, token 3:", log_probs[0, 0, 3])  # -0.2326 ✓
print("位置1, token 2:", log_probs[0, 1, 2])  # -0.3945 ✓
print("位置2, token 4:", log_probs[0, 2, 4])  # -1.8133 ✓
```

## 关键点总结

### 1. log_softmax 不改变 shape

```python
log_probs = F.log_softmax(logits, dim=-1)
# 输入: [1, 4, 50000]
# 输出: [1, 4, 50000]  ← 相同！
# 只改变数值，不改变形状
```

### 2. unsqueeze 增加维度

```python
input_ids.unsqueeze(-1)
# 输入: [1, 4]
# 输出: [1, 4, 1]  ← 增加一个维度
# -1 表示在最后增加
```

### 3. gather 输出 shape = index shape

```python
log_probs.gather(2, index)
# log_probs: [1, 4, 50000]
# index:     [1, 4, 1]
# 输出:      [1, 4, 1]  ← 与 index 相同！
```

### 4. squeeze 去掉大小为1的维度

```python
gathered.squeeze(-1)
# 输入: [1, 4, 1]
# 输出: [1, 4]  ← 去掉最后的1
```

## 为什么要这样设计？

```python
# 目标：从 [1, 4, 50000] 提取到 [1, 4]

# 方法1：循环（慢）
for i in range(4):
    token_log_probs[0, i] = log_probs[0, i, input_ids[0, i]]

# 方法2：gather（快，向量化）
token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)

# gather 需要：
# - input 和 index 有相同的维度数 → unsqueeze
# - 输出会有多余的维度 → squeeze
```

## 最终结果

```python
# 从这个：
log_probs: [1, 4, 50000]  # 每个位置对所有token的概率

# 到这个：
token_log_probs: [1, 4]   # 每个位置实际token的概率

# 数值示例：
# token_log_probs[0, 0] = -2.83  # 位置0，token=101的log概率
# token_log_probs[0, 1] = -1.52  # 位置1，token=7592的log概率
# token_log_probs[0, 2] = -0.98  # 位置2，token=2088的log概率
# token_log_probs[0, 3] = -1.25  # 位置3，token=102的log概率
```
