# PyTorch gather() 函数详解

## 代码上下文

```python
# 在 compute_log_probs 方法中
log_probs = F.log_softmax(logits, dim=-1)  # shape: [batch, seq_len, vocab_size]
token_log_probs = log_probs.gather(2, full_inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
```

## gather() 函数基础

### 函数签名

```python
torch.gather(input, dim, index)
```

**参数**：
- `input`: 源张量（要从中提取值）
- `dim`: 沿着哪个维度进行索引
- `index`: 索引张量（指定要提取哪些值）

**功能**：沿着指定维度，根据索引张量提取对应位置的值

## 简单示例

### 1D 示例

```python
import torch

# 源张量
input = torch.tensor([10, 20, 30, 40, 50])
# 索引：我想要第0、2、4个位置的值
index = torch.tensor([0, 2, 4])

result = torch.gather(input, 0, index)
# result = tensor([10, 30, 50])
```

### 2D 示例

```python
# 源张量 [3, 5]
input = torch.tensor([
    [1,  2,  3,  4,  5],   # row 0
    [6,  7,  8,  9,  10],  # row 1
    [11, 12, 13, 14, 15]   # row 2
])

# 沿着 dim=1（列方向）收集
# 对于每一行，我想要不同列的值
index = torch.tensor([
    [0, 2, 4],  # row 0: 取第0、2、4列
    [1, 3, 4],  # row 1: 取第1、3、4列
    [0, 1, 2]   # row 2: 取第0、1、2列
])

result = torch.gather(input, dim=1, index)
# result = tensor([
#     [1,  3,  5],   # input[0, [0,2,4]]
#     [7,  9,  10],  # input[1, [1,3,4]]
#     [11, 12, 13]   # input[2, [0,1,2]]
# ])
```

## 代码中的实际应用

### 数据形状分析

```python
# 假设输入文本："Hello world"
# tokenized: [101, 7592, 2088, 102]  (4 tokens)

# 1. 模型输出 logits
logits = model(**inputs).logits
# shape: [batch_size, seq_len, vocab_size]
# 例如: [1, 4, 50000]
# 含义：每个位置对词表中每个token的未归一化分数

# 2. 计算 log 概率
log_probs = F.log_softmax(logits, dim=-1)
# shape: [1, 4, 50000]
# 含义：每个位置对词表中每个token的log概率
# log_probs[0, 0, :] 是第一个位置对所有50000个token的log概率分布

# 3. 实际的 token IDs
full_inputs["input_ids"]
# shape: [1, 4]
# 值: [[101, 7592, 2088, 102]]
# 含义：实际生成的token ID序列
```

### 为什么需要 gather？

**问题**：我们有每个位置对**所有词表token**的log概率，但我们只关心**实际生成的token**的log概率。

```python
# log_probs[0, 0, :] 包含50000个值（对每个可能token的概率）
# 但我们只需要 log_probs[0, 0, 101] 这一个值（实际token=101的概率）

# 对于整个序列：
# 位置0: 需要 log_probs[0, 0, 101]
# 位置1: 需要 log_probs[0, 1, 7592]
# 位置2: 需要 log_probs[0, 2, 2088]
# 位置3: 需要 log_probs[0, 3, 102]
```

### 详细步骤拆解

#### Step 1: unsqueeze(-1)

```python
full_inputs["input_ids"]
# shape: [1, 4]
# [[101, 7592, 2088, 102]]

full_inputs["input_ids"].unsqueeze(-1)
# shape: [1, 4, 1]
# [[[101],
#   [7592],
#   [2088],
#   [102]]]

# 为什么？因为 gather 需要 index 和 input 有相同的维度数
```

#### Step 2: gather(2, ...)

```python
log_probs.shape  # [1, 4, 50000]
index.shape      # [1, 4, 1]

# gather(dim=2, index) 的含义：
# 沿着第2维（vocab维度）进行索引
# 对于每个 [batch, seq_pos] 位置，从50000个值中选出index指定的那个

result = log_probs.gather(2, index)
# shape: [1, 4, 1]

# 具体操作：
# result[0, 0, 0] = log_probs[0, 0, 101]    # 第0个位置，token=101的log概率
# result[0, 1, 0] = log_probs[0, 1, 7592]   # 第1个位置，token=7592的log概率
# result[0, 2, 0] = log_probs[0, 2, 2088]   # 第2个位置，token=2088的log概率
# result[0, 3, 0] = log_probs[0, 3, 102]    # 第3个位置，token=102的log概率
```

#### Step 3: squeeze(-1)

```python
result.shape  # [1, 4, 1]

result.squeeze(-1)
# shape: [1, 4]
# [[-0.5, -1.2, -0.8, -0.3]]  # 每个位置实际token的log概率

# 去掉最后一个维度，得到干净的结果
```

## 可视化示例

### 完整的数据流

```python
# 输入序列: "Hi !"
# Token IDs: [101, 2023, 999, 102]

# ============ Step 1: 模型输出 ============
logits = [
    # 位置0的logits（对所有50000个token）
    [0.1, 0.2, ..., 5.0(token=101), ..., 0.3],  # 50000个值
    
    # 位置1的logits
    [0.3, 0.1, ..., 4.2(token=2023), ..., 0.2],
    
    # 位置2的logits
    [0.2, 0.4, ..., 3.8(token=999), ..., 0.1],
    
    # 位置3的logits
    [0.5, 0.3, ..., 4.5(token=102), ..., 0.4]
]
# shape: [1, 4, 50000]

# ============ Step 2: Log Softmax ============
log_probs = F.log_softmax(logits, dim=-1)
# 对每个位置的50000个值做softmax，然后取log
# shape: [1, 4, 50000]

# ============ Step 3: Gather ============
# 我们只需要实际token的log概率
token_ids = [[101, 2023, 999, 102]]  # shape: [1, 4]

# gather 操作：
token_log_probs = [
    log_probs[0, 0, 101],    # 位置0，token=101的log概率
    log_probs[0, 1, 2023],   # 位置1，token=2023的log概率
    log_probs[0, 2, 999],    # 位置2，token=999的log概率
    log_probs[0, 3, 102]     # 位置3，token=102的log概率
]
# shape: [1, 4]
```

## gather 的维度规则

```python
# 通用规则：
output[i][j][k] = input[i][j][index[i][j][k]]  # 当 dim=2 时

# 对于我们的代码：
# log_probs:    [batch, seq_len, vocab_size]
# index:        [batch, seq_len, 1]
# output:       [batch, seq_len, 1]

# 具体映射：
output[b][s][0] = log_probs[b][s][index[b][s][0]]
                = log_probs[b][s][token_id]
```

## 为什么不直接索引？

你可能会想：为什么不直接用 `log_probs[0, 0, token_id]`？

```python
# ❌ 不能这样做（对于批量数据）
for i in range(batch_size):
    for j in range(seq_len):
        token_id = input_ids[i, j]
        result[i, j] = log_probs[i, j, token_id]  # 循环太慢！

# ✅ gather 是向量化操作，一次性完成
result = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
# 快速、高效、GPU友好
```

## 实际数值示例

```python
import torch
import torch.nn.functional as F

# 模拟场景：vocab_size=10（简化）
batch_size, seq_len, vocab_size = 1, 3, 10

# 模型输出的 logits
logits = torch.randn(1, 3, 10)
# 例如：
# [[[0.5, 1.2, -0.3, 2.1, 0.8, -0.5, 1.0, 0.3, -0.2, 0.7],  # 位置0
#   [0.3, -0.5, 1.8, 0.2, 1.5, 0.9, -0.3, 0.6, 1.1, 0.4],   # 位置1
#   [1.0, 0.4, 0.7, -0.2, 0.5, 1.3, 0.8, 1.6, 0.2, -0.1]]]  # 位置2

# 计算 log 概率
log_probs = F.log_softmax(logits, dim=-1)
# shape: [1, 3, 10]
# 每个位置的10个值和为1（概率分布）

# 实际的 token IDs
input_ids = torch.tensor([[3, 2, 7]])  # shape: [1, 3]
# 位置0的token是3，位置1的token是2，位置2的token是7

# 提取实际token的log概率
token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
# shape: [1, 3]

print("Token log probs:", token_log_probs)
# 结果：[[-0.82, -1.15, -0.73]]
# 含义：
# - 位置0生成token=3的log概率是-0.82
# - 位置1生成token=2的log概率是-1.15
# - 位置2生成token=7的log概率是-0.73

# 验证：
print("Manual check:")
print("Position 0, token 3:", log_probs[0, 0, 3])  # 应该是-0.82
print("Position 1, token 2:", log_probs[0, 1, 2])  # 应该是-1.15
print("Position 2, token 7:", log_probs[0, 2, 7])  # 应该是-0.73
```

## 总结

**`gather` 函数的作用**：

1. **从高维分布中提取特定值**
   - 输入：每个位置对所有50000个token的概率
   - 输出：每个位置对实际生成token的概率

2. **向量化操作**
   - 避免Python循环
   - GPU并行计算
   - 高效处理批量数据

3. **在我们的代码中**
   ```python
   token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)
   ```
   - 从 `[batch, seq, vocab]` 的概率分布中
   - 根据 `input_ids` 的索引
   - 提取出 `[batch, seq]` 的实际token概率

**核心思想**：从"所有可能"中精确提取"实际发生"的那个值。
