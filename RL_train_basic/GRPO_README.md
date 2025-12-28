# GRPO (Group Relative Policy Optimization) 训练脚本

## 概述

GRPO是一种新的强化学习算法，专门为大语言模型的训练而设计。相比传统的PPO算法，GRPO有以下主要优势：

1. **无需价值函数**：移除了critic网络，减少了内存和计算开销
2. **组内相对优化**：通过对同一提示的多个响应进行组内归一化来估计优势
3. **可编程奖励函数**：使用可验证的奖励函数，无需人工标注数据
4. **更高效的训练**：相比PPO减少了约50%的内存和计算需求

## 核心算法原理

### GRPO vs PPO 主要区别

| 特性 | PPO | GRPO |
|------|-----|------|
| 价值函数 | 需要critic网络 | 不需要 |
| 优势估计 | 使用GAE | 组内归一化 |
| 数据需求 | 需要标注数据或偏好数据 | 只需可编程奖励函数 |
| 内存使用 | 高（策略+价值网络） | 低（仅策略网络） |
| 训练稳定性 | 依赖价值函数准确性 | 通过组内比较保证稳定性 |

### GRPO算法流程

1. **组采样**：为每个提示生成多个候选响应（通常8个）
2. **奖励计算**：使用可编程函数评估每个响应
3. **组内归一化**：计算组内平均值和标准差，归一化奖励
4. **策略更新**：使用归一化奖励作为优势进行策略梯度更新

### 数学公式

GRPO的核心目标函数：

```
J_GRPO(θ) = E[1/G * Σ(min(r_i(θ) * A_i, clip(r_i(θ), 1-ε, 1+ε) * A_i))] - β * D_KL[π_θ || π_ref]
```

其中：
- `r_i(θ) = π_θ(o_i|q) / π_old(o_i|q)` 是重要性采样比率
- `A_i = (R_i - mean(R)) / std(R)` 是组内归一化的优势估计
- `G` 是组大小
- `β` 是KL散度惩罚系数

## 代码结构

### 主要类和函数

1. **PolicyNetwork**: 仅包含策略网络，移除了价值函数
2. **TextEnv**: 支持组内评估的文本环境
3. **GRPOMemoryBuffer**: 专门为GRPO设计的经验缓冲区
4. **GRPOAgent**: 实现GRPO算法的代理

### 关键修改点

#### 1. 移除价值函数
```python
# PPO中有价值函数头
self.value_head = nn.Linear(self.backbone.config.hidden_size, 1)

# GRPO中移除了价值函数，只保留策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        super(PolicyNetwork, self).__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
```

#### 2. 组内奖励归一化
```python
def evaluate_group_responses(self, prompt, responses):
    # 计算原始奖励
    raw_rewards = []
    for response in responses:
        total_reward = sum(reward_func(prompt, response) for reward_func in self.reward_functions)
        raw_rewards.append(total_reward)
    
    # GRPO核心：组内归一化
    mean_reward = np.mean(raw_rewards)
    std_reward = np.std(raw_rewards) + 1e-8
    normalized_rewards = [(r - mean_reward) / std_reward for r in raw_rewards]
    
    return normalized_rewards, raw_rewards
```

#### 3. 组响应生成
```python
def generate_group_responses(self, prompt, tokenizer, group_size=8):
    responses = []
    for _ in range(group_size):
        # 为每个提示生成多个不同的响应
        response = self.generate_single_response(prompt, tokenizer, temperature=0.8)
        responses.append(response)
    return responses
```

#### 4. 可编程奖励函数
```python
# 示例奖励函数
def length_reward(prompt, response):
    """基于长度的奖励"""
    target_length = 50
    actual_length = len(response.split())
    return max(0, 1.0 - abs(actual_length - target_length) / target_length)

def coherence_reward(prompt, response):
    """基于连贯性的奖励"""
    coherence_words = ["因为", "所以", "然而", "但是"]
    count = sum(1 for word in coherence_words if word in response)
    return min(1.0, count * 0.2)
```

## 配置参数

```python
# GRPO特有参数
GROUP_SIZE = 8          # 每个提示生成的响应数量
TEMPERATURE = 0.8       # 采样温度，控制多样性
BETA_KL = 0.1          # KL散度惩罚系数

# 通用参数
LEARNING_RATE = 1e-5
CLIP_RATIO = 0.2
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
```

## 使用方法

1. **准备数据**：准备提示列表和奖励函数
2. **运行训练**：
   ```bash
   python grpo_qwen_train.py
   ```
3. **监控训练**：观察原始奖励、归一化奖励和损失曲线

## 优势和适用场景

### 优势
- **内存效率**：相比PPO减少约50%内存使用
- **训练稳定**：组内归一化提供稳定的梯度信号
- **无需标注**：只需要可验证的奖励函数
- **适合推理任务**：特别适合数学、编程等有明确正确答案的任务

### 适用场景
- 数学问题求解
- 代码生成和调试
- 逻辑推理任务
- 任何有可验证输出的任务

### 不适用场景
- 创意写作（难以定义客观奖励）
- 主观性强的任务
- 需要长期依赖的复杂对话

## 实验结果

训练过程中会生成以下可视化：
- 平均原始奖励曲线
- 平均归一化奖励曲线  
- 总损失曲线

模型会定期保存到：
- `grpo_qwen_model_iter_{iteration}.pt`
- `grpo_qwen_model_final.pt`

## 进一步优化

1. **奖励函数设计**：根据具体任务设计更精确的奖励函数
2. **温度调节**：动态调整采样温度
3. **组大小优化**：根据计算资源调整组大小
4. **多轮对话**：扩展到多轮对话场景