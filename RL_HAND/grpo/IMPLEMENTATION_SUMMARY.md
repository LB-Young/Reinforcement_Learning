# GRPO实现总结

## 完成的工作

### 1. 核心算法实现 (`grpo.py`)

实现了完整的GRPO (Group Relative Policy Optimization) 训练脚本，包含以下核心特性：

#### 🔥 Group-based相对奖励机制
```python
def compute_relative_rewards(self, rewards, group_size):
    """GRPO核心：组内相对奖励计算"""
    batch_size = rewards.shape[0]
    rewards_grouped = rewards.view(-1, group_size)  # [num_groups, group_size]
    
    # 组内基线（均值）
    group_baselines = rewards_grouped.mean(dim=1, keepdim=True)
    
    # 相对奖励 = 绝对奖励 - 组内均值
    relative_rewards = rewards_grouped - group_baselines
    
    # 组内标准化（可选）
    if self.config.use_group_normalization:
        group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        relative_rewards = relative_rewards / group_std
    
    return relative_rewards.view(-1), group_baselines.repeat(1, group_size).view(-1)
```

#### 🔥 Policy-Only架构
```python
# GRPO只需要策略网络，无需价值网络
self.policy_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL)
self.ref_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL)  # 参考策略
self.reward_model = AutoModelForSequenceClassification.from_pretrained(REWARD_MODEL)

# 单一优化器
self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=LEARNING_RATE)
```

#### 🔥 多回复生成策略
```python
def generate_responses(self, prompts):
    """为每个prompt生成多个回复进行组内比较"""
    for prompt in prompts:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.policy_model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=GROUP_SIZE,  # 🔥 关键：每个prompt生成多个回复
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # 提取生成的回复部分
        gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
        responses = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        all_responses.extend(responses)
        all_prompts.extend([prompt] * GROUP_SIZE)
```

#### 🔥 Token级别PPO损失
```python
def compute_policy_loss(self, new_log_probs, old_log_probs, advantages, mask):
    """GRPO使用Token级别的PPO损失"""
    # Importance Sampling Ratio (Token级别)
    log_ratio = (new_log_probs - old_log_probs) * mask
    ratio = torch.exp(log_ratio)
    
    # PPO Clip Loss (对称裁剪)
    adv_t = advantages.unsqueeze(1)  # 广播优势到每个Token
    surr1 = ratio * adv_t
    surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * adv_t
    policy_loss = -torch.min(surr1, surr2)
    
    # KL散度惩罚
    kl_div = (new_log_probs - ref_log_probs)
    
    # 组合损失并对Mask求均值
    loss_map = (policy_loss + KL_COEF * kl_div) * mask
    return loss_map.sum() / mask.sum()
```

#### 🔥 优势函数标准化
```python
def compute_advantages(self, relative_rewards):
    """全局标准化优势函数"""
    # 相对奖励已经是组内标准化的，这里进行全局标准化
    advantages = (relative_rewards - relative_rewards.mean()) / (relative_rewards.std() + 1e-8)
    return advantages
```

#### 🔥 多轮策略更新
```python
# GRPO更新循环
for _ in range(GRPO_EPOCHS):
    # 重新计算当前策略的log概率
    new_log_probs, _, entropy = self.get_token_log_probs(
        self.policy_model, prompts_expanded, responses, self.device_policy
    )
    
    # 计算策略损失
    policy_loss = self.compute_policy_loss(
        new_log_probs, old_log_probs, advantages, mask, ref_log_probs
    )
    
    # 更新策略
    self.optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
    self.optimizer.step()
```

### 2. 显存优化策略

#### 双GPU架构
```python
# GPU设备分配（针对5060ti-16G * 2的配置）
self.device_policy = torch.device("cuda:0")    # 策略模型
self.device_ref = torch.device("cuda:1")       # 参考模型
self.device_reward = torch.device("cuda:1")    # 奖励模型
```

#### 显存管理
```python
def cleanup_memory(self):
    """显存清理策略"""
    # 1. 删除张量变量
    del inputs, outputs, logits, probs, labels
    
    # 2. 清理梯度缓存
    self.optimizer.zero_grad(set_to_none=True)
    
    # 3. 清理PyTorch缓存
    torch.cuda.empty_cache()
    
    # 4. 强制垃圾回收
    gc.collect()
```

### 3. Token级别精确计算

```python
def get_token_log_probs(self, model, prompts, responses, device):
    """获取Token级别的log_probs、Mask和Entropy"""
    full_texts = [p + r for p, r in zip(prompts, responses)]
    inputs = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
    
    # 计算Prompt长度，仅在Response区域计算
    prompt_lens = [len(self.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    
    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]  # Shift对齐
    labels = inputs["input_ids"][:, 1:]
    
    # Token级别log概率
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    
    # 熵计算（仅在response区域）
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    
    # 制作Mask：1仅在Response区域且非Padding处
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, p_len in enumerate(prompt_lens):
        mask[i, p_len:] = (labels[i, p_len:] != self.tokenizer.pad_token_id)
    
    return token_log_probs, mask, entropy
```

### 4. 可视化工具集成

#### GRPO专用指标图表
```python
def plot_grpo_metrics(losses, rewards, entropies, save_path):
    """绘制GRPO的3个核心指标"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Loss曲线
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    
    # Reward曲线  
    ax2.plot(rewards, 'g-', linewidth=2)
    ax2.set_title('Average Reward')
    
    # Entropy曲线
    ax3.plot(entropies, 'r-', linewidth=2)
    ax3.set_title('Policy Entropy')
```

## 算法原理

### GRPO核心思想
GRPO通过组内相对比较来学习，避免了绝对奖励的偏差问题：

1. **相对奖励**: 使用组内相对奖励而非绝对奖励
2. **Policy-Only**: 无需价值网络，简化架构
3. **组内比较**: 同一prompt的不同回复进行比较
4. **Token级别**: 在token级别计算损失和梯度

### 数学公式

#### 相对奖励计算
```
R_rel(x_i, y_i) = R(x_i, y_i) - (1/K) * Σ R(x_i, y_j)
```
其中：
- `R(x_i, y_i)` 是绝对奖励
- `K` 是组大小（GROUP_SIZE）
- `x_i` 是prompt，`y_j` 是第j个回复

#### 优势函数
```
A_i = (R_rel_i - μ) / σ
```
其中：
- `μ` 是相对奖励的均值
- `σ` 是相对奖励的标准差

#### GRPO目标函数
```
L^GRPO(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)] + β*KL(π_θ||π_ref)
```

### 与PPO的区别

| 特性 | PPO | GRPO |
|------|-----|------|
| 架构 | Actor-Critic | Policy-Only |
| 奖励类型 | 绝对奖励 | 相对奖励 |
| 价值估计 | 价值网络 | 组内基线 |
| 优势计算 | A = R - V | A = (R_rel - μ)/σ |
| 网络数量 | 2个（Actor+Critic） | 1个（Policy） |

## 代码结构特点

### 1. 简化的架构
```python
# GRPO只需要三个模型
self.policy_model    # 策略模型（可训练）
self.ref_model      # 参考模型（固定）
self.reward_model   # 奖励模型（固定）

# 只需要一个优化器
self.optimizer = torch.optim.AdamW(self.policy_model.parameters())
```

### 2. 批量处理优化
```python
# 高效的批量生成
outputs = self.policy_model.generate(
    num_return_sequences=GROUP_SIZE,  # 批量生成多个回复
    do_sample=True,
    temperature=0.7
)

# 批量奖励计算
rewards = self.compute_rewards(prompts_expanded, responses)
```

### 3. 内存管理
```python
# 分阶段释放内存
del inputs, outputs, gen_ids  # 生成阶段后释放
torch.cuda.empty_cache()      # 清理CUDA缓存
gc.collect()                  # 强制垃圾回收
```

## 配置参数

### 核心超参数
```python
BATCH_SIZE = 2              # 批次大小（显存优化）
LEARNING_RATE = 1e-6        # 学习率
GROUP_SIZE = 4              # 每个prompt的回复数量
GRPO_EPOCHS = 4             # GRPO更新轮数
CLIP_RANGE = 0.2            # PPO裁剪范围
KL_COEF = 0.01              # KL散度系数
```

### 模型配置
```python
POLICY_MODEL = r"E:\models\Qwen\Qwen3-0___6B"                    # 策略模型
REWARD_MODEL = r"E:\models\reward-model-deberta-v3-large-v2"     # 奖励模型
```

### 训练配置
```python
NUM_EPOCHS = 1              # 训练轮数
DTYPE = torch.bfloat16      # 数据类型
OUTPUT_DIR = "grpo_output"  # 输出目录
```

## 算法优势

### 1. 简化的架构
- **无需价值网络**: 减少模型参数和训练复杂度
- **单一优化器**: 简化训练过程
- **更少的超参数**: 减少调参工作量

### 2. 相对奖励机制
- **减少奖励偏差**: 组内比较消除绝对奖励的系统性偏差
- **提高样本效率**: 每个prompt生成多个回复进行比较
- **更稳定的训练**: 相对奖励更稳定

### 3. 显存友好
- **更少的模型**: 只需要一个可训练的策略模型
- **批量优化**: 高效的批量生成和计算
- **显存管理**: 完善的显存清理机制

## 适用场景

### GRPO适合的任务
- **长文本生成**: 相对奖励机制适合长文本
- **创意写作**: 多样性回复的比较学习
- **对话系统**: 多轮对话的质量提升
- **文本改写**: 多种改写方案的比较

### GRPO的优势
- **训练效率高**: Policy-Only架构训练更快
- **显存需求低**: 相比PPO需要更少显存
- **实现简单**: 代码结构更简洁
- **效果稳定**: 相对奖励机制更稳定

### GRPO的局限性
- **需要多回复**: 必须为每个prompt生成多个回复
- **组大小敏感**: GROUP_SIZE的选择影响效果
- **计算开销**: 需要生成更多样本
- **奖励模型依赖**: 严重依赖奖励模型的质量

## 训练指标

### 记录的指标
```python
self.metrics_history = {
    'loss': [],         # 策略损失
    'reward': [],       # 平均奖励
    'entropy': []       # 策略熵
}
```

### 指标含义
- **Loss**: GRPO的策略损失，包含PPO损失和KL惩罚
- **Reward**: 奖励模型给出的平均奖励
- **Entropy**: 策略熵，反映生成的多样性

### 训练监控
```python
# 实时显示训练进度
pbar.set_description(f"L:{metrics['loss']:.4f} R:{metrics['reward']:.2f} E:{metrics['entropy']:.3f}")
```

## 实现亮点

### 1. 高效的组内比较
```python
# 同一prompt的多个回复进行组内比较
responses, prompts_expanded = self.generate_responses(batch_prompts)
# prompts_expanded: ['q1','q1','q1','q1','q2','q2','q2','q2']
# responses:        ['a1','a2','a3','a4','b1','b2','b3','b4']

# 计算相对奖励
rewards_grouped = rewards.view(num_groups, GROUP_SIZE)
relative_rewards = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
```

### 2. 精确的Token级别计算
```python
# 只在response区域计算损失
mask = torch.zeros_like(labels, dtype=torch.bool)
for i, p_len in enumerate(prompt_lens):
    mask[i, p_len:] = (labels[i, p_len:] != tokenizer.pad_token_id)

# Token级别的损失计算
loss_map = (policy_loss + KL_COEF * kl_div) * mask
step_loss = loss_map.sum() / mask.sum()
```

### 3. 完善的显存管理
```python
# 分阶段显存清理
def train_step(self, batch_prompts):
    # ... 计算过程 ...
    
    # 显式清理显存
    del new_log_probs, entropy, masked_entropy, log_ratio, ratio
    del surr1, surr2, policy_loss, kl_div, loss_map, step_loss
    
    self.optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    gc.collect()
```

### 4. 自动化训练管理
```python
# 自动保存和备份
save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
self.policy_model.save_pretrained(save_path)
self.tokenizer.save_pretrained(save_path)

# 备份训练脚本
current_script = os.path.abspath(__file__)
target_script = os.path.join(save_path, "train_script.py")
shutil.copy2(current_script, target_script)
```

## 与DAPO的关系

GRPO是DAPO的基础版本，DAPO在GRPO基础上进行了以下改进：

### GRPO → DAPO的演进
| 特性 | GRPO | DAPO |
|------|------|------|
| 裁剪方式 | 对称 [0.8, 1.2] | 非对称 [0.8, 1.28] |
| 损失级别 | Token-Level | Token-Level |
| KL惩罚 | 使用 (0.01) | 移除 (0.0) |
| 动态采样 | ❌ | ✅ |
| 过长处理 | ❌ | ✅ |
| 适用场景 | 长文本生成 | 长链推理 |

### 共同特点
- Policy-Only架构
- 相对奖励机制
- Token级别计算
- 组内比较学习

## 总结

GRPO实现提供了一个简化而高效的强化学习训练框架，特别适合长文本生成任务。通过Policy-Only架构和相对奖励机制，GRPO在保证训练效果的同时显著降低了计算复杂度和显存需求。代码结构清晰，显存管理完善，是DAPO算法的重要基础。