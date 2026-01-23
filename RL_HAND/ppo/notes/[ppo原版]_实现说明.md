# PPO实现总结

## 完成的工作

### 1. 核心算法实现 (`ppo.py`)

实现了完整的PPO (Proximal Policy Optimization) 训练脚本，包含以下核心特性：

#### 🔥 Actor-Critic架构
```python
# 策略网络（Actor）
self.policy_model = AutoModelForCausalLM.from_pretrained(ACTOR_MODEL)
self.policy_optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=LEARNING_RATE)

# 价值网络（Critic）
self.critic_model = AutoModelForCausalLM.from_pretrained(CRITIC_MODEL)
self.critic_optimizer = torch.optim.AdamW(self.critic_model.parameters(), lr=LEARNING_RATE)

# 参考策略网络（Reference）
self.reference_model = AutoModelForCausalLM.from_pretrained(ACTOR_MODEL)
self.reference_model.eval()
```

#### 🔥 PPO Clip损失
```python
def compute_policy_loss(self, log_probs, old_log_probs, advantages, mask):
    """PPO的核心：对称裁剪损失"""
    log_ratio = (log_probs - old_log_probs) * mask
    ratio = torch.exp(log_ratio)
    
    # PPO对称裁剪
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    # KL散度惩罚
    kl_div = (log_probs - ref_log_probs)
    loss_map = (policy_loss + 0.01 * kl_div) * mask
    return loss_map.sum() / mask.sum()
```

#### 🔥 优势函数计算
```python
def compute_values(self, prompts, responses, requires_grad=False):
    """使用critic模型计算状态价值"""
    for p, r in zip(prompts, responses):
        full_text = p + r
        inputs = self.critic_tokenizer(full_text, return_tensors="pt")
        outputs = self.critic_model(**inputs)
        # 使用最后一个token的logits作为value（简化实现）
        value = outputs.logits[0, -1, :].mean()
        values.append(value)
    return torch.stack(values)

# 计算优势：A = R - V
advantages = rewards - values
```

#### 🔥 多轮更新机制
```python
# PPO更新循环
for _ in range(GROUP_EPOCHES):
    # 重新计算当前策略的log概率
    new_log_probs = self.get_token_log_probs(self.policy_model, prompts, responses)
    
    # 计算策略损失（PPO Clip）
    policy_loss = self.compute_policy_loss(new_log_probs, old_log_probs, advantages, mask)
    
    # 更新策略网络
    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()
    
    # 计算价值损失
    new_values = self.compute_values(prompts, responses, requires_grad=True)
    value_loss = F.mse_loss(new_values, rewards)
    
    # 更新价值网络
    self.critic_optimizer.zero_grad()
    value_loss.backward()
    self.critic_optimizer.step()
```

#### 🔥 熵正则化
```python
def compute_entropy(self, prompts, responses):
    """计算策略熵，鼓励探索"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, seq_len]
    
    # 仅在response区域计算熵
    masked_entropy = entropy * mask
    avg_entropy = masked_entropy.sum() / mask.sum()
    return avg_entropy
```

### 2. 双GPU架构设计

```python
# GPU设备分配策略
self.device_policy = torch.device("cuda:0")    # 策略模型
self.device_ref = torch.device("cuda:0")       # 参考模型
self.device_critic = torch.device("cuda:1")    # 价值模型
self.device_reward = torch.device("cuda:1")    # 奖励模型
```

### 3. Token级别计算

```python
def get_token_log_probs(self, model, prompts, responses, device, tokenizer):
    """获取Token级别的log_probs并返回Mask"""
    full_texts = [p + r for p, r in zip(prompts, responses)]
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True)
    
    # 计算Prompt的长度，仅在Response区域计算损失
    prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    
    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]  # Shift对齐
    labels = inputs["input_ids"][:, 1:]
    
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
    
    # 制作Mask: 1仅在Response区域且非Padding处
    mask = torch.zeros_like(labels, dtype=torch.bool)
    for i, p_len in enumerate(prompt_lens):
        mask[i, p_len:] = (labels[i, p_len:] != tokenizer.pad_token_id)
    
    return token_log_probs, mask
```

### 4. 可视化工具集成

#### PPO专用指标图表
```python
def plot_ppo_metrics_with_entropy(policy_losses, value_losses, rewards, advantages, entropies):
    """绘制PPO的5个核心指标"""
    # 策略损失、价值损失、奖励、优势、熵
```

#### 通用训练指标
```python
def plot_training_metrics(metrics_history):
    """通用的训练指标可视化"""
    # 支持任意数量的指标动态绘制
```

## 算法原理

### PPO核心思想
PPO通过限制策略更新的幅度来保证训练稳定性，避免策略崩溃：

1. **信任域约束**: 通过裁剪比率限制策略变化
2. **Actor-Critic**: 结合策略梯度和价值函数
3. **多轮更新**: 在同一批数据上进行多次更新
4. **KL散度惩罚**: 防止策略偏离参考策略过远

### 数学公式

#### PPO-Clip目标函数
```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```
其中：
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` 是概率比率
- `A_t` 是优势函数
- `ε` 是裁剪参数（CLIP_RANGE = 0.2）

#### 价值函数损失
```
L^VF(θ) = E[(V_θ(s_t) - V_t^targ)^2]
```

#### 熵奖励
```
L^ENT(θ) = E[H(π_θ(·|s_t))]
```

#### 总目标函数
```
L(θ) = L^CLIP(θ) - c1*L^VF(θ) + c2*L^ENT(θ)
```

## 代码结构特点

### 1. 模块化设计
- 清晰的类结构和方法分离
- 独立的奖励计算、价值估计、策略更新模块
- 可复用的token级别计算函数

### 2. 显存优化
```python
# 显存管理策略
del inputs, outputs, gen_ids  # 及时释放张量
torch.cuda.empty_cache()      # 清理显存缓存
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
```

### 3. 错误处理和日志
```python
try:
    shutil.copy2(current_script, target_script)
    print(f"脚本已备份至: {target_script}")
except Exception as e:
    print(f"脚本备份失败: {e}")
```

## 配置参数

### 核心超参数
```python
LEARNING_RATE = 1e-6        # 学习率
BATCH_SIZE = 4              # 批次大小
GROUP_SIZE = 1              # 每个prompt的回复数量
GROUP_EPOCHES = 4           # PPO更新轮数
CLIP_RANGE = 0.2            # PPO裁剪范围
```

### 模型配置
```python
ACTOR_MODEL = r"E:\models\Qwen\Qwen3-0___6B"      # 策略模型
CRITIC_MODEL = r"E:\models\Qwen\Qwen3-0___6B"     # 价值模型
REWARD_MODEL = r"E:\models\reward-model-deberta-v3-large-v2"  # 奖励模型
```

### 训练配置
```python
NUM_EPOCHES = 1             # 训练轮数
DTYPE = torch.bfloat16      # 数据类型
OUTPUT_DIR = "ppo_output"   # 输出目录
```

## 与其他算法对比

### PPO vs 传统策略梯度
| 特性 | 传统PG | PPO |
|------|--------|-----|
| 稳定性 | 不稳定 | 稳定 |
| 样本效率 | 低 | 高 |
| 实现复杂度 | 简单 | 中等 |
| 超参数敏感性 | 高 | 低 |

### PPO vs GRPO vs DAPO
| 特性 | PPO | GRPO | DAPO |
|------|-----|------|------|
| 架构 | Actor-Critic | Policy-Only | Policy-Only |
| 裁剪方式 | 对称 | 对称 | 非对称 |
| 损失级别 | Token | Token | Token |
| KL惩罚 | ✅ | ✅ | ❌ |
| 动态采样 | ❌ | ❌ | ✅ |
| 适用场景 | 通用 | 长文本 | 长链推理 |

## 训练指标

### 记录的指标
```python
self.metrics_history = {
    'policy_loss': [],      # 策略损失
    'value_loss': [],       # 价值损失
    'reward': [],           # 平均奖励
    'advantage': [],        # 平均优势
    'entropy': []           # 策略熵
}
```

### 指标含义
- **Policy Loss**: 策略网络的损失，反映策略更新幅度
- **Value Loss**: 价值网络的损失，反映价值估计准确性
- **Reward**: 奖励模型给出的平均奖励
- **Advantage**: 优势函数值，反映动作的相对价值
- **Entropy**: 策略熵，反映探索程度

## 使用场景

### PPO适合的任务
- **通用对话生成**: 平衡质量和多样性
- **文本摘要**: 需要准确的价值估计
- **代码生成**: 需要稳定的训练过程
- **创意写作**: 需要保持一定的随机性

### PPO的优势
- **训练稳定**: 裁剪机制防止策略崩溃
- **样本效率高**: 多轮更新充分利用数据
- **通用性强**: 适用于各种文本生成任务
- **理论基础扎实**: 有完善的理论支撑

### PPO的局限性
- **计算开销大**: 需要训练两个网络（Actor + Critic）
- **超参数较多**: 需要调节多个超参数
- **显存需求高**: 需要同时加载多个模型
- **收敛较慢**: 相比简单方法需要更多训练步数

## 实现亮点

### 1. 双网络架构
```python
# 策略网络负责生成
policy_outputs = self.policy_model.generate(...)

# 价值网络负责评估
values = self.critic_model(**inputs)

# 参考网络提供基线
ref_log_probs = self.reference_model(**inputs)
```

### 2. 精确的Token级别计算
```python
# 只在response区域计算损失，避免prompt部分的干扰
mask[i, p_len:] = (labels[i, p_len:] != tokenizer.pad_token_id)
masked_loss = loss * mask
final_loss = masked_loss.sum() / mask.sum()
```

### 3. 完善的指标监控
```python
# 实时显示训练进度
pbar.set_description(
    f"PL:{metrics['policy_loss']:.4f} VL:{metrics['value_loss']:.4f} "
    f"R:{metrics['reward']:.2f} A:{metrics['advantage']:.2f} E:{metrics['entropy']:.3f}"
)
```

### 4. 自动化模型管理
```python
# 自动保存模型和配置
self.policy_model.save_pretrained(save_path)
self.critic_model.save_pretrained(critic_save_path)

# 自动备份训练脚本
shutil.copy2(current_script, target_script)
```

## 总结

PPO实现提供了一个稳定、高效的强化学习训练框架，特别适合需要平衡探索和利用的文本生成任务。通过Actor-Critic架构和裁剪机制，PPO在保证训练稳定性的同时实现了较高的样本效率。代码结构清晰，模块化程度高，具有良好的可维护性和扩展性。