# Utils 工具包

训练过程中的辅助工具函数集合。

## 功能

### 1. 训练指标可视化 (`plot_metrics.py`)

提供多种绘图函数，用于可视化训练过程中的各项指标。

## 使用方法

### 基础用法

```python
from utils import plot_training_metrics

# 准备指标数据
metrics_history = {
    'policy_loss': [0.5, 0.45, 0.4, 0.38, 0.35],
    'value_loss': [0.3, 0.28, 0.25, 0.23, 0.21],
    'reward': [1.0, 1.2, 1.5, 1.7, 1.9],
    'entropy': [0.8, 0.75, 0.7, 0.68, 0.65]
}

# 绘制并保存图表
plot_training_metrics(
    metrics_history,
    save_path="./outputs/training_metrics.png",
    title="My Training Metrics"
)
```

### PPO 专用绘图

```python
from utils import plot_ppo_metrics

plot_ppo_metrics(
    policy_losses=[0.5, 0.4, 0.3],
    value_losses=[0.3, 0.25, 0.2],
    rewards=[1.0, 1.5, 2.0],
    advantages=[0.1, 0.15, 0.2],
    save_path="./outputs/ppo_metrics.png"
)
```

### GRPO 专用绘图

```python
from utils import plot_grpo_metrics

plot_grpo_metrics(
    losses=[0.5, 0.4, 0.3],
    rewards=[1.0, 1.5, 2.0],
    kl_divs=[0.01, 0.015, 0.012],
    save_path="./outputs/grpo_metrics.png"
)
```

### 损失和熵绘图

```python
from utils import plot_loss_and_entropy

plot_loss_and_entropy(
    losses=[0.5, 0.4, 0.3, 0.25],
    entropies=[0.8, 0.75, 0.7, 0.68],
    save_path="./outputs/loss_entropy.png"
)
```

## 在训练脚本中集成

### PPO 训练脚本示例

```python
from utils import plot_ppo_metrics

class PPOTrainer:
    def __init__(self):
        # ... 初始化代码 ...
        # 添加指标记录
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'reward': [],
            'advantage': []
        }
    
    def train_step(self, batch_prompts):
        # ... 训练代码 ...
        metrics = {
            "policy_loss": total_policy_loss / GROUP_EPOCHES,
            "value_loss": total_value_loss / GROUP_EPOCHES,
            "reward": rewards.mean().item(),
            "advantage": advantages.mean().item()
        }
        
        # 记录指标
        for key, value in metrics.items():
            self.metrics[key].append(value)
        
        return metrics
    
    def train(self, dataset):
        # ... 训练循环 ...
        
        # 训练结束后绘制图表
        plot_ppo_metrics(
            policy_losses=self.metrics['policy_loss'],
            value_losses=self.metrics['value_loss'],
            rewards=self.metrics['reward'],
            advantages=self.metrics['advantage'],
            save_path=os.path.join(OUTPUT_DIR, "training_metrics.png")
        )
```

### GRPO 训练脚本示例

```python
from utils import plot_grpo_metrics

class GRPOTrainer:
    def __init__(self):
        # ... 初始化代码 ...
        self.metrics = {
            'loss': [],
            'reward': [],
            'kl_div': []
        }
    
    def train_step(self, batch_prompts):
        # ... 训练代码 ...
        
        # 记录指标
        self.metrics['loss'].append(metrics['loss'])
        self.metrics['reward'].append(metrics['reward'])
        # 如果计算了 KL 散度
        # self.metrics['kl_div'].append(kl_div)
        
        return metrics
    
    def train(self, dataset):
        # ... 训练循环 ...
        
        # 训练结束后绘制图表
        plot_grpo_metrics(
            losses=self.metrics['loss'],
            rewards=self.metrics['reward'],
            kl_divs=self.metrics.get('kl_div'),
            save_path=os.path.join(OUTPUT_DIR, "training_metrics.png")
        )
```

## API 参考

### `plot_training_metrics()`

通用的训练���标绘制函数，支持任意数量的指标。

**参数：**
- `metrics_history` (Dict[str, List[float]]): 指标历史字典
- `save_path` (Optional[str]): 保存路径，None 则显示图片
- `figsize` (tuple): 图片大小，默认 (15, 10)
- `title` (str): 图片标题

### `plot_ppo_metrics()`

PPO 算法专用绘图函数。

**参数：**
- `policy_losses` (List[float]): 策略损失列表
- `value_losses` (List[float]): 价值损失列表
- `rewards` (List[float]): 奖励列表
- `advantages` (List[float]): 优势值列表
- `save_path` (Optional[str]): 保存路径
- `figsize` (tuple): 图片大小，默认 (15, 10)

### `plot_grpo_metrics()`

GRPO 算法专用绘图函数。

**参数：**
- `losses` (List[float]): 损失列表
- `rewards` (List[float]): 奖励列表
- `kl_divs` (Optional[List[float]]): KL 散度列表
- `save_path` (Optional[str]): 保存路径
- `figsize` (tuple): 图片大小，默认 (15, 5)

### `plot_loss_and_entropy()`

绘制损失和熵的变化。

**参数：**
- `losses` (List[float]): 损失值列表
- `entropies` (List[float]): 熵值列表
- `save_path` (Optional[str]): 保存路径
- `figsize` (tuple): 图片大小，默认 (12, 5)

## 依赖

```bash
pip install matplotlib numpy
```

## 特性

- ✅ 自动计算趋势线
- ✅ 网格线辅助阅读
- ✅ 高分辨率输出 (300 DPI)
- ✅ 自动创建保存目录
- ✅ 支持多种指标类型
- ✅ 专门优化的 PPO/GRPO 可视化
