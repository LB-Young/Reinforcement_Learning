"""
基于PPO算法的语言模型强化学习训练脚本
使用Qwen2.5-0.5B作为策略模型，Hugging Face的奖励模型进行评估
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from collections import deque
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    AutoModelForSequenceClassification
)
from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    """
    设置随机种子以确保实验结果可复现
    
    参数:
        seed (int): 随机种子值
    
    返回:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
set_seed(42)

# 全局配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-5
CLIP_RATIO = 0.2  # PPO中的裁剪参数
GAMMA = 0.99  # 折扣因子
GAE_LAMBDA = 0.95  # GAE参数
VALUE_COEF = 0.5  # 价值函数损失系数
ENTROPY_COEF = 0.01  # 熵正则化系数
MAX_GRAD_NORM = 0.5  # 梯度裁剪阈值
UPDATE_EPOCHS = 4  # 每次收集数据后的更新次数
TARGET_KL = 0.01  # KL散度目标阈值

class TextEnv:
    """
    文本生成环境，用于与策略模型进行交互
    """
    def __init__(self, prompts, reward_model, tokenizer):
        """
        初始化文本环境
        
        参数:
            prompts (list): 提示文本列表
            reward_model (Model): 用于评估生成文本的奖励模型
            tokenizer (Tokenizer): 用于文本编码的分词器
        """
        # 提示列表
        self.prompts = prompts
        # 奖励模型
        self.reward_model = reward_model
        # 分词器
        self.tokenizer = tokenizer
        # 当前提示索引
        self.current_idx = 0
        # 完成情况标记
        self.done = False
        
    def reset(self):
        """
        重置环境状态，随机选择一个新的提示
        
        返回:
            str: 选择的提示文本
        """
        # 随机选择一个提示
        self.current_idx = random.randint(0, len(self.prompts) - 1)
        self.done = False
        # 返回当前提示
        return self.prompts[self.current_idx]
    
    def step(self, response):
        """
        执行一步环境交互，计算奖励并判断是否完成
        
        参数:
            response (str): 模型生成的文本响应
            
        返回:
            tuple: (奖励值, 是否完成, 附加信息)
        """
        # 计算奖励
        prompt = self.prompts[self.current_idx]
        inputs = self.tokenizer(prompt + response, return_tensors="pt").to(DEVICE)
        
        # 使用奖励模型计算奖励
        with torch.no_grad():
            reward_scores = self.reward_model(**inputs).logits.squeeze(-1)
            reward = reward_scores.mean().item()
        
        # 检查是否完成（这里可以根据实际需求定义完成条件）
        # 例如，可以根据生成的标记数量或特定停止标记来判断
        done = len(response.split()) >= 50 or "</s>" in response
        
        # 返回奖励、是否完成以及额外信息
        return reward, done, {"complete_text": prompt + response}
    
    def sample_batch(self, batch_size):
        """
        从提示列表中随机采样一批提示
        
        参数:
            batch_size (int): 批次大小
            
        返回:
            list: 采样的提示列表
        """
        # 随机采样一批提示
        indices = random.sample(range(len(self.prompts)), min(batch_size, len(self.prompts)))
        return [self.prompts[i] for i in indices]

class PolicyValueNetwork(nn.Module):
    """
    策略值网络 - 基于Qwen2.5-0.5B模型，同时输出动作概率和状态值
    """
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        """
        初始化策略值网络
        
        参数:
            model_name (str): 预训练模型的名称
        """
        super(PolicyValueNetwork, self).__init__()
        # 加载预训练Qwen模型
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        # 冻结部分参数（可选）
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 仅微调最后几层
        for param in self.backbone.model.layers[-2:].parameters():
            param.requires_grad = True
        
        # 词汇表大小
        self.vocab_size = self.backbone.config.vocab_size
        
        # 值函数头
        self.value_head = nn.Linear(self.backbone.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播函数
        
        参数:
            input_ids (tensor): 输入的token ID
            attention_mask (tensor, 可选): 注意力掩码
            
        返回:
            tuple: (logits, value) - 下一个token的概率分布和当前状态值
        """
        # 获取模型输出
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一层隐藏状态的最后一个token的表示
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]
        
        # 计算状态值
        value = self.value_head(last_token_hidden)
        
        # 获取下一个token的logits
        logits = outputs.logits[:, -1, :]
        
        # 返回动作概率分布和状态值
        return logits, value
    
    def get_action_and_value(self, input_ids, attention_mask=None, action=None):
        """
        获取动作、动作概率、熵和状态值
        
        参数:
            input_ids (tensor): 输入的token ID
            attention_mask (tensor, 可选): 注意力掩码
            action (tensor, 可选): 预定义的动作
            
        返回:
            tuple: (action, action_log_prob, entropy, value) - 选择的动作、动作对数概率、熵和状态值
        """
        # 获取logits和状态值
        logits, value = self.forward(input_ids, attention_mask)
        
        # 计算动作概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 计算动作的对数概率分布
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 计算熵
        entropy = -(probs * log_probs).sum(-1)
        
        # 如果未提供动作，则采样一个
        if action is None:
            action = torch.multinomial(probs, 1)
        
        # 获取选择的动作的对数概率
        action_log_prob = log_probs.gather(-1, action)
        
        return action, action_log_prob, entropy, value
    
    def generate(self, prompt, tokenizer, max_length=50):
        """
        生成文本序列
        
        参数:
            prompt (str): 输入的提示文本
            tokenizer (Tokenizer): 用于编码/解码的分词器
            max_length (int): 最大生成长度
            
        返回:
            str: 生成的文本
        """
        # 编码提示
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 生成序列
        generated_tokens = []
        for _ in range(max_length):
            with torch.no_grad():
                # 获取下一个token
                action, _, _, _ = self.get_action_and_value(input_ids, attention_mask)
                generated_tokens.append(action.item())
                
                # 更新输入序列
                input_ids = torch.cat([input_ids, action], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=DEVICE)], dim=1)
                
                # 如果生成了结束标记，就停止
                if action.item() == tokenizer.eos_token_id:
                    break
        
        # 解码生成的序列
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

class MemoryBuffer:
    """
    经验回放缓冲区，用于存储轨迹数据
    """
    def __init__(self):
        """
        初始化经验回放缓冲区
        """
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_states = []
        
    def add(self, state, action, log_prob, reward, value, done, next_state):
        """
        添加一条轨迹数据到缓冲区
        
        参数:
            state (str): 当前状态
            action (int): 选择的动作
            log_prob (float): 动作的对数概率
            reward (float): 获得的奖励
            value (float): 状态值估计
            done (bool): 是否完成
            next_state (str): 下一个状态
        """
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_states.append(next_state)
        
    def clear(self):
        """
        清空缓冲区
        """
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.next_states.clear()
        
    def compute_returns_and_advantages(self, gamma=0.99, gae_lambda=0.95, last_value=0):
        """
        计算折扣回报和广义优势估计(GAE)
        
        参数:
            gamma (float): 折扣因子
            gae_lambda (float): GAE参数
            last_value (float): 最后一个状态的值估计
            
        返回:
            tuple: (returns, advantages) - 折扣回报和优势估计列表
        """
        # 计算GAE优势估计和折扣回报
        returns = []
        advantages = []
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = self.values[i+1]
                
            delta = self.rewards[i] + gamma * next_value * (1 - self.dones[i]) - self.values[i]
            gae = delta + gamma * gae_lambda * (1 - self.dones[i]) * gae
            
            # 计算回报和优势
            returns.insert(0, gae + self.values[i])
            advantages.insert(0, gae)
            
        return returns, advantages
    
    def get_batch(self):
        """
        获取缓冲区中的所有数据
        
        返回:
            tuple: 包含所有存储数据的元组
        """
        # 返回整个批次
        return (
            self.states,
            self.actions,
            self.log_probs,
            self.rewards,
            self.values,
            self.dones,
            self.next_states
        )

class PPOAgent:
    """
    PPO代理，用于训练策略值网络
    """
    def __init__(self, policy_model, tokenizer, learning_rate=1e-5):
        """
        初始化PPO代理
        
        参数:
            policy_model (PolicyValueNetwork): 策略值网络
            tokenizer (Tokenizer): 分词器
            learning_rate (float): 学习率
        """
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 移动模型到设备
        self.policy = self.policy.to(DEVICE)
        
    def choose_action(self, state):
        """
        根据当前状态选择动作
        
        参数:
            state (str): 当前状态/提示
            
        返回:
            tuple: (token_id, token_text, log_prob, value) - 选择的token ID、token文本、对数概率和状态值
        """
        # 编码输入
        inputs = self.tokenizer(state, return_tensors="pt").to(DEVICE)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # 获取动作和值
        with torch.no_grad():
            action, log_prob, entropy, value = self.policy.get_action_and_value(input_ids, attention_mask)
        
        # 将动作解码为文本
        next_token = action.item()
        next_token_text = self.tokenizer.decode([next_token])
        
        # 返回动作、对数概率和状态值
        return next_token, next_token_text, log_prob.item(), value.item()
    
    def evaluate_actions(self, states, actions):
        """
        评估给定状态和动作的价值
        
        参数:
            states (list): 状态列表
            actions (list): 动作列表
            
        返回:
            tuple: (log_probs, entropy, values) - 对数概率、熵和状态值
        """
        # 编码输入
        batch_input_ids = []
        batch_attention_mask = []
        
        for state in states:
            inputs = self.tokenizer(state, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            batch_input_ids.append(inputs["input_ids"])
            batch_attention_mask.append(inputs.get("attention_mask", None))
        
        batch_input_ids = torch.cat(batch_input_ids, dim=0).to(DEVICE)
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0).to(DEVICE) if batch_attention_mask[0] is not None else None
        
        # 转换动作为张量
        actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(DEVICE)
        
        # 获取动作和值
        _, log_probs, entropy, values = self.policy.get_action_and_value(batch_input_ids, batch_attention_mask, actions)
        
        return log_probs, entropy, values
    
    def update(self, states, actions, old_log_probs, returns, advantages, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        """
        使用PPO算法更新策略网络
        
        参数:
            states (list): 状态列表
            actions (list): 动作列表
            old_log_probs (list): 旧策略下的动作对数概率
            returns (list): 折扣回报
            advantages (list): 优势估计
            clip_ratio (float): PPO裁剪参数
            value_coef (float): 价值损失系数
            entropy_coef (float): 熵损失系数
            
        返回:
            dict: 包含各种训练指标的字典
        """
        # 将列表转换为张量
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float).to(DEVICE)
        returns = torch.tensor(returns, dtype=torch.float).to(DEVICE)
        advantages = torch.tensor(advantages, dtype=torch.float).to(DEVICE)
        
        # 归一化优势（提高稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算策略损失、价值损失和熵
        new_log_probs, entropy, values = self.evaluate_actions(states, actions)
        
        # 计算比率：new_prob / old_prob
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 裁剪PPO目标
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # 价值损失（均方误差）
        value_loss = F.mse_loss(values.squeeze(-1), returns)
        
        # 熵损失（用于鼓励探索）
        entropy_loss = -entropy.mean()
        
        # 总损失
        loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        
        # 计算KL散度（用于早停）
        approx_kl = ((old_log_probs - new_log_probs) * ratio).mean().item()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "total_loss": loss.item(),
            "approx_kl": approx_kl
        }
    
    def learn(self, memory, update_epochs=4, target_kl=0.01):
        """
        从收集的轨迹中学习策略
        
        参数:
            memory (MemoryBuffer): 经验回放缓冲区
            update_epochs (int): 每批数据的更新次数
            target_kl (float): 目标KL散度，用于早停
            
        返回:
            dict: 训练指标的平均值
        """
        # 计算回报和优势
        returns, advantages = memory.compute_returns_and_advantages(
            gamma=GAMMA, 
            gae_lambda=GAE_LAMBDA
        )
        
        # 获取批次数据
        states, actions, old_log_probs, rewards, values, dones, next_states = memory.get_batch()
        
        # 多次更新策略
        metrics = []
        for _ in range(update_epochs):
            # 更新策略
            update_info = self.update(
                states=states,
                actions=actions,
                old_log_probs=old_log_probs,
                returns=returns,
                advantages=advantages,
                clip_ratio=CLIP_RATIO,
                value_coef=VALUE_COEF,
                entropy_coef=ENTROPY_COEF
            )
            
            metrics.append(update_info)
            
            # 如果KL散度过大，提前停止
            if update_info["approx_kl"] > target_kl:
                break
                
        # 计算平均指标
        avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
        
        return avg_metrics
    
    def save_model(self, path):
        """
        保存模型到指定路径
        
        参数:
            path (str): 保存模型的路径
        """
        # 保存模型
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        
    def load_model(self, path):
        """
        从指定路径加载模型
        
        参数:
            path (str): 模型路径
        """
        # 加载模型
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

def collect_trajectories(env, agent, num_steps=1000):
    """
    收集环境交互轨迹数据
    
    参数:
        env (TextEnv): 文本环境
        agent (PPOAgent): PPO代理
        num_steps (int): 收集的步骤数
        
    返回:
        tuple: (memory, total_rewards, num_episodes) - 经验缓冲区、总奖励和完成的episode数量
    """
    memory = MemoryBuffer()
    total_rewards = 0
    num_episodes = 0
    
    # 重置环境
    state = env.reset()
    current_state = state
    
    for _ in range(num_steps):
        # 选择动作
        action, action_text, log_prob, value = agent.choose_action(current_state)
        
        # 更新当前状态
        next_state = current_state + action_text
        
        # 执行动作
        reward, done, info = env.step(action_text)
        
        # 存储到内存
        memory.add(current_state, action, log_prob, reward, value, done, next_state)
        
        # 累积奖励
        total_rewards += reward
        
        # 更新状态
        if done:
            num_episodes += 1
            state = env.reset()
            current_state = state
        else:
            current_state = next_state
            
    return memory, total_rewards, num_episodes

def main():
    """
    主函数，执行PPO强化学习训练过程
    """
    # 加载预训练模型和分词器
    print("加载Qwen2.5-0.5B模型和分词器...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载奖励模型
    print("加载奖励模型...")
    reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name).to(DEVICE)
    
    # 创建策略网络
    print("创建策略网络...")
    policy_model = PolicyValueNetwork(model_name)
    
    # 示例提示
    prompts = [
        "请介绍一下中国的历史",
        "解释一下量子力学的基本原理",
        "如何有效地学习编程?",
        "请写一首关于春天的诗",
        "谈谈人工智能对未来社会的影响"
    ]
    
    # 创建环境
    env = TextEnv(prompts, reward_model, tokenizer)
    
    # 创建PPO代理
    agent = PPOAgent(policy_model, tokenizer, learning_rate=LEARNING_RATE)
    
    # 训练指标记录
    rewards_history = []
    avg_rewards_history = []
    loss_history = []
    
    # 训练循环
    print("开始训练...")
    num_iterations = 100  # 训练迭代次数
    steps_per_iteration = 64  # 每次迭代收集的步骤数
    
    for iteration in tqdm(range(num_iterations)):
        # 收集轨迹
        memory, total_rewards, num_episodes = collect_trajectories(env, agent, steps_per_iteration)
        
        # 计算平均奖励
        avg_reward = total_rewards / num_episodes if num_episodes > 0 else 0
        
        # 更新策略
        metrics = agent.learn(memory, update_epochs=UPDATE_EPOCHS, target_kl=TARGET_KL)
        
        # 记录指标
        rewards_history.append(total_rewards)
        avg_rewards_history.append(avg_reward)
        loss_history.append(metrics["total_loss"])
        
        # 清空内存
        memory.clear()
        
        # 打印进度
        if iteration % 10 == 0:
            print(f"迭代 {iteration}/{num_iterations}")
            print(f"总奖励: {total_rewards:.2f}")
            print(f"平均奖励: {avg_reward:.2f}")
            print(f"策略损失: {metrics['policy_loss']:.4f}")
            print(f"价值损失: {metrics['value_loss']:.4f}")
            print(f"熵: {metrics['entropy']:.4f}")
            print(f"KL散度: {metrics['approx_kl']:.4f}")
            print("-" * 50)
            
            # 生成一些示例文本
            prompt = random.choice(prompts)
            generated_text = policy_model.generate(prompt, tokenizer)
            print(f"提示: {prompt}")
            print(f"生成: {generated_text}")
            print("=" * 50)
        
        # 保存模型
        if iteration % 20 == 0:
            agent.save_model(f"RL_train/ppo_qwen_model_iter_{iteration}.pt")
    
    # 保存最终模型
    agent.save_model("RL_train/ppo_qwen_model_final.pt")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title("总奖励")
    plt.xlabel("迭代")
    
    plt.subplot(1, 3, 2)
    plt.plot(avg_rewards_history)
    plt.title("平均奖励")
    plt.xlabel("迭代")
    
    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title("总损失")
    plt.xlabel("迭代")
    
    plt.tight_layout()
    plt.savefig("RL_train/ppo_qwen_training_curves.png")
    plt.show()

if __name__ == "__main__":
    main() 