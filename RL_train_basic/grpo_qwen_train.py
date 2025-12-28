"""
基于GRPO算法的语言模型强化学习训练脚本
使用Qwen2.5-0.5B作为策略模型，通过组内相对优化进行训练
GRPO (Group Relative Policy Optimization) 不需要价值函数，通过组内归一化来估计优势
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
CLIP_RATIO = 0.2  # GRPO中的裁剪参数
GAMMA = 0.99  # 折扣因子（GRPO中可能不需要，但保留以防需要）
GROUP_SIZE = 8  # GRPO组大小：每个提示生成的响应数量
TEMPERATURE = 0.8  # 采样温度，控制生成多样性
ENTROPY_COEF = 0.01  # 熵正则化系数
MAX_GRAD_NORM = 0.5  # 梯度裁剪阈值
UPDATE_EPOCHS = 4  # 每次收集数据后的更新次数
TARGET_KL = 0.01  # KL散度目标阈值
BETA_KL = 0.1  # KL散度惩罚系数

class TextEnv:
    """
    文本生成环境，用于与策略模型进行交互
    支持GRPO的组内评估机制
    """
    def __init__(self, prompts, reward_functions, tokenizer):
        """
        初始化文本环境
        
        参数:
            prompts (list): 提示文本列表
            reward_functions (list): 可编程奖励函数列表
            tokenizer (Tokenizer): 用于文本编码的分词器
        """
        self.prompts = prompts
        self.reward_functions = reward_functions
        self.tokenizer = tokenizer
        self.current_idx = 0
        
    def reset(self):
        """
        重置环境状态，随机选择一个新的提示
        
        返回:
            str: 选择的提示文本
        """
        self.current_idx = random.randint(0, len(self.prompts) - 1)
        return self.prompts[self.current_idx]
    
    def evaluate_group_responses(self, prompt, responses):
        """
        评估一组响应并计算组内相对奖励
        这是GRPO的核心：通过组内归一化来计算优势
        
        参数:
            prompt (str): 输入提示
            responses (list): 响应列表
            
        返回:
            list: 归一化后的奖励列表
        """
        # 计算每个响应的原始奖励
        raw_rewards = []
        for response in responses:
            total_reward = 0
            for reward_func in self.reward_functions:
                total_reward += reward_func(prompt, response)
            raw_rewards.append(total_reward)
        
        # GRPO核心：组内归一化
        # 计算组内平均值和标准差
        mean_reward = np.mean(raw_rewards)
        std_reward = np.std(raw_rewards) + 1e-8  # 避免除零
        
        # 归一化奖励（这就是GRPO的优势估计）
        normalized_rewards = [(r - mean_reward) / std_reward for r in raw_rewards]
        
        return normalized_rewards, raw_rewards
    
    def sample_batch(self, batch_size):
        """
        从提示列表中随机采样一批提示
        
        参数:
            batch_size (int): 批次大小
            
        返回:
            list: 采样的提示列表
        """
        indices = random.sample(range(len(self.prompts)), min(batch_size, len(self.prompts)))
        return [self.prompts[i] for i in indices]

class PolicyNetwork(nn.Module):
    """
    策略网络 - 基于Qwen2.5-0.5B模型
    注意：GRPO不需要价值函数，所以这里只有策略网络
    """
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        """
        初始化策略网络
        
        参数:
            model_name (str): 预训练模型的名称
        """
        super(PolicyNetwork, self).__init__()
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
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播函数
        
        参数:
            input_ids (tensor): 输入的token ID
            attention_mask (tensor, 可选): 注意力掩码
            
        返回:
            tensor: 下一个token的概率分布logits
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取下一个token的logits
        logits = outputs.logits[:, -1, :]
        
        return logits
    
    def get_action_and_log_prob(self, input_ids, attention_mask=None, action=None, temperature=1.0):
        """
        获取动作和动作对数概率
        
        参数:
            input_ids (tensor): 输入的token ID
            attention_mask (tensor, 可选): 注意力掩码
            action (tensor, 可选): 预定义的动作
            temperature (float): 采样温度
            
        返回:
            tuple: (action, action_log_prob, entropy) - 选择的动作、动作对数概率、熵
        """
        # 获取logits
        logits = self.forward(input_ids, attention_mask)
        
        # 应用温度缩放
        logits = logits / temperature
        
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
        
        return action, action_log_prob, entropy
    
    def generate_group_responses(self, prompt, tokenizer, group_size=GROUP_SIZE, max_length=50, temperature=TEMPERATURE):
        """
        为单个提示生成一组响应
        这是GRPO的关键：为每个提示生成多个候选响应
        
        参数:
            prompt (str): 输入的提示文本
            tokenizer (Tokenizer): 用于编码/解码的分词器
            group_size (int): 组大小
            max_length (int): 最大生成长度
            temperature (float): 采样温度
            
        返回:
            list: 生成的响应列表
        """
        responses = []
        
        for _ in range(group_size):
            # 编码提示
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # 生成序列
            generated_tokens = []
            for _ in range(max_length):
                with torch.no_grad():
                    # 获取下一个token
                    action, _, _ = self.get_action_and_log_prob(input_ids, attention_mask, temperature=temperature)
                    generated_tokens.append(action.item())
                    
                    # 更新输入序列
                    input_ids = torch.cat([input_ids, action], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long, device=DEVICE)], dim=1)
                    
                    # 如果生成了结束标记，就停止
                    if action.item() == tokenizer.eos_token_id:
                        break
            
            # 解码生成的序列
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(generated_text)
        
        return responses

class GRPOMemoryBuffer:
    """
    GRPO专用的经验回放缓冲区
    存储组内的轨迹数据和归一化奖励
    """
    def __init__(self):
        """
        初始化GRPO经验回放缓冲区
        """
        self.prompts = []
        self.responses = []
        self.actions = []  # token序列
        self.log_probs = []  # 对应的对数概率
        self.normalized_rewards = []  # GRPO归一化奖励
        self.raw_rewards = []  # 原始奖励
        
    def add_group(self, prompt, responses, action_sequences, log_prob_sequences, normalized_rewards, raw_rewards):
        """
        添加一组轨迹数据到缓冲区
        
        参数:
            prompt (str): 提示
            responses (list): 响应列表
            action_sequences (list): 动作序列列表
            log_prob_sequences (list): 对数概率序列列表
            normalized_rewards (list): 归一化奖励列表
            raw_rewards (list): 原始奖励列表
        """
        for i in range(len(responses)):
            self.prompts.append(prompt)
            self.responses.append(responses[i])
            self.actions.append(action_sequences[i])
            self.log_probs.append(log_prob_sequences[i])
            self.normalized_rewards.append(normalized_rewards[i])
            self.raw_rewards.append(raw_rewards[i])
        
    def clear(self):
        """
        清空缓冲区
        """
        self.prompts.clear()
        self.responses.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.normalized_rewards.clear()
        self.raw_rewards.clear()
        
    def get_batch(self):
        """
        获取缓冲区中的所有数据
        
        返回:
            tuple: 包含所有存储数据的元组
        """
        return (
            self.prompts,
            self.responses,
            self.actions,
            self.log_probs,
            self.normalized_rewards,
            self.raw_rewards
        )

class GRPOAgent:
    """
    GRPO代理，用于训练策略网络
    """
    def __init__(self, policy_model, tokenizer, learning_rate=1e-5):
        """
        初始化GRPO代理
        
        参数:
            policy_model (PolicyNetwork): 策略网络
            tokenizer (Tokenizer): 分词器
            learning_rate (float): 学习率
        """
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 移动模型到设备
        self.policy = self.policy.to(DEVICE)
        
        # 保存参考策略用于KL散度计算
        self.reference_policy = PolicyNetwork().to(DEVICE)
        self.reference_policy.load_state_dict(self.policy.state_dict())
        self.reference_policy.eval()
        
    def collect_group_trajectories(self, env, prompts_batch):
        """
        收集一批提示的组轨迹数据
        
        参数:
            env (TextEnv): 文本环境
            prompts_batch (list): 提示批次
            
        返回:
            GRPOMemoryBuffer: 填充的内存缓冲区
        """
        memory = GRPOMemoryBuffer()
        
        for prompt in prompts_batch:
            # 为每个提示生成一组响应
            responses = self.policy.generate_group_responses(prompt, self.tokenizer)
            
            # 收集动作序列和对数概率
            action_sequences = []
            log_prob_sequences = []
            
            for response in responses:
                # 重新计算生成过程中的动作和对数概率
                full_text = prompt + response
                inputs = self.tokenizer(full_text, return_tensors="pt").to(DEVICE)
                input_ids = inputs["input_ids"]
                
                # 分离提示和响应的token
                prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
                prompt_length = prompt_inputs["input_ids"].shape[1]
                
                response_tokens = input_ids[0, prompt_length:].tolist()
                response_log_probs = []
                
                # 计算每个响应token的对数概率
                for i in range(len(response_tokens)):
                    context_ids = input_ids[:, :prompt_length + i]
                    target_token = torch.tensor([[response_tokens[i]]], device=DEVICE)
                    
                    with torch.no_grad():
                        _, log_prob, _ = self.policy.get_action_and_log_prob(context_ids, action=target_token)
                        response_log_probs.append(log_prob.item())
                
                action_sequences.append(response_tokens)
                log_prob_sequences.append(response_log_probs)
            
            # 评估组响应并获取归一化奖励
            normalized_rewards, raw_rewards = env.evaluate_group_responses(prompt, responses)
            
            # 添加到内存缓冲区
            memory.add_group(prompt, responses, action_sequences, log_prob_sequences, normalized_rewards, raw_rewards)
        
        return memory
    
    def update_policy(self, memory, clip_ratio=0.2, entropy_coef=0.01, beta_kl=0.1):
        """
        使用GRPO算法更新策略网络
        
        参数:
            memory (GRPOMemoryBuffer): 经验缓冲区
            clip_ratio (float): 裁剪参数
            entropy_coef (float): 熵损失系数
            beta_kl (float): KL散度惩罚系数
            
        返回:
            dict: 包含各种训练指标的字典
        """
        # 获取批次数据
        prompts, responses, actions, old_log_probs, normalized_rewards, raw_rewards = memory.get_batch()
        
        if len(prompts) == 0:
            return {"policy_loss": 0, "entropy": 0, "total_loss": 0, "approx_kl": 0}
        
        # 计算新的对数概率和熵
        total_policy_loss = 0
        total_entropy_loss = 0
        total_kl_loss = 0
        num_tokens = 0
        
        for i in range(len(prompts)):
            prompt = prompts[i]
            response = responses[i]
            action_seq = actions[i]
            old_log_prob_seq = old_log_probs[i]
            advantage = normalized_rewards[i]  # GRPO中，归一化奖励就是优势
            
            # 重新计算当前策略下的对数概率
            full_text = prompt + response
            inputs = self.tokenizer(full_text, return_tensors="pt").to(DEVICE)
            input_ids = inputs["input_ids"]
            
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            prompt_length = prompt_inputs["input_ids"].shape[1]
            
            new_log_probs = []
            entropies = []
            ref_log_probs = []
            
            for j, token in enumerate(action_seq):
                context_ids = input_ids[:, :prompt_length + j]
                target_token = torch.tensor([[token]], device=DEVICE)
                
                # 当前策略
                _, new_log_prob, entropy = self.policy.get_action_and_log_prob(context_ids, action=target_token)
                new_log_probs.append(new_log_prob)
                entropies.append(entropy)
                
                # 参考策略（用于KL散度）
                with torch.no_grad():
                    _, ref_log_prob, _ = self.reference_policy.get_action_and_log_prob(context_ids, action=target_token)
                    ref_log_probs.append(ref_log_prob)
            
            # 转换为张量
            new_log_probs = torch.stack(new_log_probs).squeeze()
            old_log_probs_tensor = torch.tensor(old_log_prob_seq, device=DEVICE)
            ref_log_probs = torch.stack(ref_log_probs).squeeze()
            entropies = torch.stack(entropies).squeeze()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            
            # GRPO策略损失（类似PPO但使用归一化奖励作为优势）
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # 熵损失
            entropy_loss = -entropies.mean()
            
            # KL散度损失（相对于参考策略）
            kl_loss = (new_log_probs - ref_log_probs).mean()
            
            total_policy_loss += policy_loss
            total_entropy_loss += entropy_loss
            total_kl_loss += kl_loss
            num_tokens += len(action_seq)
        
        # 平均损失
        avg_policy_loss = total_policy_loss / len(prompts)
        avg_entropy_loss = total_entropy_loss / len(prompts)
        avg_kl_loss = total_kl_loss / len(prompts)
        
        # 总损失
        total_loss = avg_policy_loss + entropy_coef * avg_entropy_loss + beta_kl * avg_kl_loss
        
        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
        self.optimizer.step()
        
        return {
            "policy_loss": avg_policy_loss.item(),
            "entropy": avg_entropy_loss.item(),
            "kl_loss": avg_kl_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": avg_kl_loss.item()
        }
    
    def learn(self, memory, update_epochs=4, target_kl=0.01):
        """
        从收集的轨迹中学习策略
        
        参数:
            memory (GRPOMemoryBuffer): 经验回放缓冲区
            update_epochs (int): 每批数据的更新次数
            target_kl (float): 目标KL散度，用于早停
            
        返回:
            dict: 训练指标的平均值
        """
        metrics = []
        for _ in range(update_epochs):
            # 更新策略
            update_info = self.update_policy(
                memory=memory,
                clip_ratio=CLIP_RATIO,
                entropy_coef=ENTROPY_COEF,
                beta_kl=BETA_KL
            )
            
            metrics.append(update_info)
            
            # 如果KL散度过大，提前停止
            if update_info["approx_kl"] > target_kl:
                break
                
        # 计算平均指标
        if metrics:
            avg_metrics = {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}
        else:
            avg_metrics = {"policy_loss": 0, "entropy": 0, "kl_loss": 0, "total_loss": 0, "approx_kl": 0}
        
        return avg_metrics
    
    def save_model(self, path):
        """
        保存模型到指定路径
        
        参数:
            path (str): 保存模型的路径
        """
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
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# 定义可编程奖励函数
def length_reward(prompt, response):
    """
    基于响应长度的奖励函数
    鼓励生成适当长度的响应
    """
    target_length = 50  # 目标长度
    actual_length = len(response.split())
    
    # 长度接近目标时给予更高奖励
    length_diff = abs(actual_length - target_length)
    reward = max(0, 1.0 - length_diff / target_length)
    
    return reward

def coherence_reward(prompt, response):
    """
    基于连贯性的奖励函数
    简单的启发式：检查是否包含常见的连接词
    """
    coherence_words = ["因为", "所以", "然而", "但是", "而且", "另外", "首先", "其次", "最后", "总之"]
    
    count = sum(1 for word in coherence_words if word in response)
    reward = min(1.0, count * 0.2)  # 最多1.0分
    
    return reward

def completeness_reward(prompt, response):
    """
    基于完整性的奖励函数
    检查响应是否看起来完整（以句号结尾等）
    """
    if response.strip().endswith(("。", "！", "？", ".", "!", "?")):
        return 1.0
    else:
        return 0.5

def relevance_reward(prompt, response):
    """
    基于相关性的奖励函数
    简单检查响应是否与提示相关
    """
    # 提取提示中的关键词
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    
    # 计算重叠度
    overlap = len(prompt_words.intersection(response_words))
    relevance_score = min(1.0, overlap / max(1, len(prompt_words) * 0.3))
    
    return relevance_score

def main():
    """
    主函数，执行GRPO强化学习训练过程
    """
    # 加载预训练模型和分词器
    print("加载Qwen2.5-0.5B模型和分词器...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建策略网络
    print("创建策略网络...")
    policy_model = PolicyNetwork(model_name)
    
    # 示例提示
    prompts = [
        "请介绍一下中国的历史",
        "解释一下量子力学的基本原理",
        "如何有效地学习编程?",
        "请写一首关于春天的诗",
        "谈谈人工智能对未来社会的影响",
        "描述一下你理想中的未来城市",
        "解释什么是可持续发展",
        "如何保持身心健康?",
        "介绍一下中国的传统文化",
        "谈谈科技对教育的影响"
    ]
    
    # 定义奖励函数
    reward_functions = [
        length_reward,
        coherence_reward,
        completeness_reward,
        relevance_reward
    ]
    
    # 创建环境
    env = TextEnv(prompts, reward_functions, tokenizer)
    
    # 创建GRPO代理
    agent = GRPOAgent(policy_model, tokenizer, learning_rate=LEARNING_RATE)
    
    # 训练指标记录
    rewards_history = []
    avg_rewards_history = []
    loss_history = []
    
    # 训练循环
    print("开始GRPO训练...")
    num_iterations = 50  # 训练迭代次数
    prompts_per_iteration = 4  # 每次迭代处理的提示数量
    
    for iteration in tqdm(range(num_iterations)):
        # 采样一批提示
        prompts_batch = env.sample_batch(prompts_per_iteration)
        
        # 收集组轨迹
        memory = agent.collect_group_trajectories(env, prompts_batch)
        
        # 计算平均奖励
        _, _, _, _, normalized_rewards, raw_rewards = memory.get_batch()
        if raw_rewards:
            avg_raw_reward = np.mean(raw_rewards)
            avg_normalized_reward = np.mean(normalized_rewards)
        else:
            avg_raw_reward = 0
            avg_normalized_reward = 0
        
        # 更新策略
        metrics = agent.learn(memory, update_epochs=UPDATE_EPOCHS, target_kl=TARGET_KL)
        
        # 记录指标
        rewards_history.append(avg_raw_reward)
        avg_rewards_history.append(avg_normalized_reward)
        loss_history.append(metrics["total_loss"])
        
        # 清空内存
        memory.clear()
        
        # 打印进度
        if iteration % 10 == 0:
            print(f"迭代 {iteration}/{num_iterations}")
            print(f"平均原始奖励: {avg_raw_reward:.4f}")
            print(f"平均归一化奖励: {avg_normalized_reward:.4f}")
            print(f"策略损失: {metrics['policy_loss']:.4f}")
            print(f"熵损失: {metrics['entropy']:.4f}")
            print(f"KL损失: {metrics['kl_loss']:.4f}")
            print(f"总损失: {metrics['total_loss']:.4f}")
            print("-" * 50)
            
            # 生成一些示例文本
            prompt = random.choice(prompts)
            responses = policy_model.generate_group_responses(prompt, tokenizer, group_size=3, max_length=30)
            print(f"提示: {prompt}")
            for i, response in enumerate(responses):
                print(f"响应{i+1}: {response}")
            print("=" * 50)
        
        # 保存模型
        if iteration % 20 == 0:
            agent.save_model(f"RL_train/grpo_qwen_model_iter_{iteration}.pt")
    
    # 保存最终模型
    agent.save_model("RL_train/grpo_qwen_model_final.pt")
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title("平均原始奖励")
    plt.xlabel("迭代")
    
    plt.subplot(1, 3, 2)
    plt.plot(avg_rewards_history)
    plt.title("平均归一化奖励")
    plt.xlabel("迭代")
    
    plt.subplot(1, 3, 3)
    plt.plot(loss_history)
    plt.title("总损失")
    plt.xlabel("迭代")
    
    plt.tight_layout()
    plt.savefig("RL_train/grpo_qwen_training_curves.png")
    plt.show()

if __name__ == "__main__":
    main()