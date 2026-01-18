"""
由于采样了一条完整的trajectory之后再进行policy_update，所以应该是一种MC的方法；
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import gym


class PolicyyNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size):
        super(PolicyyNetwork, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=num_actions)
        
    def forward(self, x):
        return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=1)
    
class Agent:
    def __init__(self, n_states, n_actions, gamma=0.9, learning_rate=5e-4):
        self.gamma = gamma
        self.n_states = n_states
        self.n_actions = n_actions
        self.model = PolicyyNetwork(n_states, n_actions, 128)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        pass

    def choose_action(self, state):     # 返回概率最大的action和对应的概率
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(state)
        highest_prob_action = np.random.choice(self.n_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
    
    def discounted_future_reward(self, rewards):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt += (self.gamma**pw)*r
                pw += 1
            discounted_rewards.append(Gt)
        # len(discounted_rewards) == len(rewards)
        return discounted_rewards
    
    def update_policy(self, rewards, log_probs):
        """
        使用策略梯度方法更新深度神经网络策略
        
        实现REINFORCE算法（蒙特卡洛策略梯度）：
        1. 计算每个时间步的折扣回报Gt
        2. 对回报进行标准化（归一化）
        3. 计算策略梯度并更新模型参数
        
        计算公式：
        1. 折扣回报计算: G_t = Σ(γ^k * r_{t+k}) for k=0 to T-t
           其中γ是折扣因子，r是即时奖励，T是轨迹长度
        
        2. 回报标准化: G̃_t = (G_t - μ_G) / (σ_G + ε)
           其中μ_G是所有回报的均值，σ_G是标准差，ε是小常数防止除零
        
        3. 策略梯度: ∇_θJ(θ) = E[-log(π_θ(a_t|s_t)) * G̃_t]
           最终的损失函数: L(θ) = Σ(-log(π_θ(a_t|s_t)) * G̃_t)
        
        参数:
        - rewards: 一个episode中每步获得的奖励列表
        - log_probs: 相应动作的对数概率列表
        """
        # 计算每个时间步的折扣未来回报
        discounted_rewards = self.discounted_future_reward(rewards)
        # 转换为张量以便进行计算
        discounted_rewards = torch.tensor(discounted_rewards)
        # 标准化回报以减少方差（提高训练稳定性）
        discounted_rewards = (discounted_rewards - discounted_rewards.mean())/(discounted_rewards.std() + 1e-9)
        
        # 计算每个时间步的策略梯度
        policy_grads = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            # 策略梯度计算: -log(π_θ(a_t|s_t)) * G_t
            # 负号是因为我们要进行梯度上升（最大化期望回报）
            # 而优化器默认执行梯度下降（最小化损失）
            policy_grads.append(-log_prob * Gt)
        
        # 清除之前累积的梯度
        self.optimizer.zero_grad()
        # 将所有时间步的梯度求和
        policy_grad = torch.stack(policy_grads).sum()
        # 反向传播计算梯度
        policy_grad.backward()
        # 更新策略网络参数
        self.optimizer.step()


env = gym.make("CartPole-v0")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

max_episodes = 2000
max_steps = 500

complete_episode =0
finished_flag = False

agent = Agent(n_states, n_actions)

num_steps = []
avg_num_steps = []
all_rewards = []

for episode in range(max_episodes):
    state = env.reset()[0]
    log_probs = []
    rewards = []
    for step in range(max_steps):
        # $\pi_\theta(a_t|s_t)$
        action, log_prob = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        log_probs.append(log_prob)
        rewards.append(reward)
        
        if done:
            # 完成一次 episode/rollout，得到一次完整的 trajectory
            agent.update_policy(rewards, log_probs)
            num_steps.append(step)
            avg_num_steps.append(np.mean(num_steps[-10:]))
            all_rewards.append(sum(rewards))
            if episode % 100 == 0:
                print(f'episode: {episode}, total reward: {sum(rewards)}, average_reward: {np.mean(all_rewards)}, length: {step}')
            break
        state = next_state
plt.plot(num_steps)
plt.plot(avg_num_steps)
plt.legend(['num_steps', 'avg_steps'])
plt.xlabel('episode')
plt.show()