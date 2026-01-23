"""
本代码中  当前状态的即时受益 + 计算的下一时刻状态值的最大值 与当前状态值计算差值；所以本质上为Q-learning
"""


from collections import namedtuple
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import gym

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory:     # 存储最新的一段数据
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0  # 下一条数据存放的索引

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)    # 数据存放到索引先None占位
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):       # 定义DQN模型，输入为状态值，输出为action值，输入状态为4维，输出action为向左或者向右；
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
    

class Agent:
    def __init__(self, n_states, n_actions, eta=0.5, gamma=0.99, capacity=10000, batch_size=32):
        self.n_states = n_states
        self.n_actions = n_actions
        self.eta = eta
        self.gamma = gamma
        self.capacity = capacity
        self.batch_size = batch_size

        self.memory = ReplayMemory(capacity=self.capacity)
        self.model = DQN(n_actions=self.n_actions, n_states=self.n_states)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch_datas = self.memory.sample(self.batch_size)
        batch_datas = Transition(*zip(*batch_datas))
        state_batch = torch.cat(batch_datas.state)      # 输入状态
        action_batch = torch.cat(batch_datas.action)    # 实际trajectory中采样的action
        reward_batch = torch.cat(batch_datas.reward)    # 实际trajectory中采样的reward
        non_final_next_state_batch = torch.cat([s for s in batch_datas.next_state if s is not None])   # 实际trajectory采样的下一个状态；去掉了下一个state为None的数据
        none_final_mask = torch.ByteTensor(tuple(map(lambda s:s is not None, batch_datas.next_state)))  # 下一个状态的mask
        next_state_values = torch.zeros(self.batch_size)

        self.model.eval()
        next_state_values[none_final_mask] = self.model(non_final_next_state_batch).max(dim=1)[0].detach()  # 下一个状态的状态值
        expected_state_action_values = reward_batch + self.gamma * next_state_values    # 期望的状态值为即时收益+下一时刻状态值
        state_action_values = self.model(state_batch).gather(dim=1, index=action_batch)     # 当前时刻的状态值

        self.model.train()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_q_function(self):
        self._replay()

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
    
    def choose_action(self, state, episode):    
        eps = 0.5 * 1 / (1 + episode)       # episode值越来越大，随机选action的概率越来越小，更多选择模型的输出action
        if random.random() < eps:
            action = torch.IntTensor([[random.randrange(self.n_actions)]])
        else:
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)
        return action
    

env = gym.make("CartPole-v0")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

max_episode = 500
max_steps = 200

complete_episode =0
finished_flag = False

agent = Agent(n_states, n_actions)
frames = []

for episode in range(max_episode):  # 与环境交互N轮trajectory
    state = env.reset()[0]      # trajectory的初始状态,array类型：array([-0.04162894,  0.01775856,  0.02309508, -0.02457856], dtype=float32)
    state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)    # 转换为tensor类型
    for step in range(max_steps):   # 每一轮trajectory最多玩200轮
        # if finished_flag:
        #     frames.append(env.render(mode='rgb_array'))
        
        action = agent.choose_action(state, episode)    # agent根据当前状态选择一个action
        next_state, _, done, _, _ = env.step(action.item())     # 环境空间根据state和action选择next_state
        if step == 199:
            done = True     # 如果玩了200轮以上，则认为在第200轮失败
        if done:    # 如果cartpole失败则done为True
            next_state = None   # 失败时没有next_state

            if step < 195:      # 如果失败的时候trajectory累计步数小于195，奖励为-1；
                reward = torch.FloatTensor([-1.])
                complete_episode = 0
            else:               # 如果失败的时候trajectory累计步数大于195，奖励为1；
                reward = torch.FloatTensor([1.])
                complete_episode += 1
        else:   # 如果没有结束，奖励为0，继续下一步；
            reward=torch.FloatTensor([0])
            next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
            next_state = next_state.unsqueeze(0)

        agent.memorize(state, action, next_state, reward)   # 将当前trajectory的当前step数据缓存
        agent.update_q_function()   # 更新q_function
        state = next_state

        if done:
            print(f'episode: {episode}, steps: {step}')
            break

    if finished_flag:
        break
        
    if complete_episode >= 10:
        finished_flag = True
        print('连续成功10轮')