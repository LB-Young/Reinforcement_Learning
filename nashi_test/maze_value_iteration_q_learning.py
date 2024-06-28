"""
策略迭代算法步骤：
1、初始化pi、初始化Q;
2、依据策略pi采样，每走一步就更新一步Q（采用Q(s,:)的最大值赋值给当前的Q(s,a)）;
判断迭代终止条件，循环使用pi迭代生成trajectory，更新Q；
固本方法为off policy方案；  
##TODO 待实现
"""


import numpy as np
import gym
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self):
        self.state = 0
        pass
    
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        if action == 0:
            self.state -= 3
        elif action == 1:
            self.state += 1
        elif action == 2:
            self.state += 3
        elif action == 3:
            self.state -= 1
        done = False
        reward = 0
        if self.state == 8:
            done = True
            reward = 1
        # state, reward, done, _
        return self.state, reward, done, {}
    

class Agent:
    def __init__(self):
        self.action_space = list(range(4))
        self.theta_0 = np.asarray([[np.nan, 1, 1, np.nan],      # s0
                      [np.nan, 1, np.nan, 1],      # s1
                      [np.nan, np.nan, 1, 1],      # s2
                      [1, np.nan, np.nan, np.nan], # s3 
                      [np.nan, 1, 1, np.nan],      # s4
                      [1, np.nan, np.nan, 1],      # s5
                      [np.nan, 1, np.nan, np.nan], # s6 
                      [1, 1, np.nan, 1]]           # s7
                     )
        self.pi = self._cvt_theta_to_pi()
        # self.pi = self._softmax_cvt_theta_to_pi()
        # self.theta = self.theta_0

        self.Q = np.random.rand(*self.theta_0.shape) * self.theta_0
        self.eta = 0.1
        self.gamma = 0.9
        self.eps = 0.5
        
    def _cvt_theta_to_pi(self):
        m, n = self.theta_0.shape
        pi = np.zeros((m, n))
        for r in range(m):
            pi[r, :] = self.theta_0[r, :] / np.nansum(self.theta_0[r, :])
        return np.nan_to_num(pi)
    
    def get_action(self, s):
        # eps, explore
        if np.random.rand() < self.eps:
            action = np.random.choice(self.action_space, p=self.pi[s, :])
        else:
            # 1-eps, exploit
            action = np.nanargmax(self.Q[s, :])
        return action
        
    def sarsa(self, s, a, r, s_next, a_next):
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])
    def q_learning(self, s, a, r, s_next):
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * np.nanmax(self.Q[s_next, :]) - self.Q[s, a])


maze = MazeEnv()
agent = Agent()
epoch = 0
while True:
    old_Q = np.nanmax(agent.Q, axis=1)
    s = maze.reset()
    a = agent.get_action(s)
    s_a_history = [[s, np.nan]]
    while True:
        # s, a 
        s_a_history[-1][1] = a
        s_next, reward, done, _ = maze.step(a, )
        # s_next, a_next
        s_a_history.append([s_next, np.nan])
        if done:
            a_next = np.nan
        else:
            a_next = agent.get_action(s_next)
        # print(s, a, reward, s_next, a_next)
        # agent.sarsa(s, a, reward, s_next, a_next)
        agent.q_learning(s, a, reward, s_next, )
        # print(agent.pi)
        if done:
            break
        else:
            a = a_next
            s = maze.state

    # s_s_history, agent.Q
    update = np.sum(np.abs(np.nanmax(agent.Q, axis=1) - old_Q))
    epoch +=1
    agent.eps /= 2
    print(epoch, update, len(s_a_history))
    if epoch > 1000 or update < 1e-4:
        break