"""
价值迭代sarsa：
1、初始化pi、Q；
2、根据初始化的pi走一步，即刻使用TD算法，使用下一时刻的Q(s_next,a_next)对Q(s,a)进行更新；循环N步直至达到终点；
判断迭代终止条件，循环使用pi迭代生成trajectory，更新Q；
固本方法为on policy方案；
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self):
        self.state = 0
    
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
        done =False
        reward = 0
        if self.state == 8:
            done = True
            reward = 1
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
    
    # def _softmax_cvt_theta_to_pi(self, beta=1.):
    #     m, n = self.theta.shape
    #     pi = np.zeros((m, n))
    #     exp_theta = np.exp(self.theta*beta)
    #     for r in range(m):
    #         pi[r, :] = exp_theta[r, :] / np.nansum(exp_theta[r, :])
    #     return np.nan_to_num(pi)
    
    def get_action(self, s):
        # eps, explore
        if np.random.rand() < self.eps:
            action = np.random.choice(self.action_space, p=self.pi[s, :])
        else:
            # 1-eps, exploit
            action = np.nanargmax(self.Q[s, :])
        return action
        
    def sarsa(self, s, a, r, s_next, a_next):
        # TD learning of action value: sarsa
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])


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
        agent.sarsa(s, a, reward, s_next, a_next)
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
    if epoch > 100 or update < 1e-4:
        break