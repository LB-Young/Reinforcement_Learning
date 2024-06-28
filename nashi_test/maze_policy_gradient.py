"""
策略迭代算法步骤：
1、初始化pi；
2、？？这是什么算法？
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
        done = False
        if self.state == 8:
            done = True
        # state, reward, done, _
        return self.state, 1, done, {}
    

class Agent:
    def __init__(self):
        self.actions = list(range(4))
        self.theta_0 =  np.asarray(
            [[np.nan, 1, 1, np.nan],      # s0
            [np.nan, 1, np.nan, 1],      # s1
            [np.nan, np.nan, 1, 1],      # s2
            [1, np.nan, np.nan, np.nan], # s3 
            [np.nan, 1, 1, np.nan],      # s4
            [1, np.nan, np.nan, 1],      # s5
            [np.nan, 1, np.nan, np.nan], # s6 
            [1, 1, np.nan, 1]]           # s7
            )
        self.theta = self.theta_0
        self.pi = self._softmax_cvt_theta_to_pi(self.theta)
        self.eta = 0.1

    def _cvt_theta_to_pi(self):
        m, n = self.theta.shape
        pi = np.zeros((m, n))
        for r in range(m):
            pi[r, : ] = self.theta[r, :] / np.nansum(self.theta[r, :])
        return np.nan_to_num(pi)

    def _softmax_cvt_theta_to_pi(self, beta=1.):
        m, n = self.theta.shape
        pi = np.zeros((m, n))
        exp_theta = np.exp(self.theta * beta)
        for r in range(m):
            pi[r, :] = exp_theta[r, :] / np.nansum(exp_theta[r, :])
        return np.nan_to_num(pi)
    
    def update_theta(self, s_a_history):
        # 策略更新
        breakpoint()
        T = len(s_a_history) - 1
        m, n = self.theta.shape
        delta_theta = self.theta.copy()
        for i in range(m):
            for j in range(n):
                if not (np.isnan(self.theta_0[i, j])):
                    sa_i = [sa for sa in s_a_history if sa[0] == i]     # 统计当前状态被访问几次
                    sa_ij = [sa for sa in s_a_history if (sa[0] == i and sa[1] == j)]   # 当前状态采取当前action的次数
                    N_i = len(sa_i)
                    N_ij = len(sa_ij)
                    delta_theta[i, j] = (N_ij - self.pi[i, j] * N_i) / T    # 如果当前状态动作pair访问的次数多，则delta为正数；
        self.theta = self.theta + self.eta * delta_theta    # 经常访问的状态动作对，概率会变大
        return self.theta
    
    def update_pi(self):
        self.pi = self._softmax_cvt_theta_to_pi()
        return self.pi

    def choose_action(self, state):
        action = np.random.choice(self.actions, p=self.pi[state, :])
        return action
    

env = MazeEnv()
agent = Agent()

while True:
    state = env.reset()
    s_a_history = [[state, np.nan]]
    done = False
    while not done:
        action = agent.choose_action(state)
        s_a_history[-1][1] = action
        state, reward, done, _ = env.step(action)
        s_a_history.append([state, np.nan])
    agent.update_theta(s_a_history)
    pi = agent.pi.copy()
    agent.update_pi()
    delta = np.sum(np.abs(agent.pi - pi))
    print(len(s_a_history), delta)
    if delta < 1e-3:
        break