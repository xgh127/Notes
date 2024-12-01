import numpy as np
import matplotlib.pyplot as plt
from mfglib.env import Environment
from mfglib.alg import MFOMO
from mfglib.metrics import exploitability_score

class AutonomousDrivingEnvironment(Environment):
    def __init__(self, T, S, A, mu0, r_max):
        super().__init__(T=T, S=S, A=A, mu0=mu0, r_max=r_max)
        self.state_dim = S[0] * S[1]  # 假设状态空间是二维的
        self.action_dim = A[0] * A[1]  # 假设动作空间也是二维的

    def reward(self, state, action):
        return -np.sum(np.square(state['velocity'] - action['acceleration']))

    def transition(self, state, action):
        next_state = {
            'position': state['position'] + state['velocity'] * self.dt + 0.5 * action['acceleration'],
            'velocity': state['velocity'] + action['acceleration'] * self.dt
        }
        return next_state

# 定义环境参数
T = 10  # 时间范围的终点
num_positions = 100  # 位置的离散点数
num_velocities = 10  # 速度的离散点数
S = (num_positions, num_velocities)  # 状态空间
num_accelerations = 5  # 加速度的离散点数
A = (num_accelerations,)  # 动作空间
mu0 = np.ones(S[0] * S[1]) / (S[0] * S[1])  # 初始状态分布
r_max = 1  # 奖励的最大值
dt = 0.1  # 时间步长

# 创建环境实例
env = AutonomousDrivingEnvironment(T, S, A, mu0, r_max)