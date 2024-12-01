import numpy as np
import matplotlib.pyplot as plt
from mfglib.env import Environment
from mfglib.alg import MFOMO
from mfglib.metrics import exploitability_score

# 自定义自动驾驶环境
class AutonomousDrivingEnvironment(Environment):
    def __init__(self, T, S, A, mu0, r_max):
        super().__init__(T=T, S=S, A=A, mu0=mu0, r_max=r_max)
        self.state_dim = S[0] * S[1]  # 假设状态空间是二维的
        self.action_dim = A[0] * A[1]  # 假设动作空间也是二维的

    def reward(self, state, action):
        # 定义奖励函数，例如，基于速度和能耗
        return -np.sum(np.square(state) + action))

    def transition(self, state, action):
        # 定义状态转移函数，例如，根据加速度和加速度更新位置
        next_state = state + action  # 简化的转移模型
        return next_state

# 定义环境参数
T = 100  # 时间范围的终点
num_positions = 100  # 位置的离散点数
num_velocities = 10  # 速度的离散点数
num_accelerations = 5  # 加速度的离散点数
num_steering_angles = 5  # 转向角度的离散点数
S = (num_positions, num_velocities)  # 状态空间
A = (num_accelerations, num_steering_angles)  # 动作空间
mu0 = np.ones(S[0] * S[1]) / (S[0] * S[1])  # 初始状态分布
r_max = 1  # 奖励的最大值

# 创建环境实例
env = AutonomousDrivingEnvironment(T, S, A, mu0, r_max)

# 运行MF-OMO算法
mf_omo = MFOMO()
solutions, runtimes = mf_omo.solve(env, max_iter=300, verbose=True)

# 计算可利用性分数
exploitability_scores = [exploitability_score(env, sol) for sol in solutions]

# 可视化结果
plt.semilogy(runtimes, exploitability_scores)
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("Autonomous Driving Environment - MFOMO Algorithm")
plt.show()