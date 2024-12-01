from mfglib.env import Environment
from mfglib.alg import MFOMO
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt
import torch

# 设置随机种子以确保结果可复现
torch.manual_seed(0)

# 定义环境参数
n = 5  # 状态空间的维度，例如位置和速度
m = 1  # 一个标量，用于调整奖励和转移概率的尺度

# 定义奖励函数中的参数
alpha_C = 0.81  # 车辆密度相关的成本权重

# 定义奖励函数中的M1和M2，这些参数影响车辆的奖励
r1 = 2 * m * torch.rand(n, n) - m  # 与车辆状态相关的奖励参数
r2 = 2 * m * torch.rand(n, n) - m  # 与车辆动作相关的奖励参数

# 定义状态转移概率中的M1和M2，这些参数影响车辆状态的转移
p1 = 2 * m * torch.rand(n, n, n) - m  # 状态转移概率的一部分
p2 = 2 * m * torch.rand(n, n, n) - m  # 状态转移概率的另一部分

# 创建自定义环境实例
class AutonomousDrivingEnvironment(Environment):
    def __init__(self, T, S, A, mu0, r_max):
        super().__init__(T=T, S=S, A=A, mu0=mu0, r_max=r_max)
        self.state_dim = S[0] * S[1]  # 假设状态空间是二维的
        self.action_dim = A[0] * A[1]  # 假设动作空间也是二维的

    def reward(self, state, action):
        # 根据状态和动作计算奖励
        return -torch.sum(state['velocity']**2) - alpha_C * torch.sum(action['acceleration']**2)

    def transition(self, state, action):
        # 根据状态和动作更新状态
        next_state = {
            'position': state['position'] + state['velocity'] + 0.5 * action['acceleration'],
            'velocity': state['velocity'] + action['acceleration']
        }
        return next_state

# 创建环境实例
env = AutonomousDrivingEnvironment(T=4, S=(n,), A=(n,), mu0=torch.ones(n) / n, r_max=2 * m，)

# 使用MF-OMO算法求解环境
mf_omo = MFOMO()
solutions, exploitabilities, runtimes = mf_omo.solve(env, max_iter=300, verbose=True)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.semilogy(runtimes, exploitabilities, label='Exploitability Score')
plt.grid(True)
plt.xlabel('Runtime (seconds)')
plt.ylabel('Exploitability')
plt.title('Autonomous Driving Environment - MFOMO Algorithm')
plt.legend()
plt.show()