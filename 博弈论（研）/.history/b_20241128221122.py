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

# 定义奖励函数中的M1和M2，这些参数影响车辆的奖励
r1 = 2 * m * torch.rand(n, n) - m  # 与车辆状态相关的奖励参数
r2 = 2 * m * torch.rand(n, n) - m  # 与车辆动作相关的奖励参数

# 定义状态转移概率中的M1和M2，这些参数影响车辆状态的转移
p1 = 2 * m * torch.rand(n, n, n) - m  # 状态转移概率的一部分
p2 = 2 * m * torch.rand(n, n, n) - m  # 状态转移概率的另一部分

# 创建自定义环境实例
class AutonomousDrivingEnvironment(Environment):
    def __init__(self, T, S, A, mu0, r_max, reward_fn, transition_fn):
        super().__init__(T=T, S=S, A=A, mu0=mu0, r_max=r_max, reward_fn=reward_fn, transition_fn=transition_fn)
        self.state_dim = S[0] * S[1]  # 假设状态空间是二维的

    def reward(self, state, action):
        # 根据状态和动作计算奖励，反映论文中的关键成本因素
        return -torch.sum(state['velocity']**2) - 0.81 * torch.sum(action['acceleration']**2)

    def transition(self, state, action):
        # 根据状态和动作更新状态，模拟车辆动态
        next_state = {
            'position': state['position'] + state['velocity'] + 0.5 * action['acceleration'],
            'velocity': state['velocity'] + action['acceleration']
        }
        return next_state

# 注意：这里假设 state 和 action 都是字典，包含 'position', 'velocity', 'acceleration' 等键

# 定义奖励函数和状态转移函数
def reward_function(state, action):
    return -torch.sum(state['velocity']**2) - 0.81 * torch.sum(action['acceleration']**2)

def transition_function(state, action):
    next_state = {
        'position': state['position'] + state['velocity'] + 0.5 * action['acceleration'],
        'velocity': state['velocity'] + action['acceleration']
    }
    return next_state

# 创建环境实例
S = (n, n)  # 状态空间形状
A = (n,)  # 动作空间形状
env = AutonomousDrivingEnvironment(
    T=4,  # 时间范围的终点
    S=S,  # 状态空间形状
    A=A,  # 动作空间形状
    mu0=torch.ones(n) / n,  # 初始状态分布
    r_max=2 * m,  # 奖励的上限
    reward_fn=reward_function,  # 奖励函数
    transition_fn=transition_function  # 状态转移函数
)

# 使用MF-OMO算法求解环境
solns, expls, runtimes = MFOMO().solve(env, max_iter=300, verbose=True)

# 可视化可利用性分数
plt.figure(figsize=(10, 6))
plt.semilogy(runtimes, exploitability_score(env, solns), label='Exploitability Score')
plt.grid(True)
plt.xlabel('Runtime (seconds)')
plt.ylabel('Exploitability')
plt.title('Autonomous Driving Environment - MFOMO Algorithm')
plt.legend()
plt.show()