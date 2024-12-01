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
AutonomousDrivingEnvironment = Environment(
    T=4,  # 时间范围的终点
    S=(n,),  # 状态空间形状
    A=(n,),  # 动作空间形状
    mu0=torch.ones(n) / n,  # 初始状态分布
    r_max=2 * m,  # 奖励的上限
    reward_fn=lambda env, t, L_t: r1 @ L_t + r2,  # 奖励函数
    transition_fn=lambda env, t, L_t: torch.nn.Softmax(dim=-1)(p1 @ L_t + p2)  # 状态转移函数
)

# 使用MF-OMO算法求解环境
solns, expls, runtimes = MFOMO().solve(AutonomousDrivingEnvironment, max_iter=300, verbose=True)

# 可视化可利用性分数
plt.semilogy(runtimes, exploitability_score(user_defined_random_linear, solns))
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("Autonomous Driving Environment - MFOMO Algorithm")
plt.show()