import numpy as np

# 定义空间和时间步长
dx = 0.1
dt = 0.01
x_max = 10
t_max = 1

# 初始化空间和时间网格
x = np.arange(0, x_max, dx)
t = np.arange(0, t_max, dt)
X, T = np.meshgrid(x, t)

# 初始化价值函数V
V = np.zeros((len(t), len(x)))

# 定义Hamiltonian H
def H(x, v, theta, V_x, V_v, u_a, u_theta):
    # 这里只是一个示例，你需要根据你的模型定义Hamiltonian
    return L(x, v, theta, u_a, u_theta) + V_x * v + V_v * u_a

# 定义拉格朗日函数 L
def L(x, v, theta, u_a, u_theta):
    # 这里只是一个示例，你需要根据你的模型定义拉格朗日函数
    return 0.5 * v**2 + 0.1 * u_a**2

# 定义边界条件
V[0, :] = 0  # 初始时间的价值函数
V[-1, :] = (x - x_max)**2  # 终端时间的价值函数

# 向后时间步进求解HJB方程
for n in range(1, len(t)):
    for i in range(1, len(x) - 1):
        V[n, i] = min(V[n-1, i] + dt * H(x[i], V[n, i], 0, (V[n, i+1] - V[n, i-1]) / (2 * dx), 0, 0, 0),  # u_a = 0
                      V[n-1, i+1] + dt * H(x[i], V[n, i], 0, (V[n, i+1] - V[n, i-1]) / (2 * dx), 1, 0))  # u_a = 1

# 可视化结果
import matplotlib.pyplot as plt

plt.contourf(T, X, V, 20, cmap='viridis')
plt.colorbar(label='V(x, t)')
plt.xlabel('Time')
plt.ylabel('Space')
plt.title('Value Function V(x, t)')
plt.show()