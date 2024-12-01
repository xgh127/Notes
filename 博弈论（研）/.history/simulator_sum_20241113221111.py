import numpy as np
import matplotlib.pyplot as plt

# 参数设置
alpha_T = 1.0  # 时间成本权重
alpha_E = 0.1  # 能耗成本权重
alpha_C = 0.81  # 车辆密度权重
x_max = 10.0   # 位置的最大值
v_max = 5.0    # 速度的最大值
t_max = 10.0   # 时间的最大值
Nx = 100       # 位置的离散点数
Nv = 50        # 速度的离散点数
Nt = 100       # 时间的离散点数

# 时间序列生成
t = np.linspace(0, t_max, Nt)

# 网格生成
x = np.linspace(0, x_max, Nx)
v = np.linspace(0, v_max, Nv)
X, V = np.meshgrid(x, v)

# 时间步长和空间步长
dx = x[1] - x[0]
dv = v[1] - v[0]
dt = (t[-1] - t[0]) / (Nt - 1)

# 初始化价值函数V
V_original = np.zeros((Nt, Nx, Nv))
V_adjusted = np.zeros((Nt, Nx, Nv))

# 边界条件
V_original[:, :, -1] = (x - x_max)**2  # 终端时间的价值函数
V_adjusted[:, :, -1] = (x - x_max)**2  # 终端时间的价值函数

# 假设 m(x, t) 是通过某种方式计算得到的宏观分布函数
def m(x, t):
    return np.maximum(0, 1 - (x - 5)**2)  # 假设在 x=5 附近车辆密度最高

# 向后时间步进求解HJB方程
for n in range(Nt-2, -1, -1):
    for i in range(1, Nx-1):
        for j in range(1, Nv-1):
            # 计算梯度
            Vx = (V_original[n, i+1, j] - V_original[n, i-1, j]) / (2 * dx)
            Vv = (V_original[n, i, j+1] - V_original[n, i, j-1]) / (2 * dv)
            
            # 计算原始Hamiltonian
            H_original = lambda u_a: alpha_T / v[j] + alpha_E * u_a**2 + Vx * v[j] + Vv * u_a
            
            # 寻找最小化原始Hamiltonian的加速度u_a
            u_a_opt = 0  # 在这个简化模型中，我们假设最优加速度为0
            V_original[n, i, j] = V_original[n+1, i, j] + dt * H_original(u_a_opt)
            
            # 计算调整后的Hamiltonian
            H_adjusted = lambda u_a: alpha_T / v[j] + alpha_E * u_a**2 + alpha_C * m(x[i], t[n]) + Vx * v[j] + Vv * u_a
            
            # 寻找最小化调整后的Hamiltonian的加速度u_a
            V_adjusted[n, i, j] = V_adjusted[n+1, i, j] + dt * H_adjusted(u_a_opt)

# 可视化结果
plt.figure(figsize=(12, 6))

# 原始模型的等高线图
plt.subplot(1, 2, 1)
plt.contourf(X, V_original[0], levels=20, cmap='viridis')
plt.colorbar(label='V(x, v, t=0) - Original Model')
plt.xlabel('Position x')
plt.ylabel('Velocity v')
plt.title('Initial Value Function V(x, v, t=0) - Original Model')

# 调整后模型的等高线图
plt.subplot(1, 2, 2)
plt.contourf(X, V_adjusted[0], levels=20, cmap='viridis')
plt.colorbar(label='V(x, v, t=0) - Adjusted Model')
plt.xlabel('Position x')
plt.ylabel('Velocity v')
plt.title('Initial Value Function V(x, v, t=0) - Adjusted Model')

plt.tight_layout()
plt.show()