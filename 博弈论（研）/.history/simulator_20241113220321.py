import numpy as np
import matplotlib.pyplot as plt

# 参数设置
alpha_T = 1.0  # 时间成本权重
alpha_E = 0.1  # 能耗成本权重
x_max = 10.0   # 位置的最大值
v_max = 5.0    # 速度的最大值
t_max = 10.0   # 时间的最大值
Nx = 100       # 位置的离散点数
Nv = 50        # 速度的离散点数
Nt = 100       # 时间的离散点数
alpha_C = 0.0  # 车辆密度权重

# 网格生成
x = np.linspace(0, x_max, Nx)
v = np.linspace(0, v_max, Nv)
X, V = np.meshgrid(x, v)

# 时间步长和空间步长
dx = x[1] - x[0]
dv = v[1] - v[0]
dt = t_max / (Nt - 1)

# 初始化价值函数V
V = np.zeros((Nt, Nx, Nv))

# 边界条件
V[:, :, -1] = (x - x_max)**2  # 终端时间的价值函数


# 假设 m(x, t) 是通过某种方式计算得到的宏观分布函数
# 这里我们使用一个简单的示例函数来代表车辆密度
def m(x, t):
    return np.maximum(0, 1 - (x - 5)**2)  # 假设在 x=5 附近车辆密度最高

# 向后时间步进求解HJB方程
# 在求解HJB方程的循环中添加新的成本项
for n in range(Nt-2, -1, -1):
    for i in range(1, Nx-1):
        for j in range(1, Nv-1):
            # 计算梯度
            Vx = (V[n, i+1, j] - V[n, i-1, j]) / (2 * dx)
            Vv = (V[n, i, j+1] - V[n, i, j-1]) / (2 * dv)
            
            # 计算新的Hamiltonian
            H = lambda u_a: alpha_T / v[j] + alpha_E * u_a**2 + alpha_C * m(x[i], t[n]) + Vx * v[j] + Vv * u_a
            
            # 寻找最小化Hamiltonian的加速度u_a
            u_a_opt = 0  # 在这个简化模型中，我们假设最优加速度为0
            V[n, i, j] = V[n+1, i, j] + dt * H(u_a_opt)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.contourf(X, V[0], levels=20, cmap='viridis')  # 使用X和V[0]，它们都是2D数组
plt.colorbar(label='V(x, v, t=0)')
plt.xlabel('Position x')
plt.ylabel('Velocity v')
plt.title('Initial Value Function V(x, v, t=0)')
plt.show()