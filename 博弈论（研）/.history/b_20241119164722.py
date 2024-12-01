from scipy.stats import binom

n = 10000  # 试验次数
p = 0.15   # 成功概率
N = 0      # 成功次数的初始值
cumulative_prob = binom.cdf(N, n, p)  # 计算累积概率

# 当累积概率小于0.90时，继续增加N的值
while cumulative_prob < 0.90:
    N += 1
    cumulative_prob = binom.cdf(N, n, p)

print(N)