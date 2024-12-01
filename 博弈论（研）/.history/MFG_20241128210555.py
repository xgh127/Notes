from mfglib.env import Environment
from mfglib.alg import MFOMO
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt
import torch

# Define the environment
n = 5
m = 1

torch.manual_seed(0)
soft_max = torch.nn.Softmax(dim=-1)

r1 = 2 * m * torch.rand(n, n) - m  # M_1 for reward_fn
r2 = 2 * m * torch.rand(n, n) - m  # M_2 for reward_fn

p1 = 2 * m * torch.rand(n, n, n) - m  # M_1 for transition_fn
p2 = 2 * m * torch.rand(n, n, n) - m  # M_2 for transition_fn

user_defined_random_linear = Environment(
    T=4,
    S=(n,),
    A=(n,),
    mu0=torch.ones(n) / n,
    r_max=2 * m,
    reward_fn=lambda env, t, L_t: r1 @ L_t + r2,
    transition_fn=lambda env, t, L_t: soft_max(p1 @ L_t + p2),

)

# Environment
crowd_motion_instance = Environment.crowd_motion()

# Run the MF-OMO algorithm with default hyperparameters and default tolerances and plot exploitability scores
# solns, expls, runtimes = MFOMO().solve(crowd_motion_instance, max_iter=300, verbose=True)

plt.semilogy(runtimes, exploitability_score(crowd_motion_instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("crowd_motion Environment - MFOMO Algorithm")
plt.show()()
