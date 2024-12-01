from mfglib.env import Environment
from mfglib.alg import MFOMO
from mfglib.metrics import exploitability_score
import matplotlib.pyplot as plt


# Environment
rock_paper_scissors_instance = Environment.crowd_motion()

# Run the MF-OMO algorithm with default hyperparameters and default tolerances and plot exploitability scores
solns, expls, runtimes = MFOMO().solve(rock_paper_scissors_instance, max_iter=300, verbose=True)

plt.semilogy(runtimes, exploitability_score(rock_paper_scissors_instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.show()()
