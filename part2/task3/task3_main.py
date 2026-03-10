import matplotlib.pyplot as plt

from part2.grid_world import GridWorld
from part2.qtable import QTable
from part2.task2.epsilon_greedy import EpsilonGreedy
from part2.task3.qlearning import QLearning
from part2.task3.q_policy import QPolicy


# Initialise the gridworld.
gridworld = GridWorld()

# Initialise the Q-function.
qfunction = QTable()

# Run the Q-learning algorithm.
# Use more episodes so convergence is observable in a stochastic environment.
episode_returns = QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=5000)

# Visualise the Q-function.
gridworld.visualise_q_function(qfunction)

# Extract the policy from the Q-function.
policy = QPolicy(qfunction)

# Visualise the policy.
gridworld.visualise_policy(policy)

# =========================================================================================
# Convergence analysis plots
# =========================================================================================

episodes = list(range(1, len(episode_returns) + 1))

# Convergence speed: Episodes vs total reward per episode.
plt.figure(figsize=(10, 5))
plt.plot(episodes, episode_returns, color="tab:blue", linewidth=1.2)
plt.title("Episode vs Total Reward per Episode (Q-Learning)")
plt.xlabel("Episode")
plt.ylabel("Total Reward per Episode")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Efficiency of learning: Moving average of episode rewards.
window_size = 25
moving_average = []
for i in range(len(episode_returns)):
    start = max(0, i - window_size + 1)
    window = episode_returns[start : i + 1]
    moving_average.append(sum(window) / len(window))

plt.figure(figsize=(10, 5))
plt.plot(episodes, moving_average, color="tab:orange", linewidth=2.0)
plt.title(f"Moving Average Reward (window={window_size})")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Episodes to convergence
# Convergence criterion (offline):
# - Compute a target from final performance.
# - Find the earliest episode where moving average stays within a tolerance
#   band around that target for a sustained number of episodes.
tolerance_band = 0.5
patience = 50
converged_at = None

tail_window_size = min(100, len(moving_average))
final_target = sum(moving_average[-tail_window_size:]) / tail_window_size

for start in range(0, len(moving_average) - patience + 1):
    segment = moving_average[start : start + patience]
    if all(abs(value - final_target) <= tolerance_band for value in segment):
        converged_at = start + 1  # Convert to 1-based episode index.
        break

if converged_at is None:
    print(
        "Convergence not detected within the training horizon. "
        "Try increasing episodes or widening tolerance_band."
    )
else:
    print(
        f"Estimated episodes to convergence: {converged_at} "
        f"(target={final_target:.2f}, band=±{tolerance_band})"
    )

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, moving_average, color="tab:orange", linewidth=2.0, label="Moving average reward")
    plt.axvline(converged_at, color="tab:red", linestyle="--", label=f"Converged at episode {converged_at}")
    plt.title("Convergence Detection from Moving Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()