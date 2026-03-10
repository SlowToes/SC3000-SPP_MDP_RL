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
episode_returns = QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=10000)

# Visualise the Q-function.
gridworld.visualise_q_function(qfunction)

# Extract the policy from the Q-function.
policy = QPolicy(qfunction)

# Visualise the policy.
gridworld.visualise_policy(policy)