from part2.grid_world import GridWorld
from part2.qtable import QTable
from part2.task2.monte_carlo_control import MonteCarloControl
from part2.tabular_policy import TabularDeterministicPolicy
from part2.task2.epsilon_greedy import EpsilonGreedy


# Initialise the gridworld.
gridworld = GridWorld()

# Initialise the Q-function.
qfunction = QTable()

# Run the Monte Carlo control algorithm.
MonteCarloControl(gridworld, EpsilonGreedy(), qfunction).execute(episodes=10000)

# Visualise the Q-function.
gridworld.visualise_q_function(qfunction)

# Build a deterministic greedy policy from learned Q-values for visualisation.
policy = TabularDeterministicPolicy()
for state in gridworld.get_states():
    actions = gridworld.get_actions(state)
    if not actions:
        continue
    best_action = qfunction.get_argmax_q_value(state, actions)
    policy.update(state, best_action)

# Visualise the policy.
gridworld.visualise_policy(policy)
