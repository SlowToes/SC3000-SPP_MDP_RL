from part2.grid_world import GridWorld
from part2.qtable import QTable
from part2.task2.monte_carlo_control import MonteCarloControl
from part2.tabular_policy import TabularDeterministicPolicy
from part2.task2.epsilon_greedy import EpsilonGreedy


# GridWorld noise is defaulted to 0.1
gridworld = GridWorld()
qfunction = QTable()
MonteCarloControl(gridworld, EpsilonGreedy(), qfunction).execute(episodes=50000)
gridworld.visualise_q_function(qfunction)

# Build a deterministic greedy policy from learned Q-values for visualisation.
policy = TabularDeterministicPolicy()
for state in gridworld.get_states():
    actions = gridworld.get_actions(state)
    if not actions:
        continue
    best_action = qfunction.get_argmax_q_value(state, actions)
    policy.update(state, best_action)

gridworld.visualise_policy(policy)
