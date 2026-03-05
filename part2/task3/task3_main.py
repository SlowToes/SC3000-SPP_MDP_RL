from pathlib import Path
import sys

# Ensure sibling modules under part2 are importable when this file is run directly
CURRENT_DIR = Path(__file__).resolve().parent
PART2_DIR = CURRENT_DIR.parent
if str(PART2_DIR) not in sys.path:
    sys.path.insert(0, str(PART2_DIR))

from grid_world import GridWorld
from qtable import QTable
from qlearning import QLearning
from tabular_policy import TabularStochasticPolicy
from task2.epsilon_greedy import EpsilonGreedy


gridworld = GridWorld()
qfunction = QTable()
QLearning(gridworld, EpsilonGreedy(), qfunction).execute(episodes=100)
gridworld.visualise_q_function(qfunction)

# Build a deterministic greedy policy from learned Q-values for visualisation.
policy = TabularStochasticPolicy()
for state in gridworld.get_states():
    actions = gridworld.get_actions(state)
    if not actions:
        continue
    policy.update(state, actions, qfunction, epsilon=0.1)

gridworld.visualise_policy(policy)