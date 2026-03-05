from grid_world import GridWorld
from epsilon_greedy import EpsilonGreedy
from qtable import QTable
from qlearning import QLearning


def extract_greedy_outputs(env: GridWorld, qfunction: QTable):
    policy = {}
    state_values = {}
    for state in env.get_states():
        actions = env.get_actions(state)
        if not actions:
            state_values[state] = 0.0
            continue
        best_action = qfunction.get_argmax_q(state, actions)
        policy[state] = best_action
        state_values[state] = qfunction.get_q_value(state, best_action)
    return policy, state_values


gridworld = GridWorld()
qfunction = QTable()
QLearning(gridworld, EpsilonGreedy(epsilon=0.1), qfunction, alpha=0.1).execute(episodes=20000)
learned_policy, learned_values = extract_greedy_outputs(gridworld, qfunction)
gridworld.visualise_value_function(learned_values, "Task 3 - Q-Learning State Values (max_a Q(s,a))")
gridworld.visualise_policy(learned_policy, "Task 3 - Learned Policy (Q-Learning)")