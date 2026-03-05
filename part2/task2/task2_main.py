# import random
# from collections import defaultdict
# from typing import Dict, List, Tuple

# from grid_world import GridWorld
# from qtable import QTable
# from tabular_policy import TabularDeterministicPolicy
# from tabular_value_function import TabularValueFunction
# from value_iteration import ValueIteration


# def _resolve_next_state(env: GridWorld, state: Tuple[int, int], action: str) -> Tuple[int, int]:
#     if state == env.get_goal_state():
#         return state

#     x, y = state
#     if action == env.UP:
#         candidate = (x + 1, y)
#     elif action == env.DOWN:
#         candidate = (x - 1, y)
#     elif action == env.RIGHT:
#         candidate = (x, y + 1)
#     else:
#         candidate = (x, y - 1)

#     if (not env.is_valid_state(candidate)) or (candidate in env.roadblocks):
#         return state
#     return candidate


# def step(env: GridWorld, state: Tuple[int, int], action: str) -> Tuple[Tuple[int, int], float, bool]:
#     """
#     Sample one stochastic environment step without planning over transitions.
#     """
#     if env.is_terminal(state):
#         return state, 0.0, True

#     if action in (env.UP, env.DOWN):
#         slip_actions = [env.LEFT, env.RIGHT]
#     else:
#         slip_actions = [env.UP, env.DOWN]

#     r = random.random()
#     if r < 0.8:
#         chosen_action = action
#     elif r < 0.9:
#         chosen_action = slip_actions[0]
#     else:
#         chosen_action = slip_actions[1]

#     next_state = _resolve_next_state(env, state, chosen_action)
#     reward = env.get_reward(state, chosen_action, next_state)
#     done = env.is_terminal(next_state)
#     return next_state, reward, done


# def epsilon_greedy_action(
#     env: GridWorld, q_values: QTable, state: Tuple[int, int], epsilon: float
# ) -> str:
#     actions = env.get_actions(state)
#     if random.random() < epsilon:
#         return random.choice(actions)

#     best_q = float("-inf")
#     best_actions: List[str] = []
#     for action in actions:
#         q = q_values.get_q(state, action)
#         if q > best_q:
#             best_q = q
#             best_actions = [action]
#         elif q == best_q:
#             best_actions.append(action)
#     return random.choice(best_actions)


# def monte_carlo_control(
#     env: GridWorld,
#     num_episodes: int = 20000,
#     epsilon: float = 0.1,
#     max_steps: int = 500,
# ) -> Tuple[QTable, TabularDeterministicPolicy, Dict[Tuple[int, int], float]]:
#     """
#     First-visit MC control with fixed epsilon-greedy policy improvement.
#     """
#     q_values = QTable(default_value=0.0)
#     returns: Dict[Tuple[Tuple[int, int], str], List[float]] = defaultdict(list)
#     gamma = env.get_discount_factor()

#     for _ in range(num_episodes):
#         episode: List[Tuple[Tuple[int, int], str, float]] = []
#         state = env.get_initial_state()

#         for _ in range(max_steps):
#             if env.is_terminal(state):
#                 break
#             action = epsilon_greedy_action(env, q_values, state, epsilon)
#             next_state, reward, done = step(env, state, action)
#             episode.append((state, action, reward))
#             state = next_state
#             if done:
#                 break

#         visited = set()
#         G = 0.0
#         for t in range(len(episode) - 1, -1, -1):
#             s_t, a_t, r_t = episode[t]
#             G = (gamma * G) + r_t
#             if (s_t, a_t) in visited:
#                 continue
#             visited.add((s_t, a_t))
#             returns[(s_t, a_t)].append(G)
#             q_values.update(s_t, a_t, sum(returns[(s_t, a_t)]) / len(returns[(s_t, a_t)]))

#     learned_policy = TabularDeterministicPolicy()
#     learned_values: Dict[Tuple[int, int], float] = {}

#     for state in env.get_states():
#         actions = env.get_actions(state)
#         if not actions:
#             learned_values[state] = 0.0
#             continue

#         best_action = actions[0]
#         best_q = q_values.get_q(state, best_action)
#         for action in actions[1:]:
#             q = q_values.get_q(state, action)
#             if q > best_q:
#                 best_q = q
#                 best_action = action
#         learned_policy.update(state, best_action)
#         learned_values[state] = best_q

#     return q_values, learned_policy, learned_values


# def compare_policies(
#     env: GridWorld,
#     p1: TabularDeterministicPolicy,
#     p2: TabularDeterministicPolicy,
#     label1: str,
#     label2: str,
# ) -> None:
#     comparable_states = [s for s in env.get_states() if env.get_actions(s)]
#     matches = sum(
#         1 for s in comparable_states if p1.policy_table.get(s) == p2.policy_table.get(s)
#     )
#     ratio = 100.0 * matches / len(comparable_states)
#     print(f"Policy agreement ({label1} vs {label2}): {matches}/{len(comparable_states)} ({ratio:.2f}%)")


# def main() -> None:
#     random.seed(42)

#     env = GridWorld()
#     # Task 2 uses the stochastic environment (default noise=0.1).
#     env.noise = 0.1

#     print("Training Monte Carlo control (Task 2)...")
#     _, mc_policy, mc_values = monte_carlo_control(
#         env,
#         num_episodes=20000,
#         epsilon=0.1,
#         max_steps=500,
#     )
#     env.visualise_value_function(mc_values, "Task 2 - MC State Values (max_a Q(s,a))")
#     env.visualise_policy(mc_policy, "Task 2 - Learned Policy (MC Control)")

#     # Compare against Task 1 optimal policy from model-based planning.
#     planning_env = GridWorld()
#     planning_env.noise = 0.0
#     optimal_values = TabularValueFunction()
#     ValueIteration(planning_env, optimal_values).value_iteration(max_iterations=100)
#     optimal_policy = optimal_values.extract_policy(planning_env)
#     compare_policies(env, optimal_policy, mc_policy, "Task 1 Optimal", "Task 2 MC")


# if __name__ == "__main__":
#     main()
