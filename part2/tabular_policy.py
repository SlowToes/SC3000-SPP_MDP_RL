# from collections import defaultdict
from typing import Tuple
# import random

from part2.policy import DeterministicPolicy
# from part2.qtable import QTable

type State = Tuple[int, int]

class TabularDeterministicPolicy(DeterministicPolicy):
    def __init__(self):
        self.policy_table = {}

    def select_action(self, state: State) -> str:
        return self.policy_table[state]

    def update(self, state: State, action: str) -> None:
        self.policy_table[state] = action


# class TabularStochasticPolicy(StochasticPolicy):
#     def __init__(self):
#         self.policy_table = defaultdict(dict)

#     def select_action(self, state: State) -> str:
#         actions_probs = self.policy_table[state]

#         actions = list(actions_probs.keys())
#         probabilities = list(actions_probs.values())

#         return random.choices(actions, probabilities)[0]

#     def update(self, state: State, actions: List[str], qfunction: QTable, epsilon: float):
#         if not actions:
#             return

#         best_action = qfunction.get_argmax_q_value(state, actions)
#         n = len(actions)

#         self.policy_table[state] = {}

#         for action in actions:
#             if action == best_action:
#                 self.policy_table[state][action] = 1 - epsilon + (epsilon / n)
#             else:
#                 self.policy_table[state][action] = epsilon / n

#     def get_probability(self, state: State, action: str) -> float:
#         return self.policy_table[state].get(action, 0.0)