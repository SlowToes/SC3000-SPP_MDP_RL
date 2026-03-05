from typing import Tuple

from policy import DeterministicPolicy

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
#         self.policy_table = {}

#     def select_action(self, state: State) -> str:

#     def update(self, states: List[State], actions: List[str], rewards: List[float]) -> None:

#     def get_probability(self, state: State, action: str) -> float: