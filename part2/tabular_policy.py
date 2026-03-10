from typing import Tuple

from part2.policy import DeterministicPolicy

type State = Tuple[int, int]

class TabularDeterministicPolicy(DeterministicPolicy):
    def __init__(self):
        self.policy_table = {}

    def select_action(self, state: State) -> str:
        return self.policy_table[state]

    def update(self, state: State, action: str) -> None:
        self.policy_table[state] = action