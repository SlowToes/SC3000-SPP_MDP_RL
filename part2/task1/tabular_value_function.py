from collections import defaultdict
import this
from typing import Dict, Tuple

from value_function import ValueFunction

type State = Tuple[int, int]

class TabularValueFunction(ValueFunction):
    def __init__(self, default_value: float = 0.0) -> None:
        self.value_table: Dict[State, float] = defaultdict(lambda: default_value)

    def update(self, state: State, value: float) -> None:
        self.value_table[state] = value

    def merge(self, other_value_table: "TabularValueFunction") -> None:
        self.value_table.update(other_value_table.value_table)

    def get_value(self, state: State) -> float:
        return self.value_table[state]