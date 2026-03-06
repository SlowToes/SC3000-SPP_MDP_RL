from collections import defaultdict
import random
from typing import List, Tuple

from part2.qfunction import QFunction

type State = Tuple[int, int]

class QTable(QFunction):
    def __init__(self, alpha: float = 0.1, default_q_value: float = 0.0):
        self.qtable = defaultdict(lambda: default_q_value)
        self.alpha = alpha

    def update_value(self, state: State, action: str, value: float) -> None:
        self.qtable[(state, action)] = value

    def update(self, state: State, action: str, delta: float) -> None:
        self.qtable[(state, action)] = self.qtable[(state, action)] + self.alpha * delta

    def batch_update(self, states: List[State], actions: List[str], deltas: List[float]) -> None:
        for state, action, delta in zip(states, actions, deltas):
            self.update(state, action, delta)

    def get_q_value(self, state: State, action: str) -> float:
        return self.qtable[(state, action)]

    def get_q_values(self, states: List[State], actions: List[str]) -> List[float]:
        return [self.get_q_value(state, action) for state, action in zip(states, actions)]

    def get_max_q_value(self, state: State, actions: List[str]) -> float:
        if not actions:
            return 0.0
        max_q = float("-inf")
        for action in actions:
            max_q = max(max_q, self.get_q_value(state, action))
        return max_q

    def get_argmax_q_value(self, state: State, actions: List[str]) -> str:
        if not actions:
            return None
        best_q = float("-inf")
        best_actions = []
        for action in actions:
            q = self.get_q_value(state, action)
            if q > best_q:
                best_q = q
                best_actions = [action]
            elif q == best_q:
                best_actions.append(action)
        return random.choice(best_actions)