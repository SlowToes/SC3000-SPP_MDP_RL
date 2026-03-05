import random
from typing import List, Tuple

type State = Tuple[int, int]

class QTable:
    def __init__(self, default_value=0.0):
        """ QTable stores action-values for (state, action) pairs """
        self.qtable = {}
        self.default_value = default_value

    def get_q(self, state: State, action: str) -> float:
        """Return Q(s,a) or default value if not seen"""
        return self.qtable.get((state, action), self.default_value)

    def update(self, state: State, action: str, value: float, alpha: float = None):
        """
        Update Q(s,a)
        - If alpha is None: replace with value (used in Policy Iteration)
        - If alpha is given: incremental update (used in MC or Q-learning)
        """
        if alpha is None:
            self.qtable[(state, action)] = value
        else:
            old_value = self.get_q(state, action)
            self.qtable[(state, action)] = old_value + alpha * (value - old_value)

    def get_max_q(self, state: State, actions: List[str]) -> float:
        """ Return max Q(s,a) over available actions """
        if not actions:
            return 0.0
        max_q = float("-inf")
        for action in actions:
            max_q = max(max_q, self.get_q(state, action))
        return max_q

    def get_argmax_q(self, state: State, actions: List[str]) -> str:
        """ Return an action that maximises Q(s,a), tie-broken uniformly """
        if not actions:
            return None
        best_q = float("-inf")
        best_actions = []
        for action in actions:
            q = self.get_q(state, action)
            if q > best_q:
                best_q = q
                best_actions = [action]
            elif q == best_q:
                best_actions.append(action)
        return random.choice(best_actions)