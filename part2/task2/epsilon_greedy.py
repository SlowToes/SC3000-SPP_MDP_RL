import random
from typing import List, Optional, Tuple

from task2.multi_armed_bandit import MultiArmedBandit

from qtable import QTable

type State = Tuple[int, int]

class EpsilonGreedy(MultiArmedBandit):
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def reset(self) -> None:
        pass

    def select(self, state: State, actions: List[str], qfunction: type[QTable]) -> Optional[str]:
        if not actions:
            return None

        # Select a random action with epsilon probability
        if random.random() < self.epsilon:
            return random.choice(actions)
        arg_max_q = qfunction.get_argmax_q_value(state, actions)
        return arg_max_q