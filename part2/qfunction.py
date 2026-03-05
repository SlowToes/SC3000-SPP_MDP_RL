from abc import abstractmethod
from typing import Tuple, List

type State = Tuple[int, int]

class QFunction:
    @abstractmethod
    def update(self, state: State, action: str, delta: float) -> None:
        """ Update Q(s, a) using: Q(s,a) ← Q(s,a) + α * delta """
        pass

    @abstractmethod
    def batch_update(self, states: List[State], actions: List[str], deltas: List[float]) -> None:
        """ Perform multiple updates at once for a list of (state, action, delta) pairs """
        pass

    @abstractmethod
    def get_q_value(self, state: State, action: str) -> float:
        """ Return the Q-value for a (state, action) pair """
        pass

    @abstractmethod
    def get_q_values(self, states: List[State], actions: List[str]) -> List[float]:
        """ Return a list of Q-values for a list of (state, action) pairs """
        pass

    @abstractmethod
    def get_max_q_value(self, state: State, actions: List[str]) -> float:
        """ Return the maximum Q-value over available actions """
        pass

    @abstractmethod
    def get_argmax_q_value(self, state: State, actions: List[str]) -> str:
        """ Return the action that maximises the Q-value over available actions """
        pass