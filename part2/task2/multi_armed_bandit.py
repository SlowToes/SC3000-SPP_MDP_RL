from abc import abstractmethod
from typing import List, Tuple

from part2.qtable import QTable

type State = Tuple[int, int]

class MultiArmedBandit():
    @abstractmethod
    def select(self, state: State, actions: List[str], qfunction: type[QTable]) -> str:
        """ Select an action for this state given from a list given a Q-function """
        pass

    @abstractmethod
    def reset(self) -> None:
        """ Reset a multi-armed bandit to its initial configuration """
        self.__init__()