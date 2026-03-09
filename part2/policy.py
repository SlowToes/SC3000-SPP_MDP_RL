from abc import ABC, abstractmethod
from typing import Tuple

type State = Tuple[int, int]

class Policy(ABC):
    @abstractmethod
    def select_action(self, state: State) -> str:
        """Given a state, select an action according to the policy."""
        pass


class DeterministicPolicy(Policy):
    @abstractmethod
    def update(self, state: State, action: str) -> None:
        """Update the policy for the given state to select the given action."""
        pass