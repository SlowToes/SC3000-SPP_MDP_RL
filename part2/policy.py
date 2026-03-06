from abc import abstractmethod
from typing import Tuple

type State = Tuple[int, int]

class Policy:
    @abstractmethod
    def select_action(self, state: State) -> str:
        """ Given a state, select an action according to the policy """
        pass


class DeterministicPolicy(Policy):
    @abstractmethod
    def update(self, state: State, action: str) -> None:
        """ Update the policy for the given state to select the given action """
        pass


# class StochasticPolicy(Policy):
#     @abstractmethod
#     def update(self, states: List[State], actions: List[str], rewards: List[float]) -> None:
#         """ Update the policy for the given state to select actions according to the given probabilities """
#         pass

#     @abstractmethod
#     def get_probability(self, state: State, action: str) -> float:
#         """ Return the probability of selecting action from state according to this policy """
#         pass