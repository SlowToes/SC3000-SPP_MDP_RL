from abc import abstractmethod
from typing import List, Tuple

type State = Tuple[int, int]
type Transition = Tuple[State, float]

class MDP:
    @abstractmethod
    def get_states(self) -> List[State]:
        """ Return all states of this MDP """
        pass

    @abstractmethod
    def is_valid_state(self, state: State) -> bool:
        """ Return true if and only if state is a valid state of this MDP """
        pass

    @abstractmethod
    def get_actions(self, state: State) -> List[str]:
        """ Return all actions with non-zero probability from this state """
        pass

    @abstractmethod
    def get_transitions(self, state: State, action: str) -> List[Transition]:
        """ Return all non-zero probability transitions for this action
            from this state, as a list of (state, probability) pairs
        """
        pass

    @abstractmethod
    def get_reward(self, state: State, action: str, next_state: State) -> float:
        """ Return the reward for transitioning from state to
            nextState via action
        """
        pass

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """ Return true if and only if state is a terminal state of this MDP """
        pass

    @abstractmethod
    def get_discount_factor(self) -> float:
        """ Return the discount factor for this MDP """
        pass

    @abstractmethod
    def get_initial_state(self) -> State:
        """ Return the initial state of this MDP """
        pass

    @abstractmethod
    def get_goal_state(self) -> State:
        """ Return the goal state of this MDP """
        pass