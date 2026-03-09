from abc import ABC, abstractmethod
from typing import Tuple

from part2.mdp import MDP
from part2.tabular_policy import TabularDeterministicPolicy

type State = Tuple[int, int]

class ValueFunction(ABC):
    @abstractmethod
    def update(self, state: State, value: float) -> None:
        """Set V(s) for one state."""
        pass

    @abstractmethod
    def merge(self, other_value_table: "ValueFunction") -> None:
        """Copy values from another ValueFunction into this one."""
        pass

    @abstractmethod
    def get_value(self, state: State) -> float:
        """Return the current estimate V(s) for a state."""
        pass

    def get_q_value(self, mdp: MDP, state: State, action: str) -> float:
        """Return the Q-value of action in state."""
        q_value = 0.0
        for (new_state, probability) in mdp.get_transitions(state, action):
            reward = mdp.get_reward(state, action, new_state)
            q_value += probability * (
                reward + (mdp.get_discount_factor() * self.get_value(new_state))
            )
            
        return q_value

    def extract_policy(self, mdp: MDP) -> TabularDeterministicPolicy:
        """Return a policy from this value function."""
        policy = TabularDeterministicPolicy()

        for state in mdp.get_states():
            max_q = float("-inf")
            for action in mdp.get_actions(state):
                q_value = self.get_q_value(mdp, state, action)

                # If this is the maximum Q-value so far,
                # set the policy for this state
                if q_value > max_q:
                    policy.update(state, action)
                    max_q = q_value

        return policy