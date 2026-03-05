from mdp import MDP
from tabular_value_function import TabularValueFunction
from qtable import QTable


class ValueIteration:
    def __init__(self, mdp: type[MDP], values: type[TabularValueFunction]):
        self.mdp = mdp
        self.values = values

    def value_iteration(self, max_iterations: int = 1000, theta: float = 1e-6) -> int:
        for i in range(max_iterations):
            delta = 0.0
            new_values = TabularValueFunction()

            for state in self.mdp.get_states():
                actions = self.mdp.get_actions(state)
                if not actions:
                    new_values.update(state, 0)
                    continue

                qtable = QTable()
                for action in actions:
                    # Calculate the value of Q(s,a)
                    new_value = 0.0
                    for (new_state, probability) in self.mdp.get_transitions(state, action):
                        reward = self.mdp.get_reward(state, action, new_state)
                        new_value += probability * (
                            reward
                            + (
                                self.mdp.get_discount_factor()
                                * self.values.get_value(new_state)
                            )
                        )

                    qtable.update_value(state, action, new_value)

                # V(s) = max_a Q(sa)
                max_q = qtable.get_max_q_value(state, actions)
                delta = max(delta, abs(self.values.get_value(state) - max_q))
                new_values.update(state, max_q)

            self.values.merge(new_values)

            # Terminate if the value function has converged
            if delta < theta:
                return i + 1

        return max_iterations