from typing import Tuple

from part2.mdp import MDP
from part2.tabular_policy import TabularDeterministicPolicy
from part2.task1.tabular_value_function import TabularValueFunction


class PolicyIteration:
    def __init__(self, mdp: MDP, policy: TabularDeterministicPolicy):
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy: TabularDeterministicPolicy, values: TabularValueFunction, theta: float = 1e-6) -> TabularValueFunction:
        """Evaluate a fixed policy until the value function converges."""
        while True:
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                # Calculate the value of V(s)
                actions = self.mdp.get_actions(state)
                if not actions:
                    new_values.update(state, 0)
                    continue
                old_value = values.get_value(state)
                new_value = values.get_q_value(self.mdp, state, policy.select_action(state))
                new_values.update(state, new_value)
                delta = max(delta, abs(old_value - new_value))

            values.merge(new_values)

            # Terminate if the value function has converged
            if delta < theta:
                break

        return values

    def policy_iteration(self, max_iterations: int = 1000, theta: float = 1e-6) -> Tuple[int, TabularValueFunction]:
        """Run policy iteration and return (iterations, converged_values)."""
        values = TabularValueFunction()

        # Initialise all non-terminal states with a default valid action.
        for state in self.mdp.get_states():
            actions = self.mdp.get_actions(state)
            if not actions:
                continue
            if state not in self.policy.policy_table:
                self.policy.update(state, actions[0])

        for i in range(1, max_iterations + 1):
            policy_changed = False
            values = self.policy_evaluation(self.policy, values, theta)

            for state in self.mdp.get_states():
                actions = self.mdp.get_actions(state)
                if not actions:
                    continue
                old_action = self.policy.select_action(state)

                best_action = old_action
                best_q = float("-inf")
                for action in actions:
                    # Calculate the value of Q(s,a)
                    q_value = values.get_q_value(self.mdp, state, action)
                    if q_value > best_q:
                        best_q = q_value
                        best_action = action
                # V(s) = argmax_a Q(s,a)
                new_action = best_action
                self.policy.update(state, new_action)
                policy_changed = True if new_action != old_action else policy_changed

            if not policy_changed:
                return i, values

        return max_iterations, values