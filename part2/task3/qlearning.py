from typing import Tuple

from part2.task3.temporal_difference_learner import TemporalDifferenceLearner

type State = Tuple[int, int]

class QLearning(TemporalDifferenceLearner):
    def state_value(self, state: State, action: str) -> float:
        max_q_value = self.qfunction.get_max_q_value(state, self.mdp.get_actions(state))
        return max_q_value