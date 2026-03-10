from collections import defaultdict
from typing import Tuple

from part2.qtable import QTable
from part2.tabular_policy import TabularDeterministicPolicy

type State = Tuple[int, int]


class QPolicy(TabularDeterministicPolicy):
    def __init__(self, qfunction: QTable):
        super().__init__()

        state_actions = defaultdict(list)
        for state, action in qfunction.qtable.keys():
            state_actions[state].append(action)

        for state, actions in state_actions.items():
            best_action = qfunction.get_argmax_q_value(state, actions)
            if best_action is not None:
                self.update(state, best_action)