from collections import defaultdict
from typing import List, Tuple

from part2.task2.model_free_learner import ModelFreeLearner
from part2.task2.multi_armed_bandit import MultiArmedBandit
from part2.mdp import MDP
from part2.qtable import QTable

type State = Tuple[int, int]

class MonteCarloControl(ModelFreeLearner):
    def __init__(self, mdp: MDP, bandit: MultiArmedBandit, qfunction: QTable):
        self.mdp = mdp
        self.bandit = bandit
        self.qfunction = qfunction

    def execute(self, episodes: int = 20000, max_episode_length: int = 500):
        returns: dict[Tuple[State, str], List[float]] = defaultdict(list)
        gamma = self.mdp.get_discount_factor()
        episode_rewards = []

        for _ in range(episodes):
            episode: List[Tuple[State, str, float]] = []
            state = self.mdp.get_initial_state()

            for _ in range(max_episode_length):
                if self.mdp.is_terminal(state):
                    break
                actions = self.mdp.get_actions(state)
                action = self.bandit.select(state, actions, self.qfunction)
                next_state, reward, done = self.mdp.execute(state, action)
                episode.append((state, action, reward))
                state = next_state
                if done:
                    break

            visited = set()
            G = 0.0
            episode_reward = 0.0
            for t in range(len(episode) - 1, -1, -1):
                s_t, a_t, r_t = episode[t]
                G = gamma * G + r_t
                if (s_t, a_t) in visited:
                    continue
                visited.add((s_t, a_t))
                returns[(s_t, a_t)].append(G)
                mean_return = sum(returns[(s_t, a_t)]) / len(returns[(s_t, a_t)])
                self.qfunction.update_value(s_t, a_t, mean_return)

            for _, _, r in episode:
                episode_reward += r
            episode_rewards.append(episode_reward)

        return episode_rewards
