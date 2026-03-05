from abc import abstractmethod
from itertools import count
from typing import Tuple

from model_free_learner import ModelFreeLearner
from task2.multi_armed_bandit import MultiArmedBandit
from mdp import MDP
from qtable import QTable

type State = Tuple[int, int]

class TemporalDifferenceLearner(ModelFreeLearner):
    def __init__(self, mdp: type[MDP], bandit: type[MultiArmedBandit], qfunction: type[QTable]):
        self.mdp = mdp
        self.bandit = bandit
        self.qfunction = qfunction

    def execute(self, episodes: int = 2000, max_episode_length: float = float("inf")):
        episode_rewards = []
        for episode in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            episode_reward = 0.0
            for step in count():
                (next_state, reward, done) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.qfunction)
                delta = self.get_delta(reward, state, action, next_state, next_action, done)
                self.qfunction.update(state, action, delta)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.get_discount_factor() ** step)

                if done or step == max_episode_length:
                    break

            episode_rewards.append(episode_reward)

        return episode_rewards

    def get_delta(self, reward: float, state: State, action: str, next_state: State, next_action: str, done: bool) -> float:
        """ Calculate the delta for the update """
        q_value = self.qfunction.get_q_value(state, action)
        next_state_value = self.state_value(next_state, next_action)
        delta = (
            reward
            + (self.mdp.get_discount_factor() * next_state_value * (1 - done))
            - q_value
        )
        return delta

    @abstractmethod
    def state_value(self, state: State, action: str) -> float:
        """ Get the value of a state """
        pass