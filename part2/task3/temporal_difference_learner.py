from abc import abstractmethod
from itertools import count

from model_free_learner import ModelFreeLearner


class TemporalDifferenceLearner(ModelFreeLearner):
    def __init__(self, mdp, qfunction, alpha=0.1):
        self.mdp = mdp
        self.qfunction = qfunction
        self.alpha = alpha

    def execute(self, episodes=2000, max_episode_length=float("inf")):
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
                delta = self.get_delta(
                    reward, state, action, next_state, next_action, done
                )
                updated_target = self.qfunction.get_q_value(state, action) + delta
                self.qfunction.update(state, action, updated_target, alpha=self.alpha)

                state = next_state
                action = next_action
                episode_reward += reward * (self.mdp.get_discount_factor() ** step)

                if done or step == max_episode_length:
                    break

            episode_rewards.append(episode_reward)

        return episode_rewards

    def get_delta(self, reward, state, action, next_state, next_action, done):
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
    def state_value(self, state, action):
        """ Get the value of a state """
        pass