from abc import ABC, abstractmethod

class ModelFreeLearner(ABC):
    @abstractmethod
    def execute(self, episodes=2000):
        """Execute the model-free learner for a given number of episodes."""
        pass