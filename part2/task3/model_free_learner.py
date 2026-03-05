from abc import abstractmethod

class ModelFreeLearner:
    @abstractmethod
    def execute(self, episodes=2000):
        pass