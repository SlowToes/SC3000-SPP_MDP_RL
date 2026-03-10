from abc import ABC, abstractmethod
from typing import List

class ModelFreeLearner(ABC):
    @abstractmethod
    def execute(self, episodes: int) -> List[float]:
        """Execute the model-free learner for a given number of episodes."""
        pass