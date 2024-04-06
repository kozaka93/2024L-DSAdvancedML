from abc import ABC, abstractmethod


class Optimizer(ABC):
    """An abstract class for ML model optimizers"""

    @abstractmethod
    def update(self, X, y, weights, predictions):
        """Update the model's weights based on the given data and model"""
        raise NotImplementedError("Subclasses must implement this method")
