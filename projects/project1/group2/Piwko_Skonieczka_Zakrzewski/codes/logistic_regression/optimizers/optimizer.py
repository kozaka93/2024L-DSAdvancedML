from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms. Defines a blueprint for optimizers
    with a method to optimize parameters given training data.
    """

    @abstractmethod
    def optimize_parms(self, X: np.ndarray, y:np.ndarray, params:np.ndarray) -> np.ndarray:
        """
        Optimize parameters of a model given training data.

        Parameters:
        - X (np.ndarray): Feature data.
        - y (np.ndarray): Target labels.
        - params (np.ndarray): Initial parameters to optimize.

        Returns:
        - np.ndarray: Optimized parameters.
        """
        pass