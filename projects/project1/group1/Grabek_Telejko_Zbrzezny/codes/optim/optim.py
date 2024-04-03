"""
optim.py

Provides the abstract class for the optimizer interface.
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# pylint: disable=invalid-name
class Optimizer(ABC):
    """Defines the general optimizer interface
    for the logistic regression problem."""

    def __init__(self) -> None:
        self._loss_history: list[float] = []
        self._global_best_weights: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[list[float], np.ndarray]:
        """Optimize the LR problem.
        To be overridden by the derived classes.

        Returns:
            tuple[list[float], np.ndarray]: The loss history and the best weights
        """

    def predict(
        self, X: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculates the odds and predicts the binary class labels.

        Returns:
            np.ndarray: The predicted binary class labels
        """
        weights = self._global_best_weights if weights is None else weights
        exp = np.exp(X @ weights)
        odds = exp / (1 + exp)
        return 1 * (odds > 0.5)

    def reset(self) -> None:
        """Reset the optimizer's state."""
        self._loss_history = []
        self._global_best_weights = None

    @property
    def loss_history(self) -> list[float]:
        """Returns the loss history of the optimizer

        Returns:
            list[float]: The loss history of the optimizer
        """
        return copy.deepcopy(self._loss_history)

    @property
    def global_best_weights(self) -> Optional[np.ndarray]:
        """Returns the globally best weights of the optimizer

        Returns:
            np.ndarray: The globally best weights of the optimizer
        """
        return copy.deepcopy(self._global_best_weights)
