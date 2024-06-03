"""Module for feature selector classes."""

from abc import ABC, abstractmethod

import numpy as np


class BaseFeatureSelector(ABC):
    """Abstract class for feature selectors."""

    def __init__(self) -> None:
        """Initialize the feature selector instance."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the feature selector model to the data.

        Arguments:
            X: Array with training features.
            y: Array with training target variable.
        """
        pass

    @abstractmethod
    def get_support(self, indices: bool = True) -> np.ndarray:
        """
        Get indices of the chosen features after the fit.

        Arguments:
            indices: If True, the return value will be an array of integers, rather than a boolean mask.

        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data with feature selector.

        Arguments:
            X: Array to transform.
        """
        return X[:, self.get_support()]
