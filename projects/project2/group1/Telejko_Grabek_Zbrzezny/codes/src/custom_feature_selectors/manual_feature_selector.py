"""Module for manual feature selector."""

import numpy as np

from src.feature_selector import BaseFeatureSelector


class ManualFeatureSelector(BaseFeatureSelector):
    """Manual feature selector."""

    def __init__(self, indices) -> None:
        """Initialize the feature selector instance."""
        self.n_features = len(indices)
        self.indices = indices

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the feature selector model to the data.

        Arguments:
            X: Array with training features.
            y: Array with training target variable.
        """
        pass

    def get_support(self, indices: bool = True) -> np.ndarray:
        """
        Get indices of the chosen features after the fit.

        Arguments:
            indices: If True, the return value will be an array of integers, rather than a boolean mask.

        """
        if indices:
            return self.indices
        return np.array(
            [True if i in self.indices else False for i in range(self.n_features)]
        )
