"""Module for recursive feature selector."""

import numpy as np
from sklearn.model_selection import train_test_split

from src.feature_selector import BaseFeatureSelector
from src.settings import SEED
from src.train import calculate_score


class SequentialFeatureSelector(BaseFeatureSelector):
    """Sequential feature selector."""

    def __init__(self, estimator, n_features) -> None:
        """Initialize the feature selector instance."""
        self.estimator = estimator
        self.n_features = n_features
        self.support_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the feature selector model to the data.

        Arguments:
            X: Array with training features.
            y: Array with training target variable.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, shuffle=True
        )
        selected_features = []
        for n in range(1, self.n_features + 1):
            best_score = -np.inf
            for i in range(X.shape[1]):
                if i in selected_features:
                    continue
                X_train_selected = X_train[:, selected_features + [i]]
                X_test_selected = X_test[:, selected_features + [i]]
                score = calculate_score(
                    self.estimator(),
                    X_train_selected,
                    X_test_selected,
                    y_train,
                    y_test,
                    n,
                )
                if score > best_score:
                    best_score = score
                    best_feature = i
            selected_features.append(best_feature)
        self.support_ = np.array(selected_features)

    def get_support(self, indices: bool = True) -> np.ndarray:
        """
        Get indices of the chosen features after the fit.

        Arguments:
            indices: If True, the return value will be an array of integers, rather than a boolean mask.

        """
        if indices:
            return np.where(self.support_)[0]
        return self.support_
