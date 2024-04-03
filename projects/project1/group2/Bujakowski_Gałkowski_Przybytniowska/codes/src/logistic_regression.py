from typing import Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.optimization_algorithms import IRLS, SGD, AdamOptim

optimizers: Dict[str, Union[SGD, AdamOptim, IRLS]] = {
    "sgd": SGD,
    "adam": AdamOptim,
    "irls": IRLS,
}


class LogisticRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iter: int = 500,
        tolerance: float = 1e-4,
        add_interactions: bool = False,
        optimizer: str = "adam",
        batch_size: Union[None, int] = None,
    ):
        self.max_iter: int = max_iter
        self.tolerance: float = tolerance
        self.add_interactions: bool = add_interactions
        self.optimizer: str = optimizer
        self.weights: Union[None, np.ndarray] = None
        self.batch_size: Union[None, int] = batch_size
        self.optimizer = optimizers[optimizer](learning_rate)
        self.history = []

    def _add_interactions(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        _, n_features = X.shape

        interactions = []
        if self.add_interactions:
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    interaction = X[:, i] * X[:, j]
                    interactions.append(interaction.reshape(-1, 1))
            X = np.hstack((X, *interactions))
        return X

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _log_likelihood(self, y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / len(y)

    def _cross_entropy(self, y: np.ndarray, p: np.ndarray) -> float:
        return -self._log_likelihood(y, p)

    def _optimize(self, X: np.ndarray, y: np.ndarray) -> None:
        if isinstance(self.optimizer, SGD):
            batch_size = 1
        else:
            batch_size = self.batch_size if self.batch_size else len(X)

        for _ in tqdm(range(self.max_iter), desc="Optimizing"):
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            old_weights = np.copy(self.weights)

            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i : i + batch_size]
                batch_y = y_shuffled[i : i + batch_size]
                batch_dw_sum = np.zeros_like(self.weights)

                probabilities = self._sigmoid(np.dot(batch_X, self.weights))

                errors = probabilities - batch_y

                batch_dw_sum = np.dot(batch_X.T, errors)

                self.weights = self.optimizer.update(
                    self.weights,
                    batch_dw_sum / len(batch_X),
                    batch_X,
                    probabilities,
                )

            probabilities = self._sigmoid(np.dot(X, self.weights))
            self.history.append(self._log_likelihood(y, probabilities))

            if np.linalg.norm(self.weights - old_weights) < self.tolerance:
                print("Stopping criteria reached after ", _ + 1, " iterations")
                break
        return

    def get_log_likelihood(self):
        return self.history

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray) -> None:
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        if self.add_interactions:
            X = self._add_interactions(X)
        X = np.insert(X, 0, 1, axis=1)

        self.weights = np.zeros(X.shape[1])
        self._optimize(X, y)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        if self.weights is None:
            raise ValueError("Fit the model before prediction")

        if self.add_interactions:
            X = self._add_interactions(X)
        X = np.insert(X, 0, 1, axis=1)

        z = np.dot(X, self.weights)
        probabilities = self._sigmoid(z)
        predictions = np.where(probabilities >= 0.5, 1, 0)
        return predictions
