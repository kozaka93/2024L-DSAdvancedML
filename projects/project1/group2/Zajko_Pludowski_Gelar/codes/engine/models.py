from itertools import combinations

import numpy as np


class LogisticRegression:

    def __init__(self) -> None:
        self.weights: np.ndarray = None
        self.use_interactions: bool = None

        self.eps = 1e-6

    def predict_proba(self, X: np.ndarray, transform_data: bool = True) -> np.ndarray:

        if transform_data:
            if self.use_interactions:
                combs = combinations([i for i in range(X.shape[1])], 2)
                for comb in combs:
                    col_1 = X[:, comb[0]]
                    col_2 = X[:, comb[1]]
                    X = np.append(X, np.expand_dims(col_1 * col_2, 1), 1)

            X = np.append(X, np.ones((X.shape[0], 1)), 1)

        return 1 / (1 + np.exp(-(X @ self.weights + self.eps)))

    def predict(self, X: np.ndarray, transform_data: bool = True) -> np.ndarray:
        y_proba = self.predict_proba(X, transform_data)
        return (y_proba > 0.5).astype(int)
