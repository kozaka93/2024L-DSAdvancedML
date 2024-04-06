import numpy as np
from optimizer import Optimizer


class LogisticRegression:
    def __init__(self):
        self.weights = None

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        z = X @ self.weights
        z = np.clip(z, -100, 100)  # Clipping to prevent numerical underflow
        return self.sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(
        self,
        X,
        y,
        optimizer: Optimizer,
        max_epochs: int = 500,
        tolerance: float = 0.001,
    ):
        self.weights = np.zeros(X.shape[1])
        weights_changes = []

        for _ in range(max_epochs):
            old_weights = np.copy(self.weights)
            new_weights = optimizer.update(X, y, self.weights, self.predict_proba(X))
            weights_change = np.linalg.norm(new_weights - old_weights)
            weights_changes.append(weights_change)  # Record the change

            self.weights = new_weights

            if weights_change < tolerance:
                break

        return weights_changes
