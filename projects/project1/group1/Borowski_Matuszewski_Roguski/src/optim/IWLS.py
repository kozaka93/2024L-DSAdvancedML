import numpy as np

from src.optim.BaseOptimizer import BaseOptimizer


class IWLS(BaseOptimizer):

    def __init__(self, model, stop_condition, lambda_=1e-10, delta=1e-10):
        super().__init__(model, stop_condition)
        self.lambda_ = lambda_
        self.delta = delta

    def optimize(self, x, y):
        logliks = []
        accuracies = []
        while not self.stop_condition(model=self.model, x=x, y=y):
            p = self.model.predict_probs(x)
            w = np.maximum(p * (1 - p), self.delta)
            z = (x @ self.model.weights + ((1 / w) * (y - p))[:, np.newaxis].T).T
            b_new = np.linalg.inv(
                (1 - self.lambda_) * (x.T @ (x * w[:, np.newaxis]))
                + self.lambda_ * np.identity(self.model.weights.shape[0])) @ x.T @ (z * w[:, np.newaxis])
            self.model.weights = np.squeeze(b_new)
            loglik, accuracy = self.score(x, y)
            logliks.append(loglik)
            accuracies.append(accuracy)
        return self.stop_condition.best_model if hasattr(self.stop_condition, 'best_model') else self.model, logliks, accuracies
