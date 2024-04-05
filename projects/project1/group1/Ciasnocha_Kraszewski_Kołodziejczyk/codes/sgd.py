import numpy as np
from optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""

    def __init__(self, learning_rate: float = 1):
        self.learning_rate = learning_rate

    def update(self, X, y, weights, predictions):
        gradient = (X.T @ (predictions - y)) / y.size
        new_weights = weights - self.learning_rate * gradient
        return new_weights
