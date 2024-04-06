import numpy as np
from optimizer import Optimizer


class ADAM(Optimizer):
    """Adaptive Moment Estimation optimizer"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, X, y, weights, predictions):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        gradient = (X.T @ (predictions - y)) / y.size
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(gradient)

        m_corr = self.m / (1 - self.beta1**self.t)
        v_corr = self.v / (1 - self.beta2**self.t)

        new_weights = weights - self.learning_rate * m_corr / (
            np.sqrt(v_corr) + self.epsilon
        )
        return new_weights
