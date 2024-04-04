import numpy as np


class LogisticRegression:
    def __init__(self):
        self.weights = None

    def predict_probs(self, x):
        if self.weights is None:
            self.weights = np.zeros(x.shape[1])
        return 1 / (1 + np.exp(-self.weights @ x.T))

    def predict(self, x):
        return (self.predict_probs(x) > 0.5).astype(int)
