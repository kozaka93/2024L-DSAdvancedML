from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score


class BaseOptimizer(metaclass=ABCMeta):
    def __init__(self, model, stop_condition, **kwargs):
        self.model = model
        self.stop_condition = stop_condition

    @abstractmethod
    def optimize(self, x, y):
        pass

    def score(self, x, y):
        prediction = np.clip(self.model.predict_probs(x), 1e-10, 1 - 1e-10)
        loglik = - np.mean(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
        accuracy = accuracy_score(y, self.model.predict(x))
        return loglik, accuracy


