from itertools import combinations
from optimizer import Optimizer
import numpy as np
from logRegClf import sigmoid

class IWLSOptimizer(Optimizer):
    def __init__(self, learning_rate=1, epsilon=1e-5):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def initialize(self, dims):
        pass

    def update(self, weights, X, y):
        predictions = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (predictions - y))

        W = np.diag(predictions*(1-predictions))
        H = np.dot(np.dot(X.T, W), X) + self.epsilon * np.eye(X.shape[1])
        
        weights -= self.learning_rate * np.dot(np.linalg.inv(H), gradient)
        return weights
