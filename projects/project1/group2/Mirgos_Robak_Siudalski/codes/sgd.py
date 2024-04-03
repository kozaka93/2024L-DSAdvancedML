import numpy as np
from logRegClf import sigmoid
from optimizer import Optimizer

class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

    def initialize(self, dims):
        pass

    def update(self, weights, X, y):

        for i in range(len(X)):
            xi = X[i]
            yi = y[i]
            prediction = sigmoid(np.dot(xi, weights))
            gradient = np.dot(xi.T, (prediction - yi))
            weights -= self.learning_rate * gradient
        return weights



