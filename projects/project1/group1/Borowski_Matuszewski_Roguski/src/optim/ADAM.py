import numpy as np

from src.optim.BaseOptimizer import BaseOptimizer


class ADAM(BaseOptimizer):
    def __init__(self, model, stop_condition, learning_rate=.01, beta1=.9, beta2=.999, epsilon=1e-8, batch_size=1):
        super().__init__(model, stop_condition)
        self.model = model
        self.stop_condition = stop_condition
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.m = 0
        self.v = 0
        self.t = 0

    def optimize(self, x, y):
        logliks = []
        accuracies = []
        while not self.stop_condition(model=self.model, x=x, y=y):
            permutation = np.random.permutation(len(y))
            for ix in range(0, len(y), self.batch_size):
                sample = permutation[ix:ix + self.batch_size]
                probs = self.model.predict_probs(x[sample, :])
                gradient = np.mean(np.mean(probs - y[sample]) * x[sample], axis=0)
                self.t += 1
                self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
                self.v = self.beta2 * self.v + (1 - self.beta2) * gradient ** 2
                m_hat = self.m / (1 - self.beta1 ** self.t)
                v_hat = self.v / (1 - self.beta2 ** self.t)
                self.model.weights = self.model.weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            loglik, accuracy = self.score(x, y)
            logliks.append(loglik)
            accuracies.append(accuracy)
        return self.stop_condition.best_model if hasattr(self.stop_condition, 'best_model') else self.model, logliks, accuracies
