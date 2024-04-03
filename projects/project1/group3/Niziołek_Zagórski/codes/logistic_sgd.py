import numpy as np
from base_model import BaseModel


class LogisticRegressionSGD(BaseModel):
    def __init__(self, learning_rate=0.001, max_iter=500, tol=1e-3):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, interactions=False):
        if interactions:
            X = self.create_interactions(X)
            self.interactions = True
        else:
            self.interactions = False
        self.log_likelihood = []

        self.beta = np.zeros(X.shape[1])
        prev_beta = np.inf
        iteration = 0
        for iteration in range(self.max_iter):
            gradient = None
            for i in range(X.shape[0]):
                z = np.dot(X[i], self.beta)
                p = self.sigmoid(z)
                gradient = (p - y[i]) * X[i]
                self.beta -= self.learning_rate * gradient
            self.log_likelihood.append(self.calculate_log_likelihood(X, y))
            if (
                np.linalg.norm(self.beta - prev_beta) < self.tol
            ):  # stopping rule based on convergence detection by the change in model parameters
                print(f"SGD stopped early - {iteration} iterations")
                break
            prev_beta = self.beta.copy()
        if iteration == self.max_iter - 1:
            print("SGD did not converge")

    def predict(self, X, threshold=0.5):
        if self.interactions:
            X = self.create_interactions(X)
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
