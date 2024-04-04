import numpy as np
from base_model import BaseModel


class LogisticRegressionIWLS(BaseModel):
    def __init__(self, max_iter=500, tol=1e-3):
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
        iteration = 0
        for iteration in range(self.max_iter):
            z = np.dot(X, self.beta)
            p = self.sigmoid(z)
            W = np.diag(p * (1 - p))
            gradient = np.dot(X.T, p - y)
            self.log_likelihood.append(self.calculate_log_likelihood(X, y))
            hessian = np.dot(X.T, np.dot(W, X))
            update = np.linalg.solve(hessian, gradient)
            self.beta -= update
            if (
                np.linalg.norm(update) < self.tol
            ):  # stopping rule based on convergence detection by the change in model parameters
                print(f"IWLS stopped early - {iteration} iterations")
                break
        if iteration == self.max_iter - 1:
            print("IWLS did not converge")

    def predict(self, X, threshold=0.5):
        if self.interactions:
            X = self.create_interactions(X)
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
