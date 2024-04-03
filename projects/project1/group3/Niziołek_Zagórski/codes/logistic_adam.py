import numpy as np

from base_model import BaseModel


class LogisticRegressionADAM(BaseModel):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        max_iter=500,
        tol=1e-3,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
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
        m = np.zeros_like(self.beta)
        v = np.zeros_like(self.beta)
        t = 0

        iteration = 0
        for iteration in range(self.max_iter):
            prev_beta = np.copy(self.beta)
            gradient = np.zeros_like(self.beta)
            for i in range(X.shape[0]):
                z = np.dot(X[i], self.beta)
                p = self.sigmoid(z)
                gradient += X[i] * (p - y[i])
            self.log_likelihood.append(self.calculate_log_likelihood(X, y))
            t += 1
            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * (gradient**2)
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            self.beta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            if (
                np.linalg.norm(self.beta - prev_beta) < self.tol
            ):  # stopping rule based on convergence detection by the change in model parameters
                print(f"Adam stopped early - {iteration} iterations")
                break
        if iteration == self.max_iter - 1:
            print("Adam did not converge")

    def predict(self, X, threshold=0.5):
        if self.interactions:
            X = self.create_interactions(X)
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
