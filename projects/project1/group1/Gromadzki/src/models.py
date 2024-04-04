import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

class LogisticRegressionIWLS:
    def __init__(self, max_iter=100, tol=1e-4, interactions=False):
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.interactions = interactions

    def _add_interactions(self, X):
        X = pd.DataFrame(X)
        m, n = X.shape
        interaction_terms = []
        for i in range(n):
            for j in range(i + 1, n):
                interaction_terms.append(X.values[:, i] * X.values[:, j])
        return np.column_stack((X.values, *interaction_terms))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_weights(self, X, coef):
        y_pred = self._sigmoid(X.dot(coef))
        return np.diag(y_pred * (1 - y_pred))

    def fit(self, X, y):
        if self.interactions:
            X = self._add_interactions(X)
            y = y.values
        else:
            X, y = X.values, y.values
        n_samples, n_features = X.shape
        intercept = np.ones((n_samples, 1))
        X = np.concatenate((intercept, X), axis=1)
        self.coef_ = np.zeros(n_features + 1)
        prev_coef = np.inf
        history = []

        for it in range(self.max_iter):
            if self.interactions:
                self.interactions = False
                history.append(log_loss(y, self.predict_proba(X, add_intercept=False)))
                self.interactions = True
            else:
                history.append(log_loss(y, self.predict_proba(X, add_intercept=False)))
            weights = self._compute_weights(X, self.coef_)

            gradient = X.T.dot(y - self._sigmoid(X.dot(self.coef_)))
            hessian = X.T.dot(weights).dot(X)

            prev_coef = np.copy(self.coef_)
            self.coef_ += np.linalg.inv(hessian).dot(gradient)

            if np.linalg.norm(self.coef_ - prev_coef) < self.tol:
                break
        return history, it + 1

    def predict_proba(self, X, add_intercept=True):
        if self.interactions:
            X = self._add_interactions(X)
        else:
            X = np.array(X)
        if add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept, X), axis=1)
        return self._sigmoid(X.dot(self.coef_))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, max_iter=100, tol=1e-4, interactions=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.interactions = interactions

    def _add_interactions(self, X):
        X = pd.DataFrame(X)
        m, n = X.shape
        interaction_terms = []
        for i in range(n):
            for j in range(i + 1, n):
                interaction_terms.append(X.values[:, i] * X.values[:, j])
        return np.column_stack((X.values, *interaction_terms))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        if self.interactions:
            X = self._add_interactions(X)
            y = y.values
        else:
            X, y = X.values, y.values
        n_samples, n_features = X.shape
        intercept = np.ones((n_samples, 1))
        X = np.concatenate((intercept, X), axis=1)
        self.coef_ = np.zeros(n_features + 1)
        history = []

        random_indices = np.random.permutation(n_samples)
        X = X[random_indices]
        y = y[random_indices]

        for it in range(self.max_iter):
            prev_coef = np.copy(self.coef_)
            if self.interactions:
                self.interactions = False
                history.append(log_loss(y, self.predict_proba(X, add_intercept=False)))
                self.interactions = True
            else:
                history.append(log_loss(y, self.predict_proba(X, add_intercept=False)))
            for i in range(n_samples):
                xi = X[i]
                yi = y[i]
                z = np.dot(xi, self.coef_)
                predicted = self._sigmoid(z)
                gradient = (predicted - yi) * xi

                self.coef_ -= self.learning_rate * gradient
            if np.linalg.norm(self.coef_ - prev_coef) < self.tol:
                break

        return history, it + 1

    def predict_proba(self, X, add_intercept=True):
        if self.interactions:
            X = self._add_interactions(X)
        else:
            X = np.array(X)
        if add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept, X), axis=1)
        return self._sigmoid(X.dot(self.coef_))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LogisticRegressionAdam:
    def __init__(
        self,
        learning_rate=0.01,
        max_iter=100,
        tol=1e-4,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        interactions=False,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.interactions = interactions
        self.m = None
        self.v = None
        self.t = 0

    def _add_interactions(self, X):
        X = pd.DataFrame(X)
        m, n = X.shape
        interaction_terms = []
        for i in range(n):
            for j in range(i + 1, n):
                interaction_terms.append(X.values[:, i] * X.values[:, j])
        return np.column_stack((X.values, *interaction_terms))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        if self.interactions:
            X = self._add_interactions(X)
            y = y.values
        else:
            X, y = X.values, y.values
        n_samples, n_features = X.shape
        intercept = np.ones((n_samples, 1))
        X = np.concatenate((intercept, X), axis=1)
        n_features += 1
        self.coef_ = np.zeros(n_features)
        self.m = np.zeros(n_features)
        self.v = np.zeros(n_features)
        prev_coef = np.inf
        history = []

        for it in range(self.max_iter):
            z = np.dot(X, self.coef_)
            y_pred = self._sigmoid(z)
            history.append(log_loss(y, y_pred))
            gradient = np.dot(X.T, (y_pred - y)) / n_samples

            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient**2)

            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)

            prev_coef = self.coef_.copy()
            self.coef_ -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            if np.linalg.norm(self.coef_ - prev_coef) < self.tol:
                break

        return history, it + 1

    def predict_proba(self, X, add_intercept=True):
        if self.interactions:
            X = self._add_interactions(X)
        else:
            X = np.array(X)
        if add_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept, X), axis=1)
        return self._sigmoid(X.dot(self.coef_))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
