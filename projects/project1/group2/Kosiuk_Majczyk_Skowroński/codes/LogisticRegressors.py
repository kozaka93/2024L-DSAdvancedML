from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from typing import List, Union


class LogisticRegressor(ABC):
    def __init__(self, intercept=True, include_interactions=False):
        self.include_interactions = include_interactions
        self.intercept = intercept
        self.beta = None

    @staticmethod
    def __log_likelihood(X, y, beta):
        return -np.mean(
            y * np.log(LogisticRegressor.__sigmoid(np.dot(X, beta)))
            + (1 - y) * np.log(1 - LogisticRegressor.__sigmoid(np.dot(X, beta)))
        )

    @staticmethod
    def __add_intercept(X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    @staticmethod
    def __add_interactions(X):
        num_features = X.shape[1]
        interactions = np.ones((X.shape[0], 0))

        for i in range(num_features):
            for j in range(i + 1, num_features):
                interaction_term = X[:, i] * X[:, j]
                interactions = np.hstack(
                    (interactions, interaction_term.reshape(-1, 1))
                )

        return np.hstack((X, interactions))

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __stopping_rule(costs, epoch, lookback=5, tol=1e-5, verbose=True):
        if len(costs) < lookback + 1:
            return False
        # print(costs[-lookback-1:-1], costs[-1])
        # print(np.abs(np.mean(costs[-lookback-1:-1])), costs[-1], np.abs(np.mean(costs[-lookback-1:-1]) - costs[-1]))
        if verbose:
            print(
                f"Epoch {epoch+1}/{len(costs)}, costs_lookback: {np.mean(costs[-lookback-1:-1])}, cost_current: {costs[-1]}, diff: {np.abs(np.mean(costs[-lookback-1:-1]) - costs[-1])}"
            )
        if np.mean(costs[-lookback - 1 : -1]) - costs[-1] > tol:
            return False
        if verbose:
            print(f"Converged after {epoch} iterations")
        return True

    @abstractmethod
    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        if type(X) == pd.DataFrame:
            X = np.array(X)
        if self.include_interactions:
            X = LogisticRegressor.__add_interactions(X)
        if self.intercept:
            X = LogisticRegressor.__add_intercept(X)
        return LogisticRegressor.__sigmoid(np.dot(X, self.beta))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class SGD(LogisticRegressor):
    def __init__(
        self,
        intercept=True,
        include_interactions=False,
        learning_rate=1e-4,
        epochs=500,
        batch_size=1,
    ):
        super().__init__(intercept, include_interactions)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.cost_iterations: List[float] = list()
        self.beta_iterations: List[np.ndarray] = list()
        self.name = "SGD"

    def fit(self, X, y, verbose=True):
        if type(X) == pd.DataFrame:
            X = np.array(X)
        
        if self.include_interactions:
            X = self._LogisticRegressor__add_interactions(X)
        
        if self.intercept:
            X = self._LogisticRegressor__add_intercept(X)

        y = np.array(y)

        self.num_features = X.shape[1]
        
        self.beta = np.zeros(self.num_features)

        self.cost_iterations.append(
            self._LogisticRegressor__log_likelihood(X, y, self.beta)
        )

        self.beta_iterations.append(self.beta)

        for epoch in range(self.epochs):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]

                predictions = self._LogisticRegressor__sigmoid(
                    np.dot(X_batch, self.beta)
                )
                error = y_batch - predictions

                gradient = np.dot(X_batch.T, error)
                self.beta += self.learning_rate * gradient / self.batch_size


            cost = self._LogisticRegressor__log_likelihood(X, y, self.beta)
            self.cost_iterations.append(cost)

            if self._LogisticRegressor__stopping_rule(
                self.cost_iterations, epoch=epoch, verbose=verbose
            ):
                break
            previous_cost = cost
            self.beta_iterations.append(self.beta)
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Cost: {cost}, beta: {self.beta}")
        self.beta = self.beta_iterations[-1]


class ADAM(LogisticRegressor):
    def __init__(
        self,
        intercept: bool = True,
        include_interactions: bool = False,
        learning_rate: float = 1e-4,
        epochs: int = 500,
        batch_size: int = 1,
        beta_m: float = 0.9,
        beta_v: float = 0.999,
        epsilon: float = 1e-8,
        verbose: bool = False,
    ):
        super().__init__(intercept, include_interactions)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.epsilon = epsilon
        self.verbose = verbose
        self.cost_iterations: List[float] = list()
        self.beta_iterations: List[np.ndarray] = list()
        self.name = "ADAM"

    def fit(self, X: Union[np.array, pd.DataFrame], y: np.array, verbose: bool = True):
        if type(X) == pd.DataFrame:
            X = np.array(X)
        if self.include_interactions:
            X = self._LogisticRegressor__add_interactions(X)
        
        if self.intercept:
            X = self._LogisticRegressor__add_intercept(X)

        y = np.array(y)
        self.num_features = X.shape[1]
        self.beta = np.zeros(self.num_features)
  
        m = np.zeros(self.num_features)
        v = np.zeros(self.num_features)

        self.cost_iterations.append(
            self._LogisticRegressor__log_likelihood(X, y, self.beta)
        )

        self.beta_iterations.append(self.beta)

        for epoch in range(self.epochs):
            
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            delta = np.zeros(self.num_features)
            for i in range(0, len(X), self.batch_size):

                X_batch = X[i : i + self.batch_size]
                y_batch = y[i : i + self.batch_size]

                predictions = self._LogisticRegressor__sigmoid(
                    np.dot(X_batch, self.beta)
                )
                error = y_batch - predictions

                gradient = np.dot(X_batch.T, error)

                m = self.beta_m * m + (1 - self.beta_m) * gradient
                v = self.beta_v * v + (1 - self.beta_v) * gradient**2

                m_ = m / (1 - self.beta_m ** (epoch + 1))
                v_ = v / (1 - self.beta_v ** (epoch + 1))

                self.beta += self.learning_rate * m_ / (np.sqrt(v_) + self.epsilon)

            # print(self.beta, X.shape, y.shape)
            cost = self._LogisticRegressor__log_likelihood(X, y, self.beta)

            self.cost_iterations.append(cost)

            if self._LogisticRegressor__stopping_rule(
                self.cost_iterations, epoch=epoch, verbose=verbose
            ):
                break
            self.beta_iterations.append(self.beta)

            if verbose:
                print(f"Epoch {epoch+1}/{self.epochs}, Cost: {cost}, beta: {self.beta}")
        self.beta = self.beta_iterations[-1]


class IWLS(LogisticRegressor):

    def __init__(self, intercept=True, include_interactions=False, epochs=500):
        super().__init__(intercept, include_interactions)
        self.num_features: int = None
        self.beta: np.ndarray = None
        self.epochs = epochs

        self.cost_iterations: List[float] = list()
        self.beta_iterations: List[np.ndarray] = list()
        self.verbose: bool = False
        self.name = "IWLS"

    def fit(self, X, y, regularisation_coef=0.1, verbose: bool = True):
        if type(X) == pd.DataFrame:
            X = np.array(X)
        if self.include_interactions:
            X = self._LogisticRegressor__add_interactions(X)
        if self.intercept:
            X = self._LogisticRegressor__add_intercept(X)
        n, self.num_features = X.shape

        # coef initialisation
        self.beta = np.zeros(self.num_features)
        odds = np.ones(n) * 0.5

        self.cost_iterations.append(
            self._LogisticRegressor__log_likelihood(X, y, self.beta)
        )
        self.beta_iterations.append(self.beta)

        for epoch in range(self.epochs):
            weights = np.diag(np.sqrt(odds * (1 - odds)) + regularisation_coef)

            self.beta = (
                np.linalg.inv(X.T @ weights @ weights @ X)
                @ X.T
                @ weights
                @ (
                    (X.T @ weights).T @ self.beta + np.linalg.inv(weights) @ (y - odds)
                )  # z
            )
            odds = 1 / (1 + np.exp(-X @ self.beta))
            cost = self._LogisticRegressor__log_likelihood(X, y, self.beta)

            self.cost_iterations.append(cost)

            

            if self._LogisticRegressor__stopping_rule(
                self.cost_iterations, epoch=epoch, verbose=verbose
            ):
                break
            self.beta_iterations.append(self.beta)

        self.beta = self.beta_iterations[-1]
