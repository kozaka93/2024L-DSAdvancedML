import warnings
from abc import ABC, abstractmethod
from itertools import combinations

import numpy as np
from tqdm import tqdm

from engine.models import LogisticRegression


class Optimizer(ABC):
    eps = 1e-6

    def __init__(
        self, stopping_n: int = 30, stopping_minimal_improvement: float = 0.1
    ) -> None:
        super().__init__()

        self.stopping_n = stopping_n
        self.stopping_minimal_improvement = stopping_minimal_improvement
        self.metric_history = np.zeros(shape=(0,))

    @abstractmethod
    def _optimize_body(
        self, model: LogisticRegression, X: np.ndarray, y: np.ndarray
    ) -> None:
        pass

    def _prepare_interaction_data(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()

        combs = combinations([i for i in range(X.shape[1])], 2)
        for comb in combs:
            col_1 = X[:, comb[0]]
            col_2 = X[:, comb[1]]
            X = np.append(X, np.expand_dims(col_1 * col_2, 1), 1)

        return X

    def _is_stopping_rule(self) -> bool:
        if len(self.metric_history) <= self.stopping_n:
            return False
        best_previous_loglik = self.metric_history[: -(self.stopping_n)].max(
            initial=-np.inf
        )
        last_logliks = self.metric_history[-self.stopping_n :]
        return (
            ~(
                (last_logliks - best_previous_loglik)
                > -best_previous_loglik * self.stopping_minimal_improvement
            )
        ).all()

    @classmethod
    def _init_weights(cls, model: LogisticRegression, size: int) -> None:
        rand_weights = np.random.normal(size=(size))
        model.weights = rand_weights

    def optimize(
        self,
        model: LogisticRegression,
        X: np.ndarray,
        y: np.ndarray,
        use_iteractions: bool = False,
        max_iter: int = 500,
    ) -> None:
        if use_iteractions:
            model.use_interactions = True
            X = self._prepare_interaction_data(X)

        X = np.append(X, np.ones((X.shape[0], 1)), 1)

        if model.weights is None:
            self._init_weights(model, X.shape[1])

        for _ in tqdm(range(max_iter)):
            permutation = np.random.permutation(X.shape[0])
            X, y = X[permutation], y[permutation]
            self._optimize_body(model, X, y)

            loglik = self.calc_loglikelihood(model, X, y)
            if (loglik > self.metric_history).all():
                best_weights = model.weights
            self.metric_history = np.append(self.metric_history, loglik)
            if self._is_stopping_rule():
                model.weights = best_weights
                break
        else:
            model.weights = best_weights
            warnings.warn("method did not coverege.")

    @classmethod
    def calc_loglikelihood(
        cls, model: LogisticRegression, X: np.ndarray, y: np.ndarray
    ) -> float:
        logits = X @ model.weights
        return (y * logits - np.log(1 + np.exp(logits))).sum()


class IWLSOptimizer(Optimizer):
    def __init__(
        self, stopping_n: int = 30, stopping_minimal_improvement: float = 0.1
    ) -> None:
        super().__init__(stopping_n, stopping_minimal_improvement)

    def _optimize_body(
        self, model: LogisticRegression, X: np.ndarray, y: np.ndarray
    ) -> None:
        weights = model.weights.reshape((-1, 1))
        y_hat = model.predict_proba(X, transform_data=False)
        W = np.identity(X.shape[0]) * 0.25
        W_inv = np.identity(X.shape[0]) * 4
        z = X @ weights + (W_inv @ (y - y_hat)).reshape((-1, 1))

        weights_new = (
            np.linalg.solve(X.T @ W @ X + 1e-10 * np.eye(X.shape[1]), X.T)
            @ W
            @ z
        )
        model.weights = np.ravel(weights_new)


class SGDOptimizer(Optimizer):
    def __init__(
        self,
        stopping_n: int = 30,
        learning_rate: float = 0.1,
        num_batches: int = 1,
        stopping_minimal_improvement: float = 0.1,
    ) -> None:
        super().__init__(stopping_n, stopping_minimal_improvement)
        self.learning_rate = learning_rate
        self.num_batches = num_batches

    def _optimize_body(
        self, model: LogisticRegression, X: np.ndarray, y: np.ndarray
    ) -> None:
        for X_batch, y_batch in zip(
            np.array_split(X, self.num_batches),
            np.array_split(y, self.num_batches),
        ):
            y_batch_hat = model.predict_proba(X_batch, transform_data=False)
            grad = (
                (y_batch - y_batch_hat)
                * y_batch_hat
                * (1 - y_batch_hat)
                @ X_batch
            )
            model.weights += grad * self.learning_rate


class ADAMOptimizer(SGDOptimizer):
    def __init__(
        self,
        stopping_n: int = 30,
        learning_rate: float = 0.1,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        num_batches: int = 1,
        stopping_minimal_improvement: float = 0.1,
    ) -> None:
        super().__init__(
            stopping_n,
            learning_rate,
            num_batches,
            stopping_minimal_improvement,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = 0
        self.s = 0
        self.step_num = 0

    def _optimize_body(
        self, model: LogisticRegression, X: np.ndarray, y: np.ndarray
    ) -> None:
        for X_batch, y_batch in zip(
            np.array_split(X, self.num_batches),
            np.array_split(y, self.num_batches),
        ):
            self.step_num += 1
            y_batch_hat = model.predict_proba(X_batch, transform_data=False)
            grad = (
                -(y_batch - y_batch_hat)
                * y_batch_hat
                * (1 - y_batch_hat)
                @ X_batch
            )
            self.m = self.beta_1 * self.m - (1 - self.beta_1) * grad
            self.s = self.beta_2 * self.s + (1 - self.beta_2) * grad**2
            m_hat = self.m / (1 - self.beta_1**self.step_num)
            s_hat = self.s / (1 - self.beta_2**self.step_num)
            model.weights += (
                self.learning_rate * m_hat / (np.sqrt(s_hat + self.eps))
            )
