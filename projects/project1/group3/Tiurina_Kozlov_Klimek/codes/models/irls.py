from typing import Tuple, Optional, List

import numpy as np

from .util import ClassMapper


class IRLS:
    _p: int
    _iter_limit: int
    _delta: float
    _tol: float
    _weights: Optional[np.ndarray]
    _beta: Optional[np.ndarray]
    _n_iter: int
    _prev_loss: float
    _loss_history: List[float]
    _mapper: ClassMapper

    def __init__(self, p: int = 2, iter_limit: int = 500, delta: float = 1E-4, tol: float = 1E-6):
        self._p = p
        self._iter_limit = iter_limit
        self._delta = delta
        self._tol = tol

        self._weights = None
        self._beta = None
        self._n_iter = 0
        self._prev_loss = 1E9
        self._loss_history: List[float] = [self._prev_loss for _ in range(0)]
        self._mapper = ClassMapper([-1, 1])

    def _update_beta(self, X, y):
        m1 = np.linalg.multi_dot([X.T, self._weights, X])
        m2 = np.linalg.multi_dot([X.T, self._weights, y.reshape((y.shape[0], 1))])

        # Fix for singular matrix
        if np.linalg.matrix_rank(m1) < m1.shape[0]:
            m1 += np.eye(m1.shape[0]) * 1E-4

        inv_m1 = np.linalg.inv(m1)
        self._beta = np.dot(inv_m1, m2).flatten()

    def _update_weights(self, X, y):
        y_pred = self.predict_proba(X, prepare=False)
        residuals = np.abs(y - y_pred)
        if self._p == 1:
            residuals[residuals < self._delta] = self._delta
        weights_diag = np.power(residuals, self._p - 2)
        self._weights = np.diag(weights_diag)

    def _nll_loss(self, y_true, y_pred):
        # Stabilize the log computation
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Compute the Negative Log Likelihood loss and normalize by the number of samples
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)
        return loss

    def _iteration(self, X, y):
        if self._n_iter >= self._iter_limit:
            raise StopIteration()

        # print(f"Running iteration {self._n_iter}")
        self._update_beta(X, y)
        self._update_weights(X, y)
        self._n_iter += 1

        # Compute the loss
        y_pred = self.predict_proba(X, prepare=False)
        loss = self._nll_loss(y, y_pred)

        self._prev_loss = loss
        self._loss_history.append(loss)

        # Check stop condition based on the change in NLL loss
        if self._n_iter > 10:
            differences = []
            last_loss = self._loss_history[-10:] 
            for i in range(1, len(last_loss)):
                differences.append(np.abs(last_loss[i] - last_loss[i-1]))
            if np.mean(differences) < self._tol:
                raise StopIteration()



    def _prepare_x(self, X):
        ones = np.ones(X.shape[0]).reshape((X.shape[0], 1))
        return np.concatenate([ones, X], 1)

    def fit(self, X, y, interactions: List[Tuple[int, int]] = None):
        yy = self._mapper.map_to_target(y)
        classes = list(np.unique(yy))
        if len(classes) != 2:
            raise ValueError("y is not a binary vector")

        if self._weights is not None or self._beta is not None or self._n_iter != 0:
            raise ValueError("Model already fitted or corrupted")

        if interactions is not None and len(interactions) > 0:
            inter_cols = []
            for v1, v2 in interactions:
                inter_col = (X[:, v1] * X[:, v2]).reshape((X.shape[0], 1))
                inter_cols.append(inter_col)
            Xint = np.concatenate(inter_cols, axis=1)
            X = np.concatenate([X, Xint], axis=1)

        if len(yy.shape) != 1:
            yy = yy.flatten()

        X = self._prepare_x(X)
        nrow = X.shape[0]
        self._weights = np.diag(np.ones(nrow))

        while True:
            try:
                self._iteration(X, yy)
            except StopIteration:
                break

    def log_odds(self, X, prepare=True):
        if self._weights is None or self._beta is None:
            raise ValueError("Start with fitting the model")
        if prepare:
            X = self._prepare_x(X)
        return np.dot(X, self._beta)

    def predict_proba(self, X, prepare=True):
        log_odds = self.log_odds(X, prepare)
        return 1 / (1 + np.exp(-log_odds))

    def predict(self, X, interactions: List[Tuple[int, int]] = None):
        if interactions is not None and len(interactions) > 0:
            inter_cols = []
            for v1, v2 in interactions:
                inter_col = (X[:, v1] * X[:, v2]).reshape((X.shape[0], 1))
                inter_cols.append(inter_col)
            Xint = np.concatenate(inter_cols, axis=1)
            X = np.concatenate([X, Xint], axis=1)

        probs = self.predict_proba(X)
        probs[probs < 0.5] = -1
        probs[probs >= 0.5] = 1
        return self._mapper.map_from_target(probs)

    def get_params(self):
        params = {
            "p-regularization": self._p,
            "iteration_limit": self._iter_limit,
            "weights": self._weights,
            "coefficients": self._beta,
            "iterations_run": self._n_iter,
            "loss_history": self._loss_history
        }

        if self._p == 1:
            params["delta"] = self._delta
        return params
