from typing import Tuple, Optional, List
import numpy as np
import math

from .util import ClassMapper


def _sigmoid(z):
    if isinstance(z, float):
        if z < -100:
            return 0

        return 1 / (1 + math.exp(-z))
    else:
        exp_z = z.copy()
        exp_z[z < -100] = 0
        exp_z[z >= -100] = 1 / (1 + np.exp(-z[z >= -100]))
        return exp_z


class SGD:
    _iter_limit: int
    _learning_rate: float
    _tol: float
    _beta: Optional[np.ndarray]
    _n_iter: int
    _prev_loss: float
    _loss_history: List[float]
    _mapper: ClassMapper

    def __init__(self,
                 iter_limit: int = 500,
                 rate: float = 1E-4,
                 tol: float = 1E-6,
                 seed: int = 0):
        self._iter_limit = iter_limit
        self._learning_rate = rate
        self._tol = tol

        self._beta = None
        self._n_iter = 0
        self._prev_loss = 1E9
        self._loss_history: List[float] = [self._prev_loss for _ in range(0)]
        self._mapper = ClassMapper([-1, 1])

        np.random.seed(seed)

    def _gradient(self, x_sample, y_sample):
        pred = _sigmoid(np.dot(x_sample, self._beta))
        y_scaled = (y_sample + 1) / 2
        return (pred - y_scaled) * x_sample

    def _update_beta(self, x_sample, y_sample):
        grad = self._gradient(x_sample, y_sample)
        self._beta -= self._learning_rate * grad

    def _nll_loss(self, y_true, y_pred):
        # Stabilize the log computation
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)  # Ensure y_pred is within (0, 1)

        # Compute the Negative Log Likelihood loss and normalize by the number of samples
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)
        return loss

    def _iteration(self, X, y):
        if self._n_iter >= self._iter_limit:
            raise StopIteration()

        combined_data = np.concatenate([X, y.reshape((len(y), 1))], axis=1)
        np.random.shuffle(combined_data)

        # previous_beta = np.copy(self._beta)
        np.apply_along_axis(lambda r: self._update_beta(r[:-1], r[-1]), 1, combined_data)

        # Calculate current NLL loss
        y_pred = self.predict_proba(X, prepare=False)  # Use raw X since it's already prepared
        current_loss = self._nll_loss(y, y_pred)

        # update_factor = 1e-3 * ((self._beta - current_beta) / self._learning_rate)

        self._prev_loss = current_loss
        self._loss_history.append(current_loss)
        self._n_iter += 1

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

        if self._n_iter != 0:
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

        ncol = X.shape[1]
        self._beta = np.zeros(ncol)

        X_copy = np.copy(X)
        y_copy = np.copy(yy)

        while True:
            try:
                self._iteration(X_copy, y_copy)
            except StopIteration:
                break

    def log_odds(self, X, prepare=True):
        if self._beta is None:
            raise ValueError("Start with fitting the model")
        if prepare:
            X = self._prepare_x(X)
        return np.dot(X, self._beta)

    def predict_proba(self, X, prepare=True):
        log_odds = self.log_odds(X, prepare)
        return _sigmoid(log_odds)

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
            "learning_rate": self._learning_rate,
            "iteration_limit": self._iter_limit,
            "coefficients": self._beta,
            "iterations_run": self._n_iter,
            "tolerance": self._tol,
            "loss_history": self._loss_history
        }

        return params
