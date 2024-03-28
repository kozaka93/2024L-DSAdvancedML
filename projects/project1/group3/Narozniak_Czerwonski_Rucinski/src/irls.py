from __future__ import annotations
from typing import Optional

import numpy as np

from irls.utils import log_likelihood, sigmoid, add_features_interactions, add_intercept_fnc, \
    check_features_interaction_correctness


class LogisticRegression:
    def __init__(self, B_init: Optional[np.ndarray] = None, tol: float = 1e-4, max_iter: int = 1000,
                 add_intercept: bool = True, feat_interactions: Optional[np.ndarray] = None):
        self.B: Optional[np.ndarray] = B_init
        self.tol: float = tol
        self.max_iter: int = max_iter
        self.add_intercept = add_intercept
        self._feat_interactions = feat_interactions
        if self._feat_interactions is not None:
            check_features_interaction_correctness(feat_interactions)
        self._fitting = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        assert X.ndim == 2, f"The expected X.ndim is 2, but is {X.ndim}"
        if y.ndim == 1:
            y = np.expand_dims(y, -1)
        self._fitting = True
        X_ = X.copy()
        if self._feat_interactions is not None:
            X_ = add_features_interactions(self._feat_interactions, X_)
        if self.add_intercept:
            X_ = add_intercept_fnc(X_)
        if self.B is None:
            self._init_weights(X_.shape[1])
        iteration = 0
        ll = np.finfo(float).max
        ll_diff = np.finfo(float).max
        while ll_diff > self.tol and iteration <= self.max_iter:
            p = self.predict_proba(X_)
            W = np.diagflat(p * (1 - p))
            xb = X_ @ self.B
            W_inv = np.diagflat(1. / np.diagonal(W))
            response = xb + W_inv @ (y - p)
            xwx_inv = np.linalg.inv(X_.T @ W @ X_)
            self.B = xwx_inv @ X_.T @ W @ response
            new_ll = log_likelihood(y, self.predict_proba(X_))
            ll_diff = ll - new_ll
            ll = new_ll
            iteration += 1
        self._fitting = False
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2, f"The expected X.ndim is 2, but is {X.ndim}"
        X_ = X.copy()
        if self._feat_interactions is not None and not self._fitting:
            X_ = add_features_interactions(self._feat_interactions, X_)
        if self.add_intercept and not self._fitting:
            X_ = add_intercept_fnc(X_)
        py1 = sigmoid(X_, self.B)
        return py1

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        assert X.ndim == 2, f"The expected X.ndim is 2, but is {X.ndim}"
        py1 = self.predict_proba(X)
        y_pred = np.where(py1 > threshold, 1, 0)
        return y_pred

    def get_params(self) -> np.ndarray:
        return self.B

    def _init_weights(self, shape):
        self.B = np.zeros(shape=shape, dtype=np.float32).reshape(-1, 1)