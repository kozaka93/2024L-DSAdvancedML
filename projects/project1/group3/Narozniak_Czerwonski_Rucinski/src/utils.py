from __future__ import annotations

import numpy as np


def neg_log_likelihood(y, y_prob):
    """Computes negative logarithmic likelihood."""
    ll = log_likelihood(y, y_prob)
    return -ll


def log_likelihood(y, y_prob, add_numerical_stability=False, eps=1e-15):
    n = y.shape[0]
    prob_y_eq_1 = y_prob
    prob_y_eq_0 = 1 - y_prob
    if add_numerical_stability:
        prob_y_eq_1 = np.clip(y_prob, eps, 1 - eps)
        prob_y_eq_0 = np.clip(1 - y_prob, eps, 1 - eps)
    loss = np.log(prob_y_eq_1).T @ y - np.log(prob_y_eq_0).T @ (1 - y)
    # alternative formulation when with X, weights and y exist
    # loss = y.T @ X @ self.B - np.sum(np.log(np.ones(shape(X.shape[0], 1) + np.exp(X @ self.B))
    return loss / n


def sigmoid(X, W):
    """Calculates sigmoid from the matrix multiplication of X and W"""
    return 1 / (1 + np.exp(-X @ W))


def add_features_interactions(_feat_interactions, X: np.ndarray) -> np.ndarray:
    # add feature interactions before adding intercept
    # WARNING: it works on the original data
    new_features = []
    for row in _feat_interactions:
        new_feature = X[:, int(row[0])] * X[:, int(row[1])]
        new_features.append(new_feature)
    new_features = np.array(new_features).T
    X = np.concatenate([X, new_features], axis=1)
    return X


def add_intercept_fnc(X):
    # WARNING: it works on the original data
    X = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
    return X


def check_features_interaction_correctness(feat_interactions):
    if feat_interactions.ndim != 2:
        raise ValueError(
            f"The feat_interactions matrix should be 2 dimensional. "
            f"Instead {feat_interactions.ndim} was given.")
    if feat_interactions.shape[1] != 2:
        raise ValueError(
            f"The feat_interactions matrix should should specify interactions between "
            f"two variable therefore have shape[1] == 2. Instead {feat_interactions.shape[1]} was given")