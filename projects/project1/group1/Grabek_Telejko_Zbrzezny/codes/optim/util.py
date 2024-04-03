"""
util.py

Utility functions for optimization algorithms.
"""

import numpy as np


def calc_pi(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Calculation of odds."""
    exp = np.exp(X @ beta)
    return exp / (1 + exp)


# pylint: disable=invalid-name
def log_likelihood(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Log likelihood function."""
    return -np.sum(X @ beta * y - np.log(1 + np.exp(X @ beta)))


# pylint: disable=invalid-name
def make_batches(
    X: np.ndarray, y: np.ndarray, batch_size: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Function creates batches for gradient descent algorithm."""
    perm = np.random.permutation(len(y))
    X_perm = X[perm, :]  # pylint: disable=invalid-name
    y_perm = y[perm]
    return (
        np.array_split(X_perm, int(X_perm.shape[0] / batch_size)),
        np.array_split(y_perm, int(len(y_perm) / batch_size)),
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.

    Arguments:
    z : Input scalar or batch of scalars

    Returns:
    activation : Sigmoid activation(s) on z
    """
    activation = 1 / (1 + np.exp(-z))
    return activation


def logistic_loss(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Logistic loss function for binary classification.

    Arguments:
    preds : Predicted values
    targets : Target values

    Returns :
    cost : The mean logistic loss value between preds and targets
    """
    # mean logistic loss
    eps = 1e-14
    y = targets
    y_hat = preds
    cost = np.mean(-y * np.log(y_hat + eps) - (1 - y) * np.log(1 - y_hat + eps))
    return cost


# pylint: disable=invalid-name
def dlogistic(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> float:
    """
    Gradient/derivative of the logistic loss.

    Arguments:
    X : Input data matrix
    Y : True target values
    W : The weights
    """
    y_pred = sigmoid(np.dot(W, X.T))
    J = X.T @ (np.expand_dims(y_pred, 1) - Y)
    J = np.mean(J, axis=1)
    return J
