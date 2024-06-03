"""Module containing utility functions for mutual information based feature selection."""

import numpy as np
from sklearn.metrics import mutual_info_score as MI


def entropy(X: np.ndarray) -> float:
    """Entropy of a given variable.

    Arguments:
        X: Array with the variable.
    """
    _, counts = np.unique(X, return_counts=True)
    prob = counts / len(X)
    return -np.sum(prob * np.log2(prob))


def CMI(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> float:
    """Conditional mutual information between X and Y given Z.

    Arguments:
        X: Array with the first variable.
        Y: Array with the second variable.
        Z: Array with the conditioning variable."""
    cmi = 0
    for z in np.unique(Z):
        cmi += MI(X[Z == z], Y[Z == z]) * (len(Z[Z == z]) / len(Z))
    return cmi


def interaction_gain(X1: np.ndarray, X2: np.ndarray, Y: np.ndarray) -> float:
    """Interaction Gain between X1 and X2 given Y.

    Arguments:
        X1: Array with the first variable.
        X2: Array with the second variable.
        Y: Array with the conditioning variable."""
    return CMI(X1, Y, X2) - MI(X1, Y)


def NMI(X: np.ndarray, Y: np.ndarray) -> float:
    """Normalized mutual information between two variables.

    Arguments:
        X: Array with the first variable.
        Y: Array with the second variable."""
    return MI(X, Y) / min(entropy(X), entropy(Y))


def MI_battiti(X: np.ndarray, S: np.ndarray, Y: np.ndarray) -> float:
    """Conditional mutual information between X and Y given S proposed by Battiti.

    Arguments:
        X: Array with the first variable.
        S: Array with already selected variables.
        Y: Array with the target variable."""
    if S.shape[1] == 0:
        print(NMI(X, Y))
        return NMI(X, Y)
    beta = 1 / S.shape[1]
    sum_mi = 0
    for i in range(S.shape[1]):
        sum_mi += NMI(X, S[:, i])
    return NMI(X, Y) - beta * sum_mi


def check_for_stopping_rule(
    ind_cand: int, X: np.ndarray, Y: np.ndarray, S: np.ndarray
) -> bool:
    """Check if stopping rule is triggered.

    Arguments:
        ind_cand: Index of the candidate feature.
        X: Array with the features.
        Y: Array with the target variable.
        S: Array with the already selected features."""
    if len(S) == 0:
        return False
    if MI_battiti(X[:, ind_cand], X[:, np.array(S).astype(int)], Y) < 0.03:
        return True
    return False
