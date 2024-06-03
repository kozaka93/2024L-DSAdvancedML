import numpy as np

from .constants import COST_OF_FEATURE, RETURN_ON_CORRECT_CLIENT


def cash_profit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_features: int,
    n_clients_to_select: int,
    scaling_factor: float,
) -> float:
    """Target function. Measure how well the model perform.

    Args:
        y_true (np.ndarray): true labels
        y_proba (np.ndarray): predicted labels
        n_features (int): number of used features
        n_clients_to_select (int): number of `1` to use from the model's prediction.
            `n_clients_to_select` clients with highest probability are used.
        scaling_factor (float): scaling factor. Used when the task is saller
            when to original one to preserve target function bounds.

    Returns:
        float: target funciton value
    """
    idx_high_prob = np.argsort(y_proba[:, 1])[-n_clients_to_select:]

    y_selected = y_true[idx_high_prob]
    reward = y_selected.sum() * RETURN_ON_CORRECT_CLIENT
    cost = n_features * COST_OF_FEATURE

    return reward * scaling_factor - cost


def precision_with_limited_observations(
    y_true: np.ndarray, y_proba: np.ndarray, n_clients_to_select: int
) -> float:
    """Calculate accuracy of choosing best `n_clients_to_select` clients.

    Args:
        y_true (np.ndarray): true labels
        y_proba (np.ndarray): predicted labels
        n_clients_to_select (int): number of clients to be chosen

    Returns:
        float: accuracy of electing clients.
    """
    idx_high_prob = np.argsort(y_proba[:, 1])[-n_clients_to_select:]
    y_selected = y_true[idx_high_prob]
    return y_selected.mean()
