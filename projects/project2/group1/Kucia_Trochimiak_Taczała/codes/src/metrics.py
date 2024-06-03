import numpy as np


def calculate_gain(
    ground_truth: np.ndarray, predictions: np.ndarray, features_count: int
) -> float:
    """
    Calculate the gain of the predictions compared to the ground truth.
    """
    preds_to_pick = int(ground_truth.shape[0] * 0.2)
    sorted_preds = np.argsort(predictions)[::-1][:preds_to_pick]
    return (
        np.sum(ground_truth[sorted_preds]) / preds_to_pick * 10_000
        - features_count * 200
    )
