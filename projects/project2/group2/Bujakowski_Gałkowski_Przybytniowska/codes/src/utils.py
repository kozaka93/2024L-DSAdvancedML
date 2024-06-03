from typing import List, Union

import numpy as np
import pandas as pd


def calculate_score(
    true_labels: Union[np.ndarray, List[int]], pred_labels: Union[np.ndarray, List[int]]
) -> float:
    """
    Calculate a custom score based on the number of correct positive label predictions.

    Parameters:
    true_labels (Union[np.ndarray, List[int]]): The ground truth binary labels (0 or 1).
    pred_labels (Union[np.ndarray, List[int]]): The predicted binary labels (0 or 1).

    Returns:
    float: The scaled score representing correct positive predictions per 1000 true positives, multiplied by 10.
    """
    count_positive_labels = np.sum(np.array(true_labels) == 1)
    correct_predictions = np.sum(np.array(pred_labels)[np.array(true_labels) == 1] == 1)
    if count_positive_labels == 0:
        return 0.0
    scaled_correct_predictions = correct_predictions / count_positive_labels * 1000
    return scaled_correct_predictions * 10


def drop_highly_correlated_columns(X: pd.DataFrame, threshold: float = 0.8) -> List:
    """
    Identify columns in a DataFrame that are highly correlated with other columns.

    Parameters:
    X (pd.DataFrame): The input DataFrame with numerical features.
    threshold (float, optional): The correlation threshold above which columns are considered highly correlated. Defaults to 0.8.

    Returns:
    list: A list of column names to be dropped due to high correlation.
    """
    corr_matrix = X.corr().abs()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return to_drop
