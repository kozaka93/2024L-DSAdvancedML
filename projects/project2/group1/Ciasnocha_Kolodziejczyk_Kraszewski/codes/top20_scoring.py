from sklearn.metrics import make_scorer
import numpy as np


def top_20_perc_scoring(y_true, y_pred_proba):
    """
    Calculates the precision, but restricted to only the top 20% of the predictions.
    Top prediction means the predictions with the highest probability of class 1.
    """
    n = len(y_true)
    n_top = int(n * 0.2)
    sort_indices = np.argsort(y_pred_proba)
    top_indices = sort_indices[-n_top:]
    top_true = y_true[top_indices].sum()
    return top_true / np.min([n_top, y_true.sum()])