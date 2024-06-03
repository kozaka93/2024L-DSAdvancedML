# this file contains self-defined metrics for evaluation of our models
# the metrics are designed to be used in the context of our task. The requirements of the metrics are:

# 1. Each TP predicted by the model gives us 10 euro
# 2. Each feature costs us 200 euro
# 3. We may only predict 1/5 values as positive classes
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin


def default_competition_metric(y_true: np.ndarray, k: int, y_pred: np.ndarray = None, y_pred_proba = None, scale_metric=True):
    """Metric to evaulate the model performance in the context of the competition

    Args:
        y_true (np.ndarray): the true values of the target variable
        y_pred (np.ndarray): the predicted values of the target variable
        k (int): number of features used by the model
        y_pred_proba (np.ndarray, optional): If provided, then y_pred is not used. `y_pred_proba` is vector with models probabilities. Only 1/5 top probabilities are taken into the account. Defaults to None.
        scale_metrics (bool, optional): If True, the metric value is divided by the number of observations and multiplied by 5000 to imitate the behaviour on the test set
    """
    assert y_pred_proba is not None or y_pred is not None
    if y_pred_proba is not None:
        assert len(y_true) == len(y_pred_proba)
        assert len(y_pred_proba.shape) == 1
    if y_pred is not None:
        assert len(y_true) == len(y_pred)
    assert k >= 0
    assert k <= 500
    assert len(y_true.shape) == 1
    
    n = len(y_true)
    
    if y_pred_proba is not None:
        # take only 1/5 top probabilities
        top_02 = np.argsort(y_pred_proba)[::-1][:n//5]
        y_pred = np.zeros(n)
        y_pred[top_02] = 1
        
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    # we should also penalize the model for predicting too many positive classes
    
    positive = np.sum(y_pred)
    abundant_positive = 0
    if positive > n//5:
        abundant_positive = positive - n//5
    
    score = TP * 10 - abundant_positive * 10
    if scale_metric:
        score = score / n
        score = score * 5000
    score -= k*200
    return score
    
def proba_competition_metric(y_true, y_pred, k):
    return default_competition_metric(y_true, y_pred, k=k, y_pred_proba=y_pred)

def make_competition_scorer(k):    
    return make_scorer(proba_competition_metric, greater_is_better=True, k = k)


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.columns]

def competition_scoring(estimator, X, y, scale_metric=True):
    assert hasattr(estimator, 'predict_proba')
    #assert hasattr(estimator, 'feature_selection'), "The estimator should have feature_selection attribute"
    
    y_proba = estimator.predict_proba(X)[:, 1]
    
    k = len(estimator.named_steps.get("feature_selection").columns) if hasattr(estimator, "named_steps") and hasattr(estimator.named_steps.get("feature_selection"), 'columns') else X.shape[1]
    
    return default_competition_metric(y, k, scale_metric=scale_metric, y_pred_proba=y_proba)