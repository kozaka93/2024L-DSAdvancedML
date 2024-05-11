import numpy as np

from sklearn.metrics import accuracy_score


def accuracy(y_true: np.array, y_pred: np.array, n_features):
    # just to mock the future functions that will use n_features
    return accuracy_score(y_true, y_pred)


def money(y_true, y_pred, n_features) -> float:
    # todo: make it work to resemble the X_test
    per_customer_benefit = 10
    per_feature_cost = 2000
    return 0
