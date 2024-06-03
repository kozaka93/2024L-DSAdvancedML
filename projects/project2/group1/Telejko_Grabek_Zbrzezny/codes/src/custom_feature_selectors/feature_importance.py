"""Module containing custom feature selectors based on feature importance."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.feature_selector import BaseFeatureSelector


class RandomForestFeatureImportanceSelector(BaseFeatureSelector):
    def __init__(
        self, n_features, n_estimators=100, max_depth=None, max_features="sqrt"
    ):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.selector = None

    def fit(self, X, y):
        self.selector = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
        )
        self.selector.fit(X, y)
        self.support_ = (
            self.selector.feature_importances_
            > np.sort(self.selector.feature_importances_)[::-1][self.n_features]
        )

    def get_support(self, indices=True):
        if indices:
            return np.where(self.support_)[0]
        return self.support_
