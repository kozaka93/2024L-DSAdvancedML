import numpy as np
from src.feature_selector import BaseFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class PermutationImportance(BaseFeatureSelector):
    """Permutation Importance feature selector."""

    def __init__(self, n_estimators=100, max_depth=5, n_feats=3) -> None:
        super().__init__()
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )
        self.result = None
        self.n_feats = n_feats

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X, y)
        self.result = permutation_importance(
            self.clf, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )

    def get_support(self, indices: bool = True) -> np.ndarray:
        return np.argsort(-self.result.importances_mean)[: self.n_feats]


class Impurity(BaseFeatureSelector):
    """Impurity based feature selector."""

    def __init__(self, n_estimators=100, max_depth=5, n_feats=3) -> None:
        super().__init__()
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth
        )
        self.result = None
        self.n_feats = n_feats

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clf.fit(X, y)
        self.result = np.argsort(-self.clf.feature_importances_)

    def get_support(self, indices: bool = True) -> np.ndarray:
        return self.result[: self.n_feats]
