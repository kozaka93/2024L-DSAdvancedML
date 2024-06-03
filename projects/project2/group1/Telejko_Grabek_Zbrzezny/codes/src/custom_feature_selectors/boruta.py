import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

from src.feature_selector import BaseFeatureSelector


class Boruta(BaseFeatureSelector):
    """Boruta based feature selector."""

    def __init__(
        self,
        additional_feat_selector=None,
        model_n_estimators=100,
        model_max_depth=5,
        boruta_n_estimators="auto",
        boruta_max_iter=10,
    ) -> None:
        super().__init__()

        self.model = RandomForestClassifier(
            n_jobs=-1,
            n_estimators=model_n_estimators,
            max_depth=model_max_depth,
            class_weight="balanced",
        )
        self.boruta_feat_selector = BorutaPy(
            verbose=2,
            estimator=self.model,
            n_estimators=boruta_n_estimators,
            max_iter=boruta_max_iter,
        )
        self.additional_feat_selector = additional_feat_selector
        self.boruta_chosen = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.boruta_feat_selector.fit(X, y)
        X_reduced = self.boruta_feat_selector.transform(X)

        if self.additional_feat_selector:
            self.additional_feat_selector.fit(X_reduced, y)

        chosen_columns = []
        for i in range(X.shape[1]):
            for j in range(X_reduced.shape[1]):
                if np.equal(X[:, i], X_reduced[:, j]).all():
                    chosen_columns.append(i)
                    break
        self.boruta_chosen = np.array(chosen_columns)

    def get_support(self, indices: bool = True) -> np.ndarray:
        if indices:
            if self.additional_feat_selector:
                return self.boruta_chosen[
                    self.additional_feat_selector.get_support(indices=True)
                ]
            else:
                return self.boruta_chosen

        if self.additional_feat_selector:
            return self.boruta_chosen[
                self.additional_feat_selector.get_support(indices=True)
            ]
        else:
            return self.boruta_chosen
