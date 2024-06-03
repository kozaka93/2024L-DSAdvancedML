from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from optuna import Trial
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .constants import (
    CLIENTS_IN_TEST_SET,
    CLIENTS_TO_SELECT_IN_TEST_SET,
    LOGISTIC_SOLVER_PENALTY_PAIRS,
)
from .data import DataProvider
from .metrics import cash_profit
from .utils import get_feature_count


class Objective(ABC):
    def __init__(
        self,
        data_provider: DataProvider,
    ):
        self.data_provider = data_provider

    def __call__(self, trial: Trial) -> float:
        X_train, y_train = self.data_provider.train_data
        X_valid, y_valid = self.data_provider.validation_data
        model_pipeline = self.get_model_pipeline(trial)
        model_pipeline.fit(X_train, y_train)
        valid_proba = model_pipeline.predict_proba(X_valid)
        return cash_profit(
            y_valid.to_numpy(),
            valid_proba,
            get_feature_count(model_pipeline),
            self.clients_to_select_in_val_set,
            self.val_scaling_factor,
        )

    def get_model_pipeline(
        self, trial: Trial, random_state: int = 1
    ) -> Pipeline:
        feature_selector = self.get_feature_selector(trial)
        model_pipeline = Pipeline(
            steps=[
                ("feature_selector", feature_selector),
                ("model", self.get_model(trial, random_state=random_state)),
            ]
        )
        return model_pipeline

    def get_feature_selector(self, trial: Trial) -> Pipeline:
        rfe_ranking_type = trial.suggest_categorical(
            "rfe_ranking_type", list(self.data_provider.rfe_ranking.keys())
        )
        chosen_ranking = self.data_provider.rfe_ranking[rfe_ranking_type]
        n_features = trial.suggest_int(
            "n_features", 1, len(chosen_ranking) // 2
        )
        return Pipeline(
            [
                (
                    "selector",
                    ColumnTransformer(
                        [
                            (
                                "selector",
                                "passthrough",
                                chosen_ranking[:n_features],
                            )
                        ],
                        remainder="drop",
                    ).set_output(transform="pandas"),
                )
            ]
        )

    @abstractmethod
    def get_model(self, trial: Trial, random_state: int) -> Any:
        pass

    @cached_property
    def train_scaling_factor(self) -> float:
        return CLIENTS_IN_TEST_SET / self.data_provider.X_train.shape[0]

    @cached_property
    def clients_to_select_in_train_set(self) -> int:
        return int(CLIENTS_TO_SELECT_IN_TEST_SET / self.train_scaling_factor)

    @cached_property
    def val_scaling_factor(self) -> float:
        return CLIENTS_IN_TEST_SET / self.data_provider.X_valid.shape[0]

    @cached_property
    def clients_to_select_in_val_set(self) -> int:
        return int(CLIENTS_TO_SELECT_IN_TEST_SET / self.val_scaling_factor)

    @cached_property
    def test_scaling_factor(self) -> float:
        return CLIENTS_IN_TEST_SET / self.data_provider.X_test.shape[0]

    @cached_property
    def clients_to_select_in_test_set(self) -> int:
        return int(CLIENTS_TO_SELECT_IN_TEST_SET / self.test_scaling_factor)


class XgboostObjective(Objective):
    def get_model(self, trial: Trial, random_state: int = 1) -> Any:
        return XGBClassifier(
            max_depth=trial.suggest_int("max_depth", 1, 5),
            n_estimators=trial.suggest_int("n_estimators", 10, 100),
            max_leaves=trial.suggest_int("max_leaves", 1, 64),
            learning_rate=trial.suggest_float(
                "learning_rate", 1e-5, 10, log=True
            ),
            verbosity=0,
            n_jobs=4,
            gamma=trial.suggest_float("gamma", 1e-5, 10, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1),
            subsample=trial.suggest_float("subsample", 0.5, 1),
            random_state=random_state,
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
        )


class LogisticRegressionObjective(Objective):
    def get_model(self, trial: Trial, random_state: int = 1) -> Any:
        solver_penalty = LOGISTIC_SOLVER_PENALTY_PAIRS[
            trial.suggest_int(
                "solver_penalty_idx", 0, len(LOGISTIC_SOLVER_PENALTY_PAIRS) - 1
            )
        ]
        params = {
            "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),
            "C": trial.suggest_float("C", 1e-1, 1e6, log=True),
            "solver": solver_penalty[0],
            "penalty": solver_penalty[1],
            "max_iter": 10000,
            "random_state": random_state,
        }
        if params["solver"] == "liblinear":
            params["intercept_scaling"] = trial.suggest_float(
                "intercept_scaling", 1e-3, 1
            )
            if params["penalty"] == "l2":
                params["dual"] = trial.suggest_categorical(
                    "dual", [True, False]
                )
        elif params["solver"] == "saga":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0, 1)

        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**params)),
            ]
        )


class SvmObjective(Objective):
    def get_model(self, trial: Trial, random_state: int = 1) -> Any:
        return Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                (
                    "model",
                    SVC(
                        C=trial.suggest_float("C", 1e-4, 1e4, log=True),
                        kernel=trial.suggest_categorical(
                            "kernel", ["linear", "poly", "rbf", "sigmoid"]
                        ),
                        degree=trial.suggest_int("degree", 1, 5),
                        gamma=trial.suggest_categorical(
                            "gamma", ["scale", "auto"]
                        ),
                        shrinking=trial.suggest_categorical(
                            "shrinking", [True, False]
                        ),
                        probability=True,
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )


class RandomForestObjective(Objective):
    def get_model(self, trial: Trial, random_state: int = 1) -> Any:
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 10, 200),
            criterion=trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]
            ),
            max_depth=trial.suggest_int("max_depth", 1, 7),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 128),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 128),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 2, 128),
            min_weight_fraction_leaf=trial.suggest_float(
                "min_weight_fraction_leaf", 1e-5, 0.5, log=True
            ),
            n_jobs=3,
            random_state=random_state,
        )


class AdaboostObjective(Objective):
    def get_model(self, trial: Trial, random_state: int = 1) -> Any:
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=trial.suggest_int("max_depth", 1, 5),
                min_samples_leaf=trial.suggest_int(
                    "min_samples_leaf", 10, 100
                ),
                random_state=random_state,
            ),
            n_estimators=trial.suggest_int("n_estimators", 10, 200),
            learning_rate=trial.suggest_float(
                "learning_rate", 1e-5, 10, log=True
            ),
            random_state=random_state,
        )
