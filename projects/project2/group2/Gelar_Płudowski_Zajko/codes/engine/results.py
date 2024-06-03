import json
import pandas as pd
import pickle as pkl

from optuna import Study, Trial
from pathlib import Path
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from .metrics import cash_profit, precision_with_limited_observations
from .objectives import Objective
from .utils import get_feature_count


class ResultService:
    def __init__(
        self,
        study: Study,
        objective: Objective,
        output_path: Path,
    ):
        self.study = study
        self.objective = objective
        self.data_provider = objective.data_provider
        self.output_path = output_path

    def generate_results(self) -> None:
        self.output_path.mkdir(exist_ok=True)
        output_models_path = self.output_path / "models"
        output_models_path.mkdir(exist_ok=True)
        self.__save_best_trial_params(self.output_path, self.study)
        models, summary_df = self.__repeat_trial(
            self.objective, self.study.best_trial, 20
        )
        self.__save_summary_df(self.output_path, summary_df)
        self.__save_models(self.output_path / "models", models)

    def __save_best_trial_params(
        self, output_path: Path, study: Study
    ) -> None:
        with open(output_path / "best_hpo.json", "w") as f:
            json.dump(study.best_params, f, indent=4)

    def __repeat_trial(
        self, objective: Objective, trial: Trial, n_repetitions: int
    ) -> tuple[list[Pipeline], pd.DataFrame]:
        X_train, y_train = self.data_provider.train_data
        X_valid, y_valid = self.data_provider.validation_data
        X_test, y_test = self.data_provider.test_data

        results = []
        models = []
        for i in tqdm(range(n_repetitions)):
            model = objective.get_model_pipeline(trial, random_state=i)
            model.fit(X_train, y_train)
            models.append(model)
            results.append(
                pd.DataFrame(
                    {
                        "objective": [type(objective).__name__] * 3,
                        "repetition": [i] * 3,
                        "sample": ["train", "valid", "test"],
                        "gain": [
                            cash_profit(
                                y,
                                model.predict_proba(X),
                                n_features=get_feature_count(model),
                                n_clients_to_select=n_clients_to_select,
                                scaling_factor=scaling_factor,
                            )
                            for (
                                X,
                                y,
                                n_clients_to_select,
                                scaling_factor,
                            ) in [
                                (
                                    X_train,
                                    y_train,
                                    objective.clients_to_select_in_train_set,
                                    objective.train_scaling_factor,
                                ),
                                (
                                    X_valid,
                                    y_valid,
                                    objective.clients_to_select_in_val_set,
                                    objective.val_scaling_factor,
                                ),
                                (
                                    X_test,
                                    y_test,
                                    objective.clients_to_select_in_test_set,
                                    objective.test_scaling_factor,
                                ),
                            ]
                        ],
                        "precision": [
                            precision_with_limited_observations(
                                y,
                                model.predict_proba(X),
                                ratio,
                            )
                            for (X, y, ratio) in [
                                (
                                    X_train,
                                    y_train,
                                    objective.clients_to_select_in_train_set,
                                ),
                                (
                                    X_valid,
                                    y_valid,
                                    objective.clients_to_select_in_val_set,
                                ),
                                (
                                    X_test,
                                    y_test,
                                    objective.clients_to_select_in_test_set,
                                ),
                            ]
                        ],
                    }
                )
            )
        return models, pd.concat(results, axis=0).reset_index(drop=True)

    def __save_summary_df(
        self, output_path: Path, summary_df: pd.DataFrame
    ) -> None:
        summary_df.to_csv(output_path / "metrics.csv", index=False)

    def __save_models(
        self, output_path: Path, model_list: list[Pipeline]
    ) -> None:
        for i, model in enumerate(model_list):
            with open(
                output_path / f"model_{i}.pkl",
                "wb",
            ) as f:
                pkl.dump(model, f)
