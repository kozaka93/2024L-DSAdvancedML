import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from engine.constants import (
    DATA_PATH,
    BORUTA_FEATURES_PATH,
    CLIENTS_TO_SELECT_IN_TEST_SET,
    RESULTS_PATH,
    RFE_RANKING_PATH,
    LOGISTIC_SOLVER_PENALTY_PAIRS,
)
from engine.data import DataProvider


def load_results(result_path: Path = RESULTS_PATH) -> pd.DataFrame:
    """
    Load the results from the given result_path directory.

    Args:
        result_path (Path): The path to the directory containing the result files.

    Returns:
        pd.DataFrame: The concatenated dataframe of all the result files.
    """
    result_files = result_path.rglob("metrics.csv")
    dfs = [pd.read_csv(file) for file in result_files]
    results = pd.concat(dfs)
    return results


def select_features_from_boruta(
    X: pd.DataFrame,
    rfe_ranking_type: Literal["random_forest", "elastic_net"],
    n_features: int,
    rfe_ranking_path: Path = RFE_RANKING_PATH,
) -> pd.DataFrame:
    """
    Selects the top 'n_features' features from the input dataframe. Assumes the dataframe columns are already limited to those remaining after boruta

    Args:
        X: The input dataframe containing the features.
        rfe_ranking_type: The type of ranking to use for feature selection. Must be either "random_forest" or "elastic_net".
        n_features: The number of top features to select.
        rfe_ranking_path: The path to the file containing the Boruta feature rankings.

    Returns:
        The dataframe with only the selected top features.
    """
    with open(rfe_ranking_path, "r") as f:
        rfe_rankings = json.load(f)
    ranking = rfe_rankings[rfe_ranking_type]
    return X.loc[:, np.array(ranking) <= n_features]


def select_features_from_all(
    X: pd.DataFrame,
    rfe_ranking_type: Literal["random_forest", "elastic_net"],
    n_features: int,
    rfe_ranking_path: Path = RFE_RANKING_PATH,
    boruta_features_path: Path = BORUTA_FEATURES_PATH,
) -> pd.DataFrame:
    """
    Selects features from the given full DataFrame using Boruta feature selection and RFE ranking.

    Args:
        X (pd.DataFrame): The input DataFrame.
        rfe_ranking_type (Literal["random_forest", "elastic_net"]): The type of RFE ranking to use.
        n_features (int): The number of features to select.
        rfe_ranking_path (Path, optional): The path to the RFE ranking file. Defaults to RFE_RANKING_PATH.
        boruta_features_path (Path, optional): The path to the Boruta features file. Defaults to BORUTA_FEATURES_PATH.

    Returns:
        pd.DataFrame: The DataFrame with selected features.
    """

    with open(boruta_features_path, "r") as f:
        boruta_features = np.array(json.load(f)["boruta_features"])
    X = X.iloc[:, boruta_features]
    return select_features_from_boruta(
        X, rfe_ranking_type, n_features, rfe_ranking_path
    )


def predict_and_save_results(model: BaseEstimator, X: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts the probabilities using the given model and saves the results and indices of chosen features to files in the required format

    Args:
        model (BaseEstimator): The trained model used for prediction.
        X (pd.DataFrame): The input data for prediction.

    Returns:
        pd.DataFrame: The predicted probabilities.
    """

    selected_columns = pd.Series([int(col.removeprefix("col_")) + 1 for col in X.columns])
    positive_indices = np.argsort(model.predict_proba(X)[:, 0])[
        :CLIENTS_TO_SELECT_IN_TEST_SET
    ] + 1
    (RESULTS_PATH / "test").mkdir(parents=True, exist_ok=True)
    pd.Series(positive_indices).to_csv(
        "results/test/313343_obs.txt", index=False, header=False
    )
    selected_columns.to_csv("results/test/313343_vars.txt", index=False, header=False)


def get_best_classifier(
    objective_name: Literal[
        "AdaboostObjective",
        "RandomForestObjective",
        "SvmObjective",
        "LogisticRegressionObjective",
        "XgboostObjective",
    ],
    random_state: int = 42,
) -> dict:
    """
    Get the best classifier based on the specified objective name.

    Args:
        objective_name (Literal): The name of the objective for selecting the best classifier.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        dict: A dictionary containing the best classifier model, RFE ranking type, and number of features.

    Raises:
        ValueError: If the objective name is not recognized.
    """
    if objective_name == "AdaboostObjective":
        params_path = RESULTS_PATH / "AdaboostObjective_adaboost" / "best_hpo.json"
        with open(params_path, "r") as f:
            params = json.load(f)
        model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=random_state,
            ),
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=random_state,
        )
    elif objective_name == "LogisticRegressionObjective":
        params_path = (
            RESULTS_PATH
            / "LogisticRegressionObjective_logistic_regression"
            / "best_hpo.json"
        )
        with open(params_path, "r") as f:
            params = json.load(f)
        solver, penalty = LOGISTIC_SOLVER_PENALTY_PAIRS[params["solver_penalty_idx"]]

        model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            tol=params["tol"],
            C=params["C"],
            max_iter=10_000,
            random_state=random_state,
            # intercept_scaling=params.get("intercept_scaling"),
            # dual=params.get("dual"),
            l1_ratio=params.get("l1_ratio"),
        )
        model = Pipeline([("scaler", StandardScaler()), ("classifier", model)])

    elif objective_name == "RandomForestObjective":
        params_path = (
            RESULTS_PATH / "RandomForestObjective_random_forest" / "best_hpo.json"
        )
        with open(params_path, "r") as f:
            params = json.load(f)

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_weight_fraction_leaf=params["min_weight_fraction_leaf"],
            n_jobs=-1,
            random_state=random_state,
        )
    elif objective_name == "SvmObjective":
        params_path = RESULTS_PATH / "SvmObjective_svm" / "best_hpo.json"
        with open(params_path, "r") as f:
            params = json.load(f)

        model = SVC(
            C=params["C"],
            kernel=params["kernel"],
            degree=params["degree"],
            gamma=params["gamma"],
            shrinking=params["shrinking"],
            probability=True,
            max_iter=2_000,
            random_state=random_state,
        )
        model = Pipeline([("scaler", MinMaxScaler()), ("classifier", model)])
    elif objective_name == "XgboostObjective":
        params_path = RESULTS_PATH / "XgboostObjective_xgboost" / "best_hpo.json"
        with open(params_path, "r") as f:
            params = json.load(f)

        model = XGBClassifier(
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            max_leaves=params["max_leaves"],
            learning_rate=params["learning_rate"],
            gamma=params["gamma"],
            colsample_bytree=params["colsample_bytree"],
            subsample=params["subsample"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Objective name {objective_name} not recognized")
    return {
        "model": model,
        "rfe_ranking_type": params["rfe_ranking_type"],
        "n_features": params["n_features"],
    }


def main() -> None:
    """
    Main function that generates test predictions.

    This function loads the results, creates classifier that achived best scores, trains it on the whole labeled dataset, runs and saves predictions on the test set.

    Parameters:
    None

    Returns:
    None
    """

    results_df = load_results()
    scores = (
        results_df[results_df["sample"] == "test"]
        .groupby("objective")
        .agg({"gain": "mean"})
    )
    best_objective = scores.nlargest(1, "gain").index[0]

    data_provider = DataProvider(DATA_PATH)
    X, y = data_provider.joined_data
    bundle = get_best_classifier(best_objective)
    model = bundle["model"]
    rfe_ranking_type = bundle["rfe_ranking_type"]
    n_features = bundle["n_features"]

    X_subset = select_features_from_boruta(X, rfe_ranking_type, n_features)
    model.fit(X_subset, y)
    X_test = pd.read_csv(DATA_PATH / "test_final.csv")
    X_test_subset = select_features_from_all(X_test, rfe_ranking_type, n_features)

    predict_and_save_results(model, X_test_subset)


if __name__ == "__main__":
    main()
