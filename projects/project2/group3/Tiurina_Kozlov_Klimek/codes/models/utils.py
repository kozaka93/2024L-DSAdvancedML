import os
from typing import Optional, List, Callable, Tuple, NamedTuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cleanup_dataset_apply_standard_scaler(
        df: pd.DataFrame,
        target: Optional[str | int] = None
) -> pd.DataFrame:
    """
    Apply StandardScaler to the features

    :param df: DataFrame - input data
    :param target: str - target column name

    :return: DataFrame - cleaned data
    """
    df = df.copy()

    features = list(df.columns)
    if target is not None:
        features.remove(target)
    scaler = StandardScaler()

    if target is not None:
        df[features] = scaler.fit_transform(df[features])

    return df


ApplyModelMethod = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, Optional[int]], pd.Series]


def select_customers(
        predicted_probabilities: pd.Series,
        threshold_num: int = 1000
) -> pd.Series:
    """
    Select customers based on the predicted probabilities

    :param predicted_probabilities: Series - predicted probabilities
    :param threshold_num: int - number of customers to select
    :return: Series - selected customers, a 0 or 1 for each customer
    """
    probability_threshold = predicted_probabilities.quantile(1 - threshold_num / len(predicted_probabilities))
    return pd.Series(predicted_probabilities > probability_threshold, index=predicted_probabilities.index)


def compute_score(
        predicted: pd.Series,
        actual: pd.Series,
        feature_num: int,
        should_penalize_feature_num: bool = True,
        threshold_num: int = 1000
) -> int:
    """
    Compute score based on the number of correctly predicted customers and the number of variables used.
    :param predicted: pd.Series - Predicted values
    :param actual: pd.Series - Actual values
    :param feature_num: int - Number of variables used
    :param should_penalize_feature_num: bool - Should penalize the number of variables used
    :param threshold_num: int - Number of customers to select
    :return: int - Score
    """
    correct_instances_num = len(np.intersect1d(np.where(predicted.values == 1), np.where(actual.values == 1)))
    score = 10 * correct_instances_num + (
        # Since 200 is for 1000 customers, we need to scale 200 to threshold_num
        (-200 * feature_num * threshold_num / 1000)
        if should_penalize_feature_num
        else 0
    )
    return max(0, score)


def max_score(
        threshold_num: int = 1000
) -> int:
    """
    Compute the maximum score that can be achieved
    :param threshold_num: int - Number of customers to select
    :return: int - Maximum score
    """
    return 10 * threshold_num


GenerateFeatureInteractionsMethod = Callable[[pd.DataFrame], pd.DataFrame]


def generate_feature_interactions_noop(X: pd.DataFrame) -> pd.DataFrame:
    """
    No operation function to generate feature interactions
    :param X: DataFrame - input data
    :return: DataFrame - input data
    """
    return X


def generate_feature_interactions_quadratic(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generate quadratic feature interactions
    :param X: DataFrame - input data
    :return: DataFrame - input data with quadratic feature interactions
    """
    return pd.DataFrame(
        np.hstack([X.values, X.values ** 2]),
        columns=list(map(str, X.columns)) + [f"{col}^2" for col in X.columns],
        index=X.index
    )


def train_and_evaluate_model(
        model: ApplyModelMethod,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        selected_features: List[str | int],
        generate_feature_interactions: GenerateFeatureInteractionsMethod = generate_feature_interactions_noop,
        should_penalize_feature_num: bool = True,
        should_return_accuracy: bool = False,
        random_state: int = 0,
) -> Union[int, Tuple[int, float]]:
    """
    Train and evaluate the model
    1. Select features using the selected feature selection method
    2. Train the model
    3. Select customers
    4. Compute the score

    :param model: ApplyModelMethod - model to apply
    :param X_train: DataFrame - training data
    :param y_train: Series - training target data
    :param X_test: DataFrame - test data
    :param y_test: Series - test target data
    :param selected_features: List[str | int] - selected features
    :param generate_feature_interactions: GenerateFeatureInteractionsMethod - method to generate feature interactions
    :param should_penalize_feature_num: bool - should penalize the number of variables used
    :param should_return_accuracy: bool - should return accuracy
    :param random_state: int - random state
    :return: int - score
    """
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    X_train = generate_feature_interactions(X_train)
    X_test = generate_feature_interactions(X_test)

    predicted_probabilities = model(X_train, y_train, X_test, random_state)
    # Number of customers to select is the number of customers who bought the product
    threshold_num = np.sum(y_test.values)
    selected_customers = select_customers(predicted_probabilities, threshold_num)

    score = compute_score(
        selected_customers,
        y_test,
        len(selected_features),
        should_penalize_feature_num,
        threshold_num
    )

    if should_return_accuracy:
        y_class_assignment = predicted_probabilities > 0.5
        accuracy = accuracy_score(y_test, y_class_assignment)
        return score, accuracy

    return score


class ExperimentResult(NamedTuple):
    score: int
    accuracy: float


def run_experiment(
        experiment_save_path: str,
        experiment_name: str,
        model: ApplyModelMethod,
        X: pd.DataFrame,
        y: pd.Series,
        selected_features: List[str | int],
        generate_feature_interactions: GenerateFeatureInteractionsMethod = generate_feature_interactions_noop,
        iterations: int = 10,
        target_max_score: int = 10000
) -> List[ExperimentResult]:
    """
    Run an experiment
    1. Split the data into training and test sets
    2. Train and evaluate the model
    3. Save the results

    :param experiment_save_path: str - path to save the results
    :param experiment_name: str - name of the experiment
    :param model: ApplyModelMethod - model to apply
    :param X: DataFrame - data
    :param y: Series - target data
    :param selected_features: List[str | int] - selected features
    :param generate_feature_interactions: GenerateFeatureInteractionsMethod - method to generate feature interactions
    :param iterations: int - number of iterations
    :param target_max_score: int - target maximum score to scale the results
    :return: DataFrame - results
    """
    results = []
    for i in range(iterations):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        scale = target_max_score / max_score(threshold_num=np.sum(y_test.values))
        score, accuracy = train_and_evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            selected_features,
            generate_feature_interactions,
            should_penalize_feature_num=True,
            should_return_accuracy=True,
            random_state=i
        )
        results.append(ExperimentResult(score * scale, accuracy))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(experiment_save_path, f'{experiment_name}.csv'), index=False)

    return results
