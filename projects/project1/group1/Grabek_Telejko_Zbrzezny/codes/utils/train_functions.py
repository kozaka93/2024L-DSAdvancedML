"""
train_functions.py

Training and cross-validation functions for models evaluation.
"""

import time
import warnings
from typing import Dict

import numpy as np
from datasets.dataset_model import Dataset
from datasets.preprocess_helpers import split_with_preprocess
from optim import ADAM, GD, IWLS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 123,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Train models for a given train and test sets and return log-likelihood history and
    final balanced accuracy for each model.

    Arguments:
        X_train : Array with training data
        y_train : Array with target for training data
        X_test : Array with test data
        y_test : Array with target for test data
        seed : Random seed for reproducibility

    Returns:
        l_vals_dict : Dictionary with log-likelihood history for a single split for each model
        acc_vals_dict : Dictionary with balanced accuracy for a single split for each model
    """
    np.random.seed(seed)
    acc_vals_dict = {}
    l_vals_dict = {}

    # Custom LR models with different optimizers
    # IWLS, SGD, ADAM
    custom_models = {
        "iwls": IWLS(n_iter=500),
        "sgd": GD(learning_rate=0.0002, n_epoch=500),
        "adam": ADAM(learning_rate=0.0002, n_epoch=500),
    }

    for name, model in custom_models.items():
        model.fit(X_train, y_train)
        l_vals_dict[name] = model.loss_history
        acc_vals_dict[name] = balanced_accuracy_score(y_test, model.predict(X_test))

    # Scikit-learn models
    scikit_models = {
        "lr": LogisticRegression(),
        "qda": QDA(),
        "lda": LDA(),
        "dt": DecisionTreeClassifier(max_depth=5),
        "rf": RandomForestClassifier(max_depth=5, min_samples_split=3),
    }

    for name, model in scikit_models.items():
        model.fit(X_train, y_train.T[0])
        y_pred = np.expand_dims(model.predict(X_test), 1)
        acc_vals_dict[name] = balanced_accuracy_score(y_test, y_pred)

    return l_vals_dict, acc_vals_dict


def cv(
    dataset: Dataset, n_splits: int = 5, seed: int = 123, **kwargs
) -> tuple[Dict[str, list[float]], Dict[str, list[float]]]:
    """Cross-validation for every model used to evaluate balanced accuracy.

    Arguments:
        dataset : Dataset object
        n_splits : Number of different splits of data
        seed : Random seed for reproducibility

    Keyword Arguments:
        interactions : If True the interactions between variables are added during preprocessing
        test_size : Size of the test set
        random_state: Random seed for reproducibilty
        vif : If true multicolinear columns will be removed using VIF
        scale : If True the data will be scaled with StandardScaler

    Returns:
        l_vals_splits_dict : Dictionary with log-likelihood history for each split for each model
        acc_vals_splits_dict : Dictionary with balanced accuracy for each split for each model
    """
    np.random.seed(seed)
    all_models = ["iwls", "sgd", "adam", "lr", "qda", "lda", "dt", "rf"]
    custom_models = ["iwls", "sgd", "adam"]
    acc_vals_splits_dict = {
        model: [None for _ in range(n_splits)] for model in all_models
    }
    l_vals_splits_dict = {
        model: [None for _ in range(n_splits)] for model in custom_models
    }

    for i in range(n_splits):
        print(f"CV split {i+1}")

        X_train, y_train, X_test, y_test = split_with_preprocess(
            dataset=dataset, **kwargs
        )
        time.sleep(1)  # to remove visual bug with tqdm
        l_vals_dict, acc_vals_dict = train_and_eval(
            X_train, y_train, X_test, y_test, seed=seed
        )

        for key in l_vals_splits_dict:
            l_vals_splits_dict[key][i] = l_vals_dict[key]

        for key in acc_vals_splits_dict:
            acc_vals_splits_dict[key][i] = acc_vals_dict[key]

    return l_vals_splits_dict, acc_vals_splits_dict
