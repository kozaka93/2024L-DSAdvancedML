"""Module for training utils including ."""

import random
from sklearn.model_selection import StratifiedKFold
import numpy as np

from src.settings import SEED


def prepare_cv_indices(n_observations, k_folds):
    """
    Function creates cross-validation indices for k folds.

    Arguments:
        n_observations: Number of observations in whole dataset used in cross-validation
        k_folds: number of folds for cross-validation

    Returns:
        splits: Training and testing indices
    """
    np.random.seed(SEED)
    random.seed(SEED)
    indices = np.arange(n_observations)
    np.random.shuffle(indices)
    fold_sizes = np.full(k_folds, n_observations // k_folds, dtype=int)
    fold_sizes[: n_observations % k_folds] += 1  # Distribute the remainder

    current = 0
    splits = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_indices, val_indices))
        current = stop

    return splits


def calculate_score(model, X_train, X_test, y_train, y_test, n_feats):
    """
    Function calculates custom score. It takes 1/5 observations from test set with highest
    probability of success and checks how many of them are truly 1. For each properly classified
    observation it adds 10 to score scaled by the fold size (e.g. for the half of the data multiplies
    it by 2). Then it dimishes score by 200 for each feature in train set.

    Arguments:
        model: model used for fit and predictions
        X_train: numpy array containing training predictors
        X_test: numpy array containing test predictors
        y_train: numpy array containing training target variable
        y_test: numpy array containing test target variable
        n_feats: number of features used in the model

    Returns:
        score: custom score value for given data and model
    """
    model.fit(X_train, y_train)
    proba_preds = model.predict_proba(X_test)
    best_indices = np.argsort(proba_preds[:, 1])[-X_test.shape[0] // 5 :]

    properly_classfied_count = np.sum(y_test[best_indices])

    print(
        f"Using {n_feats} features, we properly classified {properly_classfied_count}/{X_test.shape[0] // 5} clients."
    )

    score = 10 * properly_classfied_count * 5000 / X_test.shape[0] - 200 * n_feats
    return score


def cv(X, y, experiment_config, k_folds=5, split_indices=None):
    """
    Function performs cross validation with custom scoring function

    Arguments:
        X: numpy array with predictors
        y: numpy array with target variable
        experiment_config: dataclass instance with experiment config - model
            and feature selector (if None, no feature selection is performed)
        k_folds: number of folds for cross-validation

    Returns:
        scores: List of scores from custom metric for each cross-validation split
    """
    np.random.seed(SEED)
    random.seed(SEED)
    # fold_indices = prepare_cv_indices(n_observations=X.shape[0], k_folds=k_folds)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    fold_indices = skf.split(X, y)

    scores = []
    indices = []
    fold_id = 0
    for train_indices, test_indices in fold_indices:

        np.random.seed(SEED)
        random.seed(SEED)

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        model = experiment_config.classifier(**experiment_config.classifier_config)

        if split_indices is None:
            feature_selector = experiment_config.feature_selector
            feature_selector = feature_selector(
                **experiment_config.feature_selector_config
            )

            feature_selector.fit(X_train, y_train)
            X_train = feature_selector.transform(X_train)
            X_test = feature_selector.transform(X_test)

            indices.append(feature_selector.get_support(indices=True))

        else:
            X_train = X_train[:, split_indices[fold_id]]
            X_test = X_test[:, split_indices[fold_id]]

            indices.append(split_indices[fold_id])

        n_feats = X_train.shape[1]

        np.random.seed(SEED)
        random.seed(SEED)

        scores.append(calculate_score(model, X_train, X_test, y_train, y_test, n_feats))

        fold_id += 1

    return scores, indices
