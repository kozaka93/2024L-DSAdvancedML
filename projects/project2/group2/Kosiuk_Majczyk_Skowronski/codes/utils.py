from typing import Union

import pandas as pd
import numpy as np

from scipy.stats import ttest_ind
from scipy.stats import ks_2samp

from sklearn.metrics import precision_score



# ----------------------------------------------------------------------------------------------------------------------------
# Custom scoring functions
# ----------------------------------------------------------------------------------------------------------------------------


def custom_optuna_score(y_true, y_pred):
    top_indices = np.argsort(y_pred)[-int(len(y_pred)/5):]
    y_pred = np.zeros_like(y_pred)
    y_pred[top_indices] = 1
    return np.sum((y_pred == 1) & (y_true == 1))



def task_score(estimator, X, y, model_name=None):
    def correct_score(y_true, y_pred):
        return np.sum((y_pred == 1) & (y_true == 1))
    n = X.shape[0]
    n_features = X.shape[1]
    
    if model_name:
        y_pred_proba = estimator.predict_proba(X, model=model_name).iloc[:, 1]
    else:
        y_pred_proba = estimator.predict_proba(X)[:, 1]
    top_indices = np.argsort(y_pred_proba)[-int(n/5):]
    y_pred = np.zeros_like(y_pred_proba)
    y_pred[top_indices] = 1
    
    coef = 5000 / n
    return coef * 10 * correct_score(y, y_pred) - 200 * n_features

def task_score_cv(estimator, X, y, model_name=None, cv=5):
    def correct_score(y_true, y_pred):
        return np.sum((y_pred == 1) & (y_true == 1))
    
    # split data into 5 folds
    np.random.seed(42)
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
    X_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    scores = []

    for i in range(cv):
        X_train_cv = X_folds.copy()
        y_train_cv = y_folds.copy()
        X_val = X_train_cv.pop(i)
        y_val = y_train_cv.pop(i)
        X_train_cv = np.concatenate(X_train_cv)
        y_train_cv = np.concatenate(y_train_cv)

        n = X_val.shape[0]
        n_features = X_val.shape[1]

        estimator.fit(X_train_cv, y_train_cv)

        if model_name:
            y_pred_proba = estimator.predict_proba(X_val, model=model_name).iloc[:, 1]
        else:
            y_pred_proba = estimator.predict_proba(X_val)[:, 1]

        top_indices = np.argsort(y_pred_proba)[-int(n/5):]
        y_pred = np.zeros_like(y_pred_proba)
        y_pred[top_indices] = 1

        coef = 5000 / n
        scores.append(coef * 10 * correct_score(y_val, y_pred) - 200 * n_features)
    
    return np.mean(scores), scores



def calculate_score(
        X : Union[pd.DataFrame, np.ndarray],
        y_true : Union[pd.Series, np.ndarray],
        model,
        remedy = True,
        ):
    number_of_features = X.shape[1]
    y_pred = model.predict(X)
    correct_class_1 = np.sum((y_pred == 1) & (y_true == 1))
    remedy_coef = 5000/X.shape[0] if remedy else 1   
    return remedy_coef *10 * correct_class_1 - 200 * number_of_features

def precision_calculate_score(
        X : Union[pd.DataFrame, np.ndarray],
        y_true : Union[pd.Series, np.ndarray],
        model,
        remedy = True,
        ):
    y_pred = model.predict(X)
    return precision_score(y_true,y_pred) * calculate_score(X, y_true, model, remedy)



# ----------------------------------------------------------------------------------------------------------------------------
# Statistical tests for feature selection
# ----------------------------------------------------------------------------------------------------------------------------

def t_test_features_selection(X: pd.DataFrame, pvalue: float, target: str = "y"):
    """
    Perform t-test for each feature in the dataset. Based on the p-value, the function returns the significant features.
    Works only for binary classification - target variable should be binary and have values 0, 1.

    Parameters
    ----------
        X : DataFrame - the input data containing features and target variable
        pvalue : float - the significance level for the t-test (default is 0.05). If p-value of test <= (selected_value), the feature is significant.
        target : str - the name of the target variable (default is 'y')

    Returns
    -------
        significant_features : list - the list of significant features
    """

    assert target in X.columns, f"Target variable {target} not in the dataset"
    assert X[target].nunique() == 2, "Target variable should have 2 unique values"
    assert X[target].isin([0, 1]).all(), "Target variable should have values 0, 1"

    significant_features = []
    columns = X.columns.difference([target])
    for column in columns:
        group1 = X.loc[X[target] == 0, column]
        group2 = X.loc[X[target] == 1, column]
        t_stat, p_val = ttest_ind(group1, group2)
        if p_val <= pvalue:
            significant_features.append(column)
    return significant_features


def kolmogorov_smirnoff_selection(X: pd.DataFrame, pvalue: float, target: str = "y"):
    """
    Perform Kolmogorov-Smirnoff test for each feature in the dataset. Based on the p-value, the function returns the significant features.
    Works only for binary classification - target variable should be binary and have values 0, 1.

    Parameters
    ----------
        X : DataFrame - the input data containing features and target variable
        pvalue : float - the significance level for the Kolmogorov-Smirnoff test (default is 0.05). If p-value of test <= (selected_value), the feature is significant.
        target : str - the name of the target variable (default is 'y')

    Returns
    -------
        significant_features : list - the list of significant features
    """

    assert target in X.columns, f"Target variable {target} not in the dataset"
    assert X[target].nunique() == 2, "Target variable should have 2 unique values"
    assert X[target].isin([0, 1]).all(), "Target variable should have values 0, 1"

    significant_features = []
    columns = X.columns.difference([target])
    for column in columns:
        group1 = X.loc[X[target] == 0, column]
        group2 = X.loc[X[target] == 1, column]
        _, p_val = ks_2samp(group1, group2)
        if p_val <= pvalue:
            significant_features.append(column)
    return significant_features