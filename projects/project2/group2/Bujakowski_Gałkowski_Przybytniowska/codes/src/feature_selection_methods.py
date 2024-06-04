from typing import Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import Lasso, LogisticRegression
from xgboost import XGBClassifier


# Method 1: Feature Importance using Random Forest
def rf_feature_importance_selection(
    X: np.ndarray, y: np.ndarray, n_features: int, return_indices: bool = False
) -> Union[np.ndarray, np.ndarray]:
    """
    Select top n_features based on feature importance from a RandomForestClassifier.

    Parameters:
    X (np.ndarray): The input feature matrix.
    y (np.ndarray): The target variable.
    n_features (int): Number of top features to select.
    return_indices (bool, optional): Whether to return the indices of the selected features. Defaults to False.

    Returns:
    Union[np.ndarray, np.ndarray]: The transformed feature matrix with selected features or the indices of the selected features.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    selected_features = np.argsort(importances)[::-1][:n_features]
    if return_indices:
        return selected_features
    return X[:, selected_features]


# Method 2: Recursive Feature Elimination (RFE)
def rfe_selection(X: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    """
    Perform Recursive Feature Elimination (RFE) to select top n_features.

    Parameters:
    X (np.ndarray): The input feature matrix.
    y (np.ndarray): The target variable.
    n_features (int): Number of top features to select.

    Returns:
    np.ndarray: The transformed feature matrix with selected features.
    """
    model = LogisticRegression(random_state=42)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)
    selected_features = np.where(rfe.support_)[0]
    return X[:, selected_features]


# Method 3: SelectKBest with F-classif
def select_k_best(X: np.ndarray, y: np.ndarray, n_features: int) -> np.ndarray:
    """
    Select top n_features using the SelectKBest method with the F-classif score function.

    Parameters:
    X (np.ndarray): The input feature matrix.
    y (np.ndarray): The target variable.
    n_features (int): Number of top features to select.

    Returns:
    np.ndarray: The transformed feature matrix with selected features.
    """
    selector = SelectKBest(f_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    return X_new


# Method 4: SelectKBest with mutual_info_classif
def select_k_best_mutual_info_classif(
    X: np.ndarray, y: np.ndarray, n_features: int
) -> np.ndarray:
    """
    Select top n_features using the SelectKBest method with the mutual_info_classif score function.

    Parameters:
    X (np.ndarray): The input feature matrix.
    y (np.ndarray): The target variable.
    n_features (int): Number of top features to select.

    Returns:
    np.ndarray: The transformed feature matrix with selected features.
    """
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_new = selector.fit_transform(X, y)
    return X_new


# Method 5: Feature Importance using XGBoost
def xgb_feature_importance_selection(
    X: np.ndarray, y: np.ndarray, n_features: int, return_indices: bool = False
) -> Union[np.ndarray, np.ndarray]:
    """
    Select top n_features based on feature importance from an XGBClassifier.

    Parameters:
    X (np.ndarray): The input feature matrix.
    y (np.ndarray): The target variable.
    n_features (int): Number of top features to select.
    return_indices (bool, optional): Whether to return the indices of the selected features. Defaults to False.

    Returns:
    Union[np.ndarray, np.ndarray]: The transformed feature matrix with selected features or the indices of the selected features.
    """
    model = XGBClassifier(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    selected_features = np.argsort(importances)[::-1][:n_features]
    if return_indices:
        return selected_features
    return X[:, selected_features]
