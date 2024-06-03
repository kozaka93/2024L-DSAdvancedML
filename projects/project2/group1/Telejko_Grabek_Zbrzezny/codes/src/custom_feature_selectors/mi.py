"""Module for Mutual Information based feature selection."""

from abc import ABC
from typing import Literal, Union

import numpy as np
from sklearn.metrics import mutual_info_score as MI

from src.custom_feature_selectors.mi_utils import (CMI,
                                                   check_for_stopping_rule,
                                                   interaction_gain)
from src.feature_selector import BaseFeatureSelector


class MIFeatureSelector(BaseFeatureSelector, ABC):
    """Base class for Mutual Information based feature selectors."""

    def __init__(self, n_features: Union[int, Literal["auto"]] = "auto"):
        """Initialize the MI-based feature selector.

        Arguments:
            n_features: Number of features to select. If "auto", the stopping rule is used.
        """
        super().__init__()
        self.n_features = n_features
        self._selected = None

    def get_support(self, indices: bool = True) -> np.ndarray:
        """
        Get indices of the chosen features after the fit.

        Arguments:
            indices: If True, the return value will be an array of integers, rather than a boolean mask.
        """
        if indices:
            return np.array(self._selected)
        mask = np.zeros(len(self._selected), dtype=bool)
        mask[self._selected] = True
        return mask


class CMIM(MIFeatureSelector):
    """Conditional Mutual Information Maximization."""

    def __init__(self, n_features: Union[int, Literal["auto"]] = "auto"):
        """Initialize the CMIM feature selector.

        Arguments:
            n_features: Number of features to select. If "auto", the stopping rule is used.
        """
        super().__init__(n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the CMIM feature selector to the data.

        Arguments:
            X: Array with training features.
            y: Array with training target variable.
        """
        selected = []
        for _ in range(X.shape[1] if self.n_features == "auto" else self.n_features):
            max_cmim_value = float("-inf")
            for i in range(X.shape[1]):
                if i in selected:
                    continue
                J = MI(X[:, i], y)
                max_value = float("-inf")
                for j in selected:
                    curr_value = MI(X[:, i], X[:, j]) - CMI(X[:, i], X[:, j], y)
                    if curr_value > max_value:
                        max_value = curr_value
                if J - max_value > max_cmim_value:
                    max_cmim_value = J - max_value
                    max_idx = i

            if self.n_features == "auto" and check_for_stopping_rule(
                max_idx, X, y, selected
            ):
                break
            selected.append(max_idx)
        selected.sort()
        self._selected = selected


class JMIM(MIFeatureSelector):
    """Joint Mutual Information Maximization."""

    def __init__(self, n_features: Union[int, Literal["auto"]] = "auto"):
        """Initialize the JMIM feature selector.

        Arguments:
            n_features: Number of features to select. If "auto", the stopping rule is used.
        """
        super().__init__(n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the JMIM feature selector to the data.

        Arguments:
            X: Array with training features.
            y: Array with training target variable.
        """
        max_mi_value = float("-inf")
        for i in range(X.shape[1]):
            curr_mi = MI(X[:, i], y)
            if curr_mi > max_mi_value:
                first_idx = i
                max_mi_value = curr_mi
        selected = [first_idx]

        for _ in range(
            X.shape[1] - 1 if self.n_features == "auto" else self.n_features - 1
        ):
            max_jmim_value = float("-inf")
            for i in range(X.shape[1]):
                if i in selected:
                    continue
                min_value = float("inf")
                for j in selected:
                    curr_value = MI(X[:, j], y) + CMI(X[:, i], y, X[:, j])
                    if curr_value < min_value:
                        min_value = curr_value
                if min_value > max_jmim_value:
                    max_jmim_value = min_value
                    max_idx = i

            if self.n_features == "auto" and check_for_stopping_rule(
                max_idx, X, y, selected
            ):
                break
            selected.append(max_idx)
        selected.sort()
        self._selected = selected


class IGFS(MIFeatureSelector):
    """Information Gain Feature Selection."""

    def __init__(self, n_features: Union[int, Literal["auto"]] = "auto"):
        """Initialize the IGFS feature selector.

        Arguments:
            n_features: Number of features to select. If "auto", the stopping rule is used.
        """
        super().__init__(n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the IGFS feature selector to the data.

        Arguments:
            X: Array with training features.
            y: Array with training target variable.
        """
        selected = []
        for _ in range(X.shape[1] if self.n_features == "auto" else self.n_features):
            max_igfs_value = float("-inf")
            for i in range(X.shape[1]):
                if i in selected:
                    continue
                J = MI(X[:, i], y)
                inter_gain_sum = 0
                for j in selected:
                    inter_gain_sum += interaction_gain(X[:, i], X[:, j], y)
                inter_gain_sum /= len(selected) + 1
                if J + inter_gain_sum > max_igfs_value:
                    max_igfs_value = J + inter_gain_sum
                    max_idx = i

            if self.n_features == "auto" and check_for_stopping_rule(
                max_idx, X, y, selected
            ):
                break
            selected.append(max_idx)
        selected.sort()
        self._selected = selected
