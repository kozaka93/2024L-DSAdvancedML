import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod


class FeatureSelector(ABC):
    """
    Abstract base class for feature selection.

    Methods:
        fit(X, y)
            Fit the model to the data.
        transform(X, y)
            Transform the data by selecting features.
        fit_transform(X, y)
            Fit the model and transform the data.
    """

    def __init__(self):
        self.selected_features = None

    @abstractmethod
    def fit(self, X, y):
        pass

    def transform(self, X):
        return X.loc[:, self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class RandomForestSelector(FeatureSelector, ABC):
    """
    Feature selector using a RandomForestClassifier.

    Args:
        threshold (float or str, optional): Threshold for feature importance. Default is 'auto'.
        **rf_kwargs: Additional arguments to pass to the RandomForestClassifier.

    Attributes:
        selected_features (array): Array of selected feature names.
    """
    def __init__(self, threshold='auto', **rf_kwargs):
        super().__init__()
        self.threshold = threshold
        self.rf_kwargs = rf_kwargs

    def fit(self, X, y):
        """
        Fit the RandomForest model to the data and select features.

        Args:
            X (DataFrame): Feature matrix.
            y (array-like): Target vector.
        """
        model = RandomForestClassifier(**self.rf_kwargs)
        model.fit(X, y)

        if self.threshold == 'auto':
            importances = sorted(model.feature_importances_, reverse=True)
            self.threshold = max([importances[i] - importances[i + 1] for i in range(len(importances) - 1)])

        self.selected_features = np.array(X.loc[:, model.feature_importances_ > self.threshold].columns)


class LogisticRegressionSelector(FeatureSelector, ABC):
    """
    Feature selector using Logistic Regression with L1 regularization.

    Args:
        C (float, optional): Regularization strength. Must be a positive float. Default is 1.0.
        **lasso_kwargs: Additional arguments to pass to the Lasso model.

    Attributes:
        selected_features (array): Array of selected feature names.
    """
    def __init__(self, C=1.0, **logreg_kwargs):
        super().__init__()
        self.C = C
        self.logreg_kwargs = logreg_kwargs

    def fit(self, X, y):
        """
        Fit the Lasso model to the data and select features.

        Args:
            X (DataFrame): Feature matrix.
            y (array-like): Target vector.
        """
        model = LogisticRegression(penalty='l1', C=self.C, solver='liblinear', **self.logreg_kwargs)
        model.fit(X, y)
        self.selected_features = np.array(X.columns)[model.coef_.flatten() != 0]


class SelectorPipeline(FeatureSelector, ABC):
    """
    Pipeline for sequentially or bagging multiple feature selectors.

    Args:
        selectors (list of FeatureSelector): List of feature selectors to be applied.
        method (str, optional): Method for combining feature selectors. Default is 'pipeline'.

    Attributes:
        selected_features (array): Array of selected feature names.
    """
    def __init__(self, selectors, method='pipeline'):
        super().__init__()
        self.selectors = selectors
        self.method = method

    def fit(self, X, y):
        """
        Fit the pipeline of feature selectors to the data.

        Args:
            X (DataFrame): Feature matrix.
            y (array-like): Target vector.
        """
        if self.method == 'pipeline':
            for i, selector in enumerate(self.selectors):
                print(f'Selector {i+1}/{len(self.selectors)}')
                X = selector.fit_transform(X, y)
                self.selected_features = selector.selected_features
        elif self.method == 'bagging':
            features = []
            unique_features = set()
            for i, selector in enumerate(self.selectors):
                print(f'Selector {i+1}/{len(self.selectors)}')
                selector.fit(X, y)
                features.append(selector.selected_features)
                unique_features = unique_features.union(selector.selected_features)

            self.selected_features = []
            for feature in unique_features:
                appearances = np.sum([feature in sel_features for sel_features in features])
                if appearances > (len(self.selectors) // 2):
                    self.selected_features.append(feature)
            self.selected_features = np.array(self.selected_features)
