import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2, style='darkgrid')
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score, KFold
from itertools import combinations
from evaluation import evaluate


def get_importance(X: pd.DataFrame, y: np.array, n_features=10) -> Tuple[pd.Series, np.array]:
    """
    Get the top n feature importances of a Random Forest model.
    Args:
        X: A pandas DataFrame of features
        y: A numpy array of labels
        n_features: The number of top features to return
    """
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X, y)
    feature_names = [f"feature {i}" for i in range(X.shape[1])]

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    forest_importances = pd.Series(importances, index=feature_names)

    top_features = forest_importances.nlargest(n_features)
    top_indices = top_features.index
    top_std = std[[feature_names.index(f) for f in top_indices]]

    return top_features, top_std

def plot_importance(features: pd.Series, std: np.array) -> None:
    """
    Plot the top feature importances with standard deviations.
    Args:
        features: A pandas Series of feature importances
        std: A numpy array of standard deviations
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    features.plot.bar(yerr=std, ax=ax, capsize=4)
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()

def test_k(X: pd.DataFrame, y: np.array, models: List, k_values: List[int], scoring=evaluate) -> pd.DataFrame:
    """
    Test different numbers of features for a given model.
    Args:
        X: A pandas DataFrame of features
        y: A numpy array of labels
        models: A list of models to test
        k_values: A list of numbers of features to test
        scoring: A scoring function
    """
    results = []

    for k in k_values:
        features = SelectFromModel(RandomForestClassifier(random_state=42), max_features=k).fit(X, y).get_support(indices=True)
        X_selected = X.iloc[:, features]
        for model in models:
            scores = cross_val_score(model, X_selected, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=scoring)
            results.append({'model': model.__class__.__name__, 'k': k, 'scores': scores})

    return pd.DataFrame(results)

def plot_k(df: pd.DataFrame) -> None:
    """
    Plot the results of the find_k function.
    Args:
        df: A pandas DataFrame of results
    """
    df = df.explode('scores')
    df['scores'] = pd.to_numeric(df['scores'])

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='k', y='scores', hue='model', data=df)
    plt.xlabel('Number of features (k)')
    plt.legend(title='Model')
    plt.ylabel('Score')
    plt.show()

def test_combinations(X: pd.DataFrame, y: np.array, models: List, top_features: List, k: int, scoring=evaluate) -> pd.DataFrame:
    """
    Test different combinations of k features from top features for a given model.
    Args:
        X: A pandas DataFrame of features
        y: A numpy array of labels
        models: A list of models to test
        top_features: A list of top features
        k: The number of features to test
        scoring: A scoring function
    """
    combs = list(combinations(top_features, k))

    results = []

    for k in range(10):
        print("Experiment ", k+1)
        for comb in combs:
            X_train_comb = X.iloc[:, list(comb)]
            for model in models:
                score = cross_val_score(model, X_train_comb, y, cv=KFold(n_splits=5, shuffle=True, random_state=k), scoring=scoring)
                results.append({'model': model.__class__.__name__, 'features': comb, 'scores': score})

    return pd.DataFrame(results)

def plot_combinations(df: pd.DataFrame) -> None:
    """
    Plot the results of the test_combinations function.
    Args:
        df: A pandas DataFrame of results
    """
    df['scores'] = df['scores'].apply(lambda x: np.mean(x))
    df['features'] = df['features'].apply(lambda x: str(x))

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='features', y='scores', hue='model', data=df)
    plt.xlabel('Feature Combination')
    plt.legend(title='Model')
    plt.ylabel('Mean score')
    plt.show()