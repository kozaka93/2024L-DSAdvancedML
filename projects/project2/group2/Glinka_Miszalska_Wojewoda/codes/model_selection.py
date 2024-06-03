import itertools

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
import itertools
import warnings

from utils import evaluate
warnings.filterwarnings("ignore")


# ---------------------------------- Base models exploration ----------------------------------
models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Bagging': BaggingClassifier(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Radius Neighbors': RadiusNeighborsClassifier(),
    'Gaussian NB': GaussianNB(),
    'SVM': SVC(),
    'Catboost': CatBoostClassifier(),
    'XGBoost': XGBClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Bernoulli NB': BernoulliNB(),
    'MLP': MLPClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'SGD': SGDClassifier(loss='log_loss'),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis()
}


def explore_models(X_train, X_test, y_train, y_test):
    """
    Explore different models and feature combinations to evaluate their performance.

    Parameters:
    X_train (pd.DataFrame): The feature matrix for the training dataset.
    X_test (pd.DataFrame): The feature matrix for the testing dataset.
    y_train (pd.Series): The true labels for the training dataset.
    y_test (pd.Series): The true labels for the testing dataset.

    Returns:
    list: A list of dictionaries containing the evaluation results for each model and feature combination.
    """
    results = []
    for i in range(2, 5):
        for features in itertools.combinations([100, 101, 102, 103, 104, 105], r=i):
            X_train_subset = X_train[list(features)]
            X_test_subset = X_test[list(features)]
            for model_name, model in models.items():
                price_train, price_test = evaluate(X_train_subset, X_test_subset, y_train, y_test, model)
                row = {
                    'Model': model_name,
                    'Features': [features],
                    'Score train': price_train,
                    'Score test': price_test
                }
                results.append(row)
                pd.DataFrame(row).to_csv('results/initial_models_results.csv', mode='a', header=False, index=False)
    return results


# ---------------------------------- Fine-tuning hyperparameters ----------------------------------
def grid_search_CV(model_class: BaseEstimator, param_grid: dict, X: pd.DataFrame, y: pd.Series, feature_range: range,
                   k_folds: int = 5):
    """
    Perform grid search with cross-validation over feature subsets and hyperparameters.

    Parameters:
    model_class (BaseEstimator): The machine learning model class to be evaluated.
    param_grid (dict): Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
    X (pd.DataFrame): The feature matrix containing the data.
    y (pd.Series): The target variable.
    feature_range (range): The range of numbers of features to be tested.
    k_folds (int, optional): The number of folds for cross-validation. Default is 5.

    Returns:
    pd.DataFrame: A DataFrame containing the evaluation results for each feature subset and parameter combination.
    """
    k_fold = KFold(n_splits=k_folds)
    features = X.columns.tolist()
    mean_results = pd.DataFrame(columns=['subset', 'params', 'mean_score_train', 'mean_score_test'])
    mean_results.tail(1).to_csv(f'mean_results-{model_class.__name__}.csv', index=False)

    for r in feature_range:
        subsets = list(itertools.combinations(features, r))
        for subset in subsets:
            for params in (dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())):
                model = model_class(**params)

                scores_train = []
                scores_test = []

                for train_index, test_index in k_fold.split(X[list(subset)]):
                    X_train, X_test = X[list(subset)].iloc[train_index], X[list(subset)].iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    price_train, price_test = evaluate(X_train, X_test, y_train, y_test, model)

                    scores_train.append(price_train)
                    scores_test.append(price_test)

                mean_results = mean_results._append({
                    'subset': subset,
                    'params': params,
                    'mean_score_train': np.mean(scores_train),
                    'mean_score_test': np.mean(scores_test)
                }, ignore_index=True)

                mean_results.tail(1).to_csv(f'results/mean_results-{model.__class__.__name__}.csv', mode='a', header=False,
                                            index=False)
    return mean_results


def multiple_cv(X, y, config_sets, model_class, **additional_params):
    """
    Perform multiple cross-validation for different configurations of features and hyperparameters.

    Parameters:
    X (pd.DataFrame): The feature matrix containing the data.
    y (pd.Series): The target variable.
    config_sets (list): A list of tuples containing feature sets and corresponding parameter dictionaries.
    model_class (type): The class of the machine learning model to be evaluated.
    **additional_params: Additional parameters to be passed to the model constructor.

    Returns:
    pd.DataFrame: A DataFrame containing the evaluation results for each feature set and parameter combination.
    """
    results_df = pd.DataFrame(columns=['features', 'params', 'cv1', 'cv2', 'cv3', 'cv4', 'cv5', 'mean', 'std'])
    results_df.tail(1).to_csv(f'results/best_configs_{model_class.__name__}.csv', mode='a', index=False)
    k_fold = KFold(n_splits=5)

    for i, (features, param) in enumerate(config_sets):
        param.update(additional_params)
        all_scores = []

        for _ in range(5):
            model = model_class(**param)
            scores = []
            for train_index, test_index in k_fold.split(X):
                X_train, X_test = X[list(features)].iloc[train_index], X[list(features)].iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train.values.ravel())
                score_train, score_test = evaluate(X_train, X_test, y_train, y_test, model)
                scores.append(score_test)

            all_scores.append(np.mean(scores))
        results_df.loc[i] = [list(features)] + [param] + all_scores + [np.mean(all_scores)] + [np.std(all_scores)]
        results_df.tail(1).to_csv(f'results/best_configs_{model_class.__name__}.csv', mode='a', header=False,
                                  index=False)
    return results_df
