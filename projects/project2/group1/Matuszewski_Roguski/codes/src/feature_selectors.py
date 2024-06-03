# Interface for different feature selection methods
import pandas as pd
import numpy as np
from boruta_unreleased.boruta import BorutaPy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE, RFECV
from abc import ABC, abstractmethod
from typing import Any
from sklearn.model_selection import train_test_split
import pathlib
import json

class FeatureSelector(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X, y):
        pass

    @abstractmethod
    def get_support(self):
        pass

    @abstractmethod
    def print_support(self):
        pass

    # @abstractmethod
    # def get_importance(self):
    #     pass

class BorutaSelector(FeatureSelector):
    '''
    Boruta feature selection

    Parameters
    ----------
    estimator : Any
        The estimator to use for feature selection
    n_estimators : int, default=1000
        The number of estimators to use
    perc : int, default=100
        The percentage of features to keep
    alpha : float, default=0.05
        The alpha value
    two_step : bool, default=True
        Whether to use two step feature selection
    max_iter : int, default=100
        The maximum number of iterations
    random_state : Any, default=None
        The random state
    verbose : int, default=0
        The verbosity level
    early_stopping : bool, default=False
        Whether to use early stopping
    n_iter_no_change : int, default=20
        The number of iterations with no change
    limit : int, default=None
        The limit of features to keep

    Attributes
    ----------
    features : list
        The list of features to keep
    importance_history_ : array-like
        The importance history
    columns : list
        The list of columns

    Methods
    -------
    fit(X, y)
        Fit the feature selector
    transform(X)
        Transform the data
    fit_transform(X, y)
        Fit and transform the data
    get_support()
        Get the list of features to keep
    print_support()
        Print the number of features and the features
    get_importance()
        Get the importance of features

    '''
    def __init__(self,
                estimator: Any,
                n_estimators: int = 1000,
                perc: int = 100,
                alpha: float = 0.05,
                two_step: bool = True,
                max_iter: int = 100,
                random_state: Any | None = None,
                verbose: int = 0,
                early_stopping: bool = False,
                n_iter_no_change: int = 20,
                limit: int | None = None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.features = None
        self.limit = limit
        self.importance_history_ = None
        self.columns = None

    def fit(self, X, y):
        selector = BorutaPy(estimator=self.estimator,
                                    n_estimators=self.n_estimators,
                                    perc=self.perc,
                                    alpha=self.alpha,
                                    two_step=self.two_step,
                                    max_iter=self.max_iter,
                                    random_state=self.random_state,
                                    verbose=self.verbose,
                                    early_stopping=self.early_stopping,
                                    n_iter_no_change=self.n_iter_no_change)
        selector.fit(X, y.values.ravel())
        self.features = np.array(X.columns)[selector.support_].tolist()
        self.importance_history_ = selector.importance_history_
        self.columns = X.columns
        if self.limit and len(self.features) > self.limit:
            importance_dict = self.get_importance()
            self.features = list(importance_dict.keys())[:self.limit]

    def transform(self, X):
        return X[self.features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        return self.features
    
    def print_support(self):
        print(f"Number of features: {len(self.features)}")
        print(f"Features: {self.features}")

    def get_importance(self):
        # importance_history_ : array-like, shape [n_features, n_iters]
        # Calculate the mean importance of each feature over all iterations
        means_ = np.mean(self.importance_history_, axis=0)
        importance_dict = dict(zip(self.columns, means_))
        # Sort the dictionary by value in descending order
        importance_dict = dict(sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True))
        return importance_dict
    
# Different feature selection methods

# from sklearn.feature_selection import RFE, RFECV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import SelectFromModel
# from sklearn.decomposition import PCA

class RFESelector(FeatureSelector):
    '''
    Recursive Feature Elimination feature selection

    Parameters
    ----------
    estimator : Any
        The estimator to use for feature selection
    n_features_to_select : int, default=None
        The number of features to select
    step : int, default=1
        The number of features to remove at each step
    verbose : int, default=0
        The verbosity level

    Attributes
    ----------
    features : list
        The list of features to keep
    ranking_ : array-like
        The feature ranking
    support_ : array-like
        The feature support
    columns : list
        The list of columns

    Methods
    -------
    fit(X, y)
        Fit the feature selector
    transform(X)
        Transform the data
    fit_transform(X, y)
        Fit and transform the data
    get_support()
        Get the list of features to keep
    print_support()
        Print the number of features and the features

    '''
    def __init__(self,
                estimator: Any,
                limit: int | None = None,
                step: int = 1,
                verbose: int = 0):
        self.estimator = estimator
        self.limit = limit
        self.step = step
        self.verbose = verbose
        self.features = None
        self.ranking_ = None
        self.support_ = None
        self.columns = None

    def fit(self, X, y):
        selector = RFE(estimator=self.estimator,
                        n_features_to_select=self.limit,
                        step=self.step,
                        verbose=self.verbose)
        selector.fit(X, y.values.ravel())
        self.features = np.array(X.columns)[selector.support_].tolist()
        self.ranking_ = selector.ranking_
        self.support_ = selector.support_
        self.columns = X.columns

    def transform(self, X):
        return X[self.features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        return self.features
    
    def print_support(self):
        print(f"Number of features: {len(self.features)}")
        print(f"Features: {self.features}")

class SelectFromModelSelector(FeatureSelector):
    '''
    Select From Model feature selection

    Parameters
    ----------
    estimator : Any
        The estimator to use for feature selection
    threshold : float, default=None
        The threshold to use
    prefit : bool, default=False
        Whether the estimator is already fitted

    Attributes
    ----------
    features : list
        The list of features to keep
    support_ : array-like
        The feature support
    columns : list
        The list of columns

    Methods
    -------
    fit(X, y)
        Fit the feature selector
    transform(X)
        Transform the data
    fit_transform(X, y)
        Fit and transform the data
    get_support()
        Get the list of features to keep
    print_support()
        Print the number of features and the features

    '''
    def __init__(self,
                estimator: Any,
                threshold: float | None = None,
                prefit: bool = False,
                limit: int | None = None):
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.features = None
        self.support_ = None
        self.columns = None
        self.limit = limit

    def fit(self, X, y):
        selector = SelectFromModel(estimator=self.estimator,
                                    threshold=self.threshold,
                                    prefit=self.prefit,
                                    max_features=self.limit)
        selector.fit(X, y.values.ravel())
        self.features = np.array(X.columns)[selector.get_support()].tolist()
        self.support_ = selector.get_support()
        self.columns = X.columns

    def transform(self, X):
        return X[self.features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        return self.features
    
    def print_support(self):
        print(f"Number of features: {len(self.features)}")
        print(f"Features: {self.features}")

class RFECVSelector(FeatureSelector):
    '''
    Recursive Feature Elimination with Cross Validation feature selection

    Parameters
    ----------
    estimator : Any
        The estimator to use for feature selection
    step : int, default=1
        The number of features to remove at each step
    cv : int, default=5
        The number of cross validation folds
    scoring : str, default=None
        The scoring metric
    verbose : int, default=0
        The verbosity level

    Attributes
    ----------
    features : list
        The list of features to keep
    ranking_ : array-like
        The feature ranking
    support_ : array-like
        The feature support
    columns : list
        The list of columns

    Methods
    -------
    fit(X, y)
        Fit the feature selector
    transform(X)
        Transform the data
    fit_transform(X, y)
        Fit and transform the data
    get_support()
        Get the list of features to keep
    print_support()
        Print the number of features and the features

    '''
    def __init__(self,
                estimator: Any,
                step: int = 1,
                cv: int = 5,
                scoring: str | None = None,
                verbose: int = 0,
                limit: int | None = None):
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.features = None
        self.ranking_ = None
        self.support_ = None
        self.columns = None
        self.limit = limit

    def fit(self, X, y):
        selector = RFECV(estimator=self.estimator,
                        step=self.step,
                        cv=self.cv,
                        scoring=self.scoring,
                        verbose=self.verbose)
        selector.fit(X, y.values.ravel())
        self.features = np.array(X.columns)[selector.support_].tolist()
        self.ranking_ = selector.ranking_
        self.support_ = selector.support_
        self.columns = X.columns
        if self.limit and len(self.features) > self.limit:
            # Use self.ranking_ to select the top features
            # Sort features by ranking and select the top limit features
            self.features = np.array(X.columns)[np.argsort(self.ranking_) < self.limit].tolist()

    def transform(self, X):
        return X[self.features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        return self.features
    
    def print_support(self):
        print(f"Number of features: {len(self.features)}")
        print(f"Features: {self.features}")


class LassoSelector(FeatureSelector):
    '''
    Lasso feature selection

    Parameters
    ----------
    alpha : float, default=1.0
        The alpha value
    max_iter : int, default=1000
        The maximum number of iterations
    random_state : Any, default=None
        The random state

    Attributes
    ----------
    features : list
        The list of features to keep
    coef_ : array-like
        The feature coefficients
    columns : list
        The list of columns

    Methods
    -------
    fit(X, y)
        Fit the feature selector
    transform(X)
        Transform the data
    fit_transform(X, y)
        Fit and transform the data
    get_support()
        Get the list of features to keep
    print_support()
        Print the number of features and the features

    '''
    def __init__(self,
                alpha: float = 1.0,
                max_iter: int = 1000,
                random_state: Any | None = None,
                limit: int | None = None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.features = None
        self.coef_ = None
        self.columns = None
        self.limit = limit

    def fit(self, X, y):
        selector = LogisticRegression(C=1/self.alpha,
                                    penalty='l1',
                                    solver='liblinear',
                                    max_iter=self.max_iter,
                                    random_state=self.random_state)
        selector.fit(X, y.values.ravel())
        self.features = np.array(X.columns)[selector.coef_[0] != 0].tolist()
        self.coef_ = selector.coef_[0]
        self.columns = X.columns
        if self.limit and len(self.features) > self.limit:
            importance_dict = self.get_importance()
            self.features = list(importance_dict.keys())[:self.limit]

    def transform(self, X):
        return X[self.features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        return self.features
    
    def print_support(self):
        print(f"Number of features: {len(self.features)}")
        print(f"Features: {self.features}")

    def get_importance(self):
        importance_dict = dict(zip(self.columns, self.coef_))
        # Sort the dictionary by value in descending order
        importance_dict = dict(sorted(importance_dict.items(), key=lambda item: abs(item[1]), reverse=True))
        return importance_dict
    
class GreedyGainSelector(FeatureSelector):
    '''
    Greedy Gain feature selection - inspired by the greedy algorithm and the evolution of features.

    Parameters
    ----------
    estimator : Any
        The estimator to use for feature selection
    method : str, default='top_1'
        The method to use for feature selection
        possible values: 'top_1', 'random_improvement'
            'top_1' - select the feature which improves the effectiveness score the most
            'random_improvement' - select a feature which improves the effectiveness score and is >= the current best score with probability proportional to the score
    cv_frac : float, default=0.25
        The fraction of the data to use for cross validation
    score_frac : float, default=0.2
        The fraction of the data to use for scoring - it is essential to use a fraction of the data for scoring to simulate the real world scenario
        where there is only a limited amount of predictions available for assigning label 1
    iterations : int, default=5
        The number of iterations to use in the cross validation
    verbose : int, default=0
        The verbosity level - 0 for no output, 1 for output

    Attributes
    ----------
    features : list
        The list of features to keep
    columns : list
        The list of columns

    Methods
    -------
    fit(X, y)
        Fit the feature selector
    transform(X)
        Transform the data
    fit_transform(X, y)
        Fit and transform the data
    get_support()
        Get the list of features to keep
    print_support()
        Print the number of features and the features

    IMPORTANT NOTE:
    
    Our final prediction for grading will be based on the effectiveness score.
    This test dataset have 5000 rows, where we can put max 1000 rows with label 1.
    In other words, we can have max 1000 true positives, so we have to put 1000/5000 = 0.2 
    fraction of the data for scoring (score_frac).

    Our training data by default have 4000 rows(final test 1000 rows) , with cv_frac = 0.25,
    we have 1000 rows for test (validation) dataset for cross validation. score_frac = 0.2 means
    that we will take 200 rows from the test dataset with the highest probability of 1
    to calculate the effectiveness score. It is important not to use all validation data for scoring
    because in the real world scenario we have only a limited amount of predictions
    available for assigning label 1.

    In final test scenario we will use effectiveness score:
    effectiveness = tp * tp_reward - n_features_cost * n_features
    where tp_reward = 10 and n_features_cost = 200

    In our cv scenario we will use the same effectiveness score, but we will calculate it
    effectiveness = tp * tp_reward - score_frac * n_features_cost * n_features

    We use score_frac here because maximum tp in the cv scenario is 0.2 * 1000 = 200 (5 times less
    than in the final test scenario), so we have to decrease the cost for features the same way. 
    
    '''

    def __init__(self,
                estimator: Any,
                method: str = 'top_1',
                cv_frac: float = 0.25,
                score_frac: float = 0.2,
                iterations: int = 5,
                verbose: int = 0,
                forward: bool = True,
                logs: bool = True,
                prefix: str = None
                ):
        self.estimator = estimator
        self.method = method
        self.cv_frac = cv_frac
        self.score_frac = score_frac
        self.iterations = iterations
        self.verbose = verbose
        self.forward = forward
        self.features = None
        self.columns = None
        self.logs = logs
        self.prefix = prefix

    def _get_effectiveness_score(self, y_test, y_pred, n_features, tp_reward=10, n_features_cost=200, score_frac=0.2):
        tp = sum((y_test == 1) & (y_pred == 1))
        effectiveness = tp * tp_reward - score_frac * n_features_cost * n_features
        return effectiveness
    
    def fit(self, X, y):
        if self.method == 'top_1' or self.method == 'random_improvement':
            if self.logs:
                logs_path = pathlib.Path('logs')
                logs_path.mkdir(parents=True, exist_ok=True)
                if self.prefix:
                    logs_path = logs_path / f'{self.prefix}_{self.estimator.__class__.__name__}_{self.method}_{self.cv_frac}_{self.score_frac}_{self.iterations}_{self.forward}.json'
                else:
                    logs_path = logs_path / f'{self.estimator.__class__.__name__}_{self.method}_{self.cv_frac}_{self.score_frac}_{self.iterations}_{self.forward}.json'
                # check if  the json file exists
                if logs_path.exists():
                    with open(logs_path, 'r') as f:
                        logs = json.load(f)
                    i = len(logs)
                else:
                    i = 0
                    logs = {}
                print(f"Continue from iteration {i}")
            # curr_best_score = -np.inf
            # next_best_score = -np.inf
            if self.forward:
                if logs:
                    curr_best_score = logs[str(i)]['next_best_score']
                    next_best_score = curr_best_score
                    best_column_set = logs[str(i)]['best_column_set'] + [logs[str(i)]['selected_column']]
                else:
                    best_column_set = []
                    curr_best_score = -np.inf
                    next_best_score = -np.inf
            else:
                if logs:
                    curr_best_score = logs[str(i)]['next_best_score']
                    next_best_score = curr_best_score
                    best_column_set = logs[str(i)]['best_column_set']
                    best_column_set.remove(logs[str(i)]['selected_column'])
                else:
                    best_column_set = list(X.columns)
                    scores = []
                    for seed in range(self.iterations):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.cv_frac, random_state=seed, stratify=y)
                        self.estimator.fit(X_train, y_train.values.ravel())
                        y_pred = self.estimator.predict_proba(X_test)
                        # Take y_true and y_pred from the X_test.shape[0] * score_frac rows with the highest probability
                        df_r = pd.DataFrame({'y': y_test.values.ravel(), 'y_pred': y_pred[:, 1]})
                        df_r = df_r.sort_values('y_pred', ascending=False).reset_index(drop=True)[:int(X_test.shape[0] * self.score_frac)]
                        # _y_pred - np.array of ones
                        _y_pred = np.ones(len(df_r))
                        _y = np.array(df_r['y'])
                        score = self._get_effectiveness_score(_y, _y_pred, len(best_column_set))
                        scores.append(score)
                    curr_best_score = np.mean(scores)
                    next_best_score = curr_best_score
            # Logs with keys: 'curr_best_score', 'next_best_score', 'best_column_set', 'selected_column', 'selected_column', 'selected_column_score', 'columns', 'scores'
                # logs = {'i':[], 'curr_best_score': [], 'next_best_score': [], 'best_column_set': [], 'selected_column': [], 'selected_column_score': [], 'columns': [], 'scores': []}
            while next_best_score >= curr_best_score:
                if self.forward and len(best_column_set) == len(X.columns):
                    break
                elif not self.forward and len(best_column_set) == 0:
                    break
                df_scores = pd.DataFrame(columns=['column', 'score'])
                for col in X.columns:
                    if self.forward:
                        if col in best_column_set:
                            continue
                        column_set = best_column_set + [col]
                    else:
                        if col not in best_column_set:
                            continue
                        column_set = best_column_set.copy()
                        column_set.remove(col)
                    X_temp = X[column_set]
                    scores = []
                    for seed in range(self.iterations):
                        X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=self.cv_frac, random_state=seed, stratify=y)
                        self.estimator.fit(X_train, y_train.values.ravel())
                        y_pred = self.estimator.predict_proba(X_test)
                        # Take y_true and y_pred from the X_test.shape[0] * score_frac rows with the highest probability
                        df_r = pd.DataFrame({'y': y_test.values.ravel(), 'y_pred': y_pred[:, 1]})
                        df_r = df_r.sort_values('y_pred', ascending=False).reset_index(drop=True)[:int(X_test.shape[0] * self.score_frac)]
                        # _y_pred - np.array of ones
                        _y_pred = np.ones(len(df_r))
                        _y = np.array(df_r['y'])
                        score = self._get_effectiveness_score(_y, _y_pred, len(column_set))
                        scores.append(score)
                    mean_score = np.mean(scores)
                    df_scores = pd.concat([df_scores, pd.DataFrame({'column': [col], 'score': [mean_score]})], ignore_index=True)
                next_best_score = df_scores['score'].max()
                if self.verbose == 1:
                    print("--------------------")
                    print(f"Current best score: {curr_best_score}")
                    print(f"Current selected features: {best_column_set}")
                    print(f"Head of scores dataframe:\n {df_scores.sort_values('score', ascending=False).head()}")
                if self.logs:
                    i += 1
                    log = {}
                    log['curr_best_score'] = 0 if curr_best_score == -np.inf else curr_best_score
                    log['next_best_score'] = next_best_score
                    log['best_column_set'] = best_column_set
                    log['columns'] = list(df_scores.sort_values('score', ascending=False)['column'].values)
                    log['scores'] = list(df_scores.sort_values('score', ascending=False)['score'].values)

                if next_best_score >= curr_best_score:
                    if self.method == 'top_1':
                        selected_column = df_scores[df_scores['score'] == next_best_score]['column'].values[0]
                        selected_column_score = df_scores[df_scores['score'] == next_best_score]['score'].values[0]
                    elif self.method == 'random_improvement':
                        if curr_best_score == -np.inf:
                            selected_column = df_scores[df_scores['score'] == next_best_score]['column'].values[0]
                            selected_column_score = df_scores[df_scores['score'] == next_best_score]['score'].values[0]
                        else:
                            df_scores_improvement = df_scores[df_scores['score'] >= curr_best_score]
                            # Select random column from the columns with score >= curr_best_score with probability proportional to the score
                            offset = min(df_scores_improvement['score'])
                            positiveweights = [x - offset + 1 for x in df_scores_improvement['score']]
                            df_scores_improvement['score_weights'] = positiveweights
                            selected_column = df_scores_improvement.sample(random_state=42, weights='score_weights')['column'].values[0]
                            # selected_column = df_scores_improvement.sample(random_state=42)['column'].values[0]
                            selected_column_score = df_scores_improvement[df_scores_improvement['column'] == selected_column]['score'].values[0]
                    if self.verbose == 1:
                        print(f"Selected column: {selected_column}")
                        print(f"Selected column score: {selected_column_score}")
                    if self.logs:
                        log['selected_column'] = selected_column
                        log['selected_column_score'] = selected_column_score
                        logs[i] = log
                    if self.forward:
                        best_column_set = best_column_set + [selected_column]
                    else:
                        best_column_set.remove(selected_column)
                    # best_column_set.append(selected_column)
                    curr_best_score = selected_column_score
                if self.logs:
                    # If not exist, create logs dictionary using pathlib
                    # logs_path = pathlib.Path('logs')
                    # logs_path.mkdir(parents=True, exist_ok=True)
                    # if self.prefix:
                    #     logs_path = logs_path / f'{self.prefix}_{self.estimator.__class__.__name__}_{self.method}_{self.cv_frac}_{self.score_frac}_{self.iterations}_{self.forward}.json'
                    # else:
                    #     logs_path = logs_path / f'{self.estimator.__class__.__name__}_{self.method}_{self.cv_frac}_{self.score_frac}_{self.iterations}_{self.forward}.json'
                    with open(logs_path, 'w') as f:
                        json.dump(logs, f)
            self.features = best_column_set
            self.columns = X.columns
            # if self.logs:
            #     # If not exist, create logs dictionary using pathlib
            #     logs_path = pathlib.Path('logs')
            #     logs_path.mkdir(parents=True, exist_ok=True)
            #     if self.prefix:
            #         logs_path = logs_path / f'{self.prefix}_{self.estimator.__class__.__name__}_{self.method}_{self.cv_frac}_{self.score_frac}_{self.iterations}_{self.forward}.json'
            #     else:
            #         logs_path = logs_path / f'{self.estimator.__class__.__name__}_{self.method}_{self.cv_frac}_{self.score_frac}_{self.iterations}_{self.forward}.json'
            #     with open(logs_path, 'w') as f:
            #         json.dump(logs, f)
                
        else:
            raise ValueError('Invalid method: must be top_1 or random_improvement!')

    def transform(self, X):
        return X[self.features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self):
        return self.features
    
    def print_support(self):
        print(f"Number of features: {len(self.features)}")
        print(f"Features: {self.features}")

def remove_correlated_features(df: pd.DataFrame, threshold: float = 0.75):
    """
    Function to remove correlated features from the dataframe

    Parameters:
        df: pd.DataFrame : The input dataframe
        threshold: float : The threshold for the correlation value

    Returns:
        df: pd.DataFrame : The dataframe with the correlated features removed
    """

    corr_matrix = df.corr().abs()
    while True:
        # Select the biggest correlation to check if stop, exclude the diagonal
        # Set 0 for the diagonal elements
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr = corr_matrix.max().max()
        if max_corr <= threshold:
            break
        idx = np.argmax(corr_matrix.max())
        # Remove the column with the biggest count of correlations
        df = df.drop(df.columns[idx], axis=1)
        corr_matrix = df.corr().abs()
    return df 

# def effectiveness_score(y_test, y_pred, n_features, tp_reward=10, n_features_cost=200):
#     tp = sum((y_test == 1) & (y_pred == 1))
#     effectiveness = tp * tp_reward - n_features_cost * n_features
#     return effectiveness

def effectiveness_score(y_true, y_pred, n_features, tp_reward=10, n_features_cost=200, score_frac=0.2):
        tp = sum((y_true == 1) & (y_pred == 1))
        effectiveness = tp * tp_reward - score_frac * n_features_cost * n_features
        return effectiveness