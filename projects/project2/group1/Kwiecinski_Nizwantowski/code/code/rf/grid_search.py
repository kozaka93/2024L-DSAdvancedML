import sys
sys.path.append('..')

import argparse
from metrics import default_competition_metric
from metrics import make_competition_scorer, competition_scoring, ColumnSelector


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

import pickle

np.random.seed(44)
from pathlib import Path


X_train = np.load('../../data/x_train.npy')
y_train = np.load('../../data/y_train.npy')
X_val = np.load('../../data/x_val.npy')
y_val = np.load('../../data/y_val.npy')



def get_all_subsets(input_list):
    subsets = []
    n = len(input_list)
    for i in range(1, 2**n):
        subset = [input_list[j] for j in range(n) if (i & (1 << j))]
        subsets.append(subset)
    return subsets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=Path, default="boruta.npy")
    parser.add_argument("--n-trails", type=int, default=500)
    parser.add_argument("--output-path", type=Path, default="boruta.csv")
    args = parser.parse_args()

    features_to_train = np.load(args.features_path)
    n_iter = args.n_trails

    
    pipe = Pipeline([
        ("feature_selection", ColumnSelector()),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=44))
    ])

    
    grid = {
        'feature_selection__columns': get_all_subsets(features_to_train),
        'model__bootstrap': [True, False],
        'model__max_depth': [10, 20, 40, 60, 80, 100, None],
        'model__max_features': ['auto', 'sqrt'],
        'model__min_samples_leaf': [1, 2, 4],
        'model__min_samples_split': [2, 5, 10],
        'model__n_estimators': [200, 400, 600, 800, 1000, 1200, 1600, 2000]    
    }

    grid_search = RandomizedSearchCV(pipe, grid, cv=5, scoring=competition_scoring, verbose=2, n_iter = n_iter)

    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)
    
    pipe.set_params(**grid_search.best_params_)
    pipe.fit(X_train, y_train)


    y_proba = pipe.predict_proba(X_val)[:, 1]
    print(f"Score of the best params on the validation set: {competition_scoring(pipe, X_val, y_val, scale_metric=True)}")

    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv(args.output_path)
    print(f"Saved outputs to file {args.output_path}")









