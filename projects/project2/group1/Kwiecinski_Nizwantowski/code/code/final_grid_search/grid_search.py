import sys
sys.path.append('..')

import argparse
from metrics import default_competition_metric
from metrics import make_competition_scorer, competition_scoring, ColumnSelector


import numpy as np
import pandas as pd
import json
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer

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
    parser.add_argument("--features-subset", type=Path, default=None)
    parser.add_argument("--model", type = str, choices = ["rf", "svm", "xgboost"])
    parser.add_argument("--n-trails", type=int, default=500)
    parser.add_argument("--output-path", type=Path, default="boruta.csv")
    args = parser.parse_args()

    
    n_iter = args.n_trails


    if args.features_subset is not None:
        with open(args.features_subset, 'rb') as file:
            subset_of_features = json.load(file)
    else:
        subsets_of_features = get_all_subsets(np.load(args.features_path))


    pipe = Pipeline([
            ("feature_selection", ColumnSelector()),
            ('scaler', StandardScaler())
    ])

    if args.model == "xgboost":
        model_step = ('model', xgb.XGBClassifier(random_state=44))
        
        grid = {
            'feature_selection__columns': subset_of_features,
            'model__n_estimators': [100, 500, 1000],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__colsample_bytree': [0.5, 0.75, 1]
       }   
    if args.model == "svm":
        model_step =     ('model', SVC(probability=True, random_state=44))
        grid = {
            'feature_selection__columns': subset_of_features,
            'model__C': [0.1, 1, 10, 100],
            'model__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'model__kernel': ['rbf', 'poly', 'sigmoid', 'linear']
        }
    if args.model == "rf":
        model_step =     ('model', RandomForestClassifier(n_estimators=1000, random_state=44))
        grid = {
            'feature_selection__columns': subset_of_features,
            'model__bootstrap': [True, False],
            'model__max_depth': [10, 20, 40, 60, 80, 100, None],
            'model__max_features': ['log2', 'sqrt'],
            'model__min_samples_leaf': [1, 2, 4],
            'model__min_samples_split': [2, 5, 10],
            'model__n_estimators': [200, 400, 600, 800, 1000, 1200, 1600, 2000]    
        }   



    pipe = Pipeline(pipe.steps + [model_step])
    

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









