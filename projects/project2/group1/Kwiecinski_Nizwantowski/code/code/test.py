import numpy as np
from sklearn.metrics import make_scorer
from metrics import default_competition_metric
from sklearn.dummy import DummyClassifier

np.random.seed(42)

X_train = np.load('../data/X_train.npy')
y_train = np.load('../data/y_train.npy')

proba_competition_metric = lambda y_true, y_pred, k: default_competition_metric(y_true, y_pred, k=k, y_pred_proba=y_pred)

competition_scorer = lambda k: make_scorer(proba_competition_metric, greater_is_better=True, k = k)
cls = DummyClassifier(strategy='most_frequent')
cls.fit(X_train, y_train)

default_scorer(cls, X_train, y_train)
print(default_scorer(cls, X_train, y_train))