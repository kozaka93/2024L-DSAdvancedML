import numpy as np
import pandas as pd

import logistic_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

aids = pd.read_csv('data/aids.csv')

y = aids['target']
X = aids.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

lr = logistic_regression.LogisticRegressor(descent_algorithm="iwls")

lr.fit(X_train, y_train, max_num_epoch=10000, tolerance = 1e-6, verbose=True)



sklearn_lr = LogisticRegression(max_iter=10000, tol=1e-6, solver='lbfgs')
sklearn_lr.fit(X_train, y_train)
print(sklearn_lr.score(X, y))

y_pred = sklearn_lr.predict(X_test)

print(f"Our implementation\n{lr.confusion_matrix(X_test, y_test)}")
print(f"Sklearn implementation\n{confusion_matrix(y_test, y_pred)}")

print(f"Our implementation\n{lr.balanced_accuracy(X_test, y_test)}")
print(f"Sklearn implementation\n{balanced_accuracy_score(y_pred, y_test)}")
