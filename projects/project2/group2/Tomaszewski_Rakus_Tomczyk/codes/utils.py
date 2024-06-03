import pandas as pd
import numpy as np
from pathlib import Path


def load_data(directory_path):
    data_path = Path(directory_path)
    X = pd.read_csv(data_path / 'X_train.txt', sep=' ', header=None)
    y = pd.read_csv(data_path / 'y_train.txt', header=None).rename(columns={0: 'target'})
    X_final = pd.read_csv(data_path / 'X_test.txt', sep=' ', header=None)
    return X, y.target, X_final


def reward(y_true, y_pred, n_vars):
    # calculate reward if there were 1000 customers
    cust = ((y_true == 1) & (y_pred == 1)).sum()
    print(f'Customers who took offer: {cust}/{(y_true == 1).sum()} - '
          f'{cust/(y_true == 1).sum()*100:.2f}% of all who would')
    earned = int(cust/(y_true == 1).sum()*1000) * 10
    lost = n_vars * 200
    print(f'Money gained if there were a 1000 possible customers: {earned},'
          f' money lost: {lost}, net gain: {earned - lost}')
    return earned - lost


def evaluate_model(model, X_train, y_train, X_test, y_test, n_vars=None):
    if n_vars is None:
        n_vars = X_test.shape[-1]
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f'Train accuracy: {np.mean(y_train_pred == y_train):.3f},'
          f' test accuracy: {np.mean(y_test_pred == y_test):.3f}')
    return reward(y_test, y_test_pred, n_vars)


def generate_interactions(X):
    new_variables = []
    for i in range(X.shape[-1]):
        for j in range(i+1, X.shape[-1]):
            new_variables.append(X[:, i] * X[:, j])
        # square
        new_variables.append(X[:, i]**2)
    new_variables = np.array(new_variables).T
    return np.hstack((X, new_variables))


def run_train_eval(model, feature_selector, X_train, y_train, X_test, y_test):
    X_train = feature_selector.fit_transform(X_train, y_train)
    X_test = feature_selector.transform(X_test)
    n_vars = len(feature_selector.selected_features)
    print(f'{n_vars} columns: {feature_selector.selected_features}')
    model.fit(X_train, y_train)
    return evaluate_model(model, X_train, y_train, X_test, y_test, n_vars)

def reward_20_percent(y_true, y_pred, n_vars, print_ = True):
    max_customers_to_chose = int(0.2 * y_true.shape[0])
    number_of_positive = ((y_true == 1) & (y_pred == 1)).sum()
    if print_:
        print(f'Number of positive: {number_of_positive} \ {max_customers_to_chose}')
    reward = number_of_positive * 10 - n_vars * 200
    if print_:
        print(f'Reward: {reward}')
    reward_if_1000 = number_of_positive * 10 * 1000 / max_customers_to_chose - n_vars * 200
    if print_:
        print(f'Reward if there were 1000 customers: {reward_if_1000}')
    return reward_if_1000

def evaluate_model_20_percent(model, X_test, y_test, n_vars=None, print_ = True):
    max_customers_to_chose = int(0.2 * X_test.shape[0])
    if n_vars is None:
        n_vars = X_test.shape[-1]
    proba = model.predict_proba(X_test)
    ind = np.argpartition(proba[:, 1], -max_customers_to_chose)[-max_customers_to_chose:]
    y_pred = np.zeros_like(y_test)
    y_pred[ind] = 1
    return reward_20_percent(y_test, y_pred, n_vars, print_ = print_)


def run_train_eval_20_percent(model, feature_selector, X_train, y_train, X_test, y_test, print_ = True):
    X_train = feature_selector.fit_transform(X_train, y_train)
    X_test = feature_selector.transform(X_test)
    n_vars = len(feature_selector.selected_features)
    if print_:
        print(f'{n_vars} columns: {feature_selector.selected_features}')
    model.fit(X_train, y_train)
    return evaluate_model_20_percent(model, X_test, y_test, n_vars, print_ = print_)
