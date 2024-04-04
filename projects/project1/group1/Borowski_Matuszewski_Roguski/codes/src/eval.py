import time
from typing import Tuple, List, Any

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data_loader.DataLoader import DataLoader
from src.model.LogisticRegression import LogisticRegression
from src.optim.ADAM import ADAM
from src.optim.conditions import NoLogLikOrMaxIterCondition
from src.optim.IWLS import IWLS
from src.optim.SGD import SGD


def _prepare_data(
    loader: DataLoader, dataset: str, random_state: int, return_ones_col: bool = True
) -> tuple[
    np.array,
    np.array,
    np.array,
    np.array,
]:
    np.random.seed(random_state)

    x, y = loader[dataset]

    if return_ones_col:
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    return train_test_split(x, y, test_size=0.3, stratify=y, random_state=random_state)


def _scale_data(train_x: np.array, test_x: np.array) -> tuple[np.array, np.array]:
    scaler = StandardScaler()
    scaler.fit(train_x)

    return scaler.transform(train_x), scaler.transform(test_x)


def _return_dict(
    dataset: str,
    method: str,
    start_time: float,
    end_time: float,
    test_y: np.array,
    pred_y: np.array,
    random_state: int,
    num_iters: int = -1,
) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "time": end_time - start_time,
        "accuracy": balanced_accuracy_score(test_y, pred_y),
        "random_state": random_state,
        "iters": num_iters,
    }


def eval_iwls(
    loader: DataLoader,
    dataset: str,
    random_state: int,
    max_iterations: int = 500,
    patience: int = 5,
) -> tuple[dict, Any, Any]:
    train_x, test_x, train_y, test_y = _prepare_data(
        loader, dataset, random_state, return_ones_col=True
    )
    train_x, test_x = _scale_data(train_x, test_x)

    start = time.perf_counter()

    model = LogisticRegression()
    optim = IWLS(
        model,
        NoLogLikOrMaxIterCondition(max_iterations, patience),
    )
    model, logliks, accuracies = optim.optimize(train_x, train_y)

    end = time.perf_counter()

    return _return_dict(
        dataset=dataset,
        method="IWLS",
        start_time=start,
        end_time=end,
        test_y=test_y,
        pred_y=model.predict(test_x),
        random_state=random_state,
        num_iters=optim.stop_condition.epoch,
    ), logliks, accuracies


def eval_sgd(
    loader: DataLoader,
    dataset: str,
    random_state: int,
    max_iterations: int = 500,
    patience: int = 5,
    learning_rate: float = 0.1,
    batch_size: int = 1,
) -> tuple[dict, Any, Any]:
    train_x, test_x, train_y, test_y = _prepare_data(
        loader, dataset, random_state, return_ones_col=True
    )
    train_x, test_x = _scale_data(train_x, test_x)

    start = time.perf_counter()

    model = LogisticRegression()
    optim = SGD(
        model,
        NoLogLikOrMaxIterCondition(max_iterations, patience),
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    model, logliks, accuracies = optim.optimize(train_x, train_y)

    end = time.perf_counter()

    return _return_dict(
        dataset=dataset,
        method="SGD",
        start_time=start,
        end_time=end,
        test_y=test_y,
        pred_y=model.predict(test_x),
        random_state=random_state,
        num_iters=optim.stop_condition.epoch,
    ), logliks, accuracies


def eval_adam(
    loader: DataLoader,
    dataset: str,
    random_state: int,
    max_iterations: int = 500,
    patience: int = 5,
    learning_rate: float = 0.01,
    batch_size: int = 1,
) -> tuple[dict, Any, Any]:
    train_x, test_x, train_y, test_y = _prepare_data(
        loader, dataset, random_state, return_ones_col=True
    )
    train_x, test_x = _scale_data(train_x, test_x)

    start = time.perf_counter()

    model = LogisticRegression()
    optim = ADAM(
        model,
        NoLogLikOrMaxIterCondition(max_iterations, patience),
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    model, logliks, accuracies = optim.optimize(train_x, train_y)

    end = time.perf_counter()

    return _return_dict(
        dataset=dataset,
        method="ADAM",
        start_time=start,
        end_time=end,
        test_y=test_y,
        pred_y=model.predict(test_x),
        random_state=random_state,
        num_iters=optim.stop_condition.epoch,
    ), logliks, accuracies


def eval_lda(
    loader: DataLoader,
    dataset: str,
    random_state: int,
) -> dict:
    train_x, test_x, train_y, test_y = _prepare_data(
        loader, dataset, random_state, return_ones_col=False
    )
    train_x, test_x = _scale_data(train_x, test_x)

    start = time.perf_counter()

    model = LinearDiscriminantAnalysis()
    model.fit(train_x, train_y)

    end = time.perf_counter()

    return _return_dict(
        dataset=dataset,
        method="LDA",
        start_time=start,
        end_time=end,
        test_y=test_y,
        pred_y=model.predict(test_x),
        random_state=random_state,
    )


def eval_qda(
    loader: DataLoader,
    dataset: str,
    random_state: int,
) -> dict:
    train_x, test_x, train_y, test_y = _prepare_data(
        loader, dataset, random_state, return_ones_col=False
    )
    train_x, test_x = _scale_data(train_x, test_x)

    start = time.perf_counter()

    model = QuadraticDiscriminantAnalysis(reg_param=0.1, tol=1e-8)
    model.fit(train_x, train_y)

    end = time.perf_counter()

    return _return_dict(
        dataset=dataset,
        method="QDA",
        start_time=start,
        end_time=end,
        test_y=test_y,
        pred_y=model.predict(test_x),
        random_state=random_state,
    )


def eval_tree(
    loader: DataLoader, dataset: str, random_state: int, param_grid: list = None
) -> dict:
    if param_grid is None:
        param_grid = {"max_depth": [4, 6, 10, 16, None]}

    train_x, test_x, train_y, test_y = _prepare_data(
        loader, dataset, random_state, return_ones_col=False
    )
    train_x, test_x = _scale_data(train_x, test_x)

    start = time.perf_counter()

    model = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=param_grid,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)

    end = time.perf_counter()

    return _return_dict(
        dataset=dataset,
        method="DecisionTree",
        start_time=start,
        end_time=end,
        test_y=test_y,
        pred_y=model.predict(test_x),
        random_state=random_state,
    )


def eval_random_forest(
    loader: DataLoader, dataset: str, random_state: int, param_grid: list = None
) -> dict:
    if param_grid is None:
        param_grid = {"max_depth": [4, 6, 10, 16, None]}

    train_x, test_x, train_y, test_y = _prepare_data(
        loader, dataset, random_state, return_ones_col=False
    )
    train_x, test_x = _scale_data(train_x, test_x)

    start = time.perf_counter()

    model = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=param_grid,
        n_jobs=-1,
    )
    model.fit(train_x, train_y)

    end = time.perf_counter()

    return _return_dict(
        dataset=dataset,
        method="RandomForest",
        start_time=start,
        end_time=end,
        test_y=test_y,
        pred_y=model.predict(test_x),
        random_state=random_state,
    )
