import numpy as np
import pandas as pd
from typing import Union
import time

from scipy.special import expit as sigmoid


def transform_data_type_to_np(
    X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame, pd.Series]
):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
        y = y.to_numpy().T
    return X, y


def calculate_batch_size(X: np.ndarray, batch_size: int, batch_fraction: float):
    assert isinstance(batch_size, int), "batch_size must be an integer"
    if batch_fraction is not None:
        assert 0 < batch_fraction <= 1, "batch_fraction must be between 0 and 1"
        batch_size = int(X.shape[0] * batch_fraction)
    return batch_size


def stop_criterion(gradient, tolerance, epoch, starting_time):
    if time.perf_counter() - starting_time > 60:
        print(f"Time run our at epoch {epoch}.")
        return True
    elif np.linalg.norm(gradient, ord=np.inf) < tolerance:
        print(f"Early stopping criterion reached at epoch {epoch}.")
        return True
    return False


def mini_batch_gd(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame, pd.Series],
    initial_solution: np.ndarray,
    regressor: object,
    calculate_gradient: callable,
    learning_rate: float = 0.001,
    max_num_epoch: int = 500,
    tolerance: float = 1e-6,
    batch_size: int = 32,
    batch_fraction: float = None,
    verbose: bool = False,
):
    """
    Performs mini batch gradient descent optimization.

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_weights: Initial solution for optimization.
    - calculate_gradient: Function to calculate the gradient.
    - learning_rate: Learning rate for updating the solution (default: 0.01).
    - max_num_iters: Maximum number of iterations (default: 1000).
    - tolerance: Tolerance for the stopping criterion (default: 1e-6). Stops if the L inf norm of the gradient is below this value.
    - batch_size: Size of the mini batch (default: 1).
    - batch_fraction: Fraction of the data to use in each mini batch (default: None).
    - verbose: Whether to print the solution at each iteration (default: False).

    Returns:
    - The optimized solution.
    """

    # initialization
    X, y = transform_data_type_to_np(X, y)
    current_solution = initial_solution

    batch_size = calculate_batch_size(X, batch_size, batch_fraction)
    iterations = int(X.shape[0] / batch_size)
    loss_after_epoch = [regressor.predict_and_calculate_loss(X, y, current_solution)]
    starting_time = time.perf_counter()
    for epoch in range(max_num_epoch):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)

        X, y = X[shuffled_idx], y[shuffled_idx]
        for idx in range(iterations):
            X_selected, y_selected = (
                X[idx * batch_size : (idx + 1) * batch_size],
                y[idx * batch_size : (idx + 1) * batch_size],
            )
            gradient = calculate_gradient(X_selected, y_selected, current_solution)
            gradient = gradient / batch_size
            current_solution = current_solution - learning_rate * gradient

        loss_after_epoch.append(
            regressor.predict_and_calculate_loss(X, y, current_solution)
        )
        if verbose:
            print(f"Epoch {epoch}, solution:", current_solution)

        if stop_criterion(gradient, tolerance, epoch, starting_time):
            break
    return current_solution, loss_after_epoch


def newton(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame, pd.Series],
    initial_solution: np.ndarray,
    regressor: object,
    calculate_gradient: callable,
    calculate_hessian: callable,
    max_num_epoch: int = 500,
    tolerance: float = 1e-6,
    verbose: bool = False,
):
    """
    Performs Newton method optimization using second order derivatives

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_weights: Initial solution for optimization.
    - calculate_gradient: Function to calculate the gradient.
    - calculate_hessian: Function to calculate the Hessian.
    - max_num_epoch: Maximum number of iterations (default: 1000).
    - tolerance: Tolerance for the stopping criterion (default: 1e-6). Stops if the L inf norm of the gradient is below this value.
    - verbose: Whether to print the solution at each iteration (default: False).


    Returns:
    - The optimized solution.
    """

    # initialization
    X, y = transform_data_type_to_np(X, y)
    current_solution = initial_solution
    loss_after_epoch = [regressor.predict_and_calculate_loss(X, y, current_solution)]
    starting_time = time.perf_counter()

    for epoch in range(max_num_epoch):
        gradient = calculate_gradient(X, y, current_solution)
        hessian = calculate_hessian(X, y, current_solution)
        current_solution = current_solution - np.linalg.inv(hessian) @ gradient

        loss_after_epoch.append(
            regressor.predict_and_calculate_loss(X, y, current_solution)
        )
        if verbose:
            print(f"Epoch {epoch}, solution:", current_solution)

        if stop_criterion(gradient, tolerance, epoch, starting_time):
            break
    return current_solution, loss_after_epoch


def iwls(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame, pd.Series],
    initial_solution: np.ndarray,
    regressor: object,
    max_num_epoch: int = 500,
    tolerance: float = 1e-6,
    epsilon: float = 1e-3,
    verbose: bool = False,
):
    """
    Performs iteratively reweighed least squares optimization. Uses the log-likelihood loss.

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_solution: Initial solution for optimization.
    - max_num_epoch: Maximum number of iterations (default: 1000).
    - tolerance: Tolerance for the stopping criterion (default: 1e-6). Stops if the L inf norm of the gradient is below this value.
    - verbose: Whether to print the solution at each iteration (default: False).

    Returns:
    - The optimized solution.
    """

    # initialization
    X, y = transform_data_type_to_np(X, y)
    current_solution = initial_solution
    loss_after_epoch = [regressor.predict_and_calculate_loss(X, y, current_solution)]
    starting_time = time.perf_counter()

    for epoch in range(max_num_epoch):
        P = sigmoid(X @ current_solution)
        W = np.diag(P * (1 - P))

        # prevent singular matrix
        W = W + epsilon * np.eye(W.shape[0])
        Z = X @ current_solution + np.linalg.inv(W) @ (y - P)
        H = X.T @ W @ X

        # prevent singular matrix
        H = H + epsilon * np.eye(H.shape[0])
        current_solution = np.linalg.inv(H) @ X.T @ W @ Z

        # maybe propose better stopping criterion here
        gradient = -X.T @ (y - P)

        loss_after_epoch.append(
            regressor.predict_and_calculate_loss(X, y, current_solution)
        )
        if verbose:
            print(f"Epoch {epoch}, solution:", current_solution)
            print(f"norm: {np.linalg.norm(gradient, ord=np.inf)}")
            print(f"Gradient: {gradient}")

        if stop_criterion(gradient, tolerance, epoch, starting_time):
            break
    return current_solution, loss_after_epoch


def sgd(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame, pd.Series],
    initial_solution: np.ndarray,
    regressor: object,
    calculate_gradient: callable,
    learning_rate: float = 0.001,
    max_num_epoch: int = 500,
    tolerance: float = 1e-6,
    verbose: bool = False,
):
    """
    Performs stochastic gradient descent optimization.

    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_solution: Initial solution for optimization.
    - calculate_gradient: Function to calculate the gradient.
    - learning_rate: Learning rate for updating the solution (default: 0.001).
    - max_num_iters: Maximum number of iterations (default: 1000).
    - tolerance: Tolerance for the stopping criterion (default: 1e-6). Stops if the L inf norm of the gradient is below this value.
    - verbose: Whether to print the solution at each iteration (default: False).

    Returns:
    - The optimized solution.
    """

    # initialization
    X, y = transform_data_type_to_np(X, y)
    current_solution = initial_solution
    loss_after_epoch = [regressor.predict_and_calculate_loss(X, y, current_solution)]
    starting_time = time.perf_counter()

    for epoch in range(max_num_epoch):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        X, y = X[shuffled_idx], y[shuffled_idx]
        grad_sum = np.zeros_like(current_solution)
        for X_selected, y_selected in zip(X, y):
            gradient = calculate_gradient(X_selected, y_selected, current_solution)
            grad_sum += gradient
            current_solution = current_solution - learning_rate * gradient
        loss_after_epoch.append(
            regressor.predict_and_calculate_loss(X, y, current_solution)
        )
        if verbose:
            print(f"Epoch {epoch}, solution: {current_solution}")

        gradient = grad_sum / N
        if stop_criterion(gradient, tolerance, epoch, starting_time):
            break
    return current_solution, loss_after_epoch


def adam(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.DataFrame, pd.Series],
    initial_solution: np.ndarray,
    regressor: object,
    calculate_gradient: callable,
    learning_rate: float = 0.001,
    momentum_decay: float = 0.9,
    squared_gradient_decay: float = 0.99,
    max_num_epoch: int = 500,
    tolerance: float = 1e-6,
    batch_size: int = 32,
    batch_fraction: float = None,
    epsilon: float = 1e-8,
    verbose: bool = False,
):
    """
    Performs optimization with adam algorithm.
    accuracy/ balanced accuracy/ f1 score/ precision/ recall/ roc auc score/ log loss/
    Parameters:
    - X: Input data.
    - y: Target labels.
    - initial_solution: Initial solution for optimization.
    - calculate_gradient: Function to calculate the gradient.
    - learning_rate: Learning rate for updating the solution (default: 0.001).
    - momentum_decay: Decay rate for the momentum (default: 0.9).
    - squared_gradient_decay: Decay rate for the squared gradient (default: 0.99).
    - max_num_iters: Maximum number of iterations (default: 1000).
    - tolerance: Tolerance for the stopping criterion (default: 1e-6). Stops if the L inf norm of the gradient is below this value.
    - batch_size: Size of the mini batch (default: 1).
    - batch_fraction: Fraction of the data to use in each mini batch (default: None).
    - epsilon: Small value to avoid division by zero (default: 1e-8).
    - verbose: Whether to print the solution at each iteration (default: False).

    Returns:
    - The optimized solution.
    """

    # initialization
    X, y = transform_data_type_to_np(X, y)
    current_solution = initial_solution
    momentum = np.zeros_like(initial_solution)
    squared_gradients = np.zeros_like(initial_solution)
    counter = 0
    loss_after_epoch = [regressor.predict_and_calculate_loss(X, y, current_solution)]
    starting_time = time.perf_counter()

    batch_size = calculate_batch_size(X, batch_size, batch_fraction)
    iterations = int(X.shape[0] / batch_size)

    for epoch in range(max_num_epoch):
        N, _ = X.shape
        shuffled_idx = np.random.permutation(N)
        X, y = X[shuffled_idx], y[shuffled_idx]
        for idx in range(iterations):
            X_selected, y_selected = (
                X[idx * batch_size : (idx + 1) * batch_size],
                y[idx * batch_size : (idx + 1) * batch_size],
            )
            gradient = calculate_gradient(X_selected, y_selected, current_solution)
            momentum = momentum_decay * momentum + (1 - momentum_decay) * gradient
            squared_gradients = (
                squared_gradient_decay * squared_gradients
                + (1 - squared_gradient_decay) * gradient**2
            )
            counter += 1

            # bias correction
            corrected_momentum = momentum / (1 - momentum_decay**counter)
            corrected_squared_gradients = squared_gradients / (
                1 - squared_gradient_decay**counter
            )

            current_solution = current_solution - learning_rate * corrected_momentum / (
                np.sqrt(corrected_squared_gradients) + epsilon
            )
        loss_after_epoch.append(
            regressor.predict_and_calculate_loss(X, y, current_solution)
        )
        if verbose:
            print(f"Epoch {epoch}, solution:", current_solution)

        if stop_criterion(gradient, tolerance, epoch, starting_time):
            break

    return current_solution, loss_after_epoch
