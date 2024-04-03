"""
gd.py

Gradient Descent (GD) optimizer implementation
"""

from typing import Optional

import numpy as np
from tqdm import tqdm

from .optim import Optimizer
from .util import dlogistic, log_likelihood, make_batches


# pylint: disable=invalid-name
class GD(Optimizer):
    """Stochastic Gradient Descent optimizer implementation
    with logistic loss for binary classification {0,1}"""

    def __init__(
        self,
        learning_rate: float,
        n_epoch: int = 1,
        batch_size: int = 1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps1: float = 1e-8,
        eps2: float = 1e-1,
        w_init: Optional[np.ndarray] = None,
        tolerance: int = 5,
    ) -> None:
        """
        Initializes GD optimizer with the given parameters.

        Arguments:
            learning_rate : Float value of step size to take.
            n_epoch : Maximum number of epochs, after which to stop
            batch_size : Size of the batch to use for each iteration
            w_init: Initial weights, if None precomputed from the normal equation
            tolerance: Number of epochs to wait before stopping if no improvement

        Returns:
            history : A list containing the loss value at each iteration
            best_w : The best weights corresponding to the best loss value
        """
        super().__init__()
        self.lr = learning_rate
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps1 = eps1
        self.eps2 = eps2
        self.w_init = w_init
        self.tolerance = tolerance

    def fit(self, X: np.ndarray, y: np.ndarray) -> tuple[list[float], np.ndarray]:
        """
        Runs the SGD optimizer on the given data.

        Arguments:
            X : Array with predictors
            y : Target array

        Returns:
            loss_history : A list containing the loss value at each iteration
            global_best_weights : The best weights corresponding to the best loss value
        """
        self.reset()  # resets history and best weights
        if self.w_init is None:
            self.w_init = (
                np.linalg.inv(X.T @ X + np.eye(X.shape[1]) * self.eps1) @ X.T @ y
            ).T[0]
        best_w = self.w_init.copy()
        self._global_best_weights = best_w
        best_log_like = float("inf")
        log_like = log_likelihood(X, y, np.expand_dims(best_w, 1))
        self._loss_history.append(log_like)

        early_stop = False
        no_change_counter = 0
        for _ in tqdm(range(self.n_epoch), "SGD"):
            batches = make_batches(X, y, self.batch_size)
            for j in range(len(batches[0])):
                X_sample = batches[0][j]
                Y_sample = batches[1][j]

                # compute loss gradient (J) and update weights
                J = dlogistic(X_sample, Y_sample, W=best_w)
                best_w = best_w - self.lr * J
            log_like = log_likelihood(X, y, np.expand_dims(best_w, 1))
            self._loss_history.append(log_like)
            if log_like < best_log_like - self.eps2:
                best_log_like = log_like
                self._global_best_weights = best_w
                no_change_counter = 0
            else:
                no_change_counter += 1
            if no_change_counter > self.tolerance:
                early_stop = True
                break
        if not early_stop:
            self._global_best_weights = best_w

        return self.loss_history, self.global_best_weights
