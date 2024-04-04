import numpy as np
import scipy
import pandas as pd
from itertools import combinations

from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, log_loss


class LogisticRegressionIRLS:
    def __init__(self, epochs: int = 500, interact: bool = False, early_stopping: bool = True, patience: int = 10):
        self.epochs = epochs
        self.coefficients = None
        self.interact = interact
        self.early_stopping = early_stopping
        self.patience = patience

        self.train_losses = []
        self.valid_losses = []
        self.train_balanced_accuracies = []
        self.valid_balanced_accuracies = []

        self.best_loss = np.inf
        self.best_coefficients = None
        self.best_bias = None
        self.counter = 0
        self.epochs_no_improve = 0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame = None,
            y_valid: pd.Series = None) -> None:
        if self.interact:
            X_train = self.interaction(X_train)
            if X_valid is not None:
                X_valid = self.interaction(X_valid)

        if self.early_stopping and (X_valid is None or y_valid is None):
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=17)

        num_samples, _ = X_train.shape
        mu = np.array([0.5] * num_samples, dtype=np.float32)

        for e in range(self.epochs):
            Z = self.logit(mu) + (y_train - mu) * self.logit_prime(mu)
            W = np.diag((1 / (self.logit_prime(mu) ** 2 * self.var(mu) + 1e-8)))
            self.coefficients = scipy.linalg.inv(np.array(X_train).T.dot(W).dot(np.array(X_train)))\
                .dot(np.array(X_train).T.dot(W).dot(Z))
            mu = self.predict_proba(X_train, inter=True)

            y_pred_train = self.predict(X_train, inter=True)
            self.train_balanced_accuracies.append(balanced_accuracy_score(y_train, y_pred_train))
            self.train_losses.append(log_loss(y_train, y_pred_train))

            if self.early_stopping:
                y_pred_valid = self.predict(X_valid, inter=True)
                self.valid_balanced_accuracies.append(balanced_accuracy_score(y_valid, y_pred_valid))
                valid_loss = log_loss(y_valid, y_pred_valid)
                self.valid_losses.append(valid_loss)

                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.best_coefficients = self.coefficients.copy()
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve == self.patience:
                        self.coefficients = self.best_coefficients
                        print(f'Stopping early at epoch {e}. Best loss: {self.best_loss}.')
                        break

    def predict_proba(self, X: pd.DataFrame, inter: bool = False) -> np.ndarray:
        if self.interact and not inter:
            X = self.interaction(X)
        return expit(np.array(X @ self.coefficients, dtype=np.float32))

    def predict(self, X: pd.DataFrame, inter: bool = False) -> np.ndarray:
        return np.array([1 if i > 0.5 else 0 for i in self.predict_proba(X, inter)], dtype=np.int64)

    def score(self, X: pd.DataFrame, y: pd.Series,  inter: bool = False) -> float:
        predictions = self.predict(X, inter)
        return balanced_accuracy_score(predictions, y)

    def interaction(self, X: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for col in combinations(X.columns, 2):
            new_cols[f"{col[0]}_{col[1]}"] = X[col[0]] * X[col[1]]
        new_df = pd.DataFrame(new_cols)
        result = pd.concat([X, new_df], axis=1)
        return result

    def logit(self, x):
        if isinstance(x, list):
            x = np.array(x)
        eps = 1e-8
        return np.log(x + eps / (1.0 - x + eps))

    def sigmoid(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return 1.0 / (1.0 + np.exp(-x))

    def var(self, x):
        if isinstance(x, list):
            x = np.array(x)
        return x * (1 - x)

    def logit_prime(self, x):
        if isinstance(x, list):
            x = np.array(x)
        eps = 1e-8
        return 1 / ((x + eps) * (1 - x + eps))