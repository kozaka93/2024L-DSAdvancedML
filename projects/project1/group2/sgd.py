from itertools import combinations
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split


class LogisticRegressionSGD:
    def __init__(self, learning_rate: float = 0.0001, epochs: int = 500, interact: bool = False,
                 early_stopping: bool = True, patience: int = 10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.interact = interact
        self.early_stopping = early_stopping
        self.patience = patience

        self.coefficients = None
        self.best_loss = np.inf
        self.counter = 0
        self.best_coefficients = None
        self.best_intercept = None
        self.best_accuracy = 0
        self.epochs_no_improve = 0
        self.intercept = 1e-8

        self.train_losses = []
        self.valid_losses = []
        self.train_balanced_accuracies = []
        self.valid_balanced_accuracies = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame = None,
            y_valid: pd.Series = None) -> None:
        if self.interact:
            X_train = self.interaction(X_train)
            if X_valid is not None:
                X_valid = self.interaction(X_valid)

        if self.early_stopping and (X_valid is None or y_valid is None):
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=17)

        num_samples, num_features = X_train.shape
        self.coefficients = np.random.randn(num_features) / np.sqrt(num_features)

        for e in range(self.epochs):
            for i in range(num_samples):
                y_pred = self.predict_proba(X_train.iloc[i, :], inter=True)
                gradient_coefficient = np.dot((y_pred - y_train.iloc[i]), X_train.iloc[i, :])
                gradient_intercept = np.sum(y_pred - y_train.iloc[i])

                self.coefficients -= self.learning_rate * gradient_coefficient
                self.intercept -= self.learning_rate * gradient_intercept
            
            y_pred_train = self.predict(X_train, inter=True)
            train_balanced_accuracy = balanced_accuracy_score(y_train, y_pred_train, )
            self.train_balanced_accuracies.append(train_balanced_accuracy)

            train_loss = log_loss(y_train, y_pred_train)
            self.train_losses.append(train_loss)

            if self.early_stopping:
                y_pred_valid = self.predict(X_valid, inter=True)
                self.valid_balanced_accuracies.append(balanced_accuracy_score(y_valid, y_pred_valid))
                valid_loss = log_loss(y_valid, y_pred_valid)
                self.valid_losses.append(valid_loss)

                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                    self.best_coefficients = self.coefficients.copy()
                    self.best_intercept = self.intercept
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                    if self.epochs_no_improve == self.patience:
                        self.coefficients = self.best_coefficients
                        self.intercept = self.best_intercept
                        print(f'Stopping early at epoch {e}. Best loss: {self.best_loss}.')
                        break

    def predict_proba(self, X: pd.DataFrame, inter: bool = False) -> np.ndarray:
        if self.interact and not inter:
            X = self.interaction(X)
        return np.array(expit(np.dot(X, self.coefficients) + self.intercept), dtype=np.float32)

    def predict(self, X: pd.DataFrame, inter: bool = False) -> np.ndarray:
        return np.array([1 if i > 0.5 else 0 for i in self.predict_proba(X, inter)],dtype=np.int64)

    def score(self, X: pd.DataFrame, y: pd.Series, inter: bool = False) -> float:
        predictions = self.predict(X, inter)
        return balanced_accuracy_score(predictions, y)

    def interaction(self, X: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for col in combinations(X.columns, 2):
            new_cols[f"{col[0]}_{col[1]}"] = X[col[0]] * X[col[1]]
        return pd.concat([X, pd.DataFrame(new_cols)], axis=1)
