import numpy as np
from scipy.special import expit
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from itertools import combinations
import pandas as pd


class LogisticRegressionAdam:
    def __init__(self, learning_rate: float = 0.02, beta1: float = 0.9, beta2: float = 0.999, epochs: int = 500,
                 epsilon: float = 1e-8, early_stopping: bool = True, interact: bool = False, patience: int = 10):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epochs = epochs
        self.epsilon = epsilon
        self.early_stopping = early_stopping
        self.interact = interact
        self.patience = patience

        self.coefficients = None
        self.intercept = 1e-8
        self.t = 0
        self.best_loss = np.inf
        self.m_intercept = 1e-8
        self.v_intercept = 1e-8
        self.v_coefficients = None
        self.m_coefficients = None
        self.gradient_coefs = None
        self.gradient_intercept = None
        self.best_coefficients = None
        self.best_intercept = None
        self.epochs_no_improve = 0

        self.train_losses = []
        self.valid_losses = []
        self.valid_balanced_accuracies = []
        self.train_balanced_accuracies = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame = None, y_valid: pd.Series = None):
        if self.interact:
            X_train = self.interaction(X_train)
            if X_valid is not None:
                X_valid = self.interaction(X_valid)

        if self.early_stopping and (X_valid is None or y_valid is None):
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=17)

        num_samples, num_features = X_train.shape
        self.coefficients = np.random.randn(num_features)/np.sqrt(num_features)

        for e in range(self.epochs):
            for row in range(num_samples):
                self.compute_gradients(X_train.iloc[row, :], y_train.iloc[row])
                self.update_params()

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
        z = np.dot(X, self.coefficients.T) + self.intercept
        return expit(z)

    def predict(self, X: pd.DataFrame, inter: bool = False) -> np.ndarray:
        return np.array([1 if i > 0.5 else 0 for i in self.predict_proba(X, inter)], dtype=np.int64)

    def score(self, X: pd.DataFrame, y: pd.Series, inter: bool = False) -> float:
        return balanced_accuracy_score(self.predict(X, inter), y)

    def interaction(self, X: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for col in combinations(X.columns, 2):
            new_cols[f"{col[0]}_{col[1]}"] = X[col[0]] * X[col[1]]
        return pd.concat([X, pd.DataFrame(new_cols)], axis=1)

    def compute_gradients(self, X: pd.Series, y: float) -> None:
        predictions = self.predict_proba(X, inter=True)
        error = predictions - y
        self.gradient_coefs = np.dot(X.T, error)
        self.gradient_intercept = error

    def update_params(self) -> None:
        self.t += 1
        if self.m_coefficients is None or self.v_coefficients is None:
            self.m_coefficients = np.zeros_like(self.coefficients)
            self.v_coefficients = np.zeros_like(self.coefficients)

        self.m_coefficients = self.beta1 * self.m_coefficients + (1 - self.beta1) * self.gradient_coefs
        self.m_intercept = self.beta1 * self.m_intercept + (1 - self.beta1) * self.gradient_intercept
        self.v_coefficients = self.beta2 * self.v_coefficients + (1 - self.beta2) * np.square(self.gradient_coefs)
        self.v_intercept = self.beta2 * self.v_intercept + (1 - self.beta2) * np.square(self.gradient_intercept)

        m_corrected_w = self.m_coefficients / (1 - self.beta1 ** self.t)
        m_corrected_b = self.m_intercept / (1 - self.beta1 ** self.t)
        v_corrected_w = self.v_coefficients / (1 - self.beta2 ** self.t)
        v_corrected_b = self.v_intercept / (1 - self.beta2 ** self.t)

        self.coefficients -= self.learning_rate * m_corrected_w / (np.sqrt(v_corrected_w) + self.epsilon)
        self.intercept -= self.learning_rate * m_corrected_b / (np.sqrt(v_corrected_b) + self.epsilon)

