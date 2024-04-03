import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from typing import Literal, Union

from scipy.special import expit as sigmoid

from .optimizers import mini_batch_gd, iwls, adam, sgd, newton


class LogisticRegressor:
    slots = [
        "beta",
        "prob_threshold",
        "descent_algorithm",
        "include_intercept",
        "include_interactions",
        "convergence_rate",
        "feature_names",
    ]

    def __init__(
        self,
        descent_algorithm: Literal[
            "minibatch", "newton", "iwls", "adam", "sgd"
        ] = "minibatch",
        prob_threshold: float = 0.5,
        include_intercept: bool = True,
        include_interactions: bool = False,
    ):
        """
        Initialize the LogisticRegressor class.

        Parameters:
        - descent_algorithm (str, optional): The descent algorithm to use for optimization. Defaults to "minibatch". Options are "minibatch", "newton", "iwls", "adam", "sgd".
        - prob_threshold (float, optional): The probability threshold for classification, which determines w Defaults to 0.5.
        - include_intercept (bool, optional): Whether to include intercept term in the model. Defaults to True.
        - include_interactions (bool, optional): Whether to include interaction terms in the model. Defaults to False.
        """

        assert (
            0 <= prob_threshold <= 1
        ), "prob_threshold represents a probability and must be between 0 and 1"

        self.descent_algorithm = descent_algorithm
        self.prob_threshold = prob_threshold
        self.include_intercept = include_intercept
        self.include_interactions = include_interactions
        self.beta = None
        self.convergence_rate = None
        self.feature_names = None

    def random_init_weights(self, p: int):
        self.beta = np.random.standard_normal(p)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]):
        if self.beta is None:
            raise ValueError(
                "Model has not been trained yet, please train the model first"
            )

        if X.shape[1] != len(self.beta):
            # if there are no interaction or intercept terms, then we need to add them
            X_copy = self.create_data_frame(X)
        else:
            X_copy = X

        assert X_copy.shape[1] == len(
            self.beta
        ), "Number of features in X must match the length of beta, check if you passed a X of correct shape"

        return sigmoid(X_copy @ self.beta)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        return self.predict_proba(X) > self.prob_threshold

    def minus_log_likelihood(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ):
        if X.shape[1] != len(self.beta):
            # if there are no interaction or intercept terms, then we need to add them
            X_copy = self.create_data_frame(X)
        else:
            X_copy = X

        assert X_copy.shape[1] == len(
            self.beta
        ), "Number of features in X must match the length of beta, check if you passed a X of correct shape"

        weighted_input = X_copy @ self.beta
        L = np.sum(y * weighted_input - np.log(1 + np.exp(weighted_input)))
        return -L

    def loss(
        self,
        y: Union[np.ndarray, pd.DataFrame],
        y_hat_proba: Union[np.ndarray, pd.DataFrame],
    ):
        # log likelihood loss
        return -np.sum(y * np.log(y_hat_proba) + (1 - y) * np.log(1 - y_hat_proba))

    @staticmethod
    def loss_prime(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        beta: np.ndarray,
    ):
        """
        calculates the derivative of the loss function with respect to the beta
        """
        assert X.shape[-1] == len(
            beta
        ), "Number of features in X must match the length of beta, maybe try adding an intercept or interaction terms"

        # as we know from MSO
        p = sigmoid(X @ beta)
        if y.shape == ():  # if y is a scalar, then there wont be matrix multiplication
            return -X.T * (y - p)
        return -X.T @ (y - p)

    @staticmethod
    def loss_second(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        beta: np.ndarray,
    ):
        """
        calculates the second derivative of the loss function with respect to the beta
        """
        assert X.shape[-1] == len(
            beta
        ), "Number of features in X must match the length of beta, maybe try adding an intercept or interaction terms"

        # as we also know from MSO
        p = sigmoid(X @ beta)
        W = np.diag(p * (1 - p))
        return X.T @ W @ X

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        learning_rate: float = 0.01,
        max_num_epoch: int = 500,
        tolerance: float = 1e-6,
        batch_size: int = 32,
        verbose: bool = False,
    ):

        # transform input data to include interaction terms and an intercept term
        X_copy = self.create_data_frame(X)

        self.random_init_weights(X_copy.shape[1])

        if self.descent_algorithm == "minibatch":
            self.beta, self.convergence_rate = mini_batch_gd(
                X_copy,
                y,
                initial_solution=self.beta,
                regressor=self,
                calculate_gradient=LogisticRegressor.loss_prime,
                max_num_epoch=max_num_epoch,
                tolerance=tolerance,
                batch_size=batch_size,
                verbose=verbose,
            )

        elif self.descent_algorithm == "iwls":
            self.beta, self.convergence_rate = iwls(
                X_copy,
                y,
                initial_solution=self.beta,
                regressor=self,
                max_num_epoch=max_num_epoch,
                tolerance=tolerance,
                verbose=verbose,
            )
        elif self.descent_algorithm == "adam":
            self.beta, self.convergence_rate = adam(
                X_copy,
                y,
                initial_solution=self.beta,
                regressor=self,
                calculate_gradient=LogisticRegressor.loss_prime,
                max_num_epoch=max_num_epoch,
                tolerance=tolerance,
                verbose=verbose,
            )
        elif self.descent_algorithm == "sgd":
            self.beta, self.convergence_rate = sgd(
                X_copy,
                y,
                initial_solution=self.beta,
                regressor=self,
                calculate_gradient=LogisticRegressor.loss_prime,
                max_num_epoch=max_num_epoch,
                tolerance=tolerance,
                verbose=verbose,
            )
        elif self.descent_algorithm == "newton":
            self.beta, self.convergence_rate = newton(
                X_copy,
                y,
                initial_solution=self.beta,
                regressor=self,
                calculate_gradient=LogisticRegressor.loss_prime,
                calculate_hessian=LogisticRegressor.loss_second,
                max_num_epoch=max_num_epoch,
                tolerance=tolerance,
                verbose=verbose,
            )
        else:
            raise ValueError("Invalid descent_algorithm")

    def accuracy(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ):
        return np.mean(self.predict(X) == y)

    def balanced_accuracy(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ):
        y = y.astype(bool)

        negative_class = np.sum(~y)
        positive_class = np.sum(y)

        confusion_matrix = self.confusion_matrix(X, y)
        return 0.5 * (
            confusion_matrix[0, 0] / positive_class
            + confusion_matrix[1, 1] / negative_class
        )

    def confusion_matrix(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]
    ):
        y = y.astype(bool)
        if self.include_interactions:
            y = y.reset_index(drop=True)
        y_hat = self.predict(X)

        tp = np.sum(y_hat & y)
        tn = np.sum(~y_hat & ~y)
        fp = np.sum(y_hat & ~y)
        fn = np.sum(~y_hat & y)
        return np.array([[tp, fp], [fn, tn]])

    def create_data_frame(self, X: Union[np.ndarray, pd.DataFrame]):
        """function adds interaction terms and an intercept term to the data frame if needed"""
        X = X.copy()
        if self.include_interactions:
            return self.create_data_frame_with_interactions(X)
        if self.include_intercept:
            return self.create_data_frame_with_intersept(X)
        return pd.DataFrame(X)

    def create_data_frame_with_intersept(self, X: Union[np.ndarray, pd.DataFrame]):
        """function creates a data frame with an intercept column"""
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)
        if "intercept" not in X.columns:
            X.insert(0, "intercept", 1)
        return X

    def create_data_frame_with_interactions(self, X: Union[np.ndarray, pd.DataFrame]):
        """function creates a data frame with interaction terms"""
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)

        # create the new column names

        col_names_X = list(X.columns)
        new_col_names = col_names_X.copy()
        if self.include_intercept:
            new_col_names = ["intercept"] + col_names_X

        for idx, first_variable_name in enumerate(col_names_X):
            for second_variable_name in col_names_X[idx + 1 :]:
                new_col_names.append(f"{first_variable_name}*{second_variable_name}")

        # create the interaction terms
        poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=self.include_intercept
        )
        X = poly.fit_transform(X)
        X = pd.DataFrame(X, columns=new_col_names)
        if self.feature_names is None:
            X = self.remove_collinear_features(X, 0.8)
        else:
            X = X[self.feature_names]
        return X

    def remove_collinear_features(self, x, threshold):
        """
        Objective:
            Remove collinear features in a dataframe with a correlation coefficient
            greater than the threshold. Removing collinear features can help a model
            to generalize and improves the interpretability of the model.

        Inputs:
            x: features dataframe
            threshold: features with correlations greater than this value are removed

        Output:
            dataframe that contains only the non-highly-collinear features
        """

        # Calculate the correlation matrix
        corr_matrix = x.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i + 1):
                item = corr_matrix.iloc[j : (j + 1), (i + 1) : (i + 2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the threshold
                if val >= threshold:
                    # Print the correlated features and the correlation value
                    # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                    drop_cols.append(col.values[0])

        # Drop one of each pair of correlated columns
        drops = set(drop_cols)
        self.feature_names = [col for col in x.columns if col not in drops]
        x = x.drop(columns=drops)
        return x

    def predict_and_calculate_loss(self, X, y, current_beta):
        self.beta = current_beta
        y_hat = self.predict_proba(X)
        y_hat = np.clip(y_hat, 1e-10, 1 - 1e-10)
        return self.loss(y, y_hat)
