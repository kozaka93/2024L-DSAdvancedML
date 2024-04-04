from itertools import combinations

from scipy.special import expit as sigmoid
import numpy as np


class BaseLogisticRegression:
    """
    A base class for logistic regression models.

    This class provides the framework for logistic regression modeling,
    including the option to add interaction terms between features and
    preparing input features for modeling.

    Attributes:
        weights (np.ndarray): The weights of the model.
        include_interactions (bool): Indicates whether to include interaction
            terms between features in the model.

    """

    def __init__(self, include_interactions: bool = False) -> None:
        """
        Initializes the BaseLogisticRegression class.

        Input:
         * include_interactions: bool, optional - Specifies whether to include
            interaction terms between features. Defaults to False.

        """
        self.weights = None
        self.include_interactions = include_interactions
        self.log_likelihoods = []

    def _add_interactions(self, X: np.array) -> np.array:
        """
        Adds interaction terms to the feature matrix X.

        Input:
         * X: np.ndarray - The feature matrix where each row represents an
            observation and each column represents a feature.

        Output:
         * np.ndarray: The enhanced feature matrix X with additional columns
            representing interaction terms between features.
        """
        n = X.shape[1]
        interaction_terms = np.array(
            [X[:, i] * X[:, j] for i, j in combinations(range(n), 2)]
        ).T

        return np.hstack([X, interaction_terms])

    def _prepare_features(self, X: np.array) -> np.array:
        """
        Prepares features for modeling, optionally adding interaction terms
        and an intercept term.

        Input:
         * X: np.ndarray - The original feature matrix.

        Output:
         * np.ndarray: The prepared feature matrix with an added intercept
            term and optionally interaction terms.
        """
        if self.include_interactions:
            X = self._add_interactions(X)

        X = np.hstack([np.ones((X.shape[0], 1)), X])

        return X

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the logistic regression model to the data. This method should
        be implemented by subclasses.

        Input:
         * X: np.ndarray - The feature matrix.
         * y: np.ndarray - The target variable.

        Raises:
            NotImplementedError: Indicates that the method needs to be
                implemented by subclasses.
        """
        raise NotImplementedError(
            "This method should be implemented by subclass."
            )

    def predict_proba(self, X: np.array) -> float:
        """
        Predicts probability estimates for the input samples.

        Input:
         * X: np.ndarray - The feature matrix.

        Output:
         * float: The probability of the sample for the positive class.
        """
        X = self._prepare_features(X)
        model = np.dot(X, self.weights)

        return sigmoid(model)

    def predict(self, X: np.array, threshold: float = 0.5) -> np.array:
        """
        Predicts the class labels for the input samples based on a threshold.

        Input:
         * X: np.ndarray - The feature matrix.
         * threshold: float, optional - The threshold value for deciding the
            predicted class. Defaults to 0.5.

        Output:
         * np.ndarray: The predicted class labels.
        """
        return (self.predict_proba(X) >= threshold).astype(int)


class ADAMLogisticRegression(BaseLogisticRegression):
    """
    Implements ADAM optimization algorithm for logistic regression.

    This class extends the BaseLogisticRegression class, utilizing the
    ADAM optimization technique for parameter update.
    It supports the inclusion of interaction terms between features.

    Attributes inherited from BaseLogisticRegression:
        weights (np.ndarray): The weights of the model.
        include_interactions (bool): Indicates whether to include interaction
            terms between features in the model.

    Additional Attributes:
        learning_rate: float - learning rate for the optimization algorithm
        iterations: int - number of iterations for the optimization algorithm
        beta1: float - exponential decay rate for the first moment estimates
        beta2: float - exponential decay rate for the second moment estimates
        epsilon: float - small number to prevent any division by zero
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        iterations: int = 1000,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        threshold: float = 1e-5,
        include_interactions: bool = False,
    ) -> None:
        """
        Initializes the ADAMLogisticRegression class with specified parameters.

        Input:
         * learning_rate: float - learning rate for the optimizer.
         * iterations: int - number of iterations for the optimizer.
         * beta1: float - exponential decay rate for the first moment estimates
         * beta2: float - exponential decay rate for the second moment estimate
         * epsilon: float - small constant for numerical stability.
         * include_interactions: bool, optional - Specifies whether to include
            interaction terms between features. Defaults to False.
        """
        super().__init__(include_interactions)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.threshold = threshold

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the logistic regression model using the ADAM algorithm.

        Input:
         * X: np.ndarray - The feature matrix.
         * y: np.ndarray - The target variable.
        """
        X = self._prepare_features(X)
        self.weights = np.zeros(X.shape[1])

        m = np.zeros(X.shape[1])
        v = np.zeros(X.shape[1])

        self.log_likelihoods = []
        prev_log_likelihood = -float("inf")

        for t in range(1, self.iterations + 1):
            model = np.dot(X, self.weights)
            predictions = sigmoid(model)
            errors = y - predictions
            gradient = -np.dot(X.T, errors) / len(X)

            # Calculate log likelihood
            log_likelihood = np.sum(
                np.log(np.maximum(predictions, 1e-15)) * y
                + np.log(np.maximum(1 - predictions, 1e-15)) * (1 - y)
            )

            self.log_likelihoods.append(log_likelihood)

            # Check for convergence based on change in log likelihood
            log_diff = np.abs(log_likelihood - prev_log_likelihood)
            if t > 1 and log_diff < self.threshold:
                print(f"Optimization converged after {t} iterations.")
                break

            prev_log_likelihood = log_likelihood

            m = self.beta1 * m + (1 - self.beta1) * gradient
            v = self.beta2 * v + (1 - self.beta2) * (gradient**2)
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            self.weights -= (
                self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            )


class IWLSLogisticRegression(BaseLogisticRegression):
    """
    Implements Iteratively Reweighted Least Squares (IWLS) for
    logistic regression.

    This class extends BaseLogisticRegression, utilizing the IWLS technique
    for fitting the logistic regression model. It supports the inclusion of
    interaction terms.

    Inherits attributes from BaseLogisticRegression and adds its own specific
    for the IWLS optimization method.
    """

    def __init__(
        self,
        iterations: int = 25,
        threshold: float = 1e-5,
        include_interactions: bool = False,
    ) -> None:
        """
        Initializes the IWLSLogisticRegression class with specified parameters.

        Input:
         * iterations: int - The number of iterations for the IWLS algorithm.
         * include_interactions: bool, optional - Specifies whether to
            include interaction terms between features. Defaults to False.
        """
        super().__init__(include_interactions)
        self.iterations = iterations
        self.threshold = threshold

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the logistic regression model using the IWLS algorithm.

        Input:
         * X: np.ndarray - The feature matrix.
         * y: np.ndarray - The target variable.
        """
        X = self._prepare_features(X)
        self.weights = np.zeros(X.shape[1])

        prev_log_likelihood = -float("inf")

        self.log_likelihoods = []
        for iter in range(self.iterations):
            model = np.dot(X, self.weights)
            predictions = sigmoid(model)
            W = np.diag(predictions * (1 - predictions))
            XW = X.T @ W
            try:
                H_inv = np.linalg.inv(
                    XW @ X + np.finfo(float).eps * np.eye(X.shape[1])
                )  # Adding eps to prevent singularization

                gradient = X.T @ (y - predictions)
                self.weights += H_inv @ gradient

                # Calculate log likelihood
                log_likelihood = np.sum(
                    np.log(np.maximum(predictions, 1e-15)) * y
                    + np.log(np.maximum(1 - predictions, 1e-15)) * (1 - y)
                )
                self.log_likelihoods.append(log_likelihood)
                # Check for convergence

                log_diff = np.abs(log_likelihood - prev_log_likelihood)
                if log_diff < self.threshold:
                    print(
                        f"Optimization converged after {iter + 1} iterations."
                        )
                    break

                prev_log_likelihood = log_likelihood

            except np.linalg.LinAlgError:
                break  # Loop break in case for matrix inversion problem


class SGDLogisticRegression(BaseLogisticRegression):
    """
    Implements Stochastic Gradient Descent (SGD) for logistic regression.

    This class extends BaseLogisticRegression, utilizing SGD for
    the optimization of logistic regression parameters.
    It supports mini-batch learning and the inclusion of interaction terms.

    Inherits attributes from BaseLogisticRegression and adds its own
    specific for the SGD optimization method.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 500,
        batch_size: int = 1,
        threshold: float = 1e-5,
        include_interactions: bool = False,
    ) -> None:
        """
        Initializes the SGDLogisticRegression class with specified parameters.

        Input:
         * learning_rate: float - The learning rate for the optimizer.
         * iterations: int - The number of iterations for the optimizer.
         * batch_size: int - The size of the mini-batch for the SGD algorithm.
         * include_interactions: bool, optional - Specifies whether to include
            interaction terms between features. Defaults to False.
        """
        super().__init__(include_interactions)
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.batch_size = batch_size
        self.threshold = threshold

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the logistic regression model using the SGD algorithm.

        Input:
         * X: np.ndarray - The feature matrix.
         * y: np.ndarray - The target variable.
        """
        X = self._prepare_features(X)
        self.weights = np.zeros(X.shape[1])

        n_samples = X.shape[0]
        self.log_likelihoods = []
        prev_log_likelihood = -1_000_000  # Initialize with a large value

        for iter in range(self.iterations):
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i: i + self.batch_size]
                y_batch = y[i: i + self.batch_size]

                model = np.dot(X_batch, self.weights)
                predictions = sigmoid(model)

                errors = y_batch - predictions
                gradient = -np.dot(X_batch.T, errors) / len(X_batch)

                self.weights -= self.learning_rate * gradient

            # Compute log-likelihood
            model = np.dot(X, self.weights)
            predictions = sigmoid(model)
            log_likelihood = np.sum(
                np.log(np.maximum(predictions, 1e-15)) * y
                + np.log(np.maximum(1 - predictions, 1e-15)) * (1 - y)
            )
            self.log_likelihoods.append(log_likelihood)

            # Check for convergence based on change in log likelihood
            if np.abs(log_likelihood - prev_log_likelihood) < self.threshold:
                print(f"Optimization converged after {iter + 1} iterations.")
                break

            prev_log_likelihood = log_likelihood
