import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def log_likelihood(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


class LogisticRegression:
    def __init__(self, learning_rate=0.001, batch_size=1, num_iterations=500, interactions=False,
                 seed=0, optimizer='sgd', tolerance=0.0001, stop_after=5):
        """
        Parameters
        ----------
        interactions: bool
            If true Logistic regression use the interaction of variables.
        optimizer: str
            Implemented optimizers: sgd, adam, iwls
        stop_after: int
            Stopping criteria, if for some number the
            log-likelihood is not increasing (with some tolerance), then the training is stopped
        tolerance: float
            tolerance for stopping criteria

        """
        self.bias = None
        self.weights = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.interactions = interactions
        self.optimizer = optimizer
        self.tolerance = tolerance
        self.losses = []
        self.num_features = None
        self.stop_after = stop_after

        np.random.seed(seed)

        # Adam
        self.v_dw = None
        self.v_db = 0
        self.s_dw = None
        self.s_db = 0

    def iwls(self, X, y):
        y_pred = 1 / (1 + np.exp(-np.dot(X, self.weights)) + 1e-8)
        W = np.diag(y_pred * (1 - y_pred))
        grad = np.matmul(X.T, (y_pred - y))
        hessian = np.matmul(np.matmul(X.T, W), X)
        self.weights -= np.matmul(np.linalg.pinv(hessian), grad)

    def stochastic_gradient_descent(self, Xb, yb):
        predicted = 1 / (1 + np.exp(-(np.dot(Xb, self.weights) + self.bias)))
        self.weights -= self.learning_rate * np.dot(Xb.T, predicted - yb) / len(Xb)
        self.bias -= self.learning_rate * np.mean(predicted - yb)

    def adam(self, X, y, t, beta1=0.9, beta2=0.99, eps=1e-8):
        """
        Parameters
        ----------
        t: int
            iteration number
        beta1: float
            first moment of the optimizer
        beta2: float
            second moment of the optimizer
        eps: float
            variable added for numerical stability
        """
        n_samples, n_features = X.shape
        predictions = sigmoid(np.dot(X, self.weights) + self.bias)

        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)

        self.v_dw = beta1 * self.v_dw + (1 - beta1) * dw
        self.v_db = beta1 * self.v_db + (1 - beta1) * db
        self.s_dw = beta2 * self.s_dw + (1 - beta2) * dw ** 2
        self.s_db = beta2 * self.s_db + (1 - beta2) * db ** 2

        vdw_corrected = self.v_dw / (1 - beta1 ** (t + 1))
        vdb_corrected = self.v_db / (1 - beta1 ** (t + 1))
        sdw_corrected = self.s_dw / (1 - beta2 ** (t + 1))
        sdb_corrected = self.s_db / (1 - beta2 ** (t + 1))

        self.weights = self.weights - self.learning_rate * vdw_corrected / (np.sqrt(sdw_corrected) + eps)
        self.bias = self.bias - self.learning_rate * vdb_corrected / (np.sqrt(sdb_corrected) + eps)

    def fit(self, X, y):
        self.num_features = X.shape[1]
        if self.interactions:
            X = self.add_interactions(X)
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn()
        self.v_dw = np.zeros_like(self.weights)
        self.s_dw = np.zeros_like(self.weights)

        previous_loss = np.inf
        count = 0
        for n in range(self.num_iterations):
            indices = np.random.permutation(X.shape[0])
            X_ = X[indices]
            y_ = y[indices]

            if self.optimizer == 'iwls':
                self.bias = -0.5
                self.iwls(X_, y_)

            if self.optimizer == 'sgd' or self.optimizer == 'adam':
                for i in range(0, X.shape[0], self.batch_size):
                    Xb = X_[i:i + self.batch_size]
                    yb = y_[i:i + self.batch_size]
                    if self.optimizer == 'sgd':
                        self.stochastic_gradient_descent(Xb, yb)
                    elif self.optimizer == 'adam':
                        self.adam(Xb, yb, n)

            loss = log_likelihood(y, self.predict(X))

            if loss - previous_loss <= self.tolerance and self.stop_after is not None:
                count += 1
                if count >= self.stop_after:
                    print(f"Convergence reached, iteration number: {n}")
                    self.num_iterations = n
                    break
            else:
                count = 0
            previous_loss = loss

            self.losses.append(loss)

    def add_interactions(self, X):
        """
        Add interactions (e.g. X1, X2, X3 -> X1, X2, X3, X1*X2, X1*X3, X2*X3)

        Parameters
        ----------
        X: float
            A matrix of features

        Returns
        -------
        X : numpy.ndarray
           Matrix of features with added interactions
        """
        interactions = np.zeros((X.shape[0], int(X.shape[1] * (X.shape[1] - 1) / 2)))
        t = 0
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                interactions[:, t] = X[:, i] * X[:, j]
                t += 1

        X = np.hstack((X, interactions))
        if self.optimizer == 'iwls':
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

        return X

    def predict_proba(self, X):
        if self.interactions and self.num_features == X.shape[1]:
            X = self.add_interactions(X)
        prob = 1 / (1 + np.exp(-(np.dot(X, self.weights) + self.bias)))
        return prob

    def predict(self, X, t=0.5):
        """
        Parameters
        ----------
        t: float
            Threshold of the classification
        """
        return (self.predict_proba(X) >= t).astype(int)

    def plot_loss(self):
        iterations = list(range(self.num_iterations))
        plt.plot(iterations, self.losses)
        plt.xlabel("Iterations")
        plt.ylabel("Log-likelihood")
        plt.title(f"Log-likelihood vs. iterations, optimizer: {self.optimizer.upper()}")
        plt.show()
