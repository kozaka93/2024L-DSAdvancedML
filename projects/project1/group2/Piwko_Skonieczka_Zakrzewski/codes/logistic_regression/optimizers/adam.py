import numpy as np
from logistic_regression.optimizers.optimizer import Optimizer  
from logistic_regression.utils.activations import sigmoid

class ADAM(Optimizer):
    """
    Implements the ADAM (Adaptive Moment Estimation) optimization algorithm.

    Attributes:
        learning_rate (float): The learning rate or step size to control the degree to which weights are updated during training.
        batch_size (int): The number of samples to use for each update of the model parameters.
        b1 (float): The exponential decay rate for the first moment estimates.
        b2 (float): The exponential decay rate for the second moment estimates.
        eps (float): A small scalar value to prevent division by zero in the implementation.

    Methods:
        optimize_parms(X, y, params): Updates and returns the model parameters after one iteration of ADAM on the provided data.
    """

    def __init__(self, learning_rate: float=0.001, batch_size: float=32, b1:float=0.9, b2:float=0.999, eps:float=1e-8):
        """
        Initializes the ADAM optimizer with the specified learning rate, batch size, and decay rates.

        Parameters:
            learning_rate (float): The learning rate controlling the step size in the parameter space search.
            batch_size (int): The size of the mini-batches used for the stochastic gradient updates.
            b1 (float): The exponential decay rate for the first moment estimates.
            b2 (float): The exponential decay rate for the second moment estimates.
            eps (float): Small value to prevent division by zero in the updates.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def optimize_parms(self, X: np.ndarray, y:np.ndarray, params:np.ndarray) -> np.ndarray:
        """
        Performs a single optimization step using ADAM.

        Parameters:
            X (np.ndarray): The input features for the training data.
            y (np.ndarray): The target values for the training data.
            params (np.ndarray): The current parameters of the model to be optimized.

        Returns:
            np.ndarray: The updated parameters after one iteration of optimization.
        """
        assert self.batch_size <= X.shape[0], "Batch size must be less than or equal to the number of samples."

        n = X.shape[1]
        V_params = np.zeros((n, 1))
        S_params = np.zeros((n, 1))

        n_batches = int(np.ceil(X.shape[0] / self.batch_size))
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        X_batches = np.array_split(X_shuffled, n_batches)
        y_batches = np.array_split(y_shuffled, n_batches)
        t = 0

        for X_batch, y_batch in zip(X_batches, y_batches):
            t += 1
            y_hat = sigmoid(X_batch.dot(params))
            dZ = y_hat - y_batch
            d_params = X_batch.T.dot(dZ)

            V_params = self.b1 * V_params + (1 - self.b1) * d_params
            S_params = self.b2 * S_params + (1 - self.b2) * (d_params ** 2)

            V_params_corrected = V_params / (1 - self.b1 ** t)
            S_params_corrected = S_params / (1 - self.b2 ** t)

            params -= self.learning_rate * V_params_corrected / (np.sqrt(S_params_corrected) + self.eps)

        return params
