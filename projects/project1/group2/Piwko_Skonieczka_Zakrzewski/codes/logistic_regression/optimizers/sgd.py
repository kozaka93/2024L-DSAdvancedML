import numpy as np
from logistic_regression.utils.activations import sigmoid
from logistic_regression.optimizers.optimizer import Optimizer

class SGD(Optimizer):
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.

    Attributes:
        learning_rate (float): The step size used for each iteration of the optimization.
        batch_size (int): The number of samples to use for each update of the model parameters.
        
    Methods:
        optimize_parms(X, y, params): Updates and returns the model parameters after one iteration of SGD on the provided data.
    """

    def __init__(self, learning_rate:float=0.001, batch_size:float=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def optimize_parms(self, X: np.ndarray, y:np.ndarray, params:np.ndarray) -> np.ndarray:
        """
        Performs a single optimization step using SGD.

        Parameters:
            X (np.ndarray): The input features for the training data.
            y (np.ndarray): The target values for the training data.
            params (np.ndarray): The current parameters of the model to be optimized.

        Returns:
            np.ndarray: The updated parameters after one iteration of optimization.
        """
        assert self.batch_size <= X.shape[0], "Batch size must be less than or equal to the number of samples."

        n_batches = int(np.ceil(X.shape[0] / self.batch_size))
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        X_batches = np.array_split(X_shuffled, n_batches)
        y_batches = np.array_split(y_shuffled, n_batches)
        
        for X_batch, y_batch in zip(X_batches, y_batches):
            gradient = X_batch.T.dot(y_batch - sigmoid(X_batch.dot(params)))
            params += self.learning_rate * gradient
        
        return params