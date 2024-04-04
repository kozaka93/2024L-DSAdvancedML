import numpy as np
from logistic_regression.optimizers.optimizer import Optimizer  
from logistic_regression.utils.activations import sigmoid

class IWLS(Optimizer):
    """
    Implements the Iteratively Reweighted Least Squares (IWLS) optimization algorithm.

    Attributes:
        learning_rate (float): Not used in this implementation of IWLS, included for interface compatibility.
        eps (float): A small scalar added to the diagonal of the Hessian matrix to improve numerical stability.

    Methods:
        optimize_parms(X, y, params): Updates and returns the model parameters after one iteration of IWLS on the provided data.
    """

    def __init__(self, eps:float=1e-6):
        """
        Initializes the IWLS optimizer with the specified epsilon for numerical stability.

        Parameters:
            learning_rate (float): Placeholder for interface compatibility, not used in IWLS.
            eps (float): Small value to ensure the Hessian matrix is invertible, improving numerical stability.
        """
        self.eps = eps

    def optimize_parms(self, X: np.ndarray, y:np.ndarray, params:np.ndarray) -> np.ndarray:
        """
        Performs a single optimization step using IWLS.

        Parameters:
            X (np.ndarray): The input features for the training data.
            y (np.ndarray): The target values for the training data.
            params (np.ndarray): The current parameters of the model to be optimized.

        Returns:
            np.ndarray: The updated parameters after one iteration of optimization.
        """
        y_hat = sigmoid(X.dot(params))
        W = y_hat * (1 - y_hat)
        grad = X.T.dot(y_hat - y) 
        hessian = (X * W).T.dot(X) + self.eps * np.eye(X.shape[1])  # Hessian matrix with stabilization
        params -= np.linalg.inv(hessian).dot(grad)  

        return params