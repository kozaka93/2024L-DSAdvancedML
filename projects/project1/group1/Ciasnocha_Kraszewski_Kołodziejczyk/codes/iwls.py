import numpy as np
from optimizer import Optimizer


class IWLS(Optimizer):
    """Iteratively reweighted least squares optimizer"""

    def update(self, X, y, weights, predictions):
        # Diagonal matrix of weights
        W = np.diag(predictions * (1 - predictions))
        Z = X @ weights + np.linalg.pinv(W) @ (y - predictions)

        # Solving the weighted least squares problem
        new_weights = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ Z, rcond=None)[0]
        return new_weights.ravel()
