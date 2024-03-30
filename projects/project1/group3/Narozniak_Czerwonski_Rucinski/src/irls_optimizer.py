import numpy as np


class IRLS:
    def update(self, B,  X, y):
        # Add 1 as the first column to X
        X_ = np.concatenate([np.ones(shape=(X.shape[0], 1)), X], axis=1)
        z = X_ @ B
        p = 1 / (1 + np.exp(-z))
        W = np.diagflat(p * (1 - p))
        W_inv = np.diagflat(1. / np.diagonal(W))
        response = z + W_inv @ (y - p)
        xwx_inv = np.linalg.inv(X_.T @ W @ X_)
        new_B = xwx_inv @ X_.T @ W @ response
        return new_B[1:], new_B[0]