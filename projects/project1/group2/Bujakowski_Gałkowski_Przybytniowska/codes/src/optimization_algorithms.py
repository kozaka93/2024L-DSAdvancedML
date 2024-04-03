from typing import Union

import numpy as np
from numpy.linalg import inv


class AdamOptim:
    def __init__(
        self,
        learning_rate: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.m_dw: Union[np.ndarray, None] = None
        self.v_dw: Union[np.ndarray, None] = None
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.learning_rate: float = learning_rate
        self.t: int = 0

    def update(self, w: np.ndarray, dw: np.ndarray, *args) -> np.ndarray:
        self.t += 1

        if self.m_dw is None and self.v_dw is None:
            self.m_dw = np.zeros_like(dw)
            self.v_dw = np.zeros_like(dw)

        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw**2)

        m_dw_corr = self.m_dw / (1 - self.beta1**self.t)
        v_dw_corr = self.v_dw / (1 - self.beta2**self.t)

        w = w - self.learning_rate * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        return w


class SGD:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, w: np.ndarray, dw: np.ndarray, *args) -> np.ndarray:
        return w - self.learning_rate * dw


class IRLS:
    def __init__(self, *args) -> None:
        pass

    def update(self, w: np.ndarray, dw: np.ndarray, *args) -> np.ndarray:
        X, y = args[0], args[1]

        R = np.diag(np.ravel(y * (1 - y)))
        H = np.dot(np.dot(X.T, R), X) + 1e-4 * np.eye(X.shape[1])
        H = inv(H)
        return w - np.dot(H, dw)
