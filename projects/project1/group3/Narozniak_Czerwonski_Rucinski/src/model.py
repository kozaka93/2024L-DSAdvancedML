import numpy as np

class LogisticRegression:
    def __init__(self, optimizer, other_params_to_figure_out):
        self._optimizer = optimizer

    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> list:
        pass

