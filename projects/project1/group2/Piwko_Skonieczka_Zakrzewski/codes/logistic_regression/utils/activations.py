import numpy as np

def sigmoid(z:float):
    """
    Computes the sigmoid of z, element-wise.

    Parameters:
    - z (np.ndarray): Input array or scalar for which to compute the sigmoid.

    Returns:
    - np.ndarray or scalar: The sigmoid of each element if z is an array, or the sigmoid of z if it is a scalar.
    """
    return 1 / (1 + np.exp(-z))