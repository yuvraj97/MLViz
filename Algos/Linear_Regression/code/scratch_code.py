import numpy as np


def run(X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100):
    """
    :param np.ndarray X: n x d matrix
    :param np.ndarray y: d x 1 matrix
    :param float learning_rate: (default: 0.01)
    :param int epochs: (default: 100)
    :return np.ndarray theta: d x 1 matrix
    """
    n, d = X.shape
    theta: np.ndarray = np.zeros((d, 1))  # d x 1
    for epoch in range(epochs):
        grad = X.T @ (X @ theta - y)  # (d x n) * ((n x d) * (d x 1) - (n x 1))
        theta = theta - learning_rate * grad / n  # d x 1
    return theta  # d x 1
