import numpy as np


def run(X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epsilon: float = 0.001,
        epochs: int = 100):
    """
    :param np.ndarray X: n x d matrix
    :param np.ndarray y: d x 1 matrix
    :param float learning_rate: (default: 0.01)
    :param int epsilon: (default: 0.001)
    :param int epochs: (default: 100)
    :return np.ndarray theta: d x 1 matrix
    """

    n, d = X.shape
    theta: np.ndarray = np.zeros((d, 1))  # d x 1
    prev_loss = np.inf
    for epoch in range(epochs):
        grad = X.T @ (X @ theta - y)  # (d x n) * ((n x d) * (d x 1) - (n x 1))
        theta = theta - learning_rate * grad / n  # d x 1

        loss = ((y - X @ theta) ** 2).sum() / n
        if 0 <= prev_loss - loss <= epsilon:
            return theta
        prev_loss = loss.item()

    return theta  # d x 1
