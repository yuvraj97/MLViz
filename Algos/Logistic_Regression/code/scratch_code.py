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
    theta: np.ndarray = np.zeros((d, 1))
    prev_loss = -np.inf
    for epoch in range(epochs):
        h_theta: np.ndarray = 1 / (1 + np.exp(-X @ theta))
        theta = theta + learning_rate * X.T @ (y - h_theta) / n

        loss = np.mean(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))
        if 0 <= loss - prev_loss <= epsilon:
            return theta  # d x 1
        prev_loss = loss.item()

    return theta  # d x 1
