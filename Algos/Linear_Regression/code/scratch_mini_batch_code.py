import numpy as np


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X: np.ndarray = X
        self.y: np.ndarray = y

    def get_batches(self, batch_size: int):
        n, d = self.X.shape
        n_rounds = n // batch_size
        for round_i in range(n_rounds):
            yield [
                self.X[round_i * batch_size: (round_i + 1) * batch_size],
                self.y[round_i * batch_size: (round_i + 1) * batch_size]
            ]


def run(X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
        epsilon: float = 0.001,
        batch_size: int = 10):
    """
    :param np.ndarray X: n x d matrix
    :param np.ndarray y: d x 1 matrix
    :param float learning_rate: (default: 0.01)
    :param int epochs: (default: 100)
    :param int epsilon: (default: 0.001)
    :param int batch_size: (B) (default: 10)
    :return np.ndarray theta: d x 1 matrix
    """

    n, d = X.shape
    dataset = Dataset(X, y)
    theta: np.ndarray = np.zeros((d, 1))  # d x 1
    prev_loss = np.inf
    for epoch in range(epochs):
        for batch_i, (X_batch, y_batch) in enumerate(dataset.get_batches(batch_size)):
            grad = X_batch.T @ (X_batch @ theta - y_batch)  # (d x B) * ((B x d) * (d x 1) - (B x 1))
            theta = theta - learning_rate * grad / batch_size  # d x 1

        loss = ((y - X @ theta) ** 2).sum() / batch_size
        if 0 <= prev_loss - loss <= epsilon:
            return theta
        prev_loss = loss.item()

    return theta  # d x 1
