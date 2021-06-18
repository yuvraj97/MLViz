from typing import Union, Dict
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


def run(inputs: Dict[str, Union[str, int, float, np.ndarray]]):
    """
    :param inputs: Dict[str, Union[str, int, float]] (template): {
        "X": numpy.ndarray,  # n x d
        "y": numpy.ndarray,  # n x 1
        "lr": float,
        "epochs": int,
        "epsilon": float,
        "batch_size": int
    }

    :return theta numpy.ndarray  # d x 1 matrix
    """

    X, y, learning_rate = inputs["X"], inputs["y"], inputs["lr"]
    epochs, batch_size = inputs["epochs"], inputs["batch_size"]

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    n, d = X.shape

    dataset = Dataset(X, y)
    theta: np.ndarray = np.zeros((d, 1))  # d x 1
    prev_loss = -np.inf
    for epoch in range(epochs + 1):
        for batch_i, (X_batch, y_batch) in enumerate(dataset.get_batches(batch_size)):
            h_theta: np.ndarray = 1 / (1 + np.exp(-X_batch @ theta))
            theta = theta + learning_rate * X_batch.T @ (y_batch - h_theta) / n

        h_theta: np.ndarray = 1 / (1 + np.exp(-X @ theta))
        loss = np.mean(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))
        if 0 <= loss - prev_loss <= inputs["epsilon"]:
            return theta  # d x 1
        prev_loss = loss

        yield theta, loss

    return theta  # d x 1
