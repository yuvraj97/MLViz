from typing import Union, Dict
import numpy as np
from Algos.utils.utils import get_MSE_error


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

    prev_error = np.inf
    theta: np.ndarray = np.zeros((d, 1))  # d x 1
    for epoch in range(epochs + 1):
        for batch_i, (X_batch, y_batch) in enumerate(dataset.get_batches(batch_size)):

            grad = X_batch.T @ (X_batch @ theta - y_batch)  # (d x B) * ((B x d) * (d x 1) - (B x 1))
            theta = theta - learning_rate * grad / batch_size  # d x 1

        error = get_MSE_error(y, X @ theta)
        if 0 <= prev_error - error <= inputs["epsilon"]:
            return theta
        prev_error = error

        yield theta, error

    return theta  # d x 1
