from typing import Union, Dict
import numpy as np
from Algos.utils.utils import get_MSE_error


def run(inputs: Dict[str, Union[str, int, float, np.ndarray]]):
    """
    :param inputs: Dict[str, Union[str, int, float]] (template): {
        "X": numpy.ndarray,  # n x d
        "y": numpy.ndarray,  # n x 1
        "lr": float,
        "epochs": int,
        "epsilon": float,
    }

    :return theta numpy.ndarray  # d x 1 matrix
    """

    X, y, learning_rate, epochs = inputs["X"], inputs["y"], inputs["lr"], inputs["epochs"]
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    n, d = X.shape

    prev_loss = -np.inf
    theta: np.ndarray = np.zeros((d, 1))  # d x 1
    for epoch in range(epochs + 1):
        h_theta: np.ndarray = 1 / (1 + np.exp(-X @ theta))
        theta = theta + learning_rate * X.T @ (y - h_theta) / n

        loss = np.mean(y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta))
        if 0 <= loss - prev_loss <= inputs["epsilon"]:
            break
        prev_loss = loss.item()

        yield theta, loss

    return theta
