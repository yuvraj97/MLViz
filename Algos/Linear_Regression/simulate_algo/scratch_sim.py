from typing import Union, Dict
import numpy as np
from Algos.utils.utils import get_MSE_error

def run(inputs: Dict[str, Union[str, int, float, np.ndarray]]):

    """
    :param inputs: Dict[str, Union[str, int, float]]

    inputs: {
        "n": int
        "epochs": int
        "lr": float
        "X": numpy.ndarray  # n x d
        "y": numpy.ndarray  # n x 1
    }

    :return theta numpy.ndarray  # d x 1 matrix
    """

    X, y, learning_rate, epochs = inputs["X"], inputs["y"], inputs["lr"], inputs["epochs"]
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    n, d = X.shape

    prev_error = np.inf
    theta: np.ndarray = np.zeros((d, 1))  # d x 1
    for epoch in range(epochs + 1):
        grad = X.T @ (X @ theta - y)  # (d x n) * ((n x d) * (d x 1) - (n x 1))
        theta = theta - learning_rate * grad / n  # d x 1
        error = get_MSE_error(y, X @ theta)
        if prev_error - error  < inputs["epsilon"]:
            break
        prev_error = error
        yield theta, error
