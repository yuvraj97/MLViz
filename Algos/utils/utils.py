import inspect
import numpy as np


def get_nD_regression_data(f,
                           n=10,
                           mean=0,
                           std=1,
                           coordinates_lim=(-10, 10),
                           seed=None):
    """
    :param f: function (example: lambda x1, x2: x1 + 2 * x2 + 5)
    :param n: int (number of data points)
    :param mean: float (mean of Gaussian noise)
    :param std: float (standard deviation of Gaussian noise)
    :param coordinates_lim: Tuple[int, int] (limit of out coordinates)
    :param seed: int (It specifies the order of random number)
    :return: np.ndarray (n x d)
    """

    if seed != -1:
        np.random.seed(seed)
    else:
        np.random.seed()

    dim: int = len(inspect.getfullargspec(f).args)
    X: np.ndarray = np.random.uniform(coordinates_lim[0], coordinates_lim[1], (n, dim))
    y: np.ndarray = f(*X.T).reshape(n, 1)
    noise: np.ndarray = np.random.normal(mean, std, (n, 1))
    return X, y + noise


def get_MSE_error(y: np.ndarray, y_pred: np.ndarray):
    return ((y - y_pred) ** 2).sum() / len(y)
