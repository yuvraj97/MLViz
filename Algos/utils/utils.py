import inspect
import numpy as np

def get_nD_regression_data(f,
                           n=10,
                           mean=0,
                           std=1,
                           coordinates_lim=(0, 10)
                           ):

    """
    :param f: function (example: lambda x1, x2: x1 + 2 * x2 + 5)
    :param n: int (number of data points)
    :param mean: float (mean of Gaussian noise)
    :param std: float (standard deviation of Gaussian noise)
    :param coordinates_lim: Tuple[int, int] (limit of out coordinates)
    :return: np.ndarray (n x d)
    """

    dim: int = len(inspect.getfullargspec(f).args)
    X: np.ndarray = np.random.uniform(coordinates_lim[0], coordinates_lim[1], (n, dim))
    y: np.ndarray = f(*X.T).reshape(n, 1)
    noise: np.ndarray = np.random.normal(mean, std, (n, 1))
    return X, y + noise
