import inspect
import numpy as np


def get_nD_regression_data(f,
                           n=10,
                           mean=0,
                           std=1,
                           coordinates_lim=(-10, 10),
                           seed=-1):
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


def get_nD_classification_data(
        n_classes: int,
        n=100,
        mean=0,
        std=1,
        coordinates_lim=(-10, 10),
        seed=-1):

    """
    :param n_classes: int (
            Number of class,
            example for n_classes: 2 there will be two classes +ve and -ve
        )
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

    K = 2

    l, h = coordinates_lim
    zs = []

    # First let's get the centroids of all classes
    for class_label in range(n_classes):
        z_ds = np.random.uniform(low=l, high=h, size=2)
        zs.append(z_ds)

    # Now lt's get coordinated for each dimension
    zs = np.array(zs).T
    coordinates, labels = np.zeros((n * n_classes, 2)), np.empty(n * n_classes)
    for ith_dimension in range(2):

        # It will iteratively give coordinates for nth class for ith_dimension
        for class_label in range(n_classes):
            coordinates[
                class_label * n: (class_label + 1) * n, ith_dimension
            ] = zs[ith_dimension][class_label] + np.random.normal(mean, std, n)

            labels[class_label * n: (class_label + 1) * n] = class_label

    # It will return an array like [0, 1, 2,....., n-1]
    # np.random.shuffle(idx)
    return np.array(coordinates), np.array(labels)
