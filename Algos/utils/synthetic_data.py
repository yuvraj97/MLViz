import inspect
from collections import defaultdict
from typing import List
import numpy as np
import streamlit as st
from pandas import DataFrame


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
        classes_proportions: List[float],
        n_features: int,
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
    :param classes_proportions: List[float] (proportion for each class)
    :param n_features: int (number of features)
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

    n_counts = [int(proportion * n) for proportion in classes_proportions]
    n = sum(n_counts)
    l, h = coordinates_lim

    zs = []

    # First let's get the centroids of all classes
    for class_label in range(n_classes):
        z_ds = np.random.uniform(low=l, high=h, size=n_features)
        zs.append(z_ds)

    # Now lt's get coordinated for each dimension
    zs = np.array(zs).T
    coordinates, labels = np.zeros((n, n_features)), np.empty(n)
    for ith_dimension in range(n_features):

        # It will iteratively give coordinates for nth class for ith_dimension
        for class_label in range(n_classes):
            coordinates[
                class_label * n_counts[class_label]: (class_label + 1) * n_counts[class_label], ith_dimension
            ] = zs[ith_dimension][class_label] + np.random.normal(mean, std, n_counts[class_label])

            labels[
                class_label * n_counts[class_label]: (class_label + 1) * n_counts[class_label]
            ] = class_label

    # It will return an array like [0, 1, 2,....., n-1]
    # np.random.shuffle(idx)
    return np.array(coordinates), np.array(labels).reshape((n, 1))


def split_features(features, labels):
    data = defaultdict(lambda: [])
    for idx, label in enumerate(labels):
        data[label.item()].append(features[idx])
    for label in data:
        data[label] = np.array(data[label])
    return data


def display_train_test_data(X, y, inputs, title="# Data"):
    n, d = X.shape
    st_incorrect_function = st.empty()

    st.write(title)
    st_X, st_y = st.columns([d if d < 4 else 3, 1])
    with st_X:
        df: DataFrame = DataFrame(
            data=X,
            columns=[f"x{i + 1}" for i in range(d)]
        )
        df.index += 1
        st.write(f"$\\text{{Features}}\\quad \\mathbb{{X}}_{{{n}\\times{d}}}$")
        # st.write("$\\quad$")

        # Normalization
        if inputs and st.checkbox("Normalize the Data", True, key=f"norm_data{title}"):
            norm_mean, norm_std = np.nanmean(y), np.nanstd(y)
            inputs["normalization_params"] = (norm_mean, norm_std)
            y = (y - norm_mean) / norm_std

        st.write(df)

    with st_y:
        y_isnan = np.isnan(y).reshape(np.prod(y.shape))
        if any(y_isnan):
            st_incorrect_function.error(f"""
                The function provided might be is mathematically incorrect.    
                It is failing for some values of $\\mathbb{{X}}$'s
                """)
            df: DataFrame = DataFrame(
                data=X[y_isnan],
                index=[i for i, _ in enumerate(y_isnan) if _],
                columns=[f"x{i + 1}" for i in range(d)]
            )
            df.index += 1
            st.write("$\\text{Features}\\quad \\mathbb{X}$")
            st.write("Where function $h{_\\theta}(\\mathbb{X})$ is failing")
            st.write(df)
            return

        if inputs is None or "normalization_params" not in inputs:
            df: DataFrame = DataFrame(data=y, columns=["y"])
        else:
            (norm_mean, norm_std) = inputs["normalization_params"]
            __y = np.hstack((y * norm_std + norm_mean, y))
            df: DataFrame = DataFrame(data=__y, columns=["y", "y_normalize"])

        df.index += 1
        if inputs:
            st.write(f"$y={inputs['function']}$")
            st.write(f"$+ \\mathcal{{N}}({inputs['mean']}, {inputs['std']}^2)$")
        else:
            st.write(f"Target Features $y$")

        st.write(df)

    return True, y
