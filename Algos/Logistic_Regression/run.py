from typing import Dict, Union, List
import streamlit as st
from numpy import ndarray
from pandas import DataFrame
import pandas as pd
from Algos.utils.utils import get_nD_classification_data
import math


def get_all_inputs() -> Dict[str, Union[str, int, float, List[float]]]:
    """
    Here we get all inputs from user
    :return: Dict[str, Union[str, int, float]]
    """

    method: str = st.selectbox("Which method you want to use", [
        "Batch Gradient Descent",
        "Mini Batch Gradient Descent"
    ])

    seed: int = st.sidebar.number_input("Enter seed (-1 mean seed is disabled)", -1, 1000, 0, 1)

    st.sidebar.write("### Gaussian Noise")
    st_noise = st.sidebar.empty()
    st_mean, st_std = st.sidebar.beta_columns([1, 1])
    mean: float = st_mean.slider("Mean", -100.0, 100.0, 0.0, 10.0)
    std: float = st_std.slider("Standard deviation", 0.0, 100.0, 1.0, 1.0)
    st_noise.markdown(f"$\\mathcal{{N}}(\\mu= {mean}, \\sigma^2={std}^2)$")

    st.sidebar.write("### Logistic Regression Parameters")

    st_n_classes, st_n_features = st.sidebar.beta_columns([1, 1])
    n_classes: int = st_n_classes.number_input(
        "Number of classes",
        min_value=1,
        max_value=100,
        value=2,
        step=1)

    n_features: int = st_n_features.number_input(
        "Number of Features",
        min_value=1,
        max_value=100,
        value=2,
        step=1)

    st.sidebar.write("### Classes proportions")
    classes_proportions = []
    st_classes_proportions = [st.sidebar.beta_columns([1] * 3) for _ in range(n_classes // 3 + 1)]

    j = 0
    for j in range(n_classes // 3):
        for i in range(3):
            classes_proportions.append(
                float(
                    st_classes_proportions[j][i].text_input(
                        f"C{3 * j + i + 1}",
                        "{:.3f}".format(1/n_classes),
                        key=f"proportions-{j}-{i}"
                    )
                )
            )
    for i in range(n_classes % 3):
        classes_proportions.append(
            float(
                st_classes_proportions[-1][i].text_input(
                    f"C{j * 3 + i + 1}",
                    "{:.3f}".format(1 / n_classes),
                    key=f"proportions-{-1}-{i}"
                )
            )
        )

    if not math.isclose(sum(classes_proportions), 1.0, abs_tol=0.01):
        st.error("Proportions should sum to $1$")
        raise ValueError("Algos.Logistic_Regression.run: Proportions should sum to $1$")

    st_n, st_lr = st.sidebar.beta_columns([1, 1])
    n: int = st_n.slider("N", 10, 1000, 100, 10)
    lr: float = st_lr.slider("Learning Rate", 0.0, 0.05, 0.01, 0.005)
    st_n, st_lr = st.sidebar.beta_columns([1, 1])
    st_n.success(f"$N:{n}$")
    st_lr.success(f"$lr:{lr}$")

    st_epochs, st_epsilon = st.sidebar.beta_columns([1, 1])
    epochs: int = st_epochs.slider("epochs", 1, 100, 50, 10)
    epsilon: float = st_epsilon.slider("Epsilon", 0.001, 0.1, 0.05, 0.001)
    st_epochs, st_epsilon = st.sidebar.beta_columns([1, 1])
    st_epochs.success(f"epochs$:{epochs}$")
    st_epsilon.success(f"$\\epsilon:{epsilon}$")

    batch_size = None
    if method == "Mini Batch Gradient Descent":
        batch_size = st.sidebar.number_input("Batch size", 1, n, 10, 1)

    st.sidebar.write("-----")

    lr_method: str = st.sidebar.radio("Choose method", ["Implementation From Scratch", "PyTorch Implementation"])
    sim_method: str = st.sidebar.radio("", ["Simulate", "Manually Increment Steps"], key="Algos-LR-Sim-Step")

    sim_button, step_button = None, None
    if sim_method == "Simulate":
        sim_button = st.sidebar.button("Run Simulation")
    else:
        step_button = st.sidebar.button("Step")

    d = {
        "method": method,
        "n_classes": n_classes,
        "n_features": n_features,
        "classes_proportions": classes_proportions,
        # "do_normalization": do_normalization,
        "n": n,
        "mean": mean,
        "std": std,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
        "epsilon": epsilon,
        "batch_size": batch_size,
        "lr_method": lr_method,
        "sim_method": sim_method,
        "sim_button": sim_button,
        "step_button": step_button
    }

    return d


def run(state) -> None:
    """
    Here we run the Logistic Regression Simulation
    :return: None
    """

    inputs: Dict[str, Union[str, int, float, tuple, List[float]]] = get_all_inputs()

    if inputs["n"] * inputs["n_features"] > 1000:
        st.error("Sorry but currently this app doesn't support more then 1000 data points")
        return

    X: ndarray
    y: ndarray
    X, y = get_nD_classification_data(
        n_classes=inputs["n_classes"],
        classes_proportions=inputs["classes_proportions"],
        n_features=inputs["n_features"],
        n=inputs["n"],
        mean=inputs["mean"],
        std=inputs["std"],
        seed=inputs["seed"])

    n, d = X.shape

    st.write("# Data")
    st_X, st_y = st.beta_columns([d if d < 4 else 3, 1])
    with st_X:
        df: DataFrame = pd.DataFrame(data=X,
                                     columns=[f"x{i + 1}" for i in range(d)])
        df.index += 1
        st.write(f"$\\text{{Features}}\\quad \\mathbb{{X}}_{{{n}\\times{d}}}$")
        st.write(df)
    with st_y:
        df: DataFrame = pd.DataFrame(data=y, columns=["y"])
        df.index += 1
        st.write(f"**Class Labels**")
        st.write(df)
