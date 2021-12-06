from typing import TextIO, Dict, Union, List
import streamlit as st
from numpy import ndarray
from pandas import DataFrame
from plotly.graph_objs import Figure
import pandas as pd
from Algos.utils.plots import plot_classification_data
from Algos.utils.synthetic_data import get_nD_classification_data
import math

from Algos.utils.utils import footer


def get_all_inputs() -> Dict[str, Union[str, int, float, List[float]]]:
    """
    Here we get all inputs from user
    :return: Dict[str, Union[str, int, float]]
    """

    method: str = st.sidebar.selectbox("Which method you want to use", [
        "Batch Gradient Ascent",
        "Mini Batch Gradient Ascent"
    ])

    with st.sidebar.expander("Generate n dimensional synthetic data"):
        st.write("")
        st_seed, st_n = st.columns([1, 1])
        seed: int = int(st_seed.text_input("Enter seed (-1 mean seed is disabled)", "0"))
        n: int = int(st_n.text_input("N (number of training examples)", "100"))
        n_features: int = int(st.text_input("Number of features", "2"))
        st_lower_limit, st_upper_limit = st.columns([0.5, 0.5])
        lower_limit: float = float(st_lower_limit.text_input("Lower Limit", "-10.0"))
        upper_limit: float = float(st_upper_limit.text_input("Upper Limit", "10.0"))

        st.write("### Classes proportions")
        st_prop = st.columns([1, 1])
        classes_proportions = [
            float(st_prop[0].text_input(f"Class: 1", "0.5")),
            float(st_prop[1].text_input(f"Class: 2", "0.5"))
        ]

        if not math.isclose(sum(classes_proportions), 1.0, abs_tol=0.01):
            st.error("Proportions should sum to $1$")
            raise ValueError("Algos.Logistic_Regression.run: Proportions should sum to $1$")

        st.write("### Gaussian Noise $\\mathcal{N}(\\mu,\\sigma^2)$")
        st_mean, st_std = st.columns([1, 1])
        mean: float = float(st_mean.text_input("Mean", "0.0"))
        std: float = float(st_std.text_input("Standard deviation", "1.0"))

    with st.sidebar.expander("Logistic Regression Parameters", True):

        st_lr, st_epsilon, st_epochs = st.columns([1, 1, 0.8])
        lr: float = float(st_lr.text_input("Learning Rate", "0.01"))
        epochs: int = int(st_epochs.text_input("epochs", "50"))
        epsilon: float = float(st_epsilon.text_input("Epsilon", "0.05"))

        batch_size = None
        if method == "Mini Batch Gradient Descent":
            batch_size = st.number_input("Batch size", 1, n, 10, 1)

        lr_method: str = st.radio("Choose method", ["Implementation From Scratch", "PyTorch Implementation"])
        sim_method: str = st.radio("", ["Simulate", "Manually Increment Steps"], key="Algos-LR-Sim-Step")

        sim_button, step_button = None, None
        if sim_method == "Simulate":
            sim_button = st.button("Run Simulation")
        else:
            step_button = st.button("Step")

    d = {
        "method": method,
        "seed": seed,
        "n": n,
        "n_classes": 2,
        "n_features": n_features,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "classes_proportions": classes_proportions,
        "mean": mean,
        "std": std,
        "lr": lr,
        "epsilon": epsilon,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_method": lr_method,
        "sim_method": sim_method,
        "sim_button": sim_button,
        "step_button": step_button
    }

    return d


def display_raw_code(method):

    if method == "Batch Gradient Ascent":
        f: TextIO = open("./Algos/Logistic_Regression/code/scratch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("Implementation From Scratch (Gradient Ascent)"):
            st.code(code)

        f: TextIO = open("./Algos/Logistic_Regression/code/pytorch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("PyTorch Implementation (Gradient Descent)"):
            st.code(code)

    if method == "Mini Batch Gradient Ascent":
        f: TextIO = open("./Algos/Logistic_Regression/code/scratch_mini_batch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("Implementation From Scratch (Gradient Ascent)"):
            st.code(code)

        f: TextIO = open("./Algos/Logistic_Regression/code/pytorch_code_mini_batch.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("PyTorch Implementation (Gradient Descent)"):
            st.code(code)


def run() -> None:
    """
    Here we run the Logistic Regression Simulation
    :return: None
    """

    if "Logistic Regression" not in st.session_state:
        st.session_state["Logistic Regression"] = {}

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
        seed=inputs["seed"]
    )
    n, d = X.shape

    st.write("# Data")
    st_X, st_y = st.columns([d if d < 4 else 3, 1])
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

    plt: Union[Figure, None] = None
    if inputs["n_features"] in [2, 3]:
        plt = plot_classification_data(
            X, y, title="Data",
            x_title="Feature 1", y_title="Feature 2", z_title="Feature 3"
        )
        st.plotly_chart(plt)

    st.warning("All controls are in left control panel")

    display_raw_code(inputs["method"])

    inputs["X"], inputs["y"] = X, y
    if inputs["lr_method"] == "Implementation From Scratch":
        if inputs["method"] == "Batch Gradient Ascent":
            import Algos.Logistic_Regression.simulate_algo.scratch_sim as method
        else:
            import Algos.Logistic_Regression.simulate_algo.scratch_sim_mini_batch as method
    else:
        if inputs["method"] == "Batch Gradient Ascent":
            import Algos.Logistic_Regression.simulate_algo.pytorch_sim as method
        else:
            import Algos.Logistic_Regression.simulate_algo.pytorch_sim_mini_batch as method

    if inputs["sim_method"] == "Simulate":
        import Algos.Logistic_Regression.simulation.auto_simulation as simulation
    else:
        import Algos.Logistic_Regression.simulation.iterative_simulation as simulation

    if inputs["sim_method"] == "Simulate" and inputs["sim_button"]:
        simulation.run(method.run, plt, inputs)
    if inputs["sim_method"] == "Manually Increment Steps":
        simulation.run(method.run, plt, inputs)

    footer()
