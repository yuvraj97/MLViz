import inspect
from typing import TextIO, Dict, Union

import numpy as np
import streamlit as st
from numpy import ndarray
from pandas import DataFrame
from plotly.graph_objs import Figure
import pandas as pd
from Algos.utils.plots import plotly_plot, mesh3d
from Algos.utils.preprocess import process_function
from Algos.utils.utils import get_nD_regression_data


def get_all_inputs() -> Dict[str, Union[str, int, float]]:
    """
    Here we get all inputs from user
    :return: Dict[str, Union[str, int, float]]
    """

    method: str = st.selectbox("Which method you want to use", [
        "Batch Gradient Descent",
        "Mini Batch Gradient Descent",
        "Stochastic Gradient Descent",
    ])

    seed: int = st.sidebar.number_input("Enter seed (-1 mean seed is disabled)", -1, 1000, 0, 1)

    st.sidebar.write("### Gaussian Noise")
    st_noise = st.sidebar.empty()
    st_mean, st_std = st.sidebar.beta_columns([1, 1])
    mean: float = st_mean.slider("Mean", -100.0, 100.0, 0.0, 10.0)
    std: float = st_std.slider("Standard deviation", 0.0, 100.0, 1.0, 1.0)
    st_noise.markdown(f"$\\mathcal{{N}}(\\mu= {mean}, \\sigma^2={std}^2)$")

    st.sidebar.write("### Linear Regression Parameters")
    f: str = st.sidebar.text_input("function f(X)", "2*x1 + 1")
    st_n, st_lr = st.sidebar.beta_columns([1, 1])

    # do_normalization = st.sidebar.checkbox("Normalize the Data")

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
        "function": f,
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


def sidebar_footer():
    st.sidebar.write("-----")
    with st.sidebar.beta_expander("How to get (n) dimensional data"):
        st.write(f"""
        To get $n$ dimensional data just add more features in the functions,    
        Example:    
        (1-D data) $2*x1 + 5$     
        (2-D data) $x1 ^\\wedge 2 + x2 + 3$ (this will yield in non-linear data)     
        (3-D data) $x1 + x2 + x3 + 3$      
        (4-D data) $x1 + x2 + x3 + x4 + 3$     
        and so on ...

        """)


def display_raw_code(method):

    if method == "Batch Gradient Descent":
        f: TextIO = open("./Algos/Linear_Regression/code/scratch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.beta_expander("Implementation From Scratch"):
            st.code(code)

        f: TextIO = open("./Algos/Linear_Regression/code/pytorch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.beta_expander("PyTorch Implementation"):
            st.code(code)

        f: TextIO = open("./Algos/Linear_Regression/code/pytorch_code_v2.py", "r")
        code: str = f.read()
        f.close()

        with st.beta_expander("PyTorch Implementation using Sequential module"):
            st.code(code)

    elif method == "Mini Batch Gradient Descent":

        f: TextIO = open("./Algos/Linear_Regression/code/scratch_mini_batch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.beta_expander("Implementation from scratch"):
            st.code(code)

    elif method == "Stochastic Gradient Descent":
        pass


def run(state) -> None:
    """
    Here we run the Linear Regression Simulation
    :return: None
    """

    if "lr" not in state["main"]:
        state["main"]["lr"] = {}

    inputs: Dict[str, Union[str, int, float, tuple]] = get_all_inputs()
    f = process_function(inputs["function"])  # a lambda function

    if f is False:
        st.error("Incorrect format for function")
        return

    dim: int = len(inspect.getfullargspec(f).args)

    if inputs["n"] * dim > 1000:
        st.error("Sorry but currently this app doesn't support more then 1000 data points")
        return

    X: ndarray
    y: ndarray
    X, y = get_nD_regression_data(f, n=inputs["n"], mean=inputs["mean"], std=inputs["std"], seed=inputs["seed"])
    n, d = X.shape

    st.write("# Data")
    st_X, st_y = st.beta_columns([d if d < 4 else 3, 1])
    with st_X:
        df: DataFrame = pd.DataFrame(data=X,
                                     columns=[f"x{i + 1}" for i in range(d)])
        df.index += 1
        st.write(f"$\\text{{Features}}\\quad \\mathbb{{X}}_{{{n}\\times{d}}}$")
        # st.write("$\\quad$")

        # Normalization
        if st.checkbox("Normalize the Data", True):
            norm_mean, norm_std = y.mean(), y.std()
            inputs["normalization_params"] = (norm_mean, norm_std)
            y = (y - norm_mean) / norm_std

        st.write(df)
    with st_y:
        if "normalization_params" not in inputs:
            df: DataFrame = pd.DataFrame(data=y, columns=["y"])
        else:
            (norm_mean, norm_std) = inputs["normalization_params"]
            __y = np.hstack((y * norm_std + norm_mean, y))
            df: DataFrame = pd.DataFrame(data=__y, columns=["y", "y_normalize"])

        df.index += 1
        st.write(f"$y={inputs['function']}$")
        st.write(f"$+ \\mathcal{{N}}({inputs['mean']}, {inputs['std']}^2)$")
        st.write(df)

    plt: Union[Figure, None]
    if d == 1:
        plt = plotly_plot(X.flatten(), y.flatten(), x_title="Feature", y_title="Output", title="Data")
        st.plotly_chart(plt)

    elif d == 2:
        description = {
            "title": {
                "main": "Data",
                "x": "Feature 1",
                "y": "Feature 2",
                "z": "Output"
            },
            "label": {
                "main": "Data",
            },
            "hovertemplate": "(x1, x2): (%{x}, %{y})<br>f(%{x}, %{y}): %{z}",
            "color": "green"
        }

        plt = mesh3d(X[:, 0], X[:, 1], y.flatten(),
                     description,
                     opacity=0.8)
        st.plotly_chart(plt)

    else:
        plt = None

    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    st.warning("All controls are in left control panel")

    inputs["X"], inputs["y"] = X, y
    if inputs["lr_method"] == "Implementation From Scratch":
        import Algos.Linear_Regression.simulate_algo.scratch_sim as method
    else:
        import Algos.Linear_Regression.simulate_algo.pytorch_sim as method

    if inputs["sim_method"] == "Simulate":
        import Algos.Linear_Regression.simulation.auto_simulation as simulation
    else:
        import Algos.Linear_Regression.simulation.iterative_simulation as simulation

    if inputs["sim_method"] == "Simulate" and inputs["sim_button"]:
        simulation.run(method.run, plt, inputs)
    if inputs["sim_method"] == "Manually Increment Steps":
        simulation.run(state, method.run, plt, inputs)

    st.write("-----")

    display_raw_code(inputs["method"])

    sidebar_footer()
