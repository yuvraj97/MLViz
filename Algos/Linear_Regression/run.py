import inspect
from typing import TextIO, Dict, Union

import numpy as np
import streamlit as st
from numpy import ndarray
from pandas import DataFrame
from plotly.graph_objs import Figure
import pandas as pd

from Algos.Linear_Regression.utils import plot_predition, plot_data
from Algos.utils.plots import plotly_plot, mesh3d
from Algos.utils.preprocess import process_function
from Algos.utils.synthetic_data import get_nD_regression_data, display_train_test_data
from Algos.utils.utils import intialize, footer




def display_raw_code(method):

    if method == "Batch Gradient Descent":
        f: TextIO = open("./Algos/Linear_Regression/code/scratch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("Implementation From Scratch"):
            st.code(code)

        f: TextIO = open("./Algos/Linear_Regression/code/pytorch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("PyTorch Implementation"):
            st.code(code)

        f: TextIO = open("./Algos/Linear_Regression/code/pytorch_code_v2.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("PyTorch Implementation using Sequential module"):
            st.code(code)

    elif method == "Mini Batch Gradient Descent":

        f: TextIO = open("./Algos/Linear_Regression/code/scratch_mini_batch_code.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("Implementation from scratch"):
            st.code(code)

        f: TextIO = open("./Algos/Linear_Regression/code/pytorch_code_mini_batch.py", "r")
        code: str = f.read()
        f.close()

        with st.expander("PyTorch Implementation"):
            st.code(code)


def sessionize_inputs(inputs):
    inputs_no_button = {key: inputs[key] for key in inputs if "button" not in key}
    if "inputs" not in st.session_state["Linear Regression"] or \
            st.session_state["Linear Regression"]["inputs"] != inputs_no_button:
        st.session_state["Linear Regression"] = {"inputs": inputs_no_button}


def run_simulation(inputs, plt):
    if inputs["lr_method"] == "Implementation From Scratch":
        if inputs["method"] == "Batch Gradient Descent":
            import Algos.Linear_Regression.simulate_algo.scratch_sim as method
        else:
            import Algos.Linear_Regression.simulate_algo.scratch_sim_mini_batch as method
    else:
        if inputs["method"] == "Batch Gradient Descent":
            import Algos.Linear_Regression.simulate_algo.pytorch_sim as method
        else:
            import Algos.Linear_Regression.simulate_algo.pytorch_sim_mini_batch as method

    if inputs["sim_method"] == "Simulate":
        import Algos.Linear_Regression.simulation.auto_simulation as simulation
    else:
        import Algos.Linear_Regression.simulation.iterative_simulation as simulation

    if inputs["sim_method"] == "Simulate" and inputs["sim_button"]:
        return simulation.run(method.run, plt, inputs)
    if inputs["sim_method"] == "Manually Increment Steps":
        return simulation.run(method.run, plt, inputs)


def run_scratch(inputs, plt):

    X, y = inputs["X"], inputs["y"]
    n, d = X.shape

    if inputs["lr_method"] == "Implementation From Scratch":
        is_scratch = True
        if inputs["method"] == "Batch Gradient Descent":
            import Algos.Linear_Regression.code.scratch_code as method
        else:
            import Algos.Linear_Regression.code.scratch_mini_batch_code as method
    else:
        is_scratch = False
        if inputs["method"] == "Batch Gradient Descent":
            import Algos.Linear_Regression.code.pytorch_code as method
        else:
            import Algos.Linear_Regression.code.pytorch_code_mini_batch as method

    theta = method.run(
        np.hstack((np.ones((n, 1)), X)) if is_scratch else X, y,
        learning_rate=inputs["lr"], epsilon=inputs["epsilon"], epochs=inputs["epochs"]
    )
    st.plotly_chart(plot_predition(X, theta, plt))
    return theta


def run() -> None:
    """
    Here we run the Linear Regression Simulation
    :return: None
    """

    intialize("Linear Regression")

    if "Linear Regression" not in st.session_state:
        st.session_state["Linear Regression"] = {}

    inputs: Dict[str, Union[str, int, float, tuple]] = get_all_inputs()
    sessionize_inputs(inputs)
    f = process_function(inputs["function"])  # a lambda function

    if f is None:
        st.error("Incorrect format for function")
        return

    dim: int = len(inspect.getfullargspec(f).args)

    if inputs["n"] * dim > 1000:
        st.error("Sorry but currently this app doesn't support more then 1000 data points")
        return

    X: ndarray
    y: ndarray
    X, y = get_nD_regression_data(
        f,
        n=inputs["n"],
        mean=inputs["mean"],
        std=inputs["std"],
        seed=inputs["seed"],
        coordinates_lim=[inputs["lower_limit"], inputs["upper_limit"]]
    )

    n_train = int(len(X) * inputs["training_proportion"])
    test_X,  test_y  = X[n_train:], y[n_train:]
    X, y = X[:n_train], y[:n_train]
    display_train_test_data(X, y, inputs, "# Training Data")

    with st.expander("Test Data"):
        display_train_test_data(test_X, test_y, None, "# Test Data")

    n, d = X.shape

    plt: Union[Figure, None] = plot_data(X, y)
    st.plotly_chart(plt)
    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    inputs["X"], inputs["y"] = X, y
    if inputs["simulate"]:
        theta = run_simulation(inputs, plt)
    else:
        theta = run_scratch(inputs, plt)

    st.write("-----")
    display_raw_code(inputs["method"])
    footer()
