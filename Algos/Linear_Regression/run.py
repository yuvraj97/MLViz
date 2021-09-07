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
from Algos.utils.synthetic_data import get_nD_regression_data, display_train_test_data
from Algos.utils.utils import intialize, footer


def get_all_inputs() -> Dict[str, Union[str, int, float]]:
    """
    Here we get all inputs from user
    :return: Dict[str, Union[str, int, float]]
    """

    method: str = st.sidebar.selectbox("Which method you want to use", [
        "Batch Gradient Descent",
        "Mini Batch Gradient Descent"
    ])

    with st.sidebar.expander("Generate n dimensional synthetic data"):
        st.write("")
        st_seed, st_n = st.columns([0.5, 0.5])
        seed: int = int(st_seed.text_input("Enter seed (-1 mean seed is disabled)", "0"))
        n: int = int(st_n.text_input("N (number of training examples)", "100"))

        st.write("### Features $\\mathbb{X}$")
        st_lower_limit, st_upper_limit = st.columns([0.5, 0.5])
        lower_limit: float = float(st_lower_limit.text_input("Lower Limit", "-10.0"))
        upper_limit: float = float(st_upper_limit.text_input("Upper Limit", "10.0"))

        st.markdown("")
        st.markdown("$\\mathbb{Y} =  h{_\\theta}(\\mathbb{X}) + \\mathcal{N}(\\mu, \\sigma^2)$")
        st.write("### Gaussian Noise")
        st_mean, st_std = st.columns([1, 1])
        mean: float = float(st_mean.text_input("Mean", "0.0"))
        std: float = float(st_std.text_input("Standard deviation", "1.0"))
        # st_noise.markdown(f"$\\mathbb{{Y}} =  h{{_\\theta}}(\\mathbb{{X}}) + \\mathcal{{N}}({mean}, {std}^2)$")
        # h{_\\theta}(\\mathbb{X}) = \\mathbb{{X}}\\theta + \\theta_0

        st.write("### Underlying truth $h{_\\theta}(\\mathbb{X})$")
        f: str = st.text_input("h(X)", "2*x1 + 1")
        st.write("You can also create more complex, multidimensional data")
        if st.checkbox("See how"):
            st.success("""
            Here it support most of the function, like:
            **sin(x), cos(x), e^(x), log(x), ...**   
            (If a function is supported by [numpy](https://numpy.org/doc/stable/reference/routines.math.html) you can use it here as well)   
            You can also create multidimensional data.    

            **Examples:**    
            f(x1) =  x1 + sin(x)  # **1D**    
            f(x1, x2) = e^(log(x1)) + sin(2$*$pi$*$x2)  # **2D**

            Similarly you can create $n$ dimensional data
            """)

    with st.sidebar.expander("Training Parameters"):
        training_proportion = float(st.text_input("Training data (%)", "80"))
        if training_proportion <= 0 or training_proportion >= 100:
            st.error(f"Training data (%) should be in between $0$ and $100$")
            return
        training_proportion /= 100

    with st.sidebar.expander("Linear Regression Parameters", True):

        st_lr, st_epsilon, st_epochs = st.columns([1, 1, 0.8])
        lr: float = float(st_lr.text_input("Learning Rate", "0.01"))
        epochs: int = int(st_epochs.text_input("epochs", "10"))
        epsilon: float = float(st_epsilon.text_input("Epsilon", "0.05"))

        if epochs > 30:
            st.error(f"Epochs shall be in between $0$ and $30$")
            return

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
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "mean": mean,
        "std": std,
        "function": f,
        "training_proportion": training_proportion,
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

    n, d = X.shape
    st_incorrect_function = st.empty()

    st.write("# Data")
    st_X, st_y = st.columns([d if d < 4 else 3, 1])
    with st_X:
        df: DataFrame = pd.DataFrame(data=X,
                                     columns=[f"x{i + 1}" for i in range(d)])
        df.index += 1
        st.write(f"$\\text{{Features}}\\quad \\mathbb{{X}}_{{{n}\\times{d}}}$")
        # st.write("$\\quad$")

        # Normalization
        if st.checkbox("Normalize the Data", True):
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
            df: DataFrame = pd.DataFrame(data=X[y_isnan],
                                         index=[i for i, _ in enumerate(y_isnan) if _],
                                         columns=[f"x{i + 1}" for i in range(d)])
            df.index += 1
            st.write("$\\text{Features}\\quad \\mathbb{X}$")
            st.write("Where function $h{_\\theta}(\\mathbb{X})$ is failing")
            st.write(df)
            return

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
        simulation.run(method.run, plt, inputs)
    if inputs["sim_method"] == "Manually Increment Steps":
        simulation.run(method.run, plt, inputs)

    st.write("-----")

    display_raw_code(inputs["method"])

    footer()
