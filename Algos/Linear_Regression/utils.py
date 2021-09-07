from typing import Dict, Union, TextIO

import streamlit as st
from plotly.graph_objs import Figure

from Algos.utils.plots import plotly_plot, mesh3d


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

        sim_method, sim_button, step_button = None, None, None
        simulate = st.checkbox("Use Animation", True)
        if simulate:
            sim_method: Union[str, None] = st.radio("", ["Simulate", "Manually Increment Steps"], key="Algos-LR-Sim-Step")

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
        "simulate": simulate,
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


def plot_predition(X, theta, fig):

    n, d = X.shape

    min_X, max_X = X[:, 0].min(), X[:, 0].max()

    if d == 1:
        new_fig: Figure = plotly_plot(
            [min_X, max_X],
            [
                theta[0][0] + theta[1][0] * min_X,
                theta[0][0] + theta[1][0] * max_X
            ],
            fig=fig,
            mode="lines",
            color="blue",
            do_not_change_fig=True,
            title=f"Linear Regression"
        )
        return new_fig
    elif d == 2:
        description = {
            "title": {
                "main": f"Linear Regression",
                "x": "x1",
                "y": "x2",
                "z": "y"
            },
            "label": {
                "main": "",
            },
            "hovertemplate": "(x1, x1): (%{x}, %{y})<br>f(%{x}, %{y}): %{z}"
        }
        min_X2, max_X2 = X[:, 1].min(), X[:, 1].max()
        new_fig: Figure = mesh3d(
            [min_X, min_X, max_X, max_X],
            [min_X2, max_X2, min_X2, max_X2],
            [
                theta[0][0] + theta[1][0] * min_X + theta[2][0] * min_X2,
                theta[0][0] + theta[1][0] * min_X + theta[2][0] * max_X2,
                theta[0][0] + theta[1][0] * max_X + theta[2][0] * min_X2,
                theta[0][0] + theta[1][0] * max_X + theta[2][0] * max_X2,
            ],
            description,
            fig=fig,
            opacity=0.9
        )
        return new_fig


def plot_data(X, y):

    n, d = X.shape

    if d == 1:
        return plotly_plot(
            X.flatten(), y.flatten(),
            x_title="Feature", y_title="Output", title="Data"
        )

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

        return mesh3d(
            X[:, 0], X[:, 1], y.flatten(),
            description, opacity=0.8
        )
