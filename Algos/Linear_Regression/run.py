import inspect
from typing import TextIO, Dict, Union
import streamlit as st
from numpy import ndarray
from pandas import DataFrame
from plotly.graph_objs import Figure
import pandas as pd
import time
from Algos.utils.plots import plotly_plot, mesh3d
from Algos.utils.preprocess import process_function
from Algos.utils.utils import get_nD_regression_data

def simulate(run, plt, inputs: dict):
    st_plot = st.empty()
    st_theta = st.empty()
    min_X, max_X = inputs["X"][:, 0].min(), inputs["X"][:, 0].max()
    n, d = inputs["X"].shape

    for epoch, theta in enumerate(run(inputs)):
        st_theta.write(f"$\\hat{{y}}={' + '.join(['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]).replace('x_0', '')}$")
        if d == 1:
            new_fig: Figure = plotly_plot([min_X, max_X], [theta[0][0] + theta[1][0] * min_X, theta[0][0] + theta[1][0] * max_X],
                                          fig=plt,
                                          mode="lines",
                                          color="blue",
                                          do_not_change_fig=True,
                                          title=f"Linear Regression (epoch: {epoch})")
            st_plot.plotly_chart(new_fig)
            time.sleep(1/2)

def get_all_inputs() -> Dict[str, Union[str, int, float]]:
    """
    Here we get all inputs from user
    :return: Dict[str, Union[str, int, float]]
    """
    seed: int = st.sidebar.number_input("Enter seed (-1 mean seed is disabled)", -1, 1000, 0, 1)
    st.sidebar.write("### Gaussian Noise")
    st_noise = st.sidebar.empty()
    st_mean, st_std = st.sidebar.beta_columns([1, 1])
    mean: float = st_mean.slider("Mean", -100.0, 100.0, 0.0, 10.0)
    std: float = st_std.slider("Standard deviation", 0.0, 100.0, 1.0, 1.0)
    st_noise.markdown(f"$\\mathcal{{N}}(\\mu= {mean}, \\sigma^2={std}^2)$")
    st.sidebar.write("### Linear Regression Parameters")
    f: str = st.sidebar.text_input("function f(X)", "2*x1 + 1")
    st_n, st_epochs, st_lr = st.sidebar.beta_columns([1, 1, 1])
    n: int = st_n.slider("N", 10, 1000, 100, 10)
    epochs: int = st_epochs.slider("epochs", 1, 1000, 100, 10)
    lr: float = st_lr.slider("Learning Rate", 0.0, 0.05, 0.01, 0.005)
    st_n, st_epochs, st_lr = st.sidebar.beta_columns([1, 1, 1])
    st_n.success(f"$N:{n}$")
    st_epochs.success(f"$epochs:{epochs}$")
    st_lr.success(f"$lr:{lr}$")
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

    d = {
        "function": f,
        "n": n,
        "mean": mean,
        "std": std,
        "seed": seed,
        "epochs": epochs,
        "lr": lr,
    }
    return d

def run() -> None:
    """
    Here we run the Linear Regression Simulation
    :return: None
    """

    inputs: Dict[str, Union[str, int, float]] = get_all_inputs()
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

    st_X, st_y = st.beta_columns([d if d < 4 else 3, 1])
    with st_X:
        df: DataFrame = pd.DataFrame(data=X,
                                     columns=[f"x{i+1}" for i in range(d)])
        df.index += 1
        st.write(f"$\\text{{Features}}\\quad \\mathbb{{X}}_{{{n}\\times{d}}}$")
        st.write(df)
    with st_y:
        df: DataFrame = pd.DataFrame(data=y, columns=["y"])
        df.index += 1
        st.write(f"$y={inputs['function']}$")
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

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    option: str = st.radio("Choose method", ["Implementation From Scratch", "PyTorch Implementation"])

    inputs["X"], inputs["y"] = X, y
    if option == "Implementation From Scratch":
        from Algos.Linear_Regression.scratch_sim import run
    else:
        from Algos.Linear_Regression.pytorch_sim import run

    if st.button("Simulate"):
        simulate(run, plt, inputs)

    f: TextIO = open("./Algos/Linear_Regression/scratch_code.py", "r")
    code: str = f.read()
    f.close()

    with st.beta_expander("Implementation From Scratch"):
        st.code(code)

    f: TextIO = open("./Algos/Linear_Regression/pytorch_code.py", "r")
    code: str = f.read()
    f.close()

    with st.beta_expander("PyTorch Implementation"):
        st.code(code)
