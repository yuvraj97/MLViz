import inspect
from typing import TextIO, Dict, Union
import streamlit as st
from numpy import ndarray
from plotly.graph_objs import Figure

from Algos.utils.plots import plotly_plot, mesh3d
from Algos.utils.preprocess import process_function
from Algos.utils.utils import get_nD_regression_data

def get_all_inputs() -> Dict[str, Union[str, int, float]]:
    """
    Here we get all inputs from user
    :return: Dict[str, Union[str, int, float]]
    """
    seed: int = st.sidebar.number_input("Enter seed (-1 mean seed is disabled)", 0, 1000, 0, 1)
    f: str = st.sidebar.text_input("function(f)", "2*x1 + 2*x2 + 3")
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
    n: int = st.sidebar.number_input("Number of data points(n)", 10, 1000, 100, 10)
    mean: float = st.sidebar.number_input("Mean of Gaussian noise", -100.0, 100.0, 0.0, 10.0)
    std: float = st.sidebar.number_input("Standard deviation of Gaussian noise", -10.0, 10.0, 1.0, 1.0)
    d = {
        "function": f,
        "n": n,
        "mean": mean,
        "std": std,
        "seed:": seed
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
    X, y = get_nD_regression_data(f, n=inputs["n"], mean=inputs["mean"], std=inputs["std"])
    n, d = X.shape

    st_X, st_y = st.beta_columns([d if d < 4 else 3, 1])
    with st_X:
        st.write("### Features")
        st.write(X)
    with st_y:
        st.write("### Output")
        st.write(y)

    if d == 1:
        plt = plotly_plot(X.flatten(), y.flatten(), x_title="Feature", y_title="Output", title="Data")
        st.plotly_chart(plt)

    if d == 2:
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

        plt: Figure = mesh3d(X[:, 0], X[:, 1], y.flatten(),
                             description,
                             opacity=0.8)
        st.plotly_chart(plt)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    option: str = st.radio("Choose method", ["Implementation From Scratch", "PyTorch Implementation"])

    inputs["X"], inputs["y"] = X, y
    if option == "Implementation From Scratch":
        from Algos.Linear_Regression.scratch_sim import run
        run(inputs)
    else:
        from Algos.Linear_Regression.pytorch_sim import run
        run(inputs)

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
