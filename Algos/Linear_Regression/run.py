import inspect
from typing import TextIO, Dict, Union
import streamlit as st
from numpy import ndarray

from Algos.utils.preprocess import process_function
from Algos.utils.utils import get_nD_regression_data

def get_all_inputs() -> Dict[str, Union[str, int, float]]:
    """
    Here we get all inputs from user
    :return: Dict[str, Union[str, int, float]]
    """
    f: str = st.sidebar.text_input("function(f)  use (.) for multiplication", "2*x1 + 2*x2 + 3")
    n: int = st.sidebar.number_input("Number of data points(n)", 10, 1000, 100, 10)
    mean: float = st.sidebar.number_input("Mean of Gaussian noise", -100.0, 100.0, 0.0, 10.0)
    std: float = st.sidebar.number_input("Standard deviation of Gaussian noise", -10.0, 10.0, 1.0, 1.0)
    d = {
        "function": f,
        "n": n,
        "mean": mean,
        "std": std
    }
    return d

def run() -> None:
    """
    Here we run the Linear Regression Simulation
    :return: None
    """

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
