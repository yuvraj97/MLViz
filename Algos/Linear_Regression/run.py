import inspect
from typing import Dict, Union

import numpy as np
import scipy.stats
import streamlit as st
from numpy import ndarray
from plotly.graph_objs import Figure

from Algos.Linear_Regression.utils import plot_predition, plot_data, get_all_inputs, sessionize_inputs, \
    display_raw_code, prediction_msg_to_display
from Algos.utils.plots import plotly_plot
from Algos.utils.preprocess import process_function
from Algos.utils.stats import f_test, rmse, r2
from Algos.utils.synthetic_data import get_nD_regression_data, display_train_test_data
from Algos.utils.utils import intialize, footer


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
        if inputs["method"] == "Batch Gradient Descent":
            import Algos.Linear_Regression.simulate_algo.scratch_sim as method
        else:
            import Algos.Linear_Regression.simulate_algo.scratch_sim_mini_batch as method
    else:
        if inputs["method"] == "Batch Gradient Descent":
            import Algos.Linear_Regression.simulate_algo.pytorch_sim as method
        else:
            import Algos.Linear_Regression.simulate_algo.pytorch_sim_mini_batch as method

    epochs, errors = [], []
    for epoch, (theta, error) in enumerate(method.run(inputs)):
        epochs.append(epoch)
        errors.append(error)

    st.plotly_chart(
        plotly_plot(
            epochs, errors,
            mode="lines+markers",
            x_title="epochs",
            y_title="error",
            title="Error Chart"
        )
    )

    if d in [1, 2]:
        st.plotly_chart(plot_predition(X, theta, plt, inputs))
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
    test_X, test_y = X[n_train:], y[n_train:]
    X, y = X[:n_train], y[:n_train]
    status, y_norm = display_train_test_data(X, y, inputs, "# Training Data")

    with st.expander("Test Data"):
        display_train_test_data(test_X, test_y, None, "# Test Data")

    n, d = X.shape

    if d in [1, 2]:
        plt: Union[Figure, None] = plot_data(X, y)
        st.plotly_chart(plt)
    else:
        plt = None
    # st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    st.markdown(f"""
    ## Estimated Parameters for Target Variable ($y$):
      - **Mean** $(\\hat{{\\mu}}_y) = {np.nanmean(y):.3f}$
      - **Variance** $(\\sigma^2_y)={np.nanvar(y):.3f}$
      - **Standard Deviation** $(\\sigma_y)={np.nanstd(y):.3f}$
    """)

    inputs["X"], inputs["y"] = X, y_norm
    if inputs["simulate"]:
        theta_msg = run_simulation(inputs, plt)
        if theta_msg is None:
            return
        theta, msg = theta_msg
    else:
        theta = run_scratch(inputs, plt)
        msg = prediction_msg_to_display(inputs, theta)

    n_test, d_test = test_X.shape

    X_pad = np.hstack((np.ones((n, 1)), X))
    y_hat = X_pad@theta
    X_pad_test = np.hstack((np.ones((n_test, 1)), test_X))
    y_hat_test = X_pad_test@theta

    if "normalization_params" in inputs:
        norm_mean, norm_std = inputs["normalization_params"]
    else:
        norm_mean, norm_std = 0, 1
    y_hat = y_hat * norm_std + norm_mean
    y_hat_test = y_hat_test * norm_std + norm_mean

    r_2 = r2(y, y_hat)
    f_value, p_value = f_test(y, y_hat, d, n - (d + 1))

    r_2_test = r2(test_y, y_hat_test)
    f_value_test, p_value_test = f_test(test_y, y_hat_test, d_test, n_test - (d_test + 1))

    st_left, st_middle, st_right = st.columns([1, 1, 1])
    st_left.markdown(f"""
        ## Prediction
        {msg}
        """)
    st_middle.markdown(f"""
        ## Performance   
        `Training data`    
        
        $\\text{{RMSE}}\\ :$ `{rmse(y, y_hat):.3f}`  
        $R^2\\quad\\quad:$ `{r_2:.3}`  
        $\\text{{F-value}}:$ `{f_value:.2f}`  
        $\\text{{p-value}}:$ `{p_value:.5f}`          
        """)

    if p_value < 0.01:
        st.success(f"""
        Congratulations our **features** are `statistically significant` to predict **Target variable**.
        """)
    else:
        st.warning(f"""
        Our **features** doesn't seems to be `statistically significant` to predict **Target variable**.
        """)

    st.write("---")

    st.markdown(f"""
        
    ### $\\text{{R}}^2$
    $R^2$ tells us how much of variation in **Target Variable** can be explained usind **Input Variables**  
    Now in our example $R^2$ is `{r2:.3}` that mean (using current ML algorithm) we can say:
      - `{r2*100:.2f}%` of **Target Variable** can be explained if we using our **Input Variables**.  
      - There is `{r2*100:.2f}%` reduction in variance of **Target Variable** using our **Input Variables**.  
    """)

    st.write("-----")
    display_raw_code(inputs["method"])
    footer()
