import time
import streamlit as st

from Algos.Linear_Regression.utils import plot_predition, prediction_msg_to_display
from Algos.utils.plots import plotly_plot


def run(f, plt, inputs: dict):
    """
    It will Simulate Linear Regression plot

    :param f: function (It will iteratively give us the parameters)
    :param plt: Figure (It contains our data points)
    :param inputs: dict
    :return: None
    """

    st_theta, st_error, st_plot = st.sidebar.empty(), st.empty(), st.empty()
    n, d = inputs["X"].shape

    errors = []
    epochs = []
    for epoch, (theta, error) in enumerate(f(inputs)):
        st_theta.info(prediction_msg_to_display(inputs, theta))

        if d in [1, 2]:
            st_plot.plotly_chart(plot_predition(inputs["X"], theta, plt, inputs))

        errors.append(error)
        epochs.append(epoch)
        st_error.plotly_chart(
            plotly_plot(
                epochs, errors,
                mode="lines+markers",
                x_title="epochs",
                y_title="error",
                title="Error Chart"
            )
        )
        time.sleep(1 / 4)

    msg = prediction_msg_to_display(inputs, theta)
    st_theta.success("        ## Algo Completed 😊    " + msg)

    return theta, msg
