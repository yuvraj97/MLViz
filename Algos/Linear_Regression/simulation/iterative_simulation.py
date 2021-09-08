import streamlit as st

from Algos.Linear_Regression.utils import plot_predition, prediction_msg_to_display
from Algos.utils.plots import plotly_plot


def run(f, plt, inputs):
    """
    It will plot Linear Regression Step by Step

    :param f: function (It will iteratively give us the parameters)
    :param plt: Figure (It contains our data points)
    :param inputs: dict
    :return: None
    """

    st_theta, st_error, st_plot = st.sidebar.empty(), st.empty(), st.empty()
    n, d = inputs["X"].shape

    if "errors" not in st.session_state["Linear Regression"]:
        st.session_state["Linear Regression"]["errors"] = []
    if "epochs" not in st.session_state["Linear Regression"]:
        st.session_state["Linear Regression"]["epochs"] = []

    if "steps" not in st.session_state["Linear Regression"]:
        st.session_state["Linear Regression"]["steps"] = [(theta, error) for theta, error in f(inputs)]

    if "step_i" not in st.session_state["Linear Regression"]:
        st.session_state["Linear Regression"]["step_i"] = 0

    if inputs["step_button"]:

        (theta, error) = st.session_state["Linear Regression"]["steps"][st.session_state["Linear Regression"]["step_i"]]

        epoch = len(st.session_state["Linear Regression"]["epochs"]) + 1

        if d in [1, 2]:
            st_plot.plotly_chart(plot_predition(inputs["X"], theta, plt))

        if st.session_state["Linear Regression"]["step_i"] < len(st.session_state["Linear Regression"]["steps"]) - 1:
            st.session_state["Linear Regression"]["step_i"] += 1
            st.session_state["Linear Regression"]["errors"].append(error)
            st.session_state["Linear Regression"]["epochs"].append(epoch)
            msg = prediction_msg_to_display(inputs, theta)
            st_theta.info()
        else:
            msg = prediction_msg_to_display(inputs, theta)
            st_theta.success("        ## Algo Completed 😊    " + msg)

        st_error.plotly_chart(
            plotly_plot(
                st.session_state["Linear Regression"]["epochs"],
                st.session_state["Linear Regression"]["errors"],
                mode="lines+markers",
                x_title="epochs",
                y_title="error",
                title="Error Chart"
            )
        )

        return theta, msg
