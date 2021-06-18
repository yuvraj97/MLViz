import time
import streamlit as st
from plotly.graph_objs import Figure
from Algos.utils.plots import plotly_plot, mesh3d


def run(f, plt, inputs: dict):

    """
    It will Simulate Linear Regression plot

    :param f: function (It will iteratively give us the parameters)
    :param plt: Figure (It contains our data points)
    :param inputs: dict
    :return: None
    """

    st_theta, st_error, st_plot = st.sidebar.empty(), st.empty(), st.empty()
    st_theta_completed = st.empty()

    min_X, max_X = inputs["X"][:, 0].min(), inputs["X"][:, 0].max()
    n, d = inputs["X"].shape

    errors = []
    epochs = []
    for epoch, (theta, error) in enumerate(f(inputs)):
        st_theta.info(f"""
        $\\hat{{y}}={' + '.join(
            ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
        ).replace('x_0', '')}$
        """)

        if d == 2:
            new_fig: Figure = plotly_plot(
                [min_X, max_X],
                [
                    (- theta[0].item() - theta[1].item() * min_X) / theta[2].item(),
                    (- theta[0].item() - theta[1].item() * max_X) / theta[2].item(),
                ],
                fig=plt,
                mode="lines",
                color="blue",
                do_not_change_fig=True,
                title=f"Linear Regression (epoch: {epoch})"
            )
            st_plot.plotly_chart(new_fig)

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
        time.sleep(1/4)

    s = f"""
    Algo Completed 😊    
    $\\hat{{y}}={' + '.join(
        ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
    ).replace('x_0', '')}$
    """

    st_theta.success(s)
    st_theta_completed.success(s)
