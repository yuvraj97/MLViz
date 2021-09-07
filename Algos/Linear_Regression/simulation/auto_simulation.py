import time
import streamlit as st

from Algos.Linear_Regression.utils import plot_predition
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
    st_theta_completed = st.empty()

    errors = []
    epochs = []
    for epoch, (theta, error) in enumerate(f(inputs)):
        if "normalization_params" not in inputs:
            st_theta.info(f"""
            $\\hat{{y}}={' + '.join(
                ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
            ).replace('x_0', '')}$
            """)
        else:
            norm_mean, norm_std = inputs["normalization_params"]
            st_theta.info(f"""
            **For Normalized Data:**    
            $\\hat{{y}}={' + '.join(
                ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
            ).replace('x_0', '')}$    
            **For Non Normalized Data:**    
            $\\hat{{y}}={' + '.join(
                ['{:.2f}'.format(
                    theta_i[0] * norm_std if i != 0 else theta_i[0] * norm_std + norm_mean
                ) + f'x_{i}' for i, theta_i in enumerate(theta)]
            ).replace('x_0', '')}$            
            """)

            st_plot.plotly_chart(plot_predition(inputs["X"], theta, plt))

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

    if "normalization_params" not in inputs:

        s = f"""
        Algo Completed 😊    
        $\\hat{{y}}={' + '.join(
            ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
        ).replace('x_0', '')}$
        """

        st_theta.success(s)
        st_theta_completed.success(s)

    else:
        norm_mean, norm_std = inputs["normalization_params"]

        s = f"""
        Algo Completed 😊    
        **For Normalized Data:**    
        $\\hat{{y}}={' + '.join(
            ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
        ).replace('x_0', '')}$    
        **For Non Normalized Data:**    
        $\\hat{{y}}={' + '.join(
            ['{:.2f}'.format(
                theta_i[0] * norm_std if i != 0 else theta_i[0] * norm_std + norm_mean
            ) + f'x_{i}' for i, theta_i in enumerate(theta)]
        ).replace('x_0', '')}$            
        """

        st_theta.success(s)
        st_theta_completed.success(s)

    return theta
