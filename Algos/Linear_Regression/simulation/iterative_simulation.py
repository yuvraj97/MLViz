import streamlit as st
from plotly.graph_objs import Figure
from Algos.utils.plots import plotly_plot, mesh3d

def run(state, f, plt, inputs):

    """
    It will plot Linear Regression Step by Step

    :param state: dict (current state of the program)
    :param f: function (It will iteratively give us the parameters)
    :param plt: Figure (It contains our data points)
    :param inputs: dict
    :return: None
    """

    st_theta, st_error, st_plot = st.sidebar.empty(), st.empty(), st.empty()
    st_theta_completed = st.empty()

    min_X, max_X = inputs["X"][:, 0].min(), inputs["X"][:, 0].max()
    n, d = inputs["X"].shape

    if "errors" not in state["main"]["lr"]:
        state["main"]["lr"]["errors"] = []
    if "epochs" not in state["main"]["lr"]:
        state["main"]["lr"]["epochs"] = []

    if "steps" not in state["main"]["lr"]:
        state["main"]["lr"]["steps"] = [(theta, error) for theta, error in f(inputs)]

    if "step_i" not in state["main"]["lr"]:
        state["main"]["lr"]["step_i"] = 0

    if inputs["step_button"]:

        (theta, error) = state["main"]["lr"]["steps"][state["main"]["lr"]["step_i"]]

        epoch = len(state["main"]["lr"]["epochs"]) + 1

        if d == 1:
            new_fig: Figure = plotly_plot([min_X, max_X],
                                          [theta[0][0] + theta[1][0] * min_X, theta[0][0] + theta[1][0] * max_X],
                                          fig=plt,
                                          mode="lines",
                                          color="blue",
                                          do_not_change_fig=True,
                                          title=f"Linear Regression (epoch: {epoch})")
            st_plot.plotly_chart(new_fig)
        elif d == 2:
            description = {
                "title": {
                    "main": f"Linear Regression (epoch: {epoch})",
                    "x": "x1",
                    "y": "x2",
                    "z": "y"
                },
                "label": {
                    "main": "",
                },
                "hovertemplate": "(x1, x1): (%{x}, %{y})<br>f(%{x}, %{y}): %{z}"
            }
            min_X2, max_X2 = inputs["X"][:, 1].min(), inputs["X"][:, 1].max()
            new_fig: Figure = mesh3d([min_X, min_X, max_X, max_X],
                                     [min_X2, max_X2, min_X2, max_X2],
                                     [
                                         theta[0][0] + theta[1][0] * min_X + theta[2][0] * min_X2,
                                         theta[0][0] + theta[1][0] * min_X + theta[2][0] * max_X2,
                                         theta[0][0] + theta[1][0] * max_X + theta[2][0] * min_X2,
                                         theta[0][0] + theta[1][0] * max_X + theta[2][0] * max_X2,
                                     ],
                                     description,
                                     fig=plt,
                                     opacity=0.9)
            st_plot.plotly_chart(new_fig)

        if state["main"]["lr"]["step_i"] < len(state["main"]["lr"]["steps"]) - 1:
            state["main"]["lr"]["step_i"] += 1
            state["main"]["lr"]["errors"].append(error)
            state["main"]["lr"]["epochs"].append(epoch)

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


        else:
            if "normalization_params" not in inputs:
                st_theta.success(f"""
                Algo Completed 😊    
                $\\hat{{y}}={' + '.join(
                    ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
                ).replace('x_0', '')}$
                """)

                st_theta_completed.success(f"""
                Algo Completed 😊    
                $\\hat{{y}}={' + '.join(
                    ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
                ).replace('x_0', '')}$
                """)

            else:
                norm_mean, norm_std = inputs["normalization_params"]
                st_theta.success(f"""
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
                """)

                st_theta_completed.success(f"""
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
                """)

        st_error.plotly_chart(plotly_plot(state["main"]["lr"]["epochs"],
                                          state["main"]["lr"]["errors"],
                                          mode="lines+markers",
                                          x_title="epochs",
                                          y_title="error",
                                          title="Error Chart"))
