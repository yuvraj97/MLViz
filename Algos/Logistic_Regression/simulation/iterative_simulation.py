import streamlit as st
from plotly.graph_objs import Figure
from utils.plots import plotly_plot, surface3D


def run(f, plt, inputs):

    """
    It will plot Linear Regression Step by Step

    :param f: function (It will iteratively give us the parameters)
    :param plt: Figure (It contains our data points)
    :param inputs: dict
    :return: None
    """

    st_theta, st_error, st_plot = st.sidebar.empty(), st.empty(), st.empty()
    st_theta_completed = st.empty()

    n, d = inputs["X"].shape

    if "errors" not in st.session_state["Logistic Regression"]:
        st.session_state["Logistic Regression"]["errors"] = []
    if "epochs" not in st.session_state["Logistic Regression"]:
        st.session_state["Logistic Regression"]["epochs"] = []

    if "steps" not in st.session_state["Logistic Regression"]:
        st.session_state["Logistic Regression"]["steps"] = [(theta, error) for theta, error in f(inputs)]

    if "step_i" not in st.session_state["Logistic Regression"]:
        st.session_state["Logistic Regression"]["step_i"] = 0

    if inputs["step_button"]:

        (theta, error) = st.session_state["Logistic Regression"]["steps"][
            st.session_state["Logistic Regression"]["step_i"]
        ]

        epoch = len(st.session_state["Logistic Regression"]["epochs"]) + 1

        if d == 2:
            min_X, max_X = inputs["X"][:, 0].min(), inputs["X"][:, 0].max()
            new_fig: Figure = plotly_plot(
                [min_X, max_X],
                [
                    (- theta[0].item() - theta[1].item() * min_X) / theta[2].item(),
                    (- theta[0].item() - theta[1].item() * max_X) / theta[2].item(),
                ],
                fig=plt,
                mode="lines",
                color="green",
                do_not_change_fig=True,
                title=f"Logistic Regression (epoch: {epoch})"
            )
            st_plot.plotly_chart(new_fig)
        elif d == 3:
            min_X1, max_X1 = inputs["X"][:, 0].min(), inputs["X"][:, 0].max()
            min_X2, max_X2 = inputs["X"][:, 1].min(), inputs["X"][:, 1].max()
            description = {
                "title": {
                    "main": f"Logistic Regression (epoch: {epoch})",
                    "x": "Feature 1",
                    "y": "Feature 2",
                    "z": "Feature 3"
                },
                "label": {
                    "main": "Legend",
                },
                "hovertemplate": "Feature 1: %{x}<br>Feature 2: %{y}<br>Feature 3: %{z}",
                "color": "green"
            }
            new_fig: Figure = surface3D(
                [min_X1, max_X1],
                [min_X2, max_X2],
                [
                    [
                        (- theta[0].item() - theta[1].item() * min_X1 - theta[2].item() * min_X2) / theta[3].item(),
                        (- theta[0].item() - theta[1].item() * max_X1 - theta[2].item() * min_X2) / theta[3].item(),
                    ],
                    [
                        (- theta[0].item() - theta[1].item() * min_X1 - theta[2].item() * max_X2) / theta[3].item(),
                        (- theta[0].item() - theta[1].item() * max_X1 - theta[2].item() * max_X2) / theta[3].item(),
                    ],

                ],
                description,
                fig=plt
            )
            st_plot.plotly_chart(new_fig)

        if st.session_state["Logistic Regression"]["step_i"] < len(st.session_state["Logistic Regression"]["steps"]) - 1:
            st.session_state["Logistic Regression"]["step_i"] += 1
            st.session_state["Logistic Regression"]["errors"].append(error)
            st.session_state["Logistic Regression"]["epochs"].append(epoch)

            st_theta.info(f"""
            **Decision boundary**:    
            ${' + '.join(
                ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
            ).replace('x_0', '')}=0$
            """)
        else:
            s = f"""
            Algo Completed 😊    
            **Decision boundary**:    
            ${' + '.join(
                ['{:.2f}'.format(theta_i[0]) + f'x_{i}' for i, theta_i in enumerate(theta)]
            ).replace('x_0', '')}=0$
            """

            st_theta.success(s)
            st_theta_completed.success(s)

        st_error.plotly_chart(
            plotly_plot(
                st.session_state["Logistic Regression"]["epochs"],
                st.session_state["Logistic Regression"]["errors"],
                mode="lines+markers",
                x_title="epochs",
                y_title="error",
                title="Error Chart"
            )
        )
