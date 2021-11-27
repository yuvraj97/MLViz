import math
from typing import Union, Dict, List

import streamlit as st
import pandas as pd
from pandas import DataFrame
from plotly.graph_objs import Figure

from Algos.utils.plots import plot_classification_data
from Algos.utils.synthetic_data import get_nD_classification_data


def get_rand_param_inputs():
    """
    Here we get all inputs from user
    :return: Dict[str, Union[int, float]]
    """

    with st.sidebar.expander("Generate n dimensional synthetic data"):
        st.write("")
        st_seed, st_n = st.columns([1, 1])
        seed: int = int(st_seed.text_input("Enter seed (-1 mean seed is disabled)", "0"))
        n: int = int(st_n.text_input("N (number of training examples)", "100"))
        n_features: int = int(st.text_input("Number of features", "2"))
        st_lower_limit, st_upper_limit = st.columns([0.5, 0.5])
        lower_limit: float = float(st_lower_limit.text_input("Lower Limit", "-10.0"))
        upper_limit: float = float(st_upper_limit.text_input("Upper Limit", "10.0"))

        st.write("### Classes proportions")
        st_prop = st.columns([1, 1])
        classes_proportions = [
            float(st_prop[0].text_input(f"Class: 1", "0.5")),
            float(st_prop[1].text_input(f"Class: 2", "0.5"))
        ]

        if not math.isclose(sum(classes_proportions), 1.0, abs_tol=0.01):
            st.error("Proportions should sum to $1$")
            raise ValueError("Algos.Logistic_Regression.run: Proportions should sum to $1$")

        st.write("### Gaussian Noise $\\mathcal{N}(\\mu,\\sigma^2)$")
        st_mean, st_std = st.columns([1, 1])
        mean: float = float(st_mean.text_input("Mean", "0.0"))
        std: float = float(st_std.text_input("Standard deviation", "1.0"))

    d = {
        "seed": seed,
        "n": n,
        "n_classes": 2,
        "n_features": n_features,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "classes_proportions": classes_proportions,
        "mean": mean,
        "std": std,
    }

    return d


def get_nn_def():
    if "nn_def" not in st.session_state["Neural Networks"]:
        st.session_state["Neural Networks"]["nn_def"] = {
            "n_hidden_layers": 1,
            "hidden_units_per_layer": [[True]]
        }

    st.write(f"""
    ### Create Neural Network Structure
    """)

    # st.write(st.session_state["Neural Networks"])
    layers = [1 for _ in range(st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"])]
    st_layers = st.columns(layers + [0.1])
    for li in range(len(layers)):
        layer = st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li]
        # st.write(layer)
        with st_layers[li]:
            for ui in range(len(layer)):
                st.write(f"$z_{{{ui + 1}}}^{{[{li + 1}]}}$ {'✅' if layer[ui] else '⛔'}")
                if st.button(
                        f"{'✅' if not layer[ui] else '⛔'}",
                        key=f"Remove layers for l{li}-{ui}",
                        help=f"""
                        Currently this node `{ui + 1}` of layer `{li + 1}` is `{'Activate' if layer[ui] else 'Deactivated'}`      
                        Click to `{'Deactivate' if layer[ui] else 'Activated'}` node `{ui + 1}` of layer `{li + 1}`    
                        """
                ):
                    layer[ui] = not layer[ui]
                    st.experimental_rerun()
                st.write("  ")
                st.write("  ")
            if st.button("➕", key=f"add more layers for l{li}", help=f"Add more neuron in layer `{li + 1}`"):
                layer.append(True)
                st.experimental_rerun()
    with st_layers[-1]:

        if st.button("➕", key="add more layers", help="Add one more layer"):
            st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"] += 1
            st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"].append([True])
            st.experimental_rerun()
        if st.button("❌", key="Remove Deactivated layers", help="Remove All Deactivated Neurons"):
            for li in range(len(layers)):
                st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li] = \
                    [_ for _ in st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"][li] if _]
            st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"] = \
            [_ for _ in st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"] if _]
            st.session_state["Neural Networks"]["nn_def"]["n_hidden_layers"] = \
            len(st.session_state["Neural Networks"]["nn_def"]["hidden_units_per_layer"])
            st.experimental_rerun()


def run():
    if "Neural Networks" not in st.session_state:
        st.session_state["Neural Networks"] = {}
    st.write("")
    st.write("")

    inputs = get_rand_param_inputs()
    X, y = get_nD_classification_data(
        n_classes=inputs["n_classes"],
        classes_proportions=inputs["classes_proportions"],
        n_features=inputs["n_features"],
        n=inputs["n"],
        mean=inputs["mean"],
        std=inputs["std"],
        seed=inputs["seed"]
    )
    n, d = X.shape

    plt: Union[Figure, None] = None
    if d in [2, 3]:
        plt = plot_classification_data(
            X, y, title="Data",
            x_title="Feature 1", y_title="Feature 2", z_title="Feature 3"
        )
        st.plotly_chart(plt)

    st_X, st_y = st.columns([d if d < 4 else 3, 1])
    with st_X:
        df: DataFrame = pd.DataFrame(data=X,
                                     columns=[f"x{i + 1}" for i in range(d)])
        df.index += 1
        st.write(f"$\\text{{Features}}\\quad \\mathbb{{X}}_{{{n}\\times{d}}}$")
        st.write(df)
    with st_y:
        df: DataFrame = pd.DataFrame(data=y, columns=["y"])
        df.index += 1
        st.write(f"**Class Labels**")
        st.write(df)

    st.write("---")
    get_nn_def()
    st.write("---")
