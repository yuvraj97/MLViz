import math
from typing import Union, Dict, List

import streamlit as st
import pandas as pd
from pandas import DataFrame
from plotly.graph_objs import Figure

from Algos.Neural_Networks.utils import Canvas, DrawNN, Params
from Algos.utils.plots import plot_classification_data
from Algos.utils.synthetic_data import get_nD_classification_data


def run():
    if "Neural Networks" not in st.session_state:
        st.session_state["Neural Networks"] = {}
    st.write("")
    st.write("")
    fetch_type = st.sidebar.selectbox("Get Data", ["Draw 2D Data", "Generate Random Data"])
    if fetch_type == "Draw 2D Data":
        X, y, classes_labels_mapping, norm_params = Canvas.get_canvas_data()
        if X is None: return
    elif fetch_type == "Generate Random Data":
        inputs = Params.get_random_data_inputs()
        X, y = get_nD_classification_data(
            n_classes=inputs["n_classes"],
            classes_proportions=inputs["classes_proportions"],
            n_features=inputs["n_features"],
            n=inputs["n"],
            mean=inputs["mean"],
            std=inputs["std"],
            seed=inputs["seed"]
        )
    else:
        return

    n, d = X.shape

    if fetch_type == "Generate Random Data":
        if d in [2, 3]:
            plt: Figure = plot_classification_data(
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
    DrawNN.draw_nn()
    st.write("---")
