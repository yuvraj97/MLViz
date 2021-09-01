import math
from typing import TextIO, Dict, Union, List
import streamlit as st
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from plotly.graph_objs import Figure

from Algos.K_Means_Clustering.simulation import kMean
from Algos.utils.plots import plot_classification_data
from Algos.utils.utils import get_nD_classification_data


def get_all_inputs() -> Dict[str, Union[int, float, List[float]]]:
    """
    Here we get all inputs from user
    :return: Dict[str, Union[str, int, float]]
    """

    with st.sidebar.beta_expander("Generate n dimensional synthetic data", True):
        st.write("")
        st_seed, st_n = st.beta_columns([1, 1])
        seed: int = int(st_seed.text_input("Enter seed (-1 mean seed is disabled)", "1"))
        n: int = int(st_n.text_input("N (number of training examples)", "100"))
        st_n_clusters, st_n_features = st.beta_columns([1, 1])
        n_clusters: int = int(st_n_clusters.text_input("Number of clusters", "2"))
        n_features: int = int(st_n_features.text_input("Number of features", "2"))

        st.write("### Clusters proportions")
        clusters_proportions = []
        st_clusters_proportions = [st.beta_columns([1] * 3) for _ in range(n_clusters // 3 + 1)]

        j = 0
        for j in range(n_clusters // 3):
            for i in range(3):
                clusters_proportions.append(
                    float(
                        st_clusters_proportions[j][i].text_input(
                            f"Cluster: {3 * j + i + 1}",
                            "{:.3f}".format(1 / n_clusters),
                            key=f"proportions-{j}-{i}"
                        )
                    )
                )
        for i in range(n_clusters % 3):
            clusters_proportions.append(
                float(
                    st_clusters_proportions[-1][i].text_input(
                        f"Cluster: {(j + 1) * 3 + i + 1}",
                        "{:.3f}".format(1 / n_clusters),
                        key=f"proportions-{-1}-{i}"
                    )
                )
            )

        if not math.isclose(sum(clusters_proportions), 1.0, abs_tol=0.01):
            st.error("Proportions should sum to $1$")
            raise ValueError("Algos.Logistic_Regression.run: Proportions should sum to $1$")

        st_lower_limit, st_upper_limit = st.beta_columns([0.5, 0.5])
        lower_limit: float = float(st_lower_limit.text_input("Lower Limit", "-10.0"))
        upper_limit: float = float(st_upper_limit.text_input("Upper Limit", "10.0"))

        st.write("### Gaussian Noise $\\mathcal{N}(\\mu,\\sigma^2)$")
        st_mean, st_std = st.beta_columns([1, 1])
        mean: float = float(st_mean.text_input("Mean", "0.0"))
        std: float = float(st_std.text_input("Standard deviation", "1.0"))

    d = {
        "seed": seed,
        "n": n,
        "n_clusters": n_clusters,
        "n_features": n_features,
        "clusters_proportions": clusters_proportions,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit,
        "mean": mean,
        "std": std,
    }

    return d


def run():

    inputs: Dict[str, Union[int, float, List[float]]] = get_all_inputs()

    if inputs["n"] * inputs["n_features"] > 1000:
        st.error("Sorry but currently this app doesn't support more then 1000 data points")
        return

    X: ndarray
    y: ndarray
    X, y = get_nD_classification_data(
        n_classes=inputs["n_clusters"],
        classes_proportions=inputs["clusters_proportions"],
        n_features=inputs["n_features"],
        n=inputs["n"],
        mean=inputs["mean"],
        std=inputs["std"],
        seed=inputs["seed"],
        coordinates_lim=(inputs["lower_limit"], inputs["upper_limit"])
    )
    n, d = X.shape

    st.write("# Data")
    df: DataFrame = pd.DataFrame(data=X.T,
                                 index=[f"x{i + 1}" for i in range(d)])
    df.columns += 1
    st.write(f"$\\text{{Features}}\\quad \\mathbb{{X}}_{{{n}\\times{d}}}$")
    st.write(df)

    if inputs["n_features"] in [2, 3]:
        plt: Figure = plot_classification_data(
            X, y, title="",
            x_title="Feature 1", y_title="Feature 2", z_title="Feature 3"
        )
        st.plotly_chart(plt)

    st.write("# Simulate K-Means Clustering")
    delay, K = st.beta_columns([1, 1])
    delay = delay.slider("Delay(ms)", 10, 1000, 100, 10)
    K = K.slider("Desired # of clusters",
                 min_value=1,
                 max_value=10,
                 value=inputs["n_clusters"],
                 step=1)

    if st.button("Begin Clustering !"):
        msg = "Clustering ..." if inputs["n_features"] <= 3 else \
                f"""
                Clustering ...    
                The data is ${inputs["n_features"]}$ dimensional and we can plot only up-to $3$ dimensions.    
                So plotting data points based on first $3$ dimensions.
                """
        with st.spinner(msg):
            st_labels, plot = st.empty(), st.empty()
            for labels, _ in kMean(K, X, max_epochs=10, delay=delay, seed=inputs["seed"]):
                fig = plot_classification_data(
                    X, labels, title="K-Means-Clustering",
                    x_title="Feature 1", y_title="Feature 2", z_title="Feature 3"
                )
                df: DataFrame = pd.DataFrame(data=labels.reshape((1, n)),
                                             index=["Labels"])
                df.columns += 1
                st_labels.write(df)
                plot.plotly_chart(fig)

    st.write("-----")

    f: TextIO = open("./Algos/K_Means_Clustering/code.py", "r")
    code: str = f.read()
    f.close()

    st.write("## Code")
    st.code(code)
