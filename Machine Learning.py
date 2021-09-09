import traceback
from typing import List
import streamlit as st


def main():
    """
    We will never use it in production it will came handy in development
    :return: None
    """

    algorithms: List[str] = [
        "Introduction",
        "Linear Regression",
        "Logistic Regression",
        "K Means Clustering"
    ]

    if "algorithm" in st.session_state:
        prev_idx = st.session_state["algorithm"]
    else:
        params = st.experimental_get_query_params()
        prev_idx = algorithms.index(params["algorithm"][0]) if "algorithm" in params else 0

    algorithm: str = st.selectbox("Choose Algorithm", algorithms, index=prev_idx)
    chosen_idx = algorithms.index(algorithm)
    if prev_idx != chosen_idx:
        st.session_state["algorithm"] = chosen_idx
        st.experimental_rerun()
    st.experimental_set_query_params(**{"algorithm": algorithm})

    algorithm = algorithms[prev_idx]
    exec(f"from Algos.{algorithm.replace(' ', '_')}.run import run;run()")


if __name__ == '__main__':

    try:
        main()
    except Exception as e:
        traceback.print_exc()
        st.error("Something went wrong!")
