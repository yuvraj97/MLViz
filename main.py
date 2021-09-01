import traceback
from typing import List
import streamlit as st


def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]


def main():
    """
    This is the entry point of our project
    :return: None
    """

    algorithms: List[str] = [
        "Introduction",
        "Linear Regression",
        "Logistic Regression",
        "K Means Clustering"
    ]

    st_algo, st_reset = st.beta_columns([9, 1])

    if st_reset.button("🔄", help="Reset Variables (Necessary to reset Manually Increment Steps)"):
        reset_session()

    if "algorithm" in st.session_state:
        prev_idx = st.session_state["algorithm"]
    else:
        params = st.experimental_get_query_params()
        prev_idx = algorithms.index(params["algorithm"][0]) if "algorithm" in params else 0

    algorithm: str = st_algo.selectbox("Algorithms", algorithms, index=prev_idx)
    chosen_idx = algorithms.index(algorithm)
    if prev_idx != chosen_idx:
        st.session_state["algorithm"] = chosen_idx
        st.experimental_rerun()
    st.experimental_set_query_params(**{"algorithm": algorithm})

    exec(f"from Algos.{algorithm.replace(' ', '_')}.run import run;run()")


if __name__ == '__main__':

    st.set_page_config(layout='centered', initial_sidebar_state='expanded')
    st.sidebar.markdown(
        f"""
        <a rel='noreferrer' target='_blank' href="https://www.quantml.org/">
            <img src="https://cdn.quantml.org/img/cover.webp" alt="QuantML" width="100%">
        </a><br><br>""",
        unsafe_allow_html=True
    )

    try:
        main()
    except Exception as e:
        traceback.print_exc()
        st.error("Something went wrong!")

    st.sidebar.write("-----")
    st.sidebar.write("""
    If you like this project, <br> then give it a ⭐ on [GitHub](https://github.com/yuvraj97/MLViz)
    <iframe 
        src="https://ghbtns.com/github-btn.html?user=yuvraj97&repo=MLViz&type=star&count=true&size=large" 
        frameborder="0" scrolling="0" width="170" height="30" title="GitHub">
    </iframe>""", unsafe_allow_html=True)

    st.sidebar.markdown("## Connect")
    st.sidebar.write("""
    <iframe 
        src="https://ghbtns.com/github-btn.html?user=yuvraj97&type=follow&count=true&size=large" 
        frameborder="0" scrolling="0" width="250" height="30" title="GitHub">
    </iframe>""", unsafe_allow_html=True)
    st.sidebar.markdown("""
    [Donate Here if you like this project](http://www.quantml.org/donate)    
    LinkedIn: [yuvraj97](https://www.linkedin.com/in/yuvraj97/)    
    Github: [yuvraj97](https://github.com/yuvraj97/)    
    Email: [yuvraj@quantml.org](mailto:yuvraj@quantml.org)
    """, unsafe_allow_html=True)
