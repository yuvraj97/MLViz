from typing import List
import streamlit as st
import SessionState


def main():
    """
    This is the entry point of our project
    :return: None
    """

    if state["main"] is None:
        state["main"] = {}

    options: List[str] = [
        "Visualize Algorithms",
        "Fun Projects"
    ]

    projects: List[str] = [
        "Introduction",
        "Research Paper Summarizer",
        "Linear Transformation",
        "n color",
        "Digits Classification",
        "Slicing STL"
    ]

    algorithms: List[str] = [
        "Introduction",
        "Linear Regression",
        "Logistic Regression",
        "K Means Clustering"
    ]

    st_options, st_proj_algo, st_reset = st.beta_columns([4.5, 4.5, 1])

    if st_reset.button("🔄", help="Reset Variables (Necessary to reset Manually Increment Steps)"):
        state["main"] = {}

    option: str = st_options.selectbox("Algos/Projects", options, index=0)
    if option == "Visualize Algorithms":
        proj_algo: str = st_proj_algo.selectbox("Algorithms", algorithms, index=0)
    else:
        proj_algo: str = st_proj_algo.selectbox("Fun Projects", projects, index=0)

    if option == "Visualize Algorithms":
        exec(f"from Algos.{proj_algo.replace(' ', '_')}.run import run;run(state)")
    else:
        exec(f"from Fun_Projects.{proj_algo.replace(' ', '_')}.run import run;run(state)")


if __name__ == '__main__':
    state = SessionState.get_state()

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
    except:
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
