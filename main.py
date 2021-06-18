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
        "Linear Regression",
        "Logistic Regression"
    ]

    st_selectbox, st_reset = st.beta_columns([9, 1])
    option: str = st_selectbox.selectbox("", options, index=0)
    if st_reset.button("🔄", help="Reset Variables"):
        state["main"] = {}

    if option not in options:
        return

    exec(f"from Algos.{option.replace(' ', '_')}.run import run")
    exec("run(state)")


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

    main()

    st.sidebar.markdown("### Developer")
    st.sidebar.markdown("""
    LinkedIn: <a rel='noreferrer' target='_blank' href="https://www.linkedin.com/in/yuvraj97/">yuvraj97</a><br>
    Github: <a rel='noreferrer' target='_blank' href="https://github.com/yuvraj97/">yuvraj97</a><br>
    Email: <a rel='noreferrer' target='_blank' href="mailto:yuvraj@quantml.org">yuvraj@quantml.org</a><br>
    """, unsafe_allow_html=True)
