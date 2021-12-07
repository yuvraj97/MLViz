import streamlit as st


def intialize(title: str):
    st.set_page_config(
        page_title=title,
        page_icon="https://www.quantml.org/data/img/app.webp",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": None,
            "Report a Bug": None,
        }
    )
    st.sidebar.markdown(
        f"""
        <a rel='noreferrer' target='_blank' href="https://app.quantml.org/">
            <img src="https://cdn.quantml.org/img/cover.webp" alt="QuantML" width="100%">
        </a><br><br>""",
        unsafe_allow_html=True
    )

    hamburger_correction()


def hamburger_correction():
    st.markdown("""
    <style>
    /* Set the top padding */
    .block-container
    {
        padding-top: 2rem;
    }

    /* This is to hide Streamlit footer */
    footer {visibility: hidden;}
    /*
    If you did not hide the hamburger menu completely,
    you can use the following styles to control which items on the menu to hide.
    */
    ul[data-testid=main-menu-list] > ul:nth-of-type(4), /* About */
    ul[data-testid=main-menu-list] > div:nth-of-type(1), /* 1st divider */
    ul[data-testid=main-menu-list] > div:nth-of-type(2), /* 2nd divider */
    ul[data-testid=main-menu-list] > div:nth-of-type(3) /* 3rd divider */
        {display: none;}

    /* Sidebar */
    section[data-testid=stSidebar] > div
    {
        padding-top: 1.5rem;
    }

    </style>
    """, unsafe_allow_html=True)


def footer():
    st.sidebar.write("-----")

    st.sidebar.write("""
    If you like this project, <br> then give it a ⭐ on [GitHub](https://github.com/yuvraj97/MLViz)
    <iframe
        src="https://ghbtns.com/github-btn.html?user=yuvraj97&repo=MLViz&type=star&count=true&size=large"
        frameborder="0" scrolling="0" width="170" height="30" title="GitHub">
    </iframe>
    
    <iframe
        src="https://ghbtns.com/github-btn.html?user=yuvraj97&type=follow&count=true&size=large"
        frameborder="0" scrolling="0" width="250" height="30" title="GitHub">
    </iframe>
    
    [![Patreon](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fshieldsio-patreon.vercel.app%2Fapi%3Fusername%3Dquantml%26type%3Dpatrons&style=for-the-badge)](https://patreon.com/quantml)    
    [![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yuvraj97/)    
    **Email: [`yuvraj@quantml.org`](mailto:yuvraj@quantml.org)**  
    """, unsafe_allow_html=True)


def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]
