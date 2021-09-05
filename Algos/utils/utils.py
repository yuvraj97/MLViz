import streamlit as st


def intialize(title: str):
    # st.set_page_config(layout='centered', initial_sidebar_state='expanded')
    st.sidebar.markdown(
        f"""
        <a rel='noreferrer' target='_blank' href="https://app.quantml.org/">
            <img src="https://cdn.quantml.org/img/cover.webp" alt="QuantML" width="100%">
        </a><br><br>""",
        unsafe_allow_html=True
    )

    if title:
        st_title, st_reset = st.columns([9, 1])

        st_title.title(title)
        if st_reset.button("🔄", help="Reset Variables (Necessary to reset Manually Increment Steps)"):
            reset_session()

    hamburger_correction()


def hamburger_correction():
    st.markdown("""
    <style>
    /* Set the top padding */
    .css-hi6a2p {padding-top: 1rem;}

    /* This is to hide Streamlit footer */
    footer {visibility: hidden;}
    /*
    If you did not hide the hamburger menu completely,
    you can use the following styles to control which items on the menu to hide.
    */
    ul[data-testid=main-menu-list] > li:nth-of-type(3), /* Deploy this app */
    ul[data-testid=main-menu-list] > li:nth-of-type(5), /* Documentation */
    ul[data-testid=main-menu-list] > li:nth-of-type(6), /* Ask a question */
    ul[data-testid=main-menu-list] > li:nth-of-type(7), /* Report a bug */
    ul[data-testid=main-menu-list] > li:nth-of-type(8), /* Streamlit for Teams= */
    ul[data-testid=main-menu-list] > li:nth-of-type(10), /* About */
    ul[data-testid=main-menu-list] > div:nth-of-type(1), /* 1st divider */
    ul[data-testid=main-menu-list] > div:nth-of-type(2), /* 2nd divider */
    ul[data-testid=main-menu-list] > div:nth-of-type(3) /* 3rd divider */
        {display: none;}
    </style>
    """, unsafe_allow_html=True)


def footer():
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

    # with st.sidebar.container():
    #     st.markdown(
    #         "![Yuvraj's GitHub stats](https://github-readme-stats.vercel.app/api?username=yuvraj97"
    #         "&show_icons=true&theme=radical&include_all_commits=true&count_private=true)",
    #         unsafe_allow_html=True
    #     )


def reset_session():
    for key in st.session_state.keys():
        del st.session_state[key]
