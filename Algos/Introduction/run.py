import streamlit as st


def run():

    st.write("---")

    st.markdown(
        "![Yuvraj's GitHub stats](https://github-readme-stats.vercel.app/api?username=yuvraj97&show_icons=true&theme"
        "=radical&include_all_commits=true&count_private=true)",
        unsafe_allow_html=True
    )

    st.info("""
    Here you can see **visualization of some ML algorithms**, and some **fun projects**.  
    You can see the **source code** of each section, and use the **left control panel** to interact with algorithm.
    """)

    st.write("# Algorithms (Visualizations)")

    st.success("""
    **Linear Regression:**
    - **Batch Gradient Descent** (From Scratch / PyTorch)
    - **Mini Batch Gradient Descent** (From Scratch / PyTorch)
    """)

    with st.beta_expander("Read More", False):
        st.write("""
        Here you can create synthetic (random) date, you can control the randomness of the data.   
        Using Left control panel, like **seed**, **Gaussian Noise**, you can create **n-dimensional** data.  
        You can specify the relation ship between **features($\\mathbb{X}$)** and **output($y$)** like,
        - $y=2x_1 + 2$
        - $y=2x_1 + 3x_2 + 5$
        - $y=2x_1 + 9x_2 + 7x_3 + 10$   
        
        There are much more features like **Normalization**, **Simulation**, **iterative steps** you should explore them.
        """)

    st.success("""
        **Logistic Regression:**
        - **Batch Gradient Ascent ** (From Scratch / PyTorch)
        - **Mini Batch Gradient Descent** (From Scratch / PyTorch)
        """)

    with st.beta_expander("Read More", False):
        st.write("""
        Here you can create synthetic (random) date, you can control the randomness of the data.   
        Using Left control panel, like **seed**, **Gaussian Noise**, you can create **n-dimensional** data.  
        You can specify **Number of features**, **Proportions for each classes**.   
        
        There are much more features like **Simulation**, **iterative steps**, **Regression parameters** you should explore them.
        """)

    st.success("""
    **K-Means Clustering**
    """)

    with st.beta_expander("Read More", False):
        st.write("""
        Here you can create synthetic (random) date, you can control the randomness of the data.   
        Using Left control panel, like **seed**, **Gaussian Noise**, you can create **n-dimensional** data.  
        You can specify **Number of Clusters**, **Number of features**, **Proportions for each classes**, **Desired Number of Clusters**.   
        
        Then after clicking on **Begin Clustering !** you can see **K-Means Clustering** in action.
        """)

