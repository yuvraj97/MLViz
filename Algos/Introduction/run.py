import streamlit as st

from Algos.utils.utils import intialize, footer


def run():

    intialize(None)

    st.write("# Machine Learning Algorithms (Visualizations)")

    st.markdown("""
    Here you can see **visualization of some ML algorithms**.  
    You can see the **source code** of each section, and use the **left control panel** to interact with algorithm.  
    
    ---
    """)

    #     [$1.$ **Linear Regression:**](https://app.quantml.org/machine-learning/linear-regression/)
    st.success("""
    $1.$ **Linear Regression:**
    - **Batch Gradient Descent** (From Scratch / PyTorch)
    - **Mini Batch Gradient Descent** (From Scratch / PyTorch)
    """)

    with st.expander("Read More", False):
        st.write("""
        Here you can create synthetic (random) date, you can control the randomness of the data.   
        Using Left control panel, like **seed**, **Gaussian Noise**, you can create **n-dimensional** data.  
        You can specify the relation ship between **features($\\mathbb{X}$)** and **output($y$)** like,
        - $y=2x_1 + 2$
        - $y=2x_1 + 3x_2 + 5$
        - $y=2x_1 + 9x_2 + 7x_3 + 10$   
        
        There are much more features like **Normalization**, **Simulation**, **iterative steps** you should explore them.
        """)

    #         [$2.$ **Logistic Regression:**](https://app.quantml.org/machine-learning/logistic-regression/)
    st.success("""
        $2.$ **Logistic Regression:**
        - **Batch Gradient Ascent ** (From Scratch / PyTorch)
        - **Mini Batch Gradient Descent** (From Scratch / PyTorch)
        """)

    with st.expander("Read More", False):
        st.write("""
        Here you can create synthetic (random) date, you can control the randomness of the data.   
        Using Left control panel, like **seed**, **Gaussian Noise**, you can create **n-dimensional** data.  
        You can specify **Number of features**, **Proportions for each classes**.   
        
        There are much more features like **Simulation**, **iterative steps**, **Regression parameters** you should explore them.
        """)

    #     [$3.$ **K-Means Clustering**](https://app.quantml.org/machine-learning/k-means-clustering/)
    st.success("""
    $3.$ **K-Means Clustering**
    """)

    with st.expander("Read More", False):
        st.write("""
        Here you can create synthetic (random) date, you can control the randomness of the data.   
        Using Left control panel, like **seed**, **Gaussian Noise**, you can create **n-dimensional** data.  
        You can specify **Number of Clusters**, **Number of features**, **Proportions for each classes**, **Desired Number of Clusters**.   
        
        Then after clicking on **Begin Clustering !** you can see **K-Means Clustering** in action.
        """)

    footer()
