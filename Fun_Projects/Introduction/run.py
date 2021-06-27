import streamlit as st


def run(state):
    st.info("""
    Here you can see some of my little projects.
    """)

    st.write("# Fun Projects")
    st.success("""
    **Linear Transformation**     
    - Visualize **3D** Linear Transformation
    - Visualize **2D** Linear Transformation   
    """)

    with st.beta_expander("Read More"):
        st.write("""
        Here we specify,   
        - **Transformation matrix**  
        - A **mathematical function**, like:  
            - $z = \\sqrt{9 - (x^2 + y^2)}$   
            - $z = \\sin(x) * \\cos(y)$   
            - $z = \\log(x) + e^{y}$
               
            Here we support most of the mathematical functions
        - Some **vectors**
        
        And you can visualize the **transformed** version of that **mathematical function** and those **vectors** 
        """)


    st.success("""
    **n color**     
    - Take an Image as input.
    - Reduces it's millions of colors to $n$ number of colors.
    - Uses **K-Means Clustering** Algorithm to achieve it.
    """)

    with st.beta_expander("Read More"):
        st.write("""
        An image carries millions of colors, here this program fetch $n$ colors from an image, 
        that best represent the image, then recreate image with those $n$ colors.   
        So image with millions of colors reduced to image with only $n$ colors.  
        
        It uses **K-Means Clustering** algorithm to achieve it.
        """)
