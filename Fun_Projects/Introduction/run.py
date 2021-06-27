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
