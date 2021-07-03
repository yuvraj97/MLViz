import streamlit as st


def run(state):

    st.success("""
    **Research Paper Summarizer**     
    - Give a **Research Paper** (PDF) as input.
    - Then this program will fetch the top **k** most important **sentences** from that research paper
    while maintaining a **maximum distance between sentences**.
    - It will also generate the summary of the **Research Paper** (PDF) in **k** sentences.
    - **Statistics + Dynamic Programming**.
    """)

    with st.beta_expander("Read More"):
        st.write("""
        Here you can input any **research paper** and 
        this program will fetch the top **k** most important **sentences** from that research paper.  
        So you can quickly see what the research paper is about.   
        
        Most interesting part of this program is the variable **D**.  
        **D**: Maximum distance between sentences.   
        You can specify that how far those **k** most important **sentences** should be.
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

    st.success("""
        **Digits Classification**     
        - Draw a digit from $0$ to $9$.
        - Then this program will predict that digit.
        - Uses **Neural Networks** to achieve it.
        """)

    with st.beta_expander("Read More"):
        st.write("""
        Here I used the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset to train a
        **Neural Network** (with $2$ hidden layers).  
        Which then is used to predict the digit you draw in the canvas.  
        
        Source code is available to you.
        """)

    st.success("""
        **Slicing STL**     
        - Give a **STL** file as input.
        - Then this program will create the **slices** of that STL file.
        - Then it finds the **sub cycles** in the corresponding slice.
        """)

    with st.beta_expander("Read More"):
        st.write("""
        In simple words **STL**(Standard Triangle Language) file stores information about **3D models**.  
        This format describes only the surface geometry of a three-dimensional object
        without any representation of **color**, **texture** or other common model attributes.
          
        Here this program take a binary representation of STL file as input.   
        Then it slice that STL object along z-axis, plot those slices.    
        Then it find the sub cycles in the corresponding slice, and plot those cycles. 
        """)
