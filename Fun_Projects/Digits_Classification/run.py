from typing import TextIO
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import streamlit as st
import numpy as np
from Fun_Projects.Digits_Classification.predict import predict


def run(state):

    # Create a canvas component
    st.write("""
    Draw a digit between **0** to **9**, and this program will predict your input.     
    Try to keep the drawing in center.
    """)

    canvas_result = st_canvas(
        stroke_width=22,
        stroke_color="white",
        background_color="black",
        update_streamlit=True,
        height=150,
        width=150,
        drawing_mode="freedraw",
        key="canvas",
    )
    image = canvas_result.image_data
    initializing = st.empty()
    initializing.info("Initializing ...")

    predictions = None
    if image is not None:

        initializing.empty()
        image = ImageOps.grayscale(Image.fromarray(np.uint8(image)).resize((28, 28)))
        image_np = np.array(image)
        if image_np.sum() != 0:
            predictions = predict(image_np)
            st.success(f"""
            ### The best prediction is: ${predictions[0]}$   
            2nd best prediction is: ${predictions[1]}$   
            3rd best prediction is: ${predictions[2]}$
            """)

    st.write("---")

    st.write("# Code")
    st.markdown("""
    First let's define the Neural network.    
    Here we will create a `model.py` file in that we will define our network's architecture.   
    """)

    f: TextIO = open("./Fun_Projects/Digits_Classification/model.py", "r")
    code: str = f.read()
    f.close()

    with st.beta_expander("model.py", True):
        st.code(code)

    st.markdown("""
    Now let's train our model on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset,
    and store that model in a file name `model.mnist.pytorch`.
    """)

    f: TextIO = open("./Fun_Projects/Digits_Classification/train.py", "r")
    code: str = f.read()
    f.close()

    with st.beta_expander("train.py", True):
        st.code(code)

    st.markdown("""
    Now it's time to make prediction, so here in `predict.py` we will take a `28 x 28` image as input,
    and returns the prediction.  
    Output of `predict` is an ordered list where the starting number is most likely
    and last number is least likely.  
    """)

    if predictions is not None:
        st.write(f"""
        Like in above example Output of `predict` is,  
        `{repr(list(predictions))}`
        """)

    f: TextIO = open("./Fun_Projects/Digits_Classification/predict.py", "r")
    code: str = f.read()
    f.close()

    with st.beta_expander("predict.py", True):
        st.code(code
                .replace('Fun_Projects.Digits_Classification.', '')
                .replace('Fun_Projects/Digits_Classification/', ''))
