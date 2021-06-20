from typing import Union, Dict
import numpy as np
import streamlit as st

from Fun_Projects.Linear_Algebra_Transformation.utils import str2matrix

def get_inputs():

    st.markdown("# Enter the transformation matrix")

    matrix: np.ndarray
    matrix, isValid = str2matrix(st.text_input(
        'Syntax(without quotes): "A, B; C, D" or "[A, B; C, D]" or "[[A, B]; [C, D]]" ',
        value="[2.0,  -0.5;   1.0,  0.5]"
    ))
    if isValid is False:
        st.warning("Invalid Syntax")
        return None
    if len(matrix[0]) != 2 or len(matrix[1]) != 2:
        st.warning(f"""
        Incorrect Shape.    
        Expected: (2, 2).    
        But got: ({len(matrix[0])}, {len(matrix[1])})
        """)
        return None

    a, b, c, d = float(matrix[0][0]), float(matrix[0][1]), float(matrix[1][0]), float(matrix[1][1])
    if st.sidebar.checkbox("Interact with Transformation matrix", False):
        st_a, st_b = st.sidebar.beta_columns([1, 1])
        st_c, st_d = st.sidebar.beta_columns([1, 1])
        a = st_a.slider('A', a - 10, a + 10, a, 0.1)
        b = st_b.slider('B', b - 10, b + 10, b, 0.1)
        c = st_c.slider('C', c - 10, c + 10, c, 0.1)
        d = st_d.slider('D', d - 10, d + 10, d, 0.1)
        matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1] = a, b, c, d
        st.sidebar.write("-----")


    return {
        "matrix": matrix
    }

def run(state):

    inputs: Dict[str, Union[np.ndarray]] = get_inputs()
    matrix: np.ndarray = inputs["matrix"]

    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    st.latex(f'''\\begin{{bmatrix}}
     A & B\\\\  
     C & D\\\\
    \\end{{bmatrix}}=
    \\begin{{bmatrix}}
     {a} & {b} \\\\
     {c} & {d}
    \\end{{bmatrix}}
    ''')

