from typing import Union, Dict
import numpy as np
import streamlit as st

from Fun_Projects.Linear_Algebra_Transformation.utils import str2matrix, validate_equation, str2vec, vec2str


def get_inputs(state):

    is_rerun_require = False

    st.markdown("# Enter the transformation matrix")

    matrix: np.ndarray
    matrix, isValid = str2matrix(st.text_input(
        'Syntax(without quotes): "A, B; C, D" or "[A, B; C, D]" or "[[A, B]; [C, D]]" ',
        value="[2.0,  -0.5;   1.0,  0.5]"
    ))
    if isValid is False:
        st.warning("Invalid Syntax for transformation matrix")
        return None
    if len(matrix[0]) != 2 or len(matrix[1]) != 2:
        st.warning(f"""
        Incorrect Shape.    
        Expected: (2, 2).    
        But got: ({len(matrix[0])}, {len(matrix[1])})
        """)
        return None

    a, b, c, d = float(matrix[0][0]), float(matrix[0][1]), float(matrix[1][0]), float(matrix[1][1])
    with st.sidebar.beta_expander("Interact with Transformation matrix", False):
        st_a, st_b = st.beta_columns([1, 1])
        st_c, st_d = st.beta_columns([1, 1])
        a = st_a.slider('A', a - 10, a + 10, a, 0.1)
        b = st_b.slider('B', b - 10, b + 10, b, 0.1)
        c = st_c.slider('C', c - 10, c + 10, c, 0.1)
        d = st_d.slider('D', d - 10, d + 10, d, 0.1)
        matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1] = a, b, c, d

    equation = st.sidebar.text_input("Enter Equation", value="sqrt(9+x^2)")
    equation, isValid = validate_equation(equation)
    supported_f_str = """
    Here it support most of the function, like:    
    **sin(x)**, **cos(x)**, **e^(x)**, **log(x)**, ...    
    (If a function is supported by numpy you can use it here as well)    
    
    Examples:    
    **f(x) = sin(x) $*$ cos(x)**    
    **f(x) = e^(log(x)) + sin(2$*$pi$*$x)**    
    For a complete list visit [HERE](https://numpy.org/doc/stable/reference/routines.math.html)
    """
    if isValid is False:
        st.sidebar.warning(f"""
        Invalid Syntax for equation.    
        {supported_f_str}    
        """)
        return None

    with st.sidebar.beta_expander("Allowed Functions"):
        st.success(supported_f_str)

    st.sidebar.write("## Vectors")
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]]) if "vectors" not in state["main"]["la-tf"] else \
        state["main"]["la-tf"]["vectors"]

    for i, vector in enumerate(vectors):
        st_vec, st_del = st.sidebar.beta_columns([9, 1])
        vector, status = str2vec(st_vec.text_input(f"Vector V{i+1}", vec2str(vector)))
        if status is False:
            st.warning(f"Vector V{i+1}'s syntax is incorrect")
            return None
        if False in np.equal(vectors[i], vector):
            is_rerun_require = True
        vectors[i] = vector
        if st_del.button("x", key=f"vector-{i+1}-del", help=f"Delete Vector V{i+1}"):
            vectors = np.delete(vectors, i, axis=0)
            is_rerun_require = True
            break

    st_vec, st_add = st.sidebar.beta_columns([9, 1])
    vector, status = str2vec(st_vec.text_input(f"Add this vector", "[1.0, 0.0]"))
    if status is False:
        st.warning(f"Incorrect syntax")
        return None

    if st_add.button("+", key=f"Add this vector", help=f"Add this Vector"):
        is_rerun_require = True
        vectors = np.vstack((vectors, vector))

    state["main"]["la-tf"]["vectors"] = vectors

    if is_rerun_require: st.experimental_rerun()
    return {
        "matrix": matrix,
        "equation": equation
    }

def run(state):

    if "la-tf" not in state["main"]:
        state["main"]["la-tf"] = {}

    inputs: Dict[str, Union[np.ndarray]] = get_inputs(state)
    if inputs is None: return
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
