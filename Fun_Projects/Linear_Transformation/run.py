from typing import Union, Dict, List
import numpy as np
import streamlit as st
from Fun_Projects.Linear_Transformation.Transform2D import Transform2D
from Fun_Projects.Linear_Transformation.Transform3D import Transform3D
from Fun_Projects.Linear_Transformation.utils import str2matrix, validate_equation, str2vec, vec2str


def get_2D_inputs(state):

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

    st.latex(f'''\\begin{{bmatrix}}
     A & B\\\\  
     C & D\\\\
    \\end{{bmatrix}}=
    \\begin{{bmatrix}}
     {a} & {b} \\\\
     {c} & {d}
    \\end{{bmatrix}}
    ''')

    with st.sidebar.beta_expander("Interact with Transformation matrix", False):
        st_a, st_b = st.beta_columns([1, 1])
        st_c, st_d = st.beta_columns([1, 1])
        a = st_a.slider('A', a - 10, a + 10, a, 0.1)
        b = st_b.slider('B', b - 10, b + 10, b, 0.1)
        c = st_c.slider('C', c - 10, c + 10, c, 0.1)
        d = st_d.slider('D', d - 10, d + 10, d, 0.1)
        matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1] = a, b, c, d

    equation = st.sidebar.text_input("Enter Equation", value="sqrt(9-x^2)")
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

    count = st.sidebar.number_input("Number of data points", min_value=1, max_value=100,  value=30, step=1)
    if st.sidebar.checkbox("Manually Specify Range", value=True):
        range_, status = str2vec(st.sidebar.text_input('Syntax(without quotes): "A, B" or "[A, B]"', value="[-3, 3]"))
        if status is False:
            st.sidebar.error('Invalid syntax')
            return None
    else:
        range_ = (-3, 3)

    st.sidebar.write("## Vectors")
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]]) if "vectors" not in state["main"]["la-tf"] else \
        state["main"]["la-tf"]["vectors"]

    for i, vector in enumerate(vectors):
        st_vec, st_del = st.sidebar.beta_columns([9, 1])
        vector, status = str2vec(st_vec.text_input(f"Vector V{i+1}", vec2str(vector)))
        if status is False:
            st.sidebar.warning(f"Vector V{i+1}'s syntax is incorrect")
            return None
        if len(vector) != 2:
            st.sidebar.warning(f"Vector size should be $2$ but got ${len(vector)}$")
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
        st.sidebar.warning(f"Incorrect syntax")
        return None

    if st_add.button("+", key=f"Add this vector", help=f"Add this Vector"):
        is_rerun_require = True
        vectors = np.vstack((vectors, vector))

    state["main"]["la-tf"]["vectors"] = vectors

    if is_rerun_require: st.experimental_rerun()
    return {
        "matrix": matrix,
        "equation": equation,
        "vectors": vectors,
        "count": count,
        "range": range_
    }


def get_3D_inputs(state):
    is_rerun_require = False

    st.markdown("# Enter the transformation matrix")

    matrix: np.ndarray
    matrix, isValid = str2matrix(st.text_area(
        'Syntax(without quotes): "[a11, a12, a13 ;   a21, a22, a23 ;   a31, a32, a33]"',
        value="""   [    [-1.0,  0.0,  0.0]; 
        [0.0,  -1.0,  0.0]; 
        [0.0,   0.0,  -1.0]   ]"""
    ))
    if isValid is False:
        st.warning("Invalid Syntax for transformation matrix")
        return None
    if len(matrix[0]) != 3 or len(matrix[1]) != 3 or len(matrix[2]) != 3:
        st.warning(f"""
        Incorrect Shape.    
        Expected: (3, 3, 3).    
        But got: ({len(matrix[0])}, {len(matrix[1])}, {len(matrix[2])})
        """)
        return None

    a11, a12, a13 = float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])
    a21, a22, a23 = float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])
    a31, a32, a33 = float(matrix[2][0]), float(matrix[2][1]), float(matrix[2][2])

    with st.sidebar.beta_expander("Interact with Transformation matrix", False):
        st_a11, st_a12, st_a13 = st.beta_columns([1, 1, 1])
        st_a21, st_a22, st_a23 = st.beta_columns([1, 1, 1])
        st_a31, st_a32, st_a33 = st.beta_columns([1, 1, 1])

        a11 = st_a11.slider('a11', a11 - 10, a11 + 10, a11, 0.1)
        a12 = st_a12.slider('a12', a12 - 10, a12 + 10, a12, 0.1)
        a13 = st_a13.slider('a13', a13 - 10, a13 + 10, a13, 0.1)

        a21 = st_a21.slider('a21', a21 - 10, a21 + 10, a21, 0.1)
        a22 = st_a22.slider('a22', a22 - 10, a22 + 10, a22, 0.1)
        a23 = st_a23.slider('a23', a23 - 10, a23 + 10, a23, 0.1)

        a31 = st_a31.slider('a31', a31 - 10, a31 + 10, a31, 0.1)
        a32 = st_a32.slider('a32', a32 - 10, a32 + 10, a32, 0.1)
        a33 = st_a33.slider('a33', a33 - 10, a33 + 10, a33, 0.1)

        matrix[0][0], matrix[0][1], matrix[0][2] = a11, a12, a13
        matrix[1][0], matrix[1][1], matrix[1][2] = a21, a22, a23
        matrix[2][0], matrix[2][1], matrix[2][2] = a31, a32, a33

    st.latex(f'''\\begin{{bmatrix}}
     a_{{11}} & a_{{12}} & a_{{13}}\\\\  
     a_{{21}} & a_{{21}} & a_{{21}}\\\\
     a_{{31}} & a_{{31}} & a_{{31}}\\\\
    \\end{{bmatrix}}=
    \\begin{{bmatrix}}
     {a11} & {a12} & {a13} \\\\
     {a21} & {a22} & {a23} \\\\
     {a31} & {a32} & {a33} \\\\
    \\end{{bmatrix}}
    ''')

    equation = st.sidebar.text_input("Enter equation f(x, y)=", value="sqrt[9 - (x-0)^2 - (y-0)^2] + 0")
    equation, isValid = validate_equation(equation)
    supported_f_str = """
    Here it support most of the function, like:    
    **sin(x)**, **cos(x)**, **e^(x)**, **log(x)**, ...    
    (If a function is supported by numpy you can use it here as well)    

    Examples:   
    **f(x, y) = sin(x) * cos(y)**    
    **f(x, y) = e^(log(x)) + sin(2$*$pi$*$y)**   
                           
    For a complete list visit [HERE](https://numpy.org/doc/stable/reference/routines.math.html)
    """
    if isValid is False:
        st.sidebar.warning(f"""
        **Invalid Syntax for equation.**    
        {supported_f_str}    
        """)
        return None

    with st.sidebar.beta_expander("Allowed Functions"):
        st.success(supported_f_str)

    st_opacity, st_count = st.sidebar.beta_columns([1, 2])
    opacity = st_opacity.slider('Opacity', 0.2, 1.0, 0.6, 0.05)
    count = st_count.number_input("Number of data points", min_value=1, max_value=50,  value=30, step=1)
    if st.sidebar.checkbox("Manually Specify Range", value=True):
        range_, status = str2vec(st.sidebar.text_input('Syntax(without quotes): "A, B" or "[A, B]"', value="[-3, 3]"))
        if status is False:
            st.sidebar.error('Invalid syntax')
            return None
    else:
        range_ = (-3, 3)

    st.sidebar.write("## Vectors")
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) if "vectors" not in state["main"]["la-tf"] else \
        state["main"]["la-tf"]["vectors"]

    for i, vector in enumerate(vectors):
        st_vec, st_del = st.sidebar.beta_columns([9, 1])
        vector, status = str2vec(st_vec.text_input(f"Vector V{i + 1}", vec2str(vector)))
        if status is False:
            st.sidebar.warning(f"Vector V{i + 1}'s syntax is incorrect")
            return None
        if len(vector) != 3:
            st.sidebar.warning(f"Vector size should be $3$ but got ${len(vector)}$")
            return None
        if False in np.equal(vectors[i], vector):
            is_rerun_require = True
        vectors[i] = vector
        if st_del.button("x", key=f"vector-{i + 1}-del", help=f"Delete Vector V{i + 1}"):
            vectors = np.delete(vectors, i, axis=0)
            is_rerun_require = True
            break

    st_vec, st_add = st.sidebar.beta_columns([9, 1])
    vector, status = str2vec(st_vec.text_input(f"Add this vector", "[1.0, 0.0, 0.0]"))
    if status is False:
        st.sidebar.warning(f"Incorrect syntax")
        return None

    if st_add.button("+", key=f"Add this vector", help=f"Add this Vector"):
        is_rerun_require = True
        vectors = np.vstack((vectors, vector))

    state["main"]["la-tf"]["vectors"] = vectors

    if is_rerun_require: st.experimental_rerun()
    return {
        "matrix": matrix,
        "equation": equation,
        "vectors": vectors,
        "count": count,
        "opacity": opacity,
        "range": range_
    }


def run(state):

    if "la-tf" not in state["main"]:
        state["main"]["la-tf"] = {}

    dim = st.sidebar.selectbox("3D/2D", ["3D", "2D"])
    if "dim" not in state["main"]["la-tf"]: state["main"]["la-tf"]["dim"] = dim
    if state["main"]["la-tf"]["dim"] != dim:
        state["main"]["la-tf"] = {}
        state["main"]["la-tf"]["dim"] = dim
        st.experimental_rerun()

    if dim == "2D":
        inputs: Dict[str, Union[np.ndarray, int, List[int]]] = get_2D_inputs(state)
        if inputs is None: return
        matrix: np.ndarray = inputs["matrix"]
        st.write("# Vectors")
        tf2D = Transform2D(matrix)
        if len(inputs["vectors"]) != 0:
            tf2D.add_vectors(inputs["vectors"])
        tf2D.add_equation(inputs["equation"], x_range=inputs["range"], count=inputs["count"])
        fig_orig, fig_tf, fig_combine = tf2D.fig()
        st.pyplot(fig_combine)
        st.pyplot(fig_orig)
        st.pyplot(fig_tf)
    else:
        inputs: Dict[str, Union[np.ndarray, int, List[int]]] = get_3D_inputs(state)
        if inputs is None: return
        matrix: np.ndarray = inputs["matrix"]
        st.write("# Vectors")
        tf3D = Transform3D(matrix)
        if len(inputs["vectors"]) != 0:
            tf3D.add_vectors(inputs["vectors"])
        tf3D.add_equation(
            inputs["equation"],
            x_range=inputs["range"], y_range=inputs["range"],
            count=inputs["count"], opacity=inputs["opacity"])
        fig_combine = tf3D.fig_combine()
        st.plotly_chart(fig_combine)
        if st.checkbox("Plot Side by Side", True):
            fig_sbs = tf3D.fig_side_by_side()
            st.plotly_chart(fig_sbs)
        else:
            fig_orig = tf3D.fig_orig()
            fig_tf = tf3D.fig_tf()
            st.plotly_chart(fig_orig)
            st.plotly_chart(fig_tf)
