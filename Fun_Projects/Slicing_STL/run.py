from collections import defaultdict
import numpy as np
from pandas import DataFrame
from stl import mesh, Mesh
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math
import streamlit as st
from streamlit import components


def get_cycles(xs, ys):
    _xs, _ys = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        if not (xs[i] == xs[i - 1] and ys[i] == ys[i - 1]):
            _xs.append(xs[i])
            _ys.append(ys[i])
    xs, ys = _xs, _ys
    past_coordinates = [(xs[0], ys[0])]
    shapes = []
    i = 1
    while i < len(xs):
        coor_i = (xs[i], ys[i])
        if coor_i not in past_coordinates:
            past_coordinates.append(coor_i)
        else:
            past_coordinates.append(coor_i)
            past_coordinates = past_coordinates[past_coordinates.index(coor_i):]
            if len(past_coordinates) > 3:
                shapes.append(past_coordinates[past_coordinates.index(coor_i):])
            if i + 1 < len(xs): past_coordinates = [(xs[i + 1], ys[i + 1])]
            i += 1
        i += 1
    return shapes


def mesh_plot(M: Mesh):
    fig = plt.figure()
    axes = mplot3d.Axes3D(fig)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(M.vectors))
    scale = M.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    return fig


def run(state):

    st.info("""
    In simple words **STL**(Standard Triangle Language) file stores information about **3D models**.  
    This format describes only the surface geometry of a three-dimensional object
    without any representation of **color**, **texture** or other common model attributes.  
    """)

    st.write("""
    Here this program take a binary representation of STL file as input.   
    Then it slice that STL object along z-axis, plot those slices.    
    Then it find the sub cycles in the corresponding slice, and plot those cycles.   
    
    This is the [default.stl](https://cdn.quantml.org/share/stl/default.stl) used (if no input is provided)
    """)

    with st.beta_expander("Visualize your STL file"):
        st.write("""
        We use [viewstl.com](https://www.viewstl.com/) to visualize the STL file.
        """)
        components.v1.iframe("https://www.viewstl.com/", width=660, height=700, scrolling=False)

    fh = st.file_uploader("Upload file", ['stl'])
    if fh is not None:
        M: Mesh = mesh.Mesh.from_file("stl", fh=fh)
    else:
        M: Mesh = mesh.Mesh.from_file("./Fun_Projects/Slicing_STL/default.stl")

    vectors, index = [], []
    for i, ith_vectors in enumerate(M.vectors):
        vectors.extend(ith_vectors)
        vectors.append(["-", "-", "-"])
        index.extend([
            f"Vector {3 * i + 1}",
            f"Vector {3 * i + 2}",
            f"Vector {3 * i + 3}",
            "-"])
    vectors.pop()
    index.pop()
    vectors = np.array(vectors)
    df: DataFrame = pd.DataFrame(data=vectors,
                                 columns=["x", "y", "z"],
                                 index=index)

    st_x, st_y, st_z = st.beta_columns([1, 1, 1])
    st_plt, st_df = st.beta_columns([0.55, 0.45])

    x = st_x.slider("Rotate(x)", min_value=0, max_value=360, value=90, step=1)
    y = st_y.slider("Rotate(y)", min_value=0, max_value=360, value=0, step=1)
    z = st_z.slider("Rotate(z)", min_value=0, max_value=360, value=0, step=1)
    M.rotate([-1, 0, 0], math.radians(x))
    M.rotate([0, -1, 0], math.radians(y))
    M.rotate([0, 0, -1], math.radians(z))
    st_plt.pyplot(mesh_plot(M))
    st_df.write(df)

    zs = defaultdict(lambda: [])
    for idx_i, _zs in enumerate(M.z):
        for idx_j, z in enumerate(_zs):
            zs[z].append((idx_i, idx_j))

    st.write("# Slices of object along **z-axis**")

    allowed_zs,  figures = [], []
    for z_height in sorted(zs):

        indices = np.array(zs[z_height])
        xs = [x for x in M.x[indices[:, 0], indices[:, 1]]]
        ys = [y for y in M.y[indices[:, 0], indices[:, 1]]]

        # if len(xs) < 4: continue
        if len(np.unique(xs)) < 2: continue
        if len(np.unique(ys)) < 2: continue

        fig = plt.figure()
        plt.scatter(xs, ys, c='y')
        plt.plot(xs, ys, c='b')
        plt.title(f"z_height: {z_height}")
        figures.append(fig)
        allowed_zs.append(z_height)

    for fig1, fig2 in zip(figures[::2], figures[1::2]):
        st_fig1, st_fig2 = st.beta_columns([1, 1])
        st_fig1.pyplot(fig1)
        st_fig2.pyplot(fig2)

    if len(figures) % 2 != 0:
        st_fig1, st_fig2 = st.beta_columns([1, 1])
        st_fig1.pyplot(figures[-1])

    sub_cycle_title = st.empty()
    z_height = st.selectbox("Select z-height to get the sub cycles", allowed_zs)
    sub_cycle_title.write(f"# All sub-cycles for z-height: {z_height}")

    indices = np.array(zs[z_height])
    xs = np.array([x for x in M.x[indices[:, 0], indices[:, 1]]])
    ys = np.array([y for y in M.y[indices[:, 0], indices[:, 1]]])

    shapes = get_cycles(xs, ys)

    figures = []
    for idx, shape in enumerate(shapes):
        shape = np.array(shape)
        xs = shape[:, 0]
        ys = shape[:, 1]
        fig = plt.figure()
        plt.scatter(xs, ys, c='y')
        plt.plot(xs, ys, c='b')
        plt.title(f"Sub cycle for z-height: {z_height}")
        figures.append((len(xs), fig))
    figures.sort(key=lambda ele: ele[0])
    figures.reverse()

    for (_, fig1), (_, fig2) in zip(figures[::2], figures[1::2]):
        st_fig1, st_fig2 = st.beta_columns([1, 1])
        st_fig1.pyplot(fig1)
        st_fig2.pyplot(fig2)

    if len(figures) % 2 != 0:
        st_fig1, st_fig2 = st.beta_columns([1, 1])
        st_fig1.pyplot(figures[-1][1])
