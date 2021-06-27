import numpy as np
import plotly.graph_objs as go
from plotly.graph_objs import Surface
from plotly.subplots import make_subplots
from Fun_Projects.Linear_Transformation.Plot3DVectors import Plot3DVectors as vt3D

class Transform3D:
    """
    Example:

    transform3D = np.array([
                        [-1.0,  0.0,  0.0],
                        [ 0.0, -1.0,  0.0],
                        [ 0.0,  0.0, -1.0]
                       ])
    vectors3D = np.array([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 1],
                       ])
    
    
    tf3D = Transform3D(transform3D)
    tf3D.add_vectors(vectors3D)
    tf3D.add_equation(
                        lambda x, y: np.sqrt(9 - (x-0)**2 - (y-0)**2) + 0,
                        count   = 30,
                        opacity = 0.5,
                     )
    tf3D.show()
    """
    def __init__(self, tf_matrix: np.ndarray):
        self.tf_matrix: np.ndarray = tf_matrix
        self._plot_orig_ = vt3D("Original vectors")
        self._plot_tf_ = vt3D("Transformed vectors")
        self._plot_combine_ = vt3D("Original[blue] /Transformed[red] vectors")

    @staticmethod
    def __flatten__(x: np.ndarray, y: np.ndarray, z: np.ndarray):
        vectors: np.ndarray = np.empty(shape=(3, np.product(x.shape)))
        vectors[0] = x.reshape((np.product(x.shape)))
        vectors[1] = y.reshape((np.product(y.shape)))
        vectors[2] = z.reshape((np.product(z.shape)))
        return vectors

    def add_equation(self, equation,
                     x_range=None, y_range=None,
                     count: int = 10, opacity: float = 0.5,
                     colorscale: tuple = ("Viridis", "YlOrRd")):

        if x_range is None:
            x_range, y_range, _ = self._plot_combine_.get_ranges()

        x: np.ndarray = np.linspace(x_range[0], x_range[1], count)
        y: np.ndarray = np.linspace(y_range[0], y_range[1], count)
        x, y = np.meshgrid(x, y)
        z: np.ndarray = equation(x, y)

        trace1: Surface = go.Surface(x=x, y=y, z=z, opacity=opacity, colorscale=colorscale[0], showscale=False)

        vectors = self.__flatten__(x, y, z)
        vectors_tf = self.tf_matrix @ vectors
        x, y, z = vectors_tf[0].reshape(x.shape), vectors_tf[1].reshape(y.shape), vectors_tf[2].reshape(z.shape)

        trace2: Surface = go.Surface(x=x, y=y, z=z, opacity=opacity, colorscale=colorscale[1], showscale=False)

        self._plot_orig_.set_axes_limit(*vectors)
        self._plot_orig_.add_figures([trace1])

        self._plot_tf_.set_axes_limit(*vectors_tf)
        self._plot_tf_.add_figures([trace2])

        self._plot_combine_.set_axes_limit(*np.hstack((vectors, vectors_tf)))
        self._plot_combine_.add_figures([trace1, trace2])

    def add_vectors(self, vectors: np.ndarray):
        tf_vectors = (self.tf_matrix @ vectors.T).T
        self._plot_orig_.add_vectors(vectors, color='blue')
        self._plot_combine_.add_vectors(tf_vectors, color='red')
        self._plot_combine_.add_vectors(vectors, color='blue')
        self._plot_tf_.add_vectors(tf_vectors, color='red')

    def show(self):
        self._plot_combine_.show()
        self._plot_orig_.show()
        self._plot_tf_.show()
        pass

    def fig_side_by_side(self):
        fig = make_subplots(rows=1, cols=2,
                            specs=[[{'type': 'surface'}, {'type': 'surface'}]], )
        for trace in self._plot_orig_._fig_list_:
            fig.add_trace(trace=trace, row=1, col=1)

        for trace in self._plot_tf_._fig_list_:
            fig.add_trace(trace=trace, row=1, col=2)
        fig.update_layout(width=800, height=500, showlegend=False, )
        return fig

    def fig_combine(self):
        return self._plot_combine_.fig()

    def fig_orig(self):
        return self._plot_orig_.fig()

    def fig_tf(self):
        return self._plot_tf_.fig()

    def fig(self):
        return self._plot_orig_.fig(), self._plot_tf_.fig(), self._plot_combine_.fig()
