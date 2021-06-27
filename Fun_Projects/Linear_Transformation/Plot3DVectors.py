from typing import List, Union
import numpy as np
import plotly.graph_objs as go
from plotly.graph_objs import Scatter3d, Layout, Figure, Surface
from plotly.offline import plot


class Plot3DVectors:
    """
    # Example:
    vectors = np.array([
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                       ])

    plt3D = Plot3DVectors("Vectors")
    plt3D.add_vectors(vectors)
    plt3D.show()
    """

    def __init__(self, title="3D Vectors"):
        self._title_ = title
        self._x_range_: List[int] = [0, -np.inf]
        self._y_range_: List[int] = [0, -np.inf]
        self._z_range_: List[int] = [0, -np.inf]
        self._fig_list_: List[Union[Scatter3d, Surface]] = []

    def add_vectors(self, vectors: np.ndarray, color="blue", legend="", showlegend=True):
        self.set_axes_limit(vectors[0], vectors[1], vectors[2])
        line_vectors: np.ndarray = np.zeros(shape=(3, len(vectors) * 3))
        for i in range(1, line_vectors.shape[1]):
            if i % 3 == 1:
                line_vectors[:, i] = vectors[i//3]
            if i % 3 == 2:
                line_vectors[:, i] = None

        vectors_fig: Scatter3d = go.Scatter3d(
            x=line_vectors[0],
            y=line_vectors[1],
            z=line_vectors[2],
            mode='lines',
            name=legend,
            marker=dict(color=color),
            showlegend=showlegend
        )

        scatter: Scatter3d = go.Scatter3d(
            x=vectors[:, 0],
            y=vectors[:, 1],
            z=vectors[:, 2],
            mode='markers',
            marker=dict(color=color),
            name='',
            showlegend=False
        )

        txt_plt: Scatter3d = go.Scatter3d(
            x=vectors[:, 0],
            y=vectors[:, 1],
            z=vectors[:, 2],
            mode='text',
            text=[f"[{v[0]}, {v[1]}, {v[2]}]" for v in vectors],
            marker={'opacity': 0.3},
            textfont={'size': 10, 'color': color},
            name="Co-ordinates",
            showlegend=showlegend
        )

        self._fig_list_.extend([vectors_fig, scatter, txt_plt])

    def add_figures(self, figs):
        self._fig_list_.extend(figs)

    def set_axes_limit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self._x_range_ = [min(np.min(x), self._x_range_[0]), max(np.max(x), self._x_range_[1])]
        self._y_range_ = [min(np.min(y), self._y_range_[0]), max(np.max(y), self._y_range_[1])]
        self._z_range_ = [min(np.min(z), self._z_range_[0]), max(np.max(z), self._z_range_[1])]

    def fig(self):
        layout: Layout = go.Layout(
            scene=dict(
                xaxis=dict(range=[self._x_range_[0] - 0.5, self._x_range_[1] + 0.5]),
                yaxis=dict(range=[self._y_range_[0] - 0.5, self._y_range_[1] + 0.5]),
                zaxis=dict(range=[self._z_range_[0] - 0.5, self._z_range_[1] + 0.5])
            ),
            title=dict(text=self._title_)
        )
        fig: Figure = go.Figure(data = self._fig_list_, layout = layout)
        fig.update_layout(width=800, height=800,)
        return fig

    def get_ranges(self):
        return self._x_range_, self._y_range_, self._z_range_,

    def show(self):
        plot(self.fig())
