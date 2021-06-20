import numpy as np
import matplotlib.pyplot as plt
from Fun_Projects.Linear_Transformation.Plot2DVectors import Plot2DVectors as vt2D

class Transform2D:
    """
    tf_matrix = np.array([
                        [2.0, -0.5],
                        [1.0,  0.5],
                       ])
    vectors = np.array([
                        [1.0, 0.0],
                        [0.0, 1.0],
                       ])


    tf = Transform2D(tf_matrix)
    tf.add_vectors(vectors)

    '''
    tf.add_equation(lambda x: sin(x)*cos(x),
                    x_range=(-3,3),
                    count = 100, )
    '''


    # Adding a full circle
    # First half of circle
    tf.add_equation(lambda x: np.sqrt(9-x**2),
                    x_range=(-3,3),
                    count = 100, )
    # Second half of circle
    tf.add_equation(lambda x: -np.sqrt(9-x**2),
                    x_range=(-3,3),
                    count = 100, )

    tf.show()

    """
    def __init__(self, tf_matrix: np.ndarray, vector_label: bool = True):
        self.tf_matrix = tf_matrix
        self._plot_orig_ = vt2D("Original vectors", vector_label)
        self._plot_tf_ = vt2D("Transformed vectors", vector_label)
        self._plot_combine_ = vt2D("Original[blue] /Transformed[red] vectors", vector_label)

    def add_equation(self, equation, x_range=None,
                     count: int = 30,
                     color: tuple = ("b", "r")):

        if x_range is None:
            x_range, _ = self._plot_combine_.get_ranges()
            m = max(abs(x_range[0]), abs(x_range[1]))
            x_range = [-m, m]

        x = np.linspace(x_range[0], x_range[1], count)
        y = equation(x)

        self._plot_combine_.add_multiple_points(x, y, color[0])
        self._plot_orig_.add_multiple_points(x, y, color[0])

        orig_points = np.vstack((x, y))
        tf_points = self.tf_matrix @ orig_points
        self._plot_combine_.add_multiple_points(tf_points[0], tf_points[1], color[1])
        self._plot_tf_.add_multiple_points(tf_points[0], tf_points[1], color[1])

    def add_vectors(self, vectors: np.ndarray):
        tf_vectors = (self.tf_matrix @ vectors.T).T
        self._plot_orig_.add_vectors(vectors, color='b')
        self._plot_combine_.add_vectors(tf_vectors, color='r')
        self._plot_combine_.add_vectors(vectors, color='b')
        self._plot_tf_.add_vectors(tf_vectors, color='r')

    def show(self):
        self._plot_combine_.show()
        self._plot_orig_.show()
        self._plot_tf_.show()

    def fig(self):
        return self._plot_orig_.fig(), self._plot_tf_.fig(), self._plot_combine_.fig()
