from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Plot2DVectors:
    """
    # Example:
    vectors = 5*np.array([
                        [1,0],
                        [0,1],
                        [1,1],
                        [1,2],
                        [2,1],
                       ])
    origin = np.array([0,0])

    plt2D = Plot2DVectors("Vectors")
    plt2D.add_vectors(vectors, origin)
    #plt2D.savefig()

    vector = np.array([3,3])
    plt2D.add_vector(vector)
    plt2D.show()
    """

    def __init__(self, title="", vector_label=True, head_width=0.15, head_length=0.2):
        self.vector_label: bool = vector_label
        self._x_range_: List[int] = [0, 0]
        self._y_range_: List[int] = [0, 0]
        self._fig_: Figure = plt.figure()
        self._head_width_: float = head_width
        self._head_length_: float = head_length

        plt.rc('font', size=8)
        plt.title(title)
        plt.grid()

    def add_multiple_points(self, x, y, color):
        ax = self._fig_.gca()
        ax.scatter(x, y, c=color)

        x_min, x_max = np.min(x) - 0.5, np.max(x) + 0.5
        y_min, y_max = np.min(y) - 0.5, np.max(y) + 0.5

        if x_min < self._x_range_[0]:
            self._x_range_[0] = x_min
        if x_max > self._x_range_[1]:
            self._x_range_[1] = x_max
        if y_min < self._y_range_[0]:
            self._y_range_[0] = y_min
        if y_max > self._y_range_[1]:
            self._y_range_[1] = y_max

    def add_vectors(self, vectors: np.ndarray,
                    origin=np.array([0, 0]),
                    color="b"):

        self._x_range_ = [min(vectors[:, 0].min(), self._x_range_[0]) - 0.5,
                          max(vectors[:, 0].max() + 0.5, self._x_range_[1])]
        self._y_range_ = [min(vectors[:, 1].min(), self._y_range_[0]) - 0.5,
                          max(vectors[:, 1].max() + 0.5, self._y_range_[1])]

        ax = self._fig_.gca()
        for vector in vectors:
            ax.arrow(origin[0], origin[1], vector[0] - origin[0], vector[1] - origin[1], head_width=self._head_width_,
                     head_length=self._head_length_, fc=color, ec=color, alpha=0.6)
            if self.vector_label:
                ax.text(vector[0], vector[1], str(vector), style='italic',
                        bbox={'facecolor': color, 'alpha': 0.2, 'pad': 0.5})
        ax.scatter(origin[0], origin[1], c=color)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        self.__set_axes_limit()

    def add_vector(self, vector, origin=np.array([0, 0]), color="b"):
        ax = self._fig_.gca()
        if vector[0] < self._x_range_[0]:
            self._x_range_[0] = vector[0] - 0.5
        elif vector[0] > self._x_range_[1]:
            self._x_range_[1] = vector[0] + 0.5

        if vector[1] < self._y_range_[0]:
            self._y_range_[0] = vector[1] - 0.5
        elif vector[1] > self._y_range_[1]:
            self._y_range_[1] = vector[1] + 0.5

        ax.arrow(origin[0], origin[1], vector[0] - origin[0], vector[1] - origin[1], head_width=self._head_width_,
                 head_length=self._head_length_, fc=color, ec=color, alpha=0.6)
        if self.vector_label:
            ax.text(vector[0], vector[1], str(vector), style='italic',
                    bbox={'facecolor': color, 'alpha': 0.2, 'pad': 0.5})
        ax.scatter(origin[0], origin[1], c=color)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        self.__set_axes_limit()

    def __setX_limit(self):
        ax = self._fig_.gca()
        ax.set_xlim(self._x_range_[0], self._x_range_[1])

    def __setY_limit(self):
        ax = self._fig_.gca()
        ax.set_ylim(self._y_range_[0], self._y_range_[1])

    def __set_axes_limit(self):
        self.__setX_limit()
        self.__setY_limit()

    def savefig(self, name=None):
        if name is None: name = "fig.png"
        self._fig_.set_size_inches(8, 6)
        self._fig_.savefig(name, dpi=100)

    def fig(self):
        self.__set_axes_limit()
        return self._fig_

    def get_ranges(self):
        return self._x_range_, self._y_range_

    def show(self):
        self.fig().show()
