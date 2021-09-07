from plotly.graph_objs import Figure

from Algos.utils.plots import plotly_plot, mesh3d


def plot_predition(X, theta, fig):

    n, d = X.shape

    min_X, max_X = X[:, 0].min(), X[:, 0].max()

    if d == 1:
        new_fig: Figure = plotly_plot(
            [min_X, max_X],
            [
                theta[0][0] + theta[1][0] * min_X,
                theta[0][0] + theta[1][0] * max_X
            ],
            fig=fig,
            mode="lines",
            color="blue",
            do_not_change_fig=True,
            title=f"Linear Regression"
        )
        return new_fig
    elif d == 2:
        description = {
            "title": {
                "main": f"Linear Regression",
                "x": "x1",
                "y": "x2",
                "z": "y"
            },
            "label": {
                "main": "",
            },
            "hovertemplate": "(x1, x1): (%{x}, %{y})<br>f(%{x}, %{y}): %{z}"
        }
        min_X2, max_X2 = X[:, 1].min(), X[:, 1].max()
        new_fig: Figure = mesh3d(
            [min_X, min_X, max_X, max_X],
            [min_X2, max_X2, min_X2, max_X2],
            [
                theta[0][0] + theta[1][0] * min_X + theta[2][0] * min_X2,
                theta[0][0] + theta[1][0] * min_X + theta[2][0] * max_X2,
                theta[0][0] + theta[1][0] * max_X + theta[2][0] * min_X2,
                theta[0][0] + theta[1][0] * max_X + theta[2][0] * max_X2,
            ],
            description,
            fig=fig,
            opacity=0.9
        )
        return new_fig
