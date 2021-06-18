import plotly.graph_objects as go
from plotly.graph_objs import Figure

from Algos.utils.utils import split_features


def plotly_plot(x=None, y=None, z=None,
                hovertemplate=None, legend="",
                title=None, x_title=None, y_title=None, z_title=None,
                mode="markers", color="red", marker_size=None,
                fig=None, do_not_change_fig=False,
                width=None, height=None, remove_grid=False):
    if fig is None:
        fig: Figure = go.Figure()
    if do_not_change_fig:
        fig: Figure = go.Figure(fig)

    if x is not None and y is not None and z is not None:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode=mode,
            marker=dict(size=marker_size),
            name=legend,
            hovertemplate=hovertemplate,
            line=dict(color=color)
        ))
    elif x is not None and y is not None:
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode=mode,
            marker=dict(size=marker_size),
            name=legend,
            hovertemplate=hovertemplate,
            line=dict(color=color)
        ))

    if title:
        fig.update_layout(title=title)

    if x_title and y_title and z_title:
        fig.update_layout(scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title),
        )
    elif x_title and y_title:
        fig.update_layout(xaxis_title=x_title)
        fig.update_layout(yaxis_title=y_title)

    if width is not None and height is not None:
        fig.update_layout(width=width, height=height)

    if remove_grid:
        fig.update_layout(
            xaxis=dict(
                # autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False
            ),
            yaxis=dict(
                # autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False
            )
        )
    return fig

def plot_classification_data(
        features,
        labels,
        title,
        x_title,
        y_title,
        z_title=None):

    data = split_features(features, labels)
    fig: Figure = go.Figure()
    colors = ["red", "blue", "yellow", "green", "orange", "grey", "purple", "teal"]

    for idx, label in enumerate(data):
        if features.shape[1] == 2:
            fig = plotly_plot(
                data[label][:, 0], data[label][:, 1],
                fig=fig,
                color=colors[idx % len(colors)]
            )
        else:
            fig = plotly_plot(
                data[label][:, 0], data[label][:, 1], data[label][:, 2],
                fig=fig,
                marker_size=4,
                color=colors[idx % len(colors)],
                width=700, height=500
            )
    if features.shape[1] == 2:
        fig = plotly_plot(fig=fig, x_title=x_title, y_title=y_title, title=title)
    else:
        fig = plotly_plot(fig=fig, x_title=x_title, y_title=y_title, z_title=z_title, title=title)
    return fig


def mesh3d(x, y, z,
           description: dict,
           fig: Figure = None,
           colorscale='Viridis',
           opacity=0.5,
           isMobile: bool = False) -> Figure:
    """
    description={
        "title": {
            "main": "Title of plot",
            "x": "x-axis title",
            "y": "y-axis title"
        },
        "label": {
            "main": "Legend",
        },
        "hovertemplate": "x-label (x): %{x}<br>y-label(x=%{x}): %{y}",
        "color": "green"
    }
    """

    if fig is None:
        fig: Figure = go.Figure()
    else:
        fig: Figure = go.Figure(fig)

    fig.add_trace(go.Mesh3d(x=x, y=y, z=z,
                            name=description["label"]["main"],
                            hovertemplate=description["hovertemplate"],
                            colorscale=colorscale,
                            opacity=opacity))
    if "title" in description:
        fig.update_layout(scene=dict(xaxis_title=description["title"]["x"],
                                     yaxis_title=description["title"]["y"],
                                     zaxis_title=description["title"]["z"]),
                          title=description["title"]["main"])
    fig.update_layout(showlegend=False if isMobile else True)
    return fig

def surface3D(
        x, y, z,
        description: dict,
        fig: Figure = None,
    ):

    if fig is None:
        fig: Figure = go.Figure()
    else:
        fig: Figure = go.Figure(fig)

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        name=description["label"]["main"],
        hovertemplate=description["hovertemplate"]
    ))

    if "title" in description:
        fig.update_layout(
            scene=dict(
                xaxis_title=description["title"]["x"],
                yaxis_title=description["title"]["y"],
                zaxis_title=description["title"]["z"]),
            title=description["title"]["main"]
        )
    return fig
