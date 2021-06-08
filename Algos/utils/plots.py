import plotly.graph_objects as go
from plotly.graph_objs import Figure


def plotly_plot(x=None, y=None,
                hovertemplate=None,
                legend="",
                x_title=None,
                y_title=None,
                title=None,
                fig=None,
                mode="markers",
                color="red",
                do_not_change_fig=False,
                marker_size=None,
                remove_grid=False):
    if fig is None:
        fig: Figure = go.Figure()
    if do_not_change_fig:
        fig: Figure = go.Figure(fig)

    if x is not None and y is not None:
        fig.add_trace(go.Scatter(x=x,
                                 y=y,
                                 mode=mode,
                                 marker=dict(size=marker_size),
                                 name=legend,
                                 hovertemplate=hovertemplate,
                                 line=dict(color=color)))
    if title:
        fig.update_layout(title=title)
    if x_title:
        fig.update_layout(xaxis_title=x_title)
    if y_title:
        fig.update_layout(yaxis_title=y_title)
    # if marker_size is not None:
    #     fig.update_layout(marker_size=marker_size)
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
