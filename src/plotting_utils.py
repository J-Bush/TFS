"""Plotting utilities for time series, persistence diagrams, and gradient paths."""

from typing import Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go


def _show_plotly(fig):
    """Display a Plotly figure.

    Displaying through IPython when it is available, and falling back to ``fig.show()`` otherwise.
    """
    try:
        from IPython.display import display

        display(fig)
    except Exception:
        fig.show()


def plot_time_series(
    data: Union[pd.DataFrame, np.ndarray],
    y_bounds: Tuple[float, float] = None,
    marker_style: str = None,
    marker_size: float = 6.0,
    linewidth: float = 2.5,
    underlay: Union[pd.DataFrame, np.ndarray] = None,
    underlay_alpha: float = 0.35,
    underlay_linewidth: float = 1.5,
    underlay_color: str = None,
    transparent: bool = False,
    save_fig: str = None,
    dpi: int = 300,
):
    """Plot each column of a dataset as a stacked panel.

    Parameters
    ----------
    data : DataFrame or ndarray
        The time series, one signal per column.
    y_bounds : tuple, optional
        Shared ``(ymin, ymax)`` for every panel.
    marker_style : str, optional
        A marker style (e.g. ``'.'``); if ``None``, signals are drawn as lines.
    marker_size : float, optional
        The marker size.
    linewidth : float, optional
        The line width when ``marker_style`` is ``None``.
    underlay : DataFrame or ndarray, optional
        A second series drawn faintly beneath the data, with the same number of
        columns (e.g. an underlying noise-free signal).
    underlay_alpha, underlay_linewidth, underlay_color : optional
        Styling for the underlay; ``underlay_color`` defaults to a neutral gray.
    transparent : bool, optional
        Whether to save with a transparent background.
    save_fig : str, optional
        If given, save a PDF to this path.
    dpi : int, optional
        The resolution of the saved figure.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    if underlay is not None and isinstance(underlay, np.ndarray):
        underlay = pd.DataFrame(underlay)
    u_color = underlay_color if underlay_color is not None else "0.4"

    n_signals = data.shape[1]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(k % cmap.N) for k in range(n_signals)]

    fig, axes = plt.subplots(
        n_signals, 1, figsize=(20, 2 * n_signals), squeeze=False, sharex=True
    )
    axes = axes[:, 0]

    for j, col in enumerate(data.columns):
        ax = axes[j]
        if underlay is not None and j < underlay.shape[1]:
            u_col = underlay.columns[j]
            ax.plot(
                underlay.index,
                underlay[u_col],
                linewidth=underlay_linewidth,
                color=u_color,
                alpha=underlay_alpha,
                zorder=1,
            )
        if marker_style:
            ax.plot(
                data.index, data[col], linestyle="None", marker=marker_style,
                markersize=marker_size, color=colors[j], zorder=2,
            )
        else:
            ax.plot(data.index, data[col], linewidth=linewidth, color=colors[j], zorder=2)
        ax.set_xlim(-10, len(data) + 10)
        if y_bounds:
            ax.set_ylim(y_bounds)

    if save_fig is not None:
        plt.savefig(save_fig, format="pdf", transparent=transparent, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_dataset_row(
    datasets,
    labels=("(a)", "(b)", "(c)"),
    y_bounds=(-1.5, 1.5),
    marker_style=".",
    marker_size=4,
    linewidth=1.2,
    panel_width=6.0,
    signal_height=0.7,
    label_fontsize=18,
    color=True,
    cmap="tab10",
    transparent=False,
    save_fig=None,
    dpi=300,
):
    """Plot several multi-signal datasets side by side in one labeled row.

    Each dataset is drawn as a vertical stack of its signals; the datasets share a
    common vertical scale and are placed in labeled columns. Signals are colored
    consistently across panels so the same signal can be tracked left to right.

    Parameters
    ----------
    datasets : list of DataFrame or ndarray
        The datasets to show, left to right; all must have the same number of
        columns.
    labels : sequence of str, optional
        The panel labels.
    y_bounds : tuple, optional
        The shared vertical range.
    marker_style : str, optional
        A marker style; if ``None``, signals are drawn as lines.
    marker_size, linewidth : float, optional
        Marker and line styling.
    panel_width, signal_height : float, optional
        The width of each panel and the height of each signal strip, in inches.
    label_fontsize : int, optional
        The font size of the panel labels.
    color : bool, optional
        Whether to color signals by row index.
    cmap : str, optional
        A qualitative colormap name used when ``color`` is ``True``.
    transparent : bool, optional
        Whether to save with a transparent background.
    save_fig : str, optional
        If given, save a PDF to this path.
    dpi : int, optional
        The resolution of the saved figure.

    """
    dfs = [d if isinstance(d, pd.DataFrame) else pd.DataFrame(d) for d in datasets]
    n_panels = len(dfs)
    n_signals = dfs[0].shape[1]
    if any(d.shape[1] != n_signals for d in dfs):
        raise ValueError("All datasets must have the same number of signals.")

    if color:
        cmap_obj = plt.get_cmap(cmap)
        colors = [cmap_obj(k % cmap_obj.N) for k in range(n_signals)]
    else:
        colors = [None] * n_signals

    fig, axes = plt.subplots(
        n_signals,
        n_panels,
        figsize=(panel_width * n_panels, signal_height * n_signals),
        squeeze=False,
        sharex=True,
    )

    for j, df in enumerate(dfs):
        for i, col in enumerate(df.columns):
            ax = axes[i][j]
            y = df[col].values
            x = np.arange(len(y))
            if marker_style:
                ax.plot(x, y, linestyle="None", marker=marker_style, markersize=marker_size, color=colors[i])
            else:
                ax.plot(x, y, linewidth=linewidth, color=colors[i])
            ax.set_ylim(y_bounds)
            ax.set_xlim(-5, len(y) + 5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if j < len(labels):
            axes[-1][j].annotate(
                labels[j],
                xy=(0.5, -0.6),
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=label_fontsize,
            )

    plt.subplots_adjust(wspace=0.08, hspace=0.15)
    if save_fig is not None:
        plt.savefig(save_fig, format="pdf", transparent=transparent, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_heatmap(
    data: Union[pd.DataFrame, np.ndarray],
    fontsize: int = 12,
    transparent: bool = False,
    save_fig: str = None,
    width: int = 1920,
    height: int = 1080,
    scale: float = 2.0,
):
    """Plot a dataset as a heatmap, one row per column of ``data``.

    Parameters
    ----------
    data : DataFrame or ndarray
        The data to display.
    fontsize : int, optional
        The axis tick font size.
    transparent : bool, optional
        Whether to use a transparent background.
    save_fig : str, optional
        If given, save a PDF to this path.
    width, height : int, optional
        The pixel dimensions of the saved figure.
    scale : float, optional
        The resolution multiplier for the saved figure.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    data_t = data.T

    fig = go.Figure(
        go.Heatmap(z=data_t.values, x=list(data_t.columns), colorscale="Viridis")
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(data.columns))),
        ticktext=data.columns.astype(str).tolist(),
        tickfont=dict(size=fontsize),
        autorange="reversed",
    )
    fig.update_xaxes(tickfont=dict(size=fontsize))
    if transparent:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    _show_plotly(fig)
    if save_fig:
        fig.write_image(save_fig, format="pdf", width=width, height=height, scale=scale)


def plot_covariance(
    score_df,
    restrict_threshold: float = None,
    clean_labels: bool = False,
    fontsize: int = 10,
    transparent: bool = False,
    save_fig: str = None,
    width: int = 800,
    height: int = 800,
    scale: float = 2.0,
):
    """Plot the sample covariance matrix of scores across runs.

    Parameters
    ----------
    score_df : DataFrame
        One row per run and one column per variable, as returned by
        :func:`~src.persistence_optimization.scores`.
    restrict_threshold : float, optional
        If given, restrict the matrix to variables whose mean score exceeds this
        threshold, ordered by decreasing mean score.
    clean_labels : bool, optional
        Whether to normalize labels with :func:`clean_neuron_name`.
    fontsize : int, optional
        The axis tick font size.
    transparent : bool, optional
        Whether to use a transparent background.
    save_fig : str, optional
        If given, save a PDF to this path.
    width, height : int, optional
        The pixel dimensions of the saved figure.
    scale : float, optional
        The resolution multiplier for the saved figure.

    Returns
    -------
    float
        The mean of the diagonal of the displayed covariance matrix.
    """
    columns = list(score_df.columns)
    if restrict_threshold is not None:
        means = score_df.mean()
        columns = list(means[means > restrict_threshold].sort_values(ascending=False).index)
        score_df = score_df[columns]

    cov = score_df.cov()
    labels = [str(c) for c in columns]
    if clean_labels:
        from .data_utils import clean_neuron_name

        labels = [clean_neuron_name(c) for c in columns]

    fig = go.Figure(
        go.Heatmap(z=cov.values, x=labels, y=labels, colorscale="Viridis")
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, tickfont=dict(size=fontsize))
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=fontsize))
    bg = "rgba(0,0,0,0)" if transparent else "white"
    fig.update_layout(width=width, height=height, paper_bgcolor=bg, plot_bgcolor=bg, margin=dict(l=80, b=80))
    _show_plotly(fig)
    if save_fig:
        fig.write_image(save_fig, format="pdf", width=width, height=height, scale=scale)
    return float(np.mean(np.diag(cov.values)))


def plot_diagram(
    diagram,
    range=None,
    x_range=None,
    y_range=None,
    transparent: bool = False,
    save_fig: str = None,
    width: int = 500,
    height: int = 500,
    scale: float = 3.0,
):
    """Plot a persistence diagram.

    Parameters
    ----------
    diagram : ndarray
        An array of ``(birth, death)`` pairs.
    range : tuple, optional
        A single range applied to both axes; defaults to the data extent. Pass the
        same ``range`` (see :func:`shared_diagram_range`) to two diagrams to make
        them directly comparable.
    x_range, y_range : tuple, optional
        Per-axis ranges that override ``range``.
    transparent : bool, optional
        Whether to use a transparent background.
    save_fig : str, optional
        If given, save a PDF to this path.
    width, height : int, optional
        The pixel dimensions of the saved figure.
    scale : float, optional
        The resolution multiplier for the saved figure.

    """
    if len(diagram) == 0:
        raise ValueError("The persistence diagram is empty.")

    lo, hi = np.min(diagram), np.max(diagram)
    pad = 0.02 * (hi - lo)
    base = [lo - pad, hi + pad] if range is None else list(range)
    xr = list(x_range) if x_range is not None else base
    yr = list(y_range) if y_range is not None else base

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[min(xr[0], yr[0]), max(xr[1], yr[1])],
            y=[min(xr[0], yr[0]), max(xr[1], yr[1])],
            mode="lines",
            line=dict(dash="dash", width=1, color="black"),
            showlegend=False,
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=diagram[:, 0],
            y=diagram[:, 1],
            mode="markers",
            marker=dict(size=6),
            showlegend=False,
        )
    )

    bg = "rgba(0,0,0,0)" if transparent else "white"
    fig.update_layout(
        width=width,
        height=height,
        xaxis=dict(title="Birth", range=xr, showline=True, linecolor="black", ticks="outside"),
        yaxis=dict(title="Death", range=yr, showline=True, linecolor="black", ticks="outside"),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
    )
    _show_plotly(fig)
    if save_fig:
        fig.write_image(save_fig, format="pdf", width=width, height=height, scale=scale)


def shared_diagram_range(*diagrams, pad_frac: float = 0.02):
    """Return a single ``[lo, hi]`` range covering several persistence diagrams.

    Parameters
    ----------
    *diagrams : ndarray
        One or more persistence diagrams; empty ones are ignored.
    pad_frac : float, optional
        The fractional padding added to each side.

    Returns
    -------
    list
        The shared ``[lo, hi]`` range, suitable for the ``range`` argument of
        :func:`plot_diagram`.
    """
    mins, maxs = [], []
    for d in diagrams:
        if d is None or len(d) == 0:
            continue
        arr = np.asarray(d)
        mins.append(np.min(arr))
        maxs.append(np.max(arr))
    if not mins:
        raise ValueError("All supplied diagrams are empty.")
    lo, hi = min(mins), max(maxs)
    pad = pad_frac * (hi - lo)
    return [lo - pad, hi + pad]


def plot_simplex_path(
    df,
    vertex_labels=None,
    show_points=False,
    line_color="blue",
    transparent=False,
    save_fig=None,
):
    """Plot a path through the 2-simplex as an equilateral triangle.

    Parameters
    ----------
    df : DataFrame
        A path of shape ``(N, 3)`` whose rows are points of the 2-simplex.
    vertex_labels : list of str, optional
        Labels for the three vertices.
    show_points : bool, optional
        Whether to mark individual iterates.
    line_color : str, optional
        The color of the path.
    transparent : bool, optional
        Whether to use a transparent background.
    save_fig : str, optional
        If given, save a PDF to this path.

    """
    if vertex_labels is None:
        vertex_labels = ["1", "2", "3"]
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    xy = df.values @ verts

    fig = go.Figure()
    tri = verts[[0, 1, 2, 0]]
    fig.add_trace(go.Scatter(x=tri[:, 0], y=tri[:, 1], mode="lines", line=dict(color="black", width=2), showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=xy[:, 0],
            y=xy[:, 1],
            mode="lines+markers" if show_points else "lines",
            line=dict(color=line_color, width=2),
            marker=dict(size=4),
            showlegend=False,
        )
    )
    for label, vert, off in zip(vertex_labels, verts, [[0, -0.06], [0, -0.06], [0, 0.03]]):
        fig.add_trace(
            go.Scatter(
                x=[vert[0] + off[0]],
                y=[vert[1] + off[1]],
                mode="text",
                text=[label],
                textfont=dict(size=16),
                showlegend=False,
            )
        )

    bg = "rgba(0,0,0,0)" if transparent else "white"
    fig.update_layout(
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        showlegend=False,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        margin=dict(t=20, l=20, b=20, r=20),
        width=500,
        height=500,
    )
    _show_plotly(fig)
    if save_fig:
        fig.write_image(save_fig, format="pdf", width=500, height=500, scale=3.0)


def plot_objective(
    paths,
    dist_mats,
    num_of_features=1,
    gridlines=False,
    dtick=None,
    transparent=False,
    save_fig=None,
):
    """Plot the persistence objective along the mean of one or more paths.

    Parameters
    ----------
    paths : list of DataFrame
        The gradient paths to average before evaluating the objective.
    dist_mats : ndarray
        The component distance matrices, of shape ``(p, m, m)``.
    num_of_features : int, optional
        The number of persistence pairs summed in the objective.
    gridlines : bool, optional
        Whether to draw black gridlines.
    dtick : float, optional
        The gridline spacing when ``gridlines`` is ``True``.
    transparent : bool, optional
        Whether to use a transparent background.
    save_fig : str, optional
        If given, save a PDF to this path.

    """
    from .data_utils import weighted_matrix_sum
    from .persistence_optimization import mean_gradient_path, persistence

    avg = mean_gradient_path(paths)
    values = []
    for i in range(len(avg)):
        dgm = persistence(weighted_matrix_sum(dist_mats, avg.iloc[i].values))
        if len(dgm) >= num_of_features:
            values.append(sum(dgm[k, 1] - dgm[k, 0] for k in range(num_of_features)))
        else:
            values.append(0.0)

    bg = "rgba(0,0,0,0)" if transparent else "white"
    axis = dict(showgrid=gridlines, gridcolor="black", gridwidth=1)
    if dtick is not None:
        axis["dtick"] = dtick

    fig = go.Figure(go.Scatter(y=values, mode="lines"))
    fig.update_layout(
        xaxis_title="step",
        yaxis_title="maximal persistence",
        width=600,
        height=600,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        xaxis=axis,
        yaxis=dict(axis),
    )
    _show_plotly(fig)
    if save_fig:
        fig.write_image(save_fig, format="pdf", width=600, height=600, scale=2.0)


def plot_scores_bar(
    mean_path,
    all_values=None,
    threshold=1e-3,
    clean_labels=False,
    transparent=False,
    save_fig=None,
):
    """Bar chart of the final scores that exceed a threshold.

    Parameters
    ----------
    mean_path : DataFrame
        A mean gradient path; its final row holds the scores.
    all_values : ndarray, optional
        An array of shape ``(num_trials, N, p)``; if given, error bars show the
        across-trial standard deviation of the final scores.
    threshold : float, optional
        Only variables scoring above this are shown.
    clean_labels : bool, optional
        Whether to normalize labels with :func:`clean_neuron_name`.
    transparent : bool, optional
        Whether to use a transparent background.
    save_fig : str, optional
        If given, save a PDF to this path.

    """
    last = mean_path.iloc[-1]
    keep = last[last > threshold].sort_values(ascending=False)
    if keep.empty:
        print("No scores exceed the threshold.")
        return None

    error_y = None
    if all_values is not None:
        idx = [mean_path.columns.get_loc(c) for c in keep.index]
        stds = np.std(all_values[:, -1, :][:, idx], axis=0, ddof=1)
        error_y = dict(type="data", array=stds.tolist(), visible=True)

    if clean_labels:
        from .data_utils import clean_neuron_name

        x_labels = [clean_neuron_name(c) for c in keep.index]
    else:
        x_labels = [str(c) for c in keep.index]

    fig = go.Figure(go.Bar(x=x_labels, y=keep.values, error_y=error_y))
    bg = "rgba(0,0,0,0)" if transparent else "white"
    fig.update_layout(xaxis_tickangle=-45, paper_bgcolor=bg, plot_bgcolor=bg, margin=dict(t=40, b=120), height=500)
    _show_plotly(fig)
    if save_fig:
        fig.write_image(save_fig, format="pdf", width=900, height=500, scale=3.0)
