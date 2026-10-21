"""Correlation matrix visualisation functions.

Functions
---------
plot_correlation_heatmap
    Clustered Pearson correlation heatmap with dendrogram.
plot_clustermap
    Clustered heatmap with colour-coded cluster annotations.
plot_clustermap_top
    Clustered heatmap without row reordering.
plot_combined_heatmap
    Combined inner + outer correlation heatmap with colour-coded labels.
plot_pairplot
    Seaborn pairplot (scatter matrix) for selected variables.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.colors import Normalize


def plot_correlation_heatmap(
    corr: np.ndarray, labels: list[str], title: str | None = None
) -> plt.Figure:
    """Clustered Pearson correlation heatmap with a top dendrogram.

    Parameters
    ----------
    corr : np.ndarray of shape (n_features, n_features)
        Pearson correlation matrix.
    labels : list of str
        Variable names corresponding to the rows/columns of *corr*.
    title : str or None, optional
        Figure title. If ``None``, defaults to
        ``"Pearson correlation — {n} variables"`` (default: ``None``).

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> corr = np.corrcoef(np.random.randn(50, 4).T)
    >>> fig = plot_correlation_heatmap(corr, ["x1", "x2", "x3", "x4"])
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    size = max(8, len(labels) * 0.75)
    g = sns.clustermap(
        corr,
        cmap="RdBu",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.3,
        figsize=(size, size),
        xticklabels=labels,
        yticklabels=labels,
        dendrogram_ratio=0.2,
    )
    g.ax_row_dendrogram.set_visible(False)
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha="right"
    )
    g.fig.suptitle(title or f"Pearson correlation — {len(labels)} variables", y=1.02)
    return g.fig


def plot_clustermap(
    corr: np.ndarray, labels: list[str], n_clusters: int = 3
) -> plt.Figure:
    """Clustered correlation heatmap with colour-coded cluster annotations.

    Parameters
    ----------
    corr : np.ndarray of shape (n_features, n_features)
        Pearson correlation matrix.
    labels : list of str
        Variable names corresponding to the rows/columns of *corr*.
    n_clusters : int, optional
        Number of clusters for the ``fcluster`` partition (default: 3).

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> corr = np.corrcoef(np.random.randn(50, 6).T)
    >>> fig = plot_clustermap(corr, ["x" + str(i) for i in range(6)], n_clusters=2)
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    from matplotlib.patches import Patch
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = (
        1 - corr
    ) / 2  # convert correlation to distance: d = 1 - r, scaled to [0, 1]
    np.fill_diagonal(dist, 0)
    Z = linkage(
        squareform(dist, checks=False), method="average"
    )  # build hierarchical tree
    cluster_labels = fcluster(
        Z, n_clusters, criterion="maxclust"
    )  # cut tree into k flat clusters

    palette = sns.color_palette("husl", n_clusters)
    row_colors = [palette[i - 1] for i in cluster_labels]

    size = max(8, len(labels) * 0.55)
    g = sns.clustermap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(size, size * 0.85),
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.3,
        dendrogram_ratio=0.15,
        row_colors=row_colors,
        col_colors=row_colors,
    )
    handles = [
        Patch(color=palette[i], label=f"Cluster {i + 1}") for i in range(n_clusters)
    ]
    g.fig.subplots_adjust(bottom=0.22)
    g.fig.legend(
        handles=handles,
        title="Cluster",
        loc="lower center",
        ncols=n_clusters,
        bbox_to_anchor=(0.5, 0.02),
    )
    return g.fig


def plot_clustermap_top(corr: np.ndarray, labels: list[str]) -> plt.Figure:
    """Clustered correlation heatmap without row reordering.

    Parameters
    ----------
    corr : np.ndarray of shape (n_features, n_features)
        Pearson correlation matrix.
    labels : list of str
        Variable names corresponding to the rows/columns of *corr*.

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> corr = np.corrcoef(np.random.randn(50, 4).T)
    >>> fig = plot_clustermap_top(corr, ["x1", "x2", "x3", "x4"])
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    size = max(8, len(labels) * 0.55)
    g = sns.clustermap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(size, size * 0.85),
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.3,
        dendrogram_ratio=0.15,
        row_cluster=False,
    )
    return g.fig


def plot_combined_heatmap(
    corr: np.ndarray,
    inner: list[str],
    outer: list[str],
    inner_color: str = "#3498db",
    outer_color: str = "#e67e22",
    row_cluster: bool = True,
    col_cluster: bool = True,
) -> plt.Figure:
    """Correlation heatmap for combined inner + outer variables.

    Axis tick labels are colour-coded: *inner_color* for inner metrics and
    *outer_color* for outer metrics.

    Parameters
    ----------
    corr : np.ndarray of shape (n_inner + n_outer, n_inner + n_outer)
        Pearson correlation matrix for the concatenated ``inner + outer``
        variable set.
    inner : list of str
        Labels for the inner variables (first block in *corr*).
    outer : list of str
        Labels for the outer variables (second block in *corr*).
    inner_color : str, optional
        Tick-label colour for inner variables (default: ``"#3498db"``).
    outer_color : str, optional
        Tick-label colour for outer variables (default: ``"#e67e22"``).
    row_cluster : bool, optional
        Reorder rows by hierarchical clustering (default: ``True``).
    col_cluster : bool, optional
        Reorder columns by hierarchical clustering (default: ``True``).

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> corr = np.corrcoef(np.random.randn(50, 6).T)
    >>> inner = ["x1", "x2", "x3"]
    >>> outer = ["x4", "x5", "x6"]
    >>> fig = plot_combined_heatmap(corr, inner, outer)
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    labels = inner + outer
    size = max(8, len(labels) * 0.75)
    inner_set = set(inner)
    show_dendro = row_cluster or col_cluster

    if show_dendro:
        g = sns.clustermap(
            corr,
            cmap="RdBu",
            center=0,
            vmin=-1,
            vmax=1,
            annot=False,
            linewidths=0.3,
            figsize=(size, size),
            xticklabels=labels,
            yticklabels=labels,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            dendrogram_ratio=0.2,
        )
        if row_cluster:
            g.ax_row_dendrogram.set_visible(False)
        fig, ax = g.fig, g.ax_heatmap
    else:
        fig, ax = plt.subplots(figsize=(size, size))
        sns.heatmap(
            corr,
            ax=ax,
            cmap="RdBu",
            center=0,
            vmin=-1,
            vmax=1,
            annot=False,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            linewidths=0.3,
        )

    for lbl in ax.get_xticklabels():
        lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)
        lbl.set_rotation(45)
        lbl.set_ha("right")
    for lbl in ax.get_yticklabels():
        lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)
    fig.suptitle(r"Inner \& Outer", y=1.02)
    return fig


_DENDRO_RATIO = 0.12


def _reorder_corr(corr: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Reorder a correlation matrix and its labels by average-linkage leaves."""
    from scipy.cluster.hierarchy import dendrogram, linkage

    leaves = dendrogram(linkage(corr, method="average"), no_plot=True)["leaves"]
    return corr[leaves][:, leaves], [labels[i] for i in leaves]


def _grid_heatmap(ax, corr: np.ndarray, labels: list[str], cbar: bool) -> None:
    """RdBu correlation heatmap on ``ax`` with 45 degree x tick labels."""
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar=cbar,
        linewidths=0.3,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")


def _grid_dendrogram(ax, corr: np.ndarray) -> None:
    """Draw the average-linkage dendrogram of ``corr`` on ``ax`` (axes hidden)."""
    from scipy.cluster.hierarchy import dendrogram, linkage

    with plt.rc_context({"lines.linewidth": 0.8}):
        dendrogram(
            linkage(corr, method="average"),
            ax=ax,
            orientation="top",
            above_threshold_color="black",
        )
    ax.set_axis_off()


def _color_tick_labels(
    ax, inner_set: set[str], inner_color: str, outer_color: str
) -> None:
    """Colour tick labels in ``inner_set`` with ``inner_color``, the rest with ``outer_color``."""
    for lbl in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)


def _set_corr_cbar_ticks(cbar) -> None:
    """Ticks -1..1 in steps of 0.5 with a typographic minus sign."""
    ticks = np.linspace(-1, 1, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}".replace("-", "\\textminus{}") for t in ticks])


def _grid_panel(
    fig: plt.Figure,
    spec,
    corr: np.ndarray,
    labels: list[str],
    title: str,
    inner_set: set[str],
    inner_color: str,
    outer_color: str,
    show_dendrogram: bool,
    cbar: bool,
):
    """Draw one clustered correlation heatmap (optionally with its dendrogram) in ``spec``."""
    if show_dendrogram:
        sub = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=spec,
            height_ratios=[_DENDRO_RATIO, 1 - _DENDRO_RATIO],
        )
        _grid_dendrogram(fig.add_subplot(sub[0]), corr)
        ax = fig.add_subplot(sub[1])
    else:
        ax = fig.add_subplot(spec)
    corr_ord, labels_ord = _reorder_corr(corr.copy(), labels.copy())
    _grid_heatmap(ax, corr_ord, labels_ord, cbar=cbar)
    _color_tick_labels(ax, inner_set, inner_color, outer_color)
    ax.set_title(title, fontsize=30)
    return ax


def plot_correlation_grid(
    corr_combined: np.ndarray,
    corr_inner: np.ndarray,
    corr_outer: np.ndarray,
    inner_names: list[str],
    outer_names: list[str],
    inner_color: str = "#3498db",
    outer_color: str = "#e67e22",
    show_dendrogram: bool = False,
    layout: str = "side",
) -> plt.Figure:
    """Combined figure with the inner, outer and combined correlation matrices.

    Each matrix is reordered by average-linkage clustering of its own
    correlations, and tick labels are coloured by group.

    Parameters
    ----------
    corr_combined : np.ndarray of shape (n_inner + n_outer, n_inner + n_outer)
        Correlation matrix for the concatenated inner + outer variables.
    corr_inner : np.ndarray of shape (n_inner, n_inner)
        Correlation matrix for inner variables.
    corr_outer : np.ndarray of shape (n_outer, n_outer)
        Correlation matrix for outer variables.
    inner_names : list of str
        Labels for inner variables.
    outer_names : list of str
        Labels for outer variables.
    inner_color : str, optional
        Tick-label colour for inner variables (default ``"#3498db"``).
    outer_color : str, optional
        Tick-label colour for outer variables (default ``"#e67e22"``).
    show_dendrogram : bool, optional
        If ``True``, show hierarchical dendrograms above each heatmap
        (default ``False``).
    layout : {"side", "stacked"}, optional
        ``"side"`` (default): Inner (top) and Outer (bottom) in the left
        column, the combined heatmap in the right column with its own
        colourbar.  ``"stacked"``: Inner and Outer side by side in the top
        row, the combined heatmap below, and one shared colourbar on the
        right.

    Returns
    -------
    plt.Figure

    Raises
    ------
    ValueError
        If ``layout`` is not ``"side"`` or ``"stacked"``.
    """
    if layout not in ("side", "stacked"):
        raise ValueError(f"layout must be 'side' or 'stacked', got {layout!r}.")

    labels_combined = inner_names + outer_names
    inner_set = set(inner_names)
    n_inner, n_outer = len(inner_names), len(outer_names)
    n_combined = n_inner + n_outer
    cell_size = 0.85

    panel = {
        "inner_color": inner_color,
        "outer_color": outer_color,
        "show_dendrogram": show_dendrogram,
    }

    if layout == "stacked":
        cbar_ratio = 0.05
        fig_w = max(18, n_combined * cell_size * 1.2) * (1 + cbar_ratio)
        fig_h = max(18, n_combined * cell_size * 1.0)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(
            2, 2, figure=fig, width_ratios=[1, cbar_ratio], height_ratios=[1.25, 3]
        )
        gs_top = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[0, 0], wspace=0.01
        )
        gs_bottom = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs[1, 0], width_ratios=[1, 2, 1]
        )
        cbar_ax = fig.add_subplot(gs[:, 1])
        spec_inner, spec_outer, spec_comb = gs_top[0, 0], gs_top[0, 1], gs_bottom[0, 1]
        comb_cbar = False
    else:
        fig_w = max(20, n_combined * cell_size * 1.6)
        fig_h = max(12, max(n_inner, n_outer) * cell_size * 2.2)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(
            2, 2, figure=fig, width_ratios=[1, 2], wspace=0.6, hspace=0.5
        )
        spec_inner, spec_outer, spec_comb = gs[0, 0], gs[1, 0], gs[:, 1]
        comb_cbar = True

    _grid_panel(
        fig,
        spec_inner,
        corr_inner,
        inner_names,
        "Inner",
        set(inner_names),
        cbar=False,
        **panel,
    )
    _grid_panel(
        fig,
        spec_outer,
        corr_outer,
        outer_names,
        "Outer",
        set(),
        cbar=False,
        **panel,
    )
    ax_comb = _grid_panel(
        fig,
        spec_comb,
        corr_combined,
        labels_combined,
        r"Inner \& Outer",
        inner_set,
        cbar=comb_cbar,
        **panel,
    )

    if layout == "stacked":
        sm = plt.cm.ScalarMappable(cmap="RdBu", norm=Normalize(-1, 1))
        sm.set_array([])
        _set_corr_cbar_ticks(fig.colorbar(sm, cax=cbar_ax))
    else:
        cbar_comb = ax_comb.collections[0].colorbar
        if cbar_comb:
            _set_corr_cbar_ticks(cbar_comb)

    plt.tight_layout()
    return fig


def plot_correlation_grid_stacked(*args, **kwargs) -> plt.Figure:
    """Shortcut for :func:`plot_correlation_grid` with ``layout="stacked"``."""
    return plot_correlation_grid(*args, layout="stacked", **kwargs)


def plot_pairplot(
    X: np.ndarray,
    labels: list[str],
    hue: np.ndarray | None = None,
    hue_label: str | None = None,
) -> plt.Figure:
    """Pairplot (scatter matrix) for a set of variables.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix; columns correspond to *labels*.
    labels : list of str
        Variable names (must match the column order of *X*).
    hue : np.ndarray of shape (n_samples,) or None, optional
        Categorical array used to colour points (default: ``None``).
    hue_label : str or None, optional
        Legend title for the hue variable (default: ``None``).

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 4)
    >>> fig = plot_pairplot(X, ["x1", "x2", "x3", "x4"])
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    pdf = pd.DataFrame(X, columns=labels)
    if hue is not None and hue_label:
        pdf[hue_label] = hue
    g = sns.pairplot(pdf, hue=hue_label, kind="hist", diag_kind="kde")
    if hue_label and g.legend:
        sns.move_legend(g, "outside lower center", ncols=2)
        g.fig.tight_layout(rect=[0, 0.08, 1, 1])
    else:
        g.fig.tight_layout()
    return g.fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
