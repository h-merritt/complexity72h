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

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns


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
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from matplotlib.patches import Patch

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


def plot_correlation_grid_stacked(
    corr_combined: np.ndarray,
    corr_inner: np.ndarray,
    corr_outer: np.ndarray,
    inner_names: list[str],
    outer_names: list[str],
    inner_color: str = "#3498db",
    outer_color: str = "#e67e22",
    show_dendrogram: bool = False,
) -> plt.Figure:
    """Combined figure with three correlation matrices in a stacked layout.

    Top row shows inner (left) and outer (right) heatmaps.
    Bottom row shows the combined inner+outer heatmap.

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

    Returns
    -------
    plt.Figure
    """
    from scipy.cluster.hierarchy import dendrogram, linkage

    labels_combined = inner_names + outer_names
    inner_set = set(inner_names)

    def _reorder(corr, labels):
        Z = linkage(corr, method="average")
        leaves = dendrogram(Z, no_plot=True)["leaves"]
        return corr[leaves][:, leaves], [labels[i] for i in leaves]

    def _heatmap(ax, corr, labels):
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
            cbar=False,
            linewidths=0.3,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    def _dendro_axis(ax, corr):
        Z = linkage(corr, method="average")
        with plt.rc_context({"lines.linewidth": 0.8}):
            dendrogram(Z, ax=ax, orientation="top", above_threshold_color="black")
        ax.set_axis_off()

    def _color_labels(ax, labels, inner_set, inner_color, outer_color):
        for lbl in ax.get_xticklabels():
            lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)
        for lbl in ax.get_yticklabels():
            lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)

    n_inner, n_outer = len(inner_names), len(outer_names)
    n_combined = n_inner + n_outer

    cell_size = 0.85
    cbar_ratio = 0.05
    fig_w = max(18, n_combined * cell_size * 1.2) * (1 + cbar_ratio)
    fig_h = max(18, n_combined * cell_size * 1.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        2, 2, figure=fig, width_ratios=[1, cbar_ratio], height_ratios=[1.25, 3]
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.01)
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, 0], width_ratios=[1, 2, 1]
    )
    cbar_ax = fig.add_subplot(gs[:, 1])

    if show_dendrogram:
        dendro_ratio = 0.12
        # Combined
        gs_comb = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs_bottom[0, 1],
            height_ratios=[dendro_ratio, 1 - dendro_ratio],
        )
        ax_dendro_comb = fig.add_subplot(gs_comb[0])
        ax_comb = fig.add_subplot(gs_comb[1])
        c_combined, l_combined = _reorder(corr_combined.copy(), labels_combined.copy())
        _dendro_axis(ax_dendro_comb, corr_combined)
        _heatmap(ax_comb, c_combined, l_combined)
        _color_labels(ax_comb, l_combined, inner_set, inner_color, outer_color)
        ax_comb.set_title(r"Inner \& Outer", fontsize=30)

        # Inner
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs_top[0, 0],
            height_ratios=[dendro_ratio, 1 - dendro_ratio],
        )
        ax_dendro_inner = fig.add_subplot(gs_inner[0])
        ax_inner = fig.add_subplot(gs_inner[1])
        c_inner, l_inner = _reorder(corr_inner.copy(), inner_names.copy())
        _dendro_axis(ax_dendro_inner, corr_inner)
        _heatmap(ax_inner, c_inner, l_inner)
        _color_labels(ax_inner, l_inner, set(l_inner), inner_color, outer_color)
        ax_inner.set_title("Inner", fontsize=30)

        # Outer
        gs_outer = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs_top[0, 1],
            height_ratios=[dendro_ratio, 1 - dendro_ratio],
        )
        ax_dendro_outer = fig.add_subplot(gs_outer[0])
        ax_outer = fig.add_subplot(gs_outer[1])
        c_outer, l_outer = _reorder(corr_outer.copy(), outer_names.copy())
        _dendro_axis(ax_dendro_outer, corr_outer)
        _heatmap(ax_outer, c_outer, l_outer)
        _color_labels(ax_outer, l_outer, set(), inner_color, outer_color)
        ax_outer.set_title("Outer", fontsize=30)
    else:
        # Row 0: Inner
        c_inner, l_inner = _reorder(corr_inner.copy(), inner_names.copy())
        ax_inner = fig.add_subplot(gs_top[0, 0])
        _heatmap(ax_inner, c_inner, l_inner)
        _color_labels(ax_inner, l_inner, set(l_inner), inner_color, outer_color)
        ax_inner.set_title("Inner", fontsize=30)

        # Row 0: Outer
        c_outer, l_outer = _reorder(corr_outer.copy(), outer_names.copy())
        ax_outer = fig.add_subplot(gs_top[0, 1])
        _heatmap(ax_outer, c_outer, l_outer)
        _color_labels(ax_outer, l_outer, set(), inner_color, outer_color)
        ax_outer.set_title("Outer", fontsize=30)

        # Row 1: Combined
        c_combined, l_combined = _reorder(corr_combined.copy(), labels_combined.copy())
        ax_comb = fig.add_subplot(gs_bottom[0, 1])
        _heatmap(ax_comb, c_combined, l_combined)
        _color_labels(ax_comb, l_combined, inner_set, inner_color, outer_color)
        ax_comb.set_title(r"Inner \& Outer", fontsize=30)

    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=Normalize(-1, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    ticks = np.linspace(-1, 1, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}".replace("-", "\\textminus{}") for t in ticks])

    plt.tight_layout()
    return fig


def plot_correlation_grid(
    corr_combined: np.ndarray,
    corr_inner: np.ndarray,
    corr_outer: np.ndarray,
    inner_names: list[str],
    outer_names: list[str],
    inner_color: str = "#3498db",
    outer_color: str = "#e67e22",
    show_dendrogram: bool = False,
) -> plt.Figure:
    """Combined figure with three correlation matrices in a side layout.

    Left column stacks Inner (top) and Outer (bottom) heatmaps.
    Right column shows the combined Inner+Outer heatmap with its own colorbar.

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

    Returns
    -------
    plt.Figure
    """
    from scipy.cluster.hierarchy import dendrogram, linkage

    labels_combined = inner_names + outer_names
    inner_set = set(inner_names)

    def _reorder(corr, labels):
        Z = linkage(corr, method="average")
        leaves = dendrogram(Z, no_plot=True)["leaves"]
        return corr[leaves][:, leaves], [labels[i] for i in leaves]

    def _heatmap(ax, corr, labels, cbar=True):
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

    def _dendro_axis(ax, corr):
        Z = linkage(corr, method="average")
        with plt.rc_context({"lines.linewidth": 0.8}):
            dendrogram(Z, ax=ax, orientation="top", above_threshold_color="black")
        ax.set_axis_off()

    def _color_labels(ax, labels, inner_set, inner_color, outer_color):
        for lbl in ax.get_xticklabels():
            lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)
        for lbl in ax.get_yticklabels():
            lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)

    n_inner, n_outer = len(inner_names), len(outer_names)
    n_combined = n_inner + n_outer

    cell_size = 0.85
    fig_w = max(20, n_combined * cell_size * 1.6)
    fig_h = max(12, max(n_inner, n_outer) * cell_size * 2.2)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        2, 2, figure=fig, width_ratios=[1, 2], wspace=0.6, hspace=0.5
    )

    if show_dendrogram:
        dendro_ratio = 0.12
        # Combined (right column, both rows)
        gs_comb = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs[:, 1],
            height_ratios=[dendro_ratio, 1 - dendro_ratio],
        )
        ax_dendro_comb = fig.add_subplot(gs_comb[0])
        ax_comb = fig.add_subplot(gs_comb[1])
        c_combined, l_combined = _reorder(corr_combined.copy(), labels_combined.copy())
        _dendro_axis(ax_dendro_comb, corr_combined)
        _heatmap(ax_comb, c_combined, l_combined, cbar=True)
        _color_labels(ax_comb, l_combined, inner_set, inner_color, outer_color)
        ax_comb.set_title(r"Inner \& Outer", fontsize=30)
        cbar_comb = ax_comb.collections[0].colorbar
        if cbar_comb:
            ticks = np.linspace(-1, 1, 5)
            cbar_comb.set_ticks(ticks)
            cbar_comb.set_ticklabels(
                [f"{t:.1f}".replace("-", "\\textminus{}") for t in ticks]
            )

        # Inner (top-left)
        gs_inner = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs[0, 0],
            height_ratios=[dendro_ratio, 1 - dendro_ratio],
        )
        ax_dendro_inner = fig.add_subplot(gs_inner[0])
        ax_inner = fig.add_subplot(gs_inner[1])
        c_inner, l_inner = _reorder(corr_inner.copy(), inner_names.copy())
        _dendro_axis(ax_dendro_inner, corr_inner)
        _heatmap(ax_inner, c_inner, l_inner, cbar=False)
        _color_labels(ax_inner, l_inner, set(l_inner), inner_color, outer_color)
        ax_inner.set_title("Inner", fontsize=30)

        # Outer (bottom-left)
        gs_outer = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=gs[1, 0],
            height_ratios=[dendro_ratio, 1 - dendro_ratio],
        )
        ax_dendro_outer = fig.add_subplot(gs_outer[0])
        ax_outer = fig.add_subplot(gs_outer[1])
        c_outer, l_outer = _reorder(corr_outer.copy(), outer_names.copy())
        _dendro_axis(ax_dendro_outer, corr_outer)
        _heatmap(ax_outer, c_outer, l_outer, cbar=False)
        _color_labels(ax_outer, l_outer, set(), inner_color, outer_color)
        ax_outer.set_title("Outer", fontsize=30)
    else:
        # Inner (top-left)
        c_inner, l_inner = _reorder(corr_inner.copy(), inner_names.copy())
        ax_inner = fig.add_subplot(gs[0, 0])
        _heatmap(ax_inner, c_inner, l_inner, cbar=False)
        _color_labels(ax_inner, l_inner, set(l_inner), inner_color, outer_color)
        ax_inner.set_title("Inner", fontsize=30)

        # Outer (bottom-left)
        c_outer, l_outer = _reorder(corr_outer.copy(), outer_names.copy())
        ax_outer = fig.add_subplot(gs[1, 0])
        _heatmap(ax_outer, c_outer, l_outer, cbar=False)
        _color_labels(ax_outer, l_outer, set(), inner_color, outer_color)
        ax_outer.set_title("Outer", fontsize=30)

        # Combined (right column, both rows)
        c_combined, l_combined = _reorder(corr_combined.copy(), labels_combined.copy())
        ax_comb = fig.add_subplot(gs[:, 1])
        _heatmap(ax_comb, c_combined, l_combined, cbar=True)
        _color_labels(ax_comb, l_combined, inner_set, inner_color, outer_color)
        ax_comb.set_title(r"Inner \& Outer", fontsize=30)
        cbar_comb = ax_comb.collections[0].colorbar
        if cbar_comb:
            ticks = np.linspace(-1, 1, 5)
            cbar_comb.set_ticks(ticks)
            cbar_comb.set_ticklabels(
                [f"{t:.1f}".replace("-", "\\textminus{}") for t in ticks]
            )

    plt.tight_layout()
    return fig


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
