from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(corr: np.ndarray, labels: list[str]) -> plt.Figure:
    """Clustered Pearson correlation heatmap with a top dendrogram.

    Parameters
    ----------
    corr : np.ndarray of shape (n_features, n_features)
        Pearson correlation matrix.
    labels : list of str
        Variable names corresponding to the rows/columns of *corr*.

    Returns
    -------
    plt.Figure
    """
    size = max(8, len(labels) * 0.75)
    g = sns.clustermap(
        corr,
        cmap="BrBG",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.3,
        figsize=(size, size * 0.85),
        xticklabels=labels,
        yticklabels=labels,
        dendrogram_ratio=0.2,
    )
    g.ax_row_dendrogram.set_visible(False)
    g.fig.suptitle(f"Pearson correlation — {len(labels)} variables", y=1.02)
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
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from matplotlib.patches import Patch

    dist = (1 - corr) / 2
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    cluster_labels = fcluster(Z, n_clusters, criterion="maxclust")

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
) -> plt.Figure:
    """Clustered correlation heatmap for combined inner + outer variables.

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

    Returns
    -------
    plt.Figure
    """
    labels = inner + outer
    size = max(8, len(labels) * 0.75)
    g = sns.clustermap(
        corr,
        cmap="BrBG",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.3,
        figsize=(size, size * 0.85),
        xticklabels=labels,
        yticklabels=labels,
        dendrogram_ratio=0.2,
    )
    g.ax_row_dendrogram.set_visible(False)
    inner_set = set(inner)
    for lbl in g.ax_heatmap.get_xticklabels():
        lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)
    for lbl in g.ax_heatmap.get_yticklabels():
        lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)
    g.fig.suptitle("Inner + Outer Pearson correlation", y=1.02)
    return g.fig


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
