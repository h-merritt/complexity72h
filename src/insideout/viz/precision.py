"""Precision / covariance matrix visualisation functions.

Functions
---------
plot_covariance_heatmap
    Clustered heatmap of a covariance matrix.
plot_precision_heatmap
    Heatmap of a precision (inverse covariance) matrix.
plot_precision_graph
    Force-directed graph of a precision matrix.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_covariance_heatmap(cov: np.ndarray, labels: list[str]) -> plt.Figure:
    """Clustered covariance heatmap with hierarchical dendrograms.

    Parameters
    ----------
    cov : np.ndarray of shape (n_features, n_features)
        Sample covariance matrix.
    labels : list of str
        Variable names corresponding to the rows and columns of *cov*.

    Returns
    -------
    plt.Figure
        The clustered heatmap figure.

    Examples
    --------
    >>> import numpy as np
    >>> from insideout.viz.precision import plot_covariance_heatmap
    >>> cov = np.cov(np.random.randn(50, 4).T)
    >>> fig = plot_covariance_heatmap(cov, ["x1", "x2", "x3", "x4"])
    >>> fig.show()
    """
    size = max(8, len(labels) * 0.55)
    g = sns.clustermap(
        cov,
        cmap="RdBu_r",
        center=0,
        figsize=(size, size * 0.85),
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.3,
        dendrogram_ratio=0.15,
    )
    return g.fig


def plot_precision_heatmap(precision: np.ndarray, labels: list[str]) -> plt.Figure:
    """Heatmap of a precision (inverse covariance) matrix.

    Parameters
    ----------
    precision : np.ndarray of shape (n_features, n_features)
        Precision matrix.
    labels : list of str
        Variable names corresponding to the rows and columns of *precision*.

    Returns
    -------
    plt.Figure
        The heatmap figure.

    Examples
    --------
    >>> import numpy as np
    >>> from insideout.viz.precision import plot_precision_heatmap
    >>> prec = np.linalg.pinv(np.cov(np.random.randn(50, 4).T))
    >>> fig = plot_precision_heatmap(prec, ["x1", "x2", "x3", "x4"])
    >>> fig.show()
    """
    size = max(8, len(labels) * 0.55)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        precision,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        annot=len(labels) <= 15,
        fmt=".1f",
        linewidths=0.3,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.tight_layout()
    return fig


def plot_precision_graph(
    precision: np.ndarray,
    labels: list[str],
    threshold: float = 0.0,
) -> plt.Figure:
    """Force-directed graph of a precision (or any symmetric) matrix.

    Nodes represent variables; an edge between nodes *i* and *j* is drawn
    when ``|precision[i, j]| > threshold``. Edge width and opacity scale
    with the absolute value; colour encodes sign (red = positive,
    blue = negative partial correlation).

    Parameters
    ----------
    precision : np.ndarray of shape (n_features, n_features)
        Precision matrix (or any symmetric weight matrix).
    labels : list of str
        Variable names used as node labels.
    threshold : float, optional
        Minimum absolute edge weight to display (default ``0.0``).

    Returns
    -------
    plt.Figure
        The force-directed graph figure.

    Examples
    --------
    >>> import numpy as np
    >>> from insideout.viz.precision import plot_precision_graph
    >>> prec = np.linalg.pinv(np.cov(np.random.randn(50, 4).T))
    >>> fig = plot_precision_graph(prec, ["x1", "x2", "x3", "x4"], threshold=0.1)
    >>> fig.show()
    """

    import igraph as ig
    from matplotlib.lines import Line2D

    n = len(labels)
    edges, weights, signs = [], [], []
    for i in range(n):
        for j in range(i + 1, n):  # iterate upper triangle only
            val = precision[i, j]
            if abs(val) > threshold:
                edges.append((i, j))
                weights.append(abs(val))
                signs.append(1 if val > 0 else -1)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    if not edges:
        ax.text(
            0.5,
            0.5,
            "No edges above threshold",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
        )
        ax.set_axis_off()
        return fig

    g = ig.Graph(n=n, edges=edges)
    coords = np.array(
        g.layout_kamada_kawai().coords
    )  # force-directed layout for readable node placement
    degrees = np.array(g.degree())
    max_w = max(weights)
    norm_w = [w / max_w for w in weights]

    pos_col = np.array([0.831, 0.176, 0.176])
    neg_col = np.array([0.161, 0.502, 0.725])

    for (src, tgt), nw, sign in zip(edges, norm_w, signs):
        ax.plot(
            [coords[src, 0], coords[tgt, 0]],
            [coords[src, 1], coords[tgt, 1]],
            color=pos_col if sign > 0 else neg_col,
            alpha=0.15 + 0.7 * nw,
            linewidth=0.5 + 4.5 * nw,
            solid_capstyle="round",
            zorder=1,
        )

    node_sizes = 180 + 500 * degrees / max(degrees.max(), 1)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=node_sizes,
        c="#f0f3f4",
        edgecolors="#2c3e50",
        linewidths=1.5,
        zorder=2,
    )

    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    for i, (x, y) in enumerate(coords):
        dx, dy = x - cx, y - cy
        dist = np.hypot(dx, dy) or 1e-9
        pad = 0.12
        ox, oy = pad * dx / dist, pad * dy / dist
        ax.text(
            x + ox,
            y + oy,
            labels[i],
            fontsize=7.5,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec="none"),
        )

    ax.legend(
        handles=[
            Line2D([0], [0], color=pos_col, linewidth=2.5, label="Positive"),
            Line2D([0], [0], color=neg_col, linewidth=2.5, label="Negative"),
        ],
        title="Partial correlation",
        title_fontsize=9,
        loc="lower right",
        fontsize=9,
        framealpha=0.95,
        edgecolor="#dddddd",
    )
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
