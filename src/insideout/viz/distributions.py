"""Distribution visualisation functions.

Functions
---------
plot_distribution
    Histogram + KDE + boxplot for a single variable.
plot_group_distributions
    Horizontal violin + box + strip plot for a set of variables.
plot_gender_violin
    Split violin plot comparing a metric across gender groups.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage


def plot_distribution(data: np.ndarray, name: str) -> plt.Figure:
    """Plot a histogram with KDE and a boxplot for a single variable.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples,)
        Observed values.
    name : str
        Variable name used for axis labels and titles.

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(100)
    >>> fig = plot_distribution(data, "test")
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram — {name}")
    axes[0].set_xlabel(name)
    sns.boxplot(y=data, ax=axes[1])
    axes[1].set_title(f"Boxplot — {name}")
    axes[1].set_ylabel(name)
    plt.tight_layout()
    return fig


def plot_group_distributions(
    X: np.ndarray,
    labels: list[str],
    color: str = "#3498db",
    group_name: str = "",
) -> plt.Figure:
    """Horizontal violin + box + strip plot for a set of variables.

    All variables share a common axis: metrics on the y-axis, scores on
    the x-axis. Rows maintain the order of *labels*.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix; columns correspond to *labels*.
    labels : list of str
        Variable names (must match the column order of *X*).
    color : str, optional
        Fill colour for all plot elements (default: ``"#3498db"``).
    group_name : str, optional
        Title shown above the figure (default: ``""``).

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 4)
    >>> fig = plot_group_distributions(X, ["a", "b", "c", "d"])
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    long = pd.DataFrame(X, columns=labels).melt(var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 1.2)))
    sns.violinplot(  # layer 1: kernel-density shape
        data=long,
        x="Value",
        y="Metric",
        order=labels,
        ax=ax,
        color=color,
        inner=None,
        alpha=0.45,
    )
    sns.boxplot(  # layer 2: quartile box (whiskers only, no outliers)
        data=long,
        x="Value",
        y="Metric",
        order=labels,
        ax=ax,
        color=color,
        width=0.3,
        fliersize=0,
        linewidth=1.5,
    )
    sns.stripplot(  # layer 3: raw jittered points
        data=long,
        x="Value",
        y="Metric",
        order=labels,
        ax=ax,
        color=color,
        alpha=0.25,
        jitter=True,
        size=2,
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Social Metric")
    if group_name:
        ax.set_title(group_name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_combined_distributions(
    X: np.ndarray,
    inner_names: list[str],
    outer_names: list[str],
    inner_color: str = "#3498db",
    outer_color: str = "#e67e22",
) -> plt.Figure:
    """Vertical violin + box + strip plot for combined Inner + Outer metrics.

    Metrics are ordered by hierarchical clustering of the correlation matrix,
    first all Inner metrics, then all Outer. X-axis tick labels and plot
    elements are colour-coded by group (inner / outer).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_inner + n_outer)
        Combined data matrix; first ``n_inner`` columns correspond to
        *inner_names*, remaining to *outer_names*.
    inner_names : list of str
        Variable names for the inner block (order matches first columns of X).
    outer_names : list of str
        Variable names for the outer block (order matches last columns of X).
    inner_color : str, optional
        Colour for inner-group elements (default ``"#3498db"``).
    outer_color : str, optional
        Colour for outer-group elements (default ``"#e67e22"``).

    Returns
    -------
    plt.Figure
    """
    n_inner = len(inner_names)
    n_outer = len(outer_names)
    X_inner = X[:, :n_inner]
    X_outer = X[:, n_inner:]

    def _dendro_order(cols, name_list):
        corr = np.corrcoef(cols.T)
        Z = linkage(corr, method="average")
        leaves = dendrogram(Z, no_plot=True)["leaves"]
        return [name_list[i] for i in leaves]

    inner_ordered = _dendro_order(X_inner, inner_names)
    outer_ordered = _dendro_order(X_outer, outer_names)
    order = inner_ordered + outer_ordered

    idx_map = {name: i for i, name in enumerate(inner_names)}
    idx_map.update({name: n_inner + i for i, name in enumerate(outer_names)})
    reordered_idx = [idx_map[name] for name in order]
    X_reordered = X[:, reordered_idx]

    group = ["inner"] * n_inner + ["outer"] * n_outer
    long = pd.DataFrame(X_reordered, columns=order).melt(
        var_name="Metric", value_name="Value"
    )
    long["Group"] = long["Metric"].map(dict(zip(order, group)))

    palette_colors = [inner_color] * n_inner + [outer_color] * n_outer

    fig, ax = plt.subplots(figsize=(max(16, len(order) * 0.9), 8))

    sns.violinplot(
        data=long,
        x="Metric",
        y="Value",
        order=order,
        hue="Metric",
        palette=palette_colors,
        legend=False,
        inner=None,
        alpha=0.45,
        ax=ax,
    )
    sns.boxplot(
        data=long,
        x="Metric",
        y="Value",
        order=order,
        hue="Metric",
        palette=palette_colors,
        legend=False,
        width=0.3,
        fliersize=0,
        linewidth=1.5,
        ax=ax,
    )
    sns.stripplot(
        data=long,
        x="Metric",
        y="Value",
        order=order,
        hue="Metric",
        palette=palette_colors,
        legend=False,
        alpha=0.25,
        jitter=True,
        size=2,
        ax=ax,
    )

    inner_set = set(inner_names)
    for lbl in ax.get_xticklabels():
        lbl.set_color(inner_color if lbl.get_text() in inner_set else outer_color)
        lbl.set_rotation(45)
        lbl.set_ha("right")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("")
    ax.set_ylabel("Z-Score")
    plt.tight_layout()
    sns.despine(top=True, right=True, bottom=True, left=False)
    return fig


def plot_gender_violin(
    values: np.ndarray,
    genders: np.ndarray,
    metric_name: str,
    gender_name: str = "Gender",
) -> plt.Figure:
    """Split violin plot comparing a metric across two gender groups.

    Parameters
    ----------
    values : np.ndarray of shape (n_samples,)
        Numeric metric values.
    genders : np.ndarray of shape (n_samples,)
        Categorical gender labels (e.g. ``"F"``, ``"M"``).
    metric_name : str
        Name of the metric shown on the y-axis.
    gender_name : str, optional
        Column name used for the hue legend (default: ``"Gender"``).

    Returns
    -------
    plt.Figure

    Examples
    --------
    >>> import numpy as np
    >>> vals = np.random.randn(100)
    >>> genders = np.array(["F"] * 50 + ["M"] * 50)
    >>> fig = plot_gender_violin(vals, genders, "score")
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """
    pdf = pd.DataFrame({metric_name: values, gender_name: genders})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.violinplot(
        data=pdf,
        y=metric_name,
        hue=gender_name,
        split=True,
        gap=0.05,
        ax=ax,
        inner="box",
        palette={"F": "#e74c3c", "M": "#3498db"},
    )
    ax.set_title(f"{metric_name} — by {gender_name}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import doctest

    doctest.testmod()
