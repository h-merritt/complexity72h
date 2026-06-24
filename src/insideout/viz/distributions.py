from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    """
    long = pd.DataFrame(X, columns=labels).melt(var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 1.2)))
    sns.violinplot(
        data=long,
        x="Value",
        y="Metric",
        order=labels,
        ax=ax,
        color=color,
        inner=None,
        alpha=0.45,
    )
    sns.boxplot(
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
    sns.stripplot(
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
