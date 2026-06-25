"""PCA diagnostic plots.

Functions
---------
plot_variance
    Cumulative explained-variance line plot.
plot_scatter
    Scatter plot of the first two principal components.
plot_top_loadings
    Horizontal bar chart of the top-N variables for a given PC, colour-coded
    by variable group.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from insideout.decomposition import PCAResult
from insideout.viz.utils import save_fig


def plot_variance(res: PCAResult, title: str, out_dir: Path) -> None:
    """Plot the cumulative explained variance ratio.

    Parameters
    ----------
    res : PCAResult
        Result of a PCA fit.
    title : str
        Title label (e.g. ``"Combined"``, ``"Inner"``, ``"Outer"``).
    out_dir : Path
        Directory where the figure is saved (PNG + PDF).

    Examples
    --------
    >>> res = run_pca(df)
    >>> plot_variance(res, "Combined", Path("results/plots/pca"))
    """
    cum = np.cumsum(res.explained_variance_ratio)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cum) + 1), cum, marker="o", linestyle="--")
    ax.set_title(f"Accumulated variance explained by ({title})")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Accumulated explained variance")
    ax.grid(True)
    save_fig(fig, f"{title.lower()}_cumulative_variance", out_dir)
    plt.close(fig)


def plot_scatter(res: PCAResult, title: str, out_dir: Path) -> None:
    """Scatter plot of the first two principal components.

    Parameters
    ----------
    res : PCAResult
        Result of a PCA fit.
    title : str
        Title label.
    out_dir : Path
        Directory where the figure is saved.

    Examples
    --------
    >>> plot_scatter(res, "Combined", Path("results/plots/pca"))
    """
    if res.scores.shape[1] < 2:
        return  # need at least two PCs for a 2-D scatter
    e = res.explained_variance_ratio
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x="PC1", y="PC2", data=res.scores.to_pandas(), ax=ax)
    ax.set_title(f"PCA - First two PCs ({title})")
    ax.set_xlabel(f"PC 1 ({e[0]:.2%})")
    ax.set_ylabel(f"PC 2 ({e[1]:.2%})")
    ax.grid(True)
    save_fig(fig, f"{title.lower()}_pc1_pc2_scatter", out_dir)
    plt.close(fig)


def plot_top_loadings(
    loadings: pl.DataFrame,
    pc: int,
    inner_vars: list[str],
    outer_vars: list[str],
    top_n: int,
    title: str,
    out_dir: Path,
    inner_color: str = "#3498db",
    outer_color: str = "#e67e22",
) -> None:
    """Horizontal bar chart of the top-N variables for a given PC.

    Variables belonging to the *inner* group are coloured ``#3498db``, those in
    the *outer* group are coloured ``#e67e22``.

    Parameters
    ----------
    loadings : pl.DataFrame
        PCA loadings with a ``"variable"`` column.
    pc : int
        Principal component index (1-based).
    inner_vars : list of str
        Names of inner-group variables.
    outer_vars : list of str
        Names of outer-group variables.
    top_n : int
        How many top variables to show.
    title : str
        Title label.
    out_dir : Path
        Directory where the figure is saved.
    inner_color : str, optional
        Colour for inner-group tick labels (default ``"steelblue"``).
    outer_color : str, optional
        Colour for outer-group tick labels (default ``"tomato"``).

    Examples
    --------
    >>> plot_top_loadings(res.loadings, 1, inner, outer, 10, "Combined", out_dir,
    ...                   inner_color="#3498db", outer_color="#e67e22")
    """
    col = f"PC{pc}"
    if col not in loadings.columns:
        return  # requested PC not in loadings table

    top = loadings.select(["variable", col]).sort(col, descending=True).head(top_n)
    names = top["variable"].to_list()
    vals = top[col].to_list()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=vals, y=names, color="purple", alpha=0.5, ax=ax)
    ax.set_title(f"Top {top_n} most relevant variables in {col} ({title})")
    ax.set_xlabel("Loading")
    ax.set_ylabel("Variable")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    for label in ax.get_yticklabels():
        t = label.get_text()
        if t in inner_vars:
            label.set_color(inner_color)
        elif t in outer_vars:
            label.set_color(outer_color)  # colour-code by variable group

    save_fig(fig, f"{title.lower()}_pc{pc}_top_loadings", out_dir)
    plt.close(fig)

    def block(v: str) -> str:
        return "Inner" if v in inner_vars else "Outer" if v in outer_vars else "Other"

    print(f"\nComposition of top {top_n} variables in {col} ({title}):")
    for n, v in zip(names, vals):
        print(f"  - {n} ({block(n)}): {v:.2f}")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
