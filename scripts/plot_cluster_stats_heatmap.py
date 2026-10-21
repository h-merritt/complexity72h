"""Significance heatmaps of the cluster statistical tests.

Reads ``results/clustering/statistical_tests.parquet`` (written by
``scripts/statistical_tests_clusters.py``) and draws, for each subset (columns)
and each k (x axis), the significance level of every tested variable: one pair
of figures (raw and Bonferroni p-values) for the subset's own variables and one
pair for the mental-health variables. The subset variables were used to build
the clusters, so those p-values are circular and mainly descriptive.

Output: ``results/plots/clustering/statistical_tests/{png,pdf}/``
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from omegaconf import DictConfig

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT_DIR / "results" / "clustering" / "statistical_tests.parquet"
OUT_DIR = ROOT_DIR / "results" / "plots" / "clustering" / "statistical_tests"

CMAP = ListedColormap(["white", "#c7e9c0", "#74c476", "#238b45", "#e0e0e0"])
PVAL_THRESHOLDS = [0.05, 0.01, 0.001]
PVAL_LABELS = ["n.s.", "*", "**", "***"]

legend_elements = [
    Patch(facecolor="white", edgecolor="#dddddd", label=r"n.s.  (p $\geq$ 0.05)"),
    Patch(facecolor="#c7e9c0", edgecolor="#dddddd", label=r"*  (p < 0.05)"),
    Patch(facecolor="#74c476", edgecolor="#dddddd", label=r"**  (p < 0.01)"),
    Patch(facecolor="#238b45", edgecolor="#dddddd", label=r"***  (p < 0.001)"),
    Patch(facecolor="#e0e0e0", edgecolor="#dddddd", label="no data"),
]


def _subset_label(subset: str) -> str:
    """Panel title, e.g. ``"Inner & Outer"`` -> ``"Inner + Outer variables"``."""
    return f"{subset.replace('&', '+')} variables"


def _build_figure(
    df: pl.DataFrame,
    title: str,
    filename: str,
    subsets: list[str],
    pval_col: str = "pval",
) -> None:
    """Draw one significance heatmap per subset and save it as PNG and PDF.

    Parameters
    ----------
    df : pl.DataFrame
        Rows of the statistical-test table to draw (columns ``subset``, ``k``,
        ``metric`` and ``pval_col``).
    title : str
        Figure title.
    filename : str
        Output file stem.
    subsets : list of str
        Subset display names, one panel each, in plotting order.
    pval_col : str
        ``"pval"`` (raw) or ``"pval_bonf"`` (Bonferroni corrected).
    """
    k_values = sorted(df["k"].unique().to_list())

    fig, axes = plt.subplots(1, len(subsets), figsize=(22, 6.5), squeeze=False)
    axes = axes[0]

    for idx, subset in enumerate(subsets):
        sub = df.filter(pl.col("subset") == subset)
        metrics = sorted(sub["metric"].unique().to_list())
        n_metrics = len(metrics)
        n_k = len(k_values)

        color_mat = np.full((n_metrics, n_k), 4, dtype=int)
        annot_mat = np.full((n_metrics, n_k), "", dtype=object)

        for i, m in enumerate(metrics):
            for j, k in enumerate(k_values):
                row = sub.filter((pl.col("metric") == m) & (pl.col("k") == k))
                if row.height == 0:
                    continue
                p = row[pval_col][0]
                if np.isnan(p) or np.isinf(p):
                    continue
                if p >= PVAL_THRESHOLDS[0]:
                    level = 0
                elif p >= PVAL_THRESHOLDS[1]:
                    level = 1
                elif p >= PVAL_THRESHOLDS[2]:
                    level = 2
                else:
                    level = 3
                color_mat[i, j] = level
                annot_mat[i, j] = PVAL_LABELS[level]

        ax = axes[idx]
        sns.heatmap(
            color_mat,
            ax=ax,
            cmap=CMAP,
            vmin=0,
            vmax=4,
            annot=annot_mat,
            fmt="",
            linewidths=0.5,
            linecolor="#dddddd",
            cbar=False,
            xticklabels=[f"k={k}" for k in k_values],
            yticklabels=metrics,
        )
        ax.set_title(_subset_label(subset), fontsize=14, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", rotation=0)
        ax.set_xlabel("")

    fig.suptitle(title, fontsize=16, y=0.98)
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncols=5,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 1.04),
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.87)

    for fmt in ("png", "pdf"):
        out = OUT_DIR / fmt
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{filename}.{fmt}", bbox_inches="tight", dpi=300)
        print(f"Saved {out / f'{filename}.{fmt}'}")

    plt.close(fig)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    mh_metrics = {entry.name for entry in cfg.components.mental_health}

    df = pl.read_parquet(PARQUET_PATH)
    subsets = [s for s in df["subset"].unique(maintain_order=True).to_list()]

    df_subset = df.filter(~pl.col("metric").is_in(mh_metrics))
    _build_figure(
        df_subset,
        "Cluster separation — subset metrics (raw p-values)",
        "cluster_stats_heatmap",
        subsets,
        pval_col="pval",
    )
    _build_figure(
        df_subset,
        "Cluster separation — subset metrics (Bonferroni corrected)",
        "cluster_stats_heatmap_bonf",
        subsets,
        pval_col="pval_bonf",
    )

    df_mh = df.filter(pl.col("metric").is_in(mh_metrics))
    _build_figure(
        df_mh,
        "Cluster separation — MH metrics (raw p-values)",
        "cluster_stats_heatmap_mh",
        subsets,
        pval_col="pval",
    )
    _build_figure(
        df_mh,
        "Cluster separation — MH metrics (Bonferroni corrected)",
        "cluster_stats_heatmap_mh_bonf",
        subsets,
        pval_col="pval_bonf",
    )


if __name__ == "__main__":
    main()
