from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import save_fig


def plot_clustering_metrics(
    metrics_dict: dict[str, list],
    subset_name: str,
    out_dir: str,
) -> None:
    """Plot a 2x2 grid of clustering quality metrics across k values.

    Parameters
    ----------
    metrics_dict : dict of str -> list
        Contains keys ``"k"``, ``"Inertia"``, ``"Silhouette"``,
        ``"CH_Index"``, ``"DB_Index"``. Each value is a list with one
        entry per evaluated k.
    subset_name : str
        Name of the variable subset (e.g. ``"Inner"``, ``"Combined_2"``).
    out_dir : str or Path
        Directory where the figure is saved (PNG + PDF).

    Examples
    --------
    >>> d = {"k": [2,3], "Inertia": [100, 80], "Silhouette": [0.3, 0.4],
    ...      "CH_Index": [50, 60], "DB_Index": [1.5, 1.2]}
    >>> plot_clustering_metrics(d, "Test", "results/plots/clustering")
    """
    k_range = metrics_dict["k"]
    k_valid = [x for x in k_range if x >= 2]  # silhouette/CH/DB need k>=2
    sil = metrics_dict["Silhouette"]
    ch = metrics_dict["CH_Index"]
    db = metrics_dict["DB_Index"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"{subset_name} Variables: Clustering Evaluation Metrics", fontsize=14)

    # Top-left: inertia (elbow), plotted over all k including k=1
    axes[0, 0].plot(k_range, metrics_dict["Inertia"], marker="o", color="tab:red")
    axes[0, 0].set_title("Inertia (WCSS)")
    axes[0, 0].set_xticks(k_range)

    # Top-right: silhouette score (higher is better, k>=2 only)
    axes[0, 1].plot(k_valid, sil, marker="s", color="tab:blue")
    axes[0, 1].set_title("Silhouette Score")
    axes[0, 1].set_xticks(k_valid)

    # Bottom-left: Calinski-Harabasz index (higher is better)
    axes[1, 0].plot(k_valid, ch, marker="^", color="tab:green")
    axes[1, 0].set_title("Calinski-Harabasz Index")
    axes[1, 0].set_xticks(k_valid)

    # Bottom-right: Davies-Bouldin index (lower is better)
    axes[1, 1].plot(k_valid, db, marker="d", color="tab:purple")
    axes[1, 1].set_title("Davies-Bouldin Index")
    axes[1, 1].set_xticks(k_valid)

    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_evaluation_metrics", out_dir)
    plt.close(fig)


def plot_cluster_heatmap_avg_std(
    cluster_means: np.ndarray,
    cluster_stds: np.ndarray,
    metric_names: list[str],
    cluster_labels: list[str],
    subset_name: str,
    k: int,
    out_dir: str,
    cmap: str = "Reds",
) -> None:
    """Heatmap of cluster centroids annotated with mean ± standard deviation.

    Each cell displays the mean z-score of a metric within a cluster on the
    first line and the standard deviation on the second line.

    Parameters
    ----------
    cluster_means : np.ndarray of shape (k, n_metrics)
        Mean z-score per cluster per metric.
    cluster_stds : np.ndarray of shape (k, n_metrics)
        Standard deviation per cluster per metric.
    metric_names : list of str
        Column names for the heatmap x-axis tick labels.
    cluster_labels : list of str
        Row labels for the heatmap y-axis (e.g. ``"1 (15.2%)"``).
    subset_name : str
        Name of the variable subset.
    k : int
        Number of clusters.
    out_dir : str or Path
        Directory where the figure is saved.
    cmap : str, optional
        Seaborn / matplotlib colormap name (default ``"Reds"``).

    Examples
    --------
    >>> means = np.array([[0.5, -0.3], [-0.5, 0.3]])
    >>> stds = np.array([[0.8, 0.9], [0.7, 1.0]])
    >>> plot_cluster_heatmap_avg_std(means, stds, ["A", "B"],
    ...     ["1 (50%)", "2 (50%)"], "Test", 2, "results/plots/clustering/k2")
    """
    # Build two-line annotation strings: "0.45\n±0.12" per cell
    annot = np.empty_like(cluster_means, dtype=object)
    for i in range(cluster_means.shape[0]):
        for j in range(cluster_means.shape[1]):
            annot[i, j] = f"{cluster_means[i, j]:.2f}\n±{cluster_stds[i, j]:.2f}"

    fig, ax = plt.subplots(figsize=(max(16, len(metric_names) * 1.0), max(8, k * 1.5)))
    sns.heatmap(
        cluster_means,
        annot=annot,
        fmt="",
        annot_kws={"size": 7},
        cmap=cmap,
        center=0,
        cbar_kws={"label": "Mean Z-Score"},
        linewidths=0.5,
        xticklabels=metric_names,
        yticklabels=cluster_labels,
        ax=ax,
    )
    ax.set_title(f"{subset_name} Clusters (k={k}): Average Z-Scores ± SD")
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Survey Measure")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_heatmap", out_dir)
    plt.close(fig)


def plot_cluster_distributions(
    X: np.ndarray,
    labels: np.ndarray,
    metric_names: list[str],
    subset_name: str,
    k: int,
    out_dir: str,
) -> None:
    """Violin + box + strip plots of metric distributions per cluster.

    Creates one subplot per cluster. Each subplot shows the full distribution
    of all metrics (as z-scores) for the observations belonging to that
    cluster, using a layered violin + box + strip visualisation.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_metrics)
        Standardised data matrix.
    labels : np.ndarray of shape (n_samples,)
        Cluster assignment (integer labels 0 … k-1).
    metric_names : list of str
        Names of the metrics, used as y-axis tick labels.
    subset_name : str
        Name of the variable subset.
    k : int
        Number of clusters.
    out_dir : str or Path
        Directory where the figure is saved.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(100, 4))
    >>> lbls = rng.integers(0, 2, size=100)
    >>> plot_cluster_distributions(X, lbls, ["A","B","C","D"], "Test", 2, "tmp")
    """
    n_clusters = len(np.unique(labels))
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for cluster_id in range(n_clusters):
        X_cluster = X[labels == cluster_id]
        # Melt to long format so seaborn can hue on Metric
        long_df = pd.DataFrame(X_cluster, columns=metric_names).melt(
            var_name="Metric", value_name="Value"
        )
        ax = axes_flat[cluster_id]
        # Three visualisation layers: violin (density), box (quartiles), strip (raw)
        sns.violinplot(
            data=long_df,
            x="Value",
            y="Metric",
            order=metric_names,
            ax=ax,
            inner=None,
            alpha=0.45,
            color="tab:blue",
        )
        sns.boxplot(
            data=long_df,
            x="Value",
            y="Metric",
            order=metric_names,
            ax=ax,
            width=0.3,
            fliersize=0,
            linewidth=1.5,
            color="tab:blue",
        )
        sns.stripplot(
            data=long_df,
            x="Value",
            y="Metric",
            order=metric_names,
            ax=ax,
            alpha=0.25,
            jitter=True,
            size=2,
            color="tab:blue",
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(f"Cluster {cluster_id + 1} (n={X_cluster.shape[0]})")
        ax.set_xlabel("Z-Score")
        ax.set_ylabel("")

    # Hide unused subplots when k < total grid slots
    for idx in range(n_clusters, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"{subset_name} Clusters (k={k}): Metric Distributions per Cluster",
        fontsize=14,
    )
    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_distributions", out_dir)
    plt.close(fig)


def plot_cluster_pca_scatter(
    X_pca: np.ndarray,
    cluster_labels_pct: np.ndarray,
    subset_name: str,
    k: int,
    out_dir: str,
) -> None:
    """PCA scatter plot of the first two components, coloured by cluster.

    Points are projected onto PC1 and PC2. The hue legend shows each
    cluster's label and its percentage of the total sample.

    Parameters
    ----------
    X_pca : np.ndarray of shape (n_samples, 2)
        Coordinates on the first two principal components.
    cluster_labels_pct : np.ndarray of shape (n_samples,)
        Per-point cluster label with percentage, e.g. ``"1 (15.2%)"``.
    subset_name : str
        Name of the variable subset.
    k : int
        Number of clusters.
    out_dir : str or Path
        Directory where the figure is saved.

    Examples
    --------
    >>> Xp = np.random.randn(100, 2)
    >>> lbls = np.array(["1 (50%)"] * 50 + ["2 (50%)"] * 50)
    >>> plot_cluster_pca_scatter(Xp, lbls, "Test", 2, "results/plots/clustering/k2")
    """
    plot_df = pd.DataFrame(
        {"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": cluster_labels_pct}
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        alpha=0.8,
        ax=ax,
    )
    ax.set_title(f"{subset_name} Clusters (k={k}) on Principal Components")
    ax.legend(title="Cluster (%)")
    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_pca_scatter", out_dir)
    plt.close(fig)
