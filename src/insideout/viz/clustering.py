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
    # Top-left: inertia (elbow), plotted over all k including k=1
    axes[0, 0].plot(k_range, metrics_dict["Inertia"], marker="o", color="black")
    axes[0, 0].set_title("Inertia (WCSS)")
    axes[0, 0].set_xticks(k_range)

    # Top-right: silhouette score (higher is better, k>=2 only)
    axes[0, 1].plot(k_valid, sil, marker="s", color="black")
    axes[0, 1].set_title("Silhouette Score")
    axes[0, 1].set_xticks(k_valid)

    # Bottom-left: Calinski-Harabasz index (higher is better)
    axes[1, 0].plot(k_valid, ch, marker="^", color="black")
    axes[1, 0].set_title("Calinski-Harabasz Index")
    axes[1, 0].set_xticks(k_valid)

    # Bottom-right: Davies-Bouldin index (lower is better)
    axes[1, 1].plot(k_valid, db, marker="d", color="black")
    axes[1, 1].set_title("Davies-Bouldin Index")
    axes[1, 1].set_xticks(k_valid)

    sns.despine()
    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_evaluation_metrics", out_dir)
    plt.close(fig)


def plot_cluster_heatmap_avg_std(
    cluster_means: np.ndarray,
    metric_names: list[str],
    cluster_labels: list[str],
    subset_name: str,
    k: int,
    out_dir: str,
    cmap: str = "Reds",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Heatmap of cluster centroids coloured by mean z-score.

    Parameters
    ----------
    cluster_means : np.ndarray of shape (k, n_metrics)
        Mean z-score per cluster per metric.
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
    vmin : float or None, optional
        Minimum value for the colormap (default ``None`` = data min).
    vmax : float or None, optional
        Maximum value for the colormap (default ``None`` = data max).

    Examples
    --------
    >>> means = np.array([[0.5, -0.3], [-0.5, 0.3]])
    >>> plot_cluster_heatmap_avg_std(means, ["A", "B"],
    ...     ["1 (50%)", "2 (50%)"], "Test", 2, "results/plots/clustering/k2")
    """
    fig, ax = plt.subplots(figsize=(max(14, len(metric_names) * 1.5), max(8, k * 1.5)))
    sns.heatmap(
        cluster_means,
        annot=False,
        square=True,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "Mean Z-Score"},
        linewidths=0.5,
        xticklabels=metric_names,
        yticklabels=cluster_labels,
        ax=ax,
    )
    ax.set_ylabel("Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2f}".replace("-", "\\textminus{}"))
        )

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
    ax.legend(title="Cluster (\\%)")
    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_pca_scatter", out_dir)
    plt.close(fig)


def plot_cluster_correlation_matrix(
    X: np.ndarray,
    labels: np.ndarray,
    subset_name: str,
    k: int,
    out_dir: str,
    cmap: str = "RdBu",
) -> None:
    """Subject-by-subject correlation matrix reordered by cluster assignment.

    Parameters
    ----------
    X : np.ndarray of shape (n_subjects, n_features)
        Standardised data matrix.
    labels : np.ndarray of shape (n_subjects,)
        Cluster assignment (integer labels 0 … k-1).
    subset_name : str
        Name of the variable subset.
    k : int
        Number of clusters.
    out_dir : str or Path
        Directory where the figure is saved.
    cmap : str, optional
        Seaborn / matplotlib colormap name (default ``"RdBu"``).
    """
    corr = np.corrcoef(X)

    sort_idx = np.argsort(labels)
    corr_sorted = corr[sort_idx][:, sort_idx]
    labels_sorted = labels[sort_idx]

    boundaries = [0]
    for cid in range(k):
        idx = np.where(labels_sorted == cid)[0]
        if len(idx) > 0:
            boundaries.append(idx[-1] + 1)

    n = X.shape[0]
    size = min(14, max(6, n * 0.025))
    fig, ax = plt.subplots(figsize=(size, size * 0.9))

    sns.heatmap(
        corr_sorted,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Pearson r"},
        xticklabels=False,
        yticklabels=False,
    )

    for b in boundaries[1:-1]:
        ax.axhline(b, color="black", linewidth=1.5)
        ax.axvline(b, color="black", linewidth=1.5)

    ax.set_title(f"{subset_name} Clusters (k={k}): Subject Correlation Matrix")
    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_subject_corr", out_dir)
    plt.close(fig)


def plot_cluster_means_bars(
    cluster_means: np.ndarray,
    cluster_stds: np.ndarray,
    cluster_sizes: np.ndarray,
    metric_names: list[str],
    cluster_labels: list[str],
    subset_name: str,
    k: int,
    out_dir: str,
) -> None:
    """Grouped bar plot of cluster means per metric with SE error bars.

    Parameters
    ----------
    cluster_means : np.ndarray of shape (k, n_metrics)
        Mean z-score per cluster per metric.
    cluster_stds : np.ndarray of shape (k, n_metrics)
        Standard deviation per cluster per metric.
    cluster_sizes : np.ndarray of shape (k,)
        Number of observations in each cluster.
    metric_names : list of str
        Names of the metrics (x-axis).
    cluster_labels : list of str
        Labels for each cluster (hue).
    subset_name : str
        Name of the variable subset.
    k : int
        Number of clusters.
    out_dir : str or Path
        Directory where the figure is saved.

    Examples
    --------
    >>> means = np.array([[0.5, -0.3], [-0.5, 0.3]])
    >>> stds = np.array([[0.8, 0.9], [0.7, 1.0]])
    >>> sizes = np.array([50, 50])
    >>> plot_cluster_means_bars(means, stds, sizes, ["A", "B"],
    ...     ["1 (50%)", "2 (50%)"], "Test", 2, "results/plots/clustering/k2")
    """
    cluster_ses = cluster_stds / np.sqrt(cluster_sizes)[:, np.newaxis]

    df = pd.DataFrame(cluster_means, columns=metric_names)
    df["Cluster"] = cluster_labels
    long_df = df.melt(id_vars="Cluster", var_name="Metric", value_name="Mean Z-Score")

    fig, ax = plt.subplots(figsize=(max(14, len(metric_names) * 1.0), 6))
    sns.barplot(
        data=long_df,
        x="Metric",
        y="Mean Z-Score",
        hue="Cluster",
        palette="tab10",
        legend=False,
        ax=ax,
    )
    bars = ax.patches
    for i in range(k):
        cluster_bars = bars[i::k]
        centers = [b.get_x() + b.get_width() / 2 for b in cluster_bars]
        heights = [b.get_height() for b in cluster_bars]
        ax.errorbar(
            centers,
            heights,
            yerr=cluster_ses[i],
            fmt="none",
            capsize=3,
            color="black",
            linewidth=0.8,
        )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Z-Score")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    sns.despine(top=True, right=True, bottom=True, left=False)
    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_means_bars", out_dir)
    plt.close(fig)


def plot_cluster_means_lollipop(
    cluster_means: np.ndarray,
    metric_names: list[str],
    cluster_labels: list[str],
    subset_name: str,
    k: int,
    out_dir: str,
) -> None:
    """Grouped lollipop plot of cluster means per metric.

    Parameters
    ----------
    cluster_means : np.ndarray of shape (k, n_metrics)
        Mean z-score per cluster per metric.
    metric_names : list of str
        Names of the metrics (x-axis).
    cluster_labels : list of str
        Labels for each cluster (colour).
    subset_name : str
        Name of the variable subset.
    k : int
        Number of clusters.
    out_dir : str or Path
        Directory where the figure is saved.

    Examples
    --------
    >>> means = np.array([[0.5, -0.3], [-0.5, 0.3]])
    >>> plot_cluster_means_lollipop(means, ["A", "B"],
    ...     ["1 (50%)", "2 (50%)"], "Test", 2, "results/plots/clustering/k2")
    """
    n_metrics = len(metric_names)
    palette = sns.color_palette("tab10", k)
    x = np.arange(n_metrics)
    offsets = np.linspace(-0.4, 0.4, k)

    fig, ax = plt.subplots(figsize=(max(14, n_metrics * 1.0), 6))
    for i in range(k):
        xi = x + offsets[i]
        ax.vlines(xi, 0, cluster_means[i], color=palette[i], linewidth=1.5)
        ax.scatter(
            xi,
            cluster_means[i],
            color=palette[i],
            s=50,
            zorder=3,
            label=cluster_labels[i],
        )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Mean Z-Score")
    ax.legend(title="Cluster")
    sns.despine(top=True, right=True, bottom=True, left=False)
    plt.tight_layout()
    save_fig(fig, f"{subset_name.lower()}_means_lollipop", out_dir)
    plt.close(fig)
