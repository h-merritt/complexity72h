"""Visualisation subpackage for the insideout project.

Provides functions for exploring distributions, correlations, and
precision / covariance matrices in Polars DataFrames.

Exports
-------
Distributions
plot_distribution
    Histogram + boxplot for a single column.
plot_gender_violin
    Violin plot comparing a metric across two gender groups.
plot_group_distributions
    Horizontal violin + box + strip plots for a group of variables.
plot_combined_distributions
    Violin + box + strip plot of the combined inner + outer metrics.

Correlation
plot_correlation_heatmap
    Pearson correlation heatmap for a set of columns.
plot_clustermap
    Clustered correlation heatmap with hierarchical dendrogram.
plot_clustermap_top
    Clustermap restricted to the top-N variables.
plot_combined_heatmap
    Heatmap combining two variable groups with colour-coded labels.
plot_correlation_grid
    Inner, outer and combined correlation matrices in one figure
    (``layout="side"`` or ``"stacked"``).
plot_correlation_grid_stacked
    Shortcut for ``plot_correlation_grid(..., layout="stacked")``.
plot_pairplot
    Seaborn pairplot for selected columns.

Precision / covariance
plot_covariance_heatmap
    Clustered covariance heatmap.
plot_precision_heatmap
    Heatmap of a precision (inverse covariance) matrix.
plot_precision_graph
    Force-directed graph of a precision matrix.

PCA
plot_variance
    Cumulative explained variance ratio.
plot_scatter
    Scatter plot of the first two principal components.
plot_top_loadings
    Largest loadings (unit eigenvector weights) of one PC, coloured by inner/outer.

Clustering
plot_clustering_metrics
    2x2 grid of clustering quality metrics across k values.
plot_cluster_heatmap_avg_std
    Heatmap of cluster centroids coloured by mean z-score.
plot_cluster_distributions
    Violin + box + strip plots of the metric distributions per cluster.
plot_cluster_pca_scatter
    PCA scatter plot of the first two components, coloured by cluster.
plot_cluster_correlation_matrix
    Subject-by-subject correlation matrix reordered by cluster.
plot_cluster_means_bars
    Grouped bar plot of cluster means per metric with SE error bars.
plot_cluster_means_lollipop
    Grouped lollipop plot of cluster means per metric.

Utilities
configure_plot_style
    Load matplotlib rcParams from a YAML config file.
save_fig
    Save a figure as PNG and PDF.
"""

from .clustering import (
    plot_cluster_correlation_matrix,
    plot_cluster_distributions,
    plot_cluster_heatmap_avg_std,
    plot_cluster_means_bars,
    plot_cluster_means_lollipop,
    plot_cluster_pca_scatter,
    plot_clustering_metrics,
)
from .correlation import (
    plot_clustermap,
    plot_clustermap_top,
    plot_combined_heatmap,
    plot_correlation_grid,
    plot_correlation_grid_stacked,
    plot_correlation_heatmap,
    plot_pairplot,
)
from .distributions import (
    plot_combined_distributions,
    plot_distribution,
    plot_gender_violin,
    plot_group_distributions,
)
from .pca import (
    plot_scatter,
    plot_top_loadings,
    plot_variance,
)
from .precision import (
    plot_covariance_heatmap,
    plot_precision_graph,
    plot_precision_heatmap,
)
from .utils import configure_plot_style, save_fig

__all__ = [
    "configure_plot_style",
    "plot_cluster_correlation_matrix",
    "plot_cluster_distributions",
    "plot_cluster_heatmap_avg_std",
    "plot_cluster_means_bars",
    "plot_cluster_means_lollipop",
    "plot_cluster_pca_scatter",
    "plot_clustering_metrics",
    "plot_clustermap",
    "plot_clustermap_top",
    "plot_combined_distributions",
    "plot_combined_heatmap",
    "plot_correlation_grid",
    "plot_correlation_grid_stacked",
    "plot_correlation_heatmap",
    "plot_covariance_heatmap",
    "plot_distribution",
    "plot_gender_violin",
    "plot_group_distributions",
    "plot_pairplot",
    "plot_precision_graph",
    "plot_precision_heatmap",
    "plot_scatter",
    "plot_top_loadings",
    "plot_variance",
    "save_fig",
]

if __name__ == "__main__":
    import doctest

    doctest.testmod()
