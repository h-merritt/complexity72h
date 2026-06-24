"""Visualisation subpackage for the insideout project.

Provides functions for exploring distributions, correlations, and
precision / covariance matrices in Polars DataFrames.

Exports
-------
plot_distribution
    Histogram + boxplot for a single column.
plot_gender_violin
    Violin plot comparing a metric across two gender groups.
plot_group_distributions
    Horizontal violin + box + strip plots for a group of variables.
plot_correlation_heatmap
    Pearson correlation heatmap for a set of columns.
plot_clustermap
    Clustered correlation heatmap with hierarchical dendrogram.
plot_clustermap_top
    Clustermap restricted to the top-N variables.
plot_combined_heatmap
    Heatmap combining two variable groups with colour-coded labels.
plot_pairplot
    Seaborn pairplot for selected columns.
plot_covariance_heatmap
    Clustered covariance heatmap.
plot_precision_heatmap
    Heatmap of a precision (inverse covariance) matrix.
plot_precision_graph
    Force-directed graph of a precision matrix.
configure_plot_style
    Load matplotlib rcParams from a YAML config file.
save_fig
    Save a figure as PNG and PDF.
"""

from .distributions import (
    plot_distribution,
    plot_gender_violin,
    plot_group_distributions,
)
from .correlation import (
    plot_correlation_heatmap,
    plot_clustermap,
    plot_clustermap_top,
    plot_combined_heatmap,
    plot_pairplot,
)
from .precision import (
    plot_covariance_heatmap,
    plot_precision_heatmap,
    plot_precision_graph,
)
from .pca import (
    plot_scatter,
    plot_top_loadings,
    plot_variance,
)
from .utils import configure_plot_style, save_fig

__all__ = [
    "plot_distribution",
    "plot_gender_violin",
    "plot_group_distributions",
    "plot_correlation_heatmap",
    "plot_clustermap",
    "plot_clustermap_top",
    "plot_combined_heatmap",
    "plot_pairplot",
    "plot_covariance_heatmap",
    "plot_precision_heatmap",
    "plot_precision_graph",
    "plot_variance",
    "plot_scatter",
    "plot_top_loadings",
    "configure_plot_style",
    "save_fig",
]

if __name__ == "__main__":
    import doctest

    doctest.testmod()
