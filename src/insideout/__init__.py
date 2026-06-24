"""Top-level package for the insideout project.

Provides data-loading (``insideout.data``), graph / covariance modelling
(``insideout.graph_models``), PCA decomposition
(``insideout.decomposition``), and visualisation helpers (``insideout.viz``)
for exploring tabular data with Polars DataFrames.

Exports
-------
data : module
    CSV / YAML survey data loading.
graph_models : module
    Covariance, correlation, precision, and Graphical Lasso computation.
viz : module
    Plotting functions for distributions, correlations, precision, and PCA.
decomposition : module
    PCA fitting and result dataclass.
"""

from . import data, decomposition, graph_models, viz

__all__ = ["data", "decomposition", "graph_models", "viz"]
