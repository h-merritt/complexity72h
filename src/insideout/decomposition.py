"""Principal component analysis (PCA) utilities.

Functions
---------
run_pca
    Fit PCA on a Polars DataFrame and return component loadings, scores,
    explained variance, and component weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


@dataclass
class PCAResult:
    """Output of a PCA fit.

    Attributes
    ----------
    loadings : pl.DataFrame
        Matrix of shape ``(n_features, n_components)`` with a ``"variable"``
        column naming the original features.  The entries are the **unit-norm
        eigenvectors** of the covariance matrix (``components.T``), i.e.
        weights of the variables in each PC, not correlations.  Their
        magnitudes are not comparable across PCs; to obtain the correlation
        between a standardised variable and a PC multiply a column by
        ``sqrt(eigenvalue)`` (``sqrt(n_features * explained_variance_ratio)``
        for standardised data).
    scores : pl.DataFrame
        Projection of the input data onto the principal components; shape
        ``(n_samples, n_components)``.
    explained_variance_ratio : list of float
        Fraction of total variance explained by each principal component.
    components : pl.DataFrame
        Principal component directions (the right singular vectors); shape
        ``(n_components, n_features)``.  Equal to ``loadings`` transposed.
    """

    loadings: pl.DataFrame
    scores: pl.DataFrame
    explained_variance_ratio: list[float]
    components: pl.DataFrame


def run_pca(df: pl.DataFrame, is_z_score: bool = True) -> PCAResult:
    """Fit PCA on a Polars DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input data.  Rows are samples, columns are features.
    is_z_score : bool
        If ``True`` (default), the data is assumed to already be z-scored so
        no additional standardisation is applied.  If ``False``, a
        :class:`~sklearn.preprocessing.StandardScaler` is fit first.

    Returns
    -------
    PCAResult
        Dataclass containing loadings, scores, explained variance ratios, and
        component directions.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    >>> res = run_pca(df)
    >>> [round(v, 6) for v in res.explained_variance_ratio]
    [1.0, 0.0]
    """
    X = df.to_numpy()
    X = SimpleImputer(strategy="mean").fit_transform(
        X
    )  # fill missing values with column mean
    if not is_z_score:
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(X)  # standardise to zero-mean unit-variance

    pca = PCA(n_components=None).fit(X)
    # sklearn returns components_ in Fortran order; polars builds a DataFrame
    # from a 2-D array according to its memory layout, so force C order to keep
    # the logical (n_components, n_features) orientation.
    components = np.ascontiguousarray(pca.components_)
    pc_cols = [f"PC{i + 1}" for i in range(pca.n_components_)]

    return PCAResult(
        loadings=(
            pl.DataFrame(
                np.ascontiguousarray(components.T), schema=pc_cols
            ).with_columns(  # components are rows; transpose to get variable × PC
                pl.Series("variable", df.columns)
            )
        ),
        scores=pl.DataFrame(dict(zip(pc_cols, pca.transform(X).T))),
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
        components=pl.DataFrame(components, schema=df.columns),
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
