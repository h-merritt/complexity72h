import numpy as np
import polars as pl
import pytest

from insideout.decomposition import run_pca


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 1))
    X = np.hstack([base + 0.1 * rng.normal(size=(200, 1)) for _ in range(3)])
    X = np.hstack([X, rng.normal(size=(200, 2))])
    return pl.DataFrame(X, schema=["a", "b", "c", "d", "e"])


def test_explained_variance_sums_to_one(df):
    res = run_pca(df, is_z_score=False)
    assert sum(res.explained_variance_ratio) == pytest.approx(1.0)
    assert res.explained_variance_ratio == sorted(
        res.explained_variance_ratio, reverse=True
    )


def test_loadings_are_unit_eigenvectors_and_transpose_of_components(df):
    res = run_pca(df, is_z_score=False)
    L = res.loadings.drop("variable").to_numpy()
    assert L.shape == (5, 5)
    assert res.loadings["variable"].to_list() == df.columns
    np.testing.assert_allclose(
        (L**2).sum(0), 1.0
    )  # unit norm, not scaled by sqrt(eigenvalue)
    np.testing.assert_allclose(L, res.components.to_numpy().T)


def test_scores_shape_and_orthogonality(df):
    res = run_pca(df, is_z_score=False)
    S = res.scores.to_numpy()
    assert S.shape == (200, 5)
    C = np.cov(S.T)
    np.testing.assert_allclose(C - np.diag(np.diag(C)), 0, atol=1e-10)


def test_missing_values_are_mean_imputed(df):
    X = df.to_numpy().copy()
    X[0, 0] = np.nan
    res = run_pca(pl.DataFrame(X, schema=df.columns), is_z_score=False)
    assert np.isfinite(res.scores.to_numpy()).all()
