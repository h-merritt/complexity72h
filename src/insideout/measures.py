"""Analyses of the inner and outer social measures.

Functions
---------
load_measures
    Survey measures of the inner and outer self as one numeric matrix.
cronbach_alpha
    Internal consistency of a set of items.
pc_group_overlap
    Agreement between the sign pattern of each PC and the inner/outer grouping.
loading_group_test
    Mann-Whitney comparison of the loadings of two groups of measures on each PC.
leave_one_out_pca
    Change in explained variance of each PC when one measure is removed.
measure_fc_correlations
    Edge-wise correlation of every measure with functional connectivity.
similarity_contrast
    Within- versus between-group mean similarity with a permutation test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from insideout.data.loaders import load_survey_data
from insideout.decomposition import run_pca


@dataclass
class Measures:
    """Survey measures in one matrix.

    Attributes
    ----------
    subjects : np.ndarray
        Subject IDs, shape ``(n,)``.
    X : np.ndarray
        Measure scores, shape ``(n, p)``; inner columns first.
    columns : list of str
        Column names in the survey file.
    names : list of str
        Display names.
    group : np.ndarray
        ``"inner"`` or ``"outer"`` for each measure, shape ``(p,)``.
    """

    subjects: np.ndarray
    X: np.ndarray
    columns: list[str]
    names: list[str]
    group: np.ndarray

    def names_of(self, mask: np.ndarray) -> list[str]:
        """Display names of the measures selected by a boolean mask."""
        return [n for n, k in zip(self.names, mask) if k]


def load_measures(
    csv_path: str, yaml_config_path: str, drop_na: bool = True
) -> Measures:
    """Load the inner and outer measures as a single matrix.

    Parameters
    ----------
    csv_path : str
        Path to the survey CSV.
    yaml_config_path : str
        Path to the components YAML (``inner`` and ``outer`` lists).
    drop_na : bool
        Drop subjects with missing values (default ``True``).

    Returns
    -------
    Measures
        Subjects, scores, column names, display names and group labels.
    """
    import yaml

    with open(yaml_config_path) as f:
        cfg = yaml.safe_load(f)
    subject_col = cfg["metadata"]["subject_id_col"]
    data = load_survey_data(csv_path, yaml_config_path, drop_na=drop_na)
    joined = data["inner"].join(data["outer"], on=subject_col, how="inner")
    columns, names, group = [], [], []
    for g in ("inner", "outer"):
        for item in cfg[g]:
            columns.append(item["column"])
            names.append(item["name"])
            group.append(g)
    return Measures(
        subjects=joined[subject_col].to_numpy(),
        X=joined.select(columns).to_numpy().astype(float),
        columns=columns,
        names=names,
        group=np.asarray(group),
    )


def cronbach_alpha(X: np.ndarray) -> dict[str, float]:
    """Cronbach's alpha of a set of items.

    Parameters
    ----------
    X : np.ndarray
        Scores, shape ``(n_subjects, n_items)``.

    Returns
    -------
    dict
        ``alpha`` (raw, from covariances), ``alpha_std`` (standardised, from
        the mean inter-item correlation) and ``mean_r`` (mean inter-item
        correlation).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> t = rng.normal(size=500)
    >>> X = np.c_[t + rng.normal(size=500), t + rng.normal(size=500)]
    >>> round(cronbach_alpha(X)["alpha"], 1)
    0.7
    """
    k = X.shape[1]
    cov = np.cov(X, rowvar=False)
    alpha = k / (k - 1) * (1 - np.trace(cov) / cov.sum())
    corr = np.corrcoef(X, rowvar=False)
    mean_r = corr[np.triu_indices(k, 1)].mean()
    alpha_std = k * mean_r / (1 + (k - 1) * mean_r)
    return {
        "alpha": float(alpha),
        "alpha_std": float(alpha_std),
        "mean_r": float(mean_r),
    }


def align_signs(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip items so that they correlate positively with the first PC.

    Parameters
    ----------
    X : np.ndarray
        Scores, shape ``(n_subjects, n_items)``.

    Returns
    -------
    tuple
        The sign-aligned scores and the sign (+1 or -1) applied to each item.
    """
    pc1 = run_pca(pl.DataFrame(X, schema=[str(i) for i in range(X.shape[1])]))
    w = pc1.loadings["PC1"].to_numpy()
    signs = np.where(w >= 0, 1.0, -1.0)
    return X * signs, signs


def pc_group_overlap(
    loadings: np.ndarray,
    group: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> pl.DataFrame:
    """Overlap of each PC with the inner/outer grouping of the measures.

    The sign of the loadings splits the measures into two pseudo-clusters,
    which are compared with the inner/outer labels by ARI and AMI. Both scores
    ignore which cluster is which. The p-value comes from permuting the
    inner/outer labels over the measures.

    Parameters
    ----------
    loadings : np.ndarray
        Shape ``(n_measures, n_pcs)``.
    group : np.ndarray
        Inner/outer label of each measure, shape ``(n_measures,)``.
    n_perm : int
        Number of label permutations.
    seed : int
        Seed of the permutations.

    Returns
    -------
    pl.DataFrame
        One row per PC: ``pc``, ``n_positive``, ``ari``, ``ami``, ``p_ari``
        (one-sided, ARI at least as large as observed).

    Examples
    --------
    >>> L = np.array([[1.0], [1.0], [-1.0], [-1.0]])
    >>> g = np.array(["a", "a", "b", "b"])
    >>> pc_group_overlap(L, g, n_perm=50)["ari"].to_list()
    [1.0]
    """
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(group) for _ in range(n_perm)]
    rows = []
    for j in range(loadings.shape[1]):
        pseudo = (loadings[:, j] > 0).astype(int)
        ari = adjusted_rand_score(group, pseudo)
        ami = adjusted_mutual_info_score(group, pseudo)
        null = np.array([adjusted_rand_score(p, pseudo) for p in perms])
        rows.append(
            {
                "pc": f"PC{j + 1}",
                "n_positive": int(pseudo.sum()),
                "ari": float(ari),
                "ami": float(ami),
                "p_ari": float((1 + (null >= ari).sum()) / (1 + n_perm)),
            }
        )
    return pl.DataFrame(rows)


def loading_group_test(loadings: np.ndarray, group: np.ndarray) -> pl.DataFrame:
    """Compare the loadings of two groups of measures on each PC.

    Uses the loading values, not only their signs.

    Parameters
    ----------
    loadings : np.ndarray
        Shape ``(n_measures, n_pcs)``.
    group : np.ndarray
        Label of each measure with exactly two distinct values, shape
        ``(n_measures,)``.

    Returns
    -------
    pl.DataFrame
        One row per PC: ``pc``, ``mean_<first group>``, ``mean_<second group>``
        (labels sorted alphabetically) and ``p_mannwhitney`` (two-sided).

    Examples
    --------
    >>> L = np.array([[1.0], [0.9], [0.8], [-1.0], [-0.9], [-0.8]])
    >>> g = np.array(["a"] * 3 + ["b"] * 3)
    >>> loading_group_test(L, g).columns
    ['pc', 'mean_a', 'mean_b', 'p_mannwhitney']
    """
    from scipy.stats import mannwhitneyu

    first, second = sorted(set(group))
    rows = []
    for j in range(loadings.shape[1]):
        a = loadings[group == first, j]
        b = loadings[group == second, j]
        rows.append(
            {
                "pc": f"PC{j + 1}",
                f"mean_{first}": float(a.mean()),
                f"mean_{second}": float(b.mean()),
                "p_mannwhitney": float(mannwhitneyu(a, b).pvalue),
            }
        )
    return pl.DataFrame(rows)


def leave_one_out_pca(
    X: np.ndarray, columns: list[str], n_pcs: int = 3
) -> pl.DataFrame:
    """Explained variance of the first PCs with each measure left out in turn.

    Parameters
    ----------
    X : np.ndarray
        Scores, shape ``(n_subjects, n_measures)``.
    columns : list of str
        Measure names.
    n_pcs : int
        Number of PCs reported.

    Returns
    -------
    pl.DataFrame
        One row per (left-out measure, PC): ``left_out``, ``pc``, ``full``
        (variance ratio with all measures), ``without`` (variance ratio
        without the measure) and ``delta`` (``without - full``). The ratios
        are relative to the total variance of the respective measure set.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> out = leave_one_out_pca(rng.normal(size=(50, 4)), list("abcd"), n_pcs=2)
    >>> out.shape
    (8, 5)
    """
    full = run_pca(pl.DataFrame(X, schema=columns)).explained_variance_ratio
    rows = []
    for i, name in enumerate(columns):
        keep = [c for c in range(len(columns)) if c != i]
        sub = run_pca(
            pl.DataFrame(X[:, keep], schema=[columns[c] for c in keep])
        ).explained_variance_ratio
        for j in range(n_pcs):
            rows.append(
                {
                    "left_out": name,
                    "pc": f"PC{j + 1}",
                    "full": full[j],
                    "without": sub[j],
                    "delta": sub[j] - full[j],
                }
            )
    return pl.DataFrame(rows)


def measure_fc_correlations(X: np.ndarray, fc: np.ndarray) -> np.ndarray:
    """Pearson correlation of every measure with every FC edge.

    Parameters
    ----------
    X : np.ndarray
        Measure scores, shape ``(n_subjects, n_measures)``.
    fc : np.ndarray
        FC matrices, shape ``(n_subjects, n_rois, n_rois)``.

    Returns
    -------
    np.ndarray
        Symmetric matrices of shape ``(n_measures, n_rois, n_rois)`` with a
        zero diagonal.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> fc = rng.normal(size=(30, 5, 5)); fc = fc + fc.transpose(0, 2, 1)
    >>> measure_fc_correlations(rng.normal(size=(30, 3)), fc).shape
    (3, 5, 5)
    """
    n, r, _ = fc.shape
    iu = np.triu_indices(r, 1)
    E = fc[:, iu[0], iu[1]]
    Zx = (X - X.mean(0)) / X.std(0, ddof=1)
    Ze = (E - E.mean(0)) / E.std(0, ddof=1)
    edges = Zx.T @ Ze / (n - 1)
    out = np.zeros((X.shape[1], r, r))
    out[:, iu[0], iu[1]] = edges
    out[:, iu[1], iu[0]] = edges
    return out


def similarity_contrast(
    S: np.ndarray, labels: np.ndarray, n_perm: int = 10000, seed: int = 42
) -> dict[str, float]:
    """Mean similarity within versus between the groups of a label vector.

    Parameters
    ----------
    S : np.ndarray
        Symmetric similarity matrix, shape ``(p, p)``.
    labels : np.ndarray
        Group label of each row, shape ``(p,)``.
    n_perm : int
        Number of label permutations.
    seed : int
        Seed of the permutations.

    Returns
    -------
    dict
        ``within``, ``between``, ``contrast`` (within - between) and ``p``
        (one-sided permutation p-value of the contrast).

    Examples
    --------
    >>> S = np.array([[1, .9, 0, 0], [.9, 1, 0, 0], [0, 0, 1, .9], [0, 0, .9, 1]])
    >>> similarity_contrast(S, np.array([0, 0, 1, 1]), n_perm=20)["contrast"]
    0.9
    """
    iu = np.triu_indices(S.shape[0], 1)

    def contrast(lab: np.ndarray) -> tuple[float, float]:
        same = (lab[:, None] == lab[None, :])[iu]
        v = S[iu]
        return v[same].mean(), v[~same].mean()

    w, b = contrast(labels)
    rng = np.random.default_rng(seed)
    null = np.array(
        [np.subtract(*contrast(rng.permutation(labels))) for _ in range(n_perm)]
    )
    return {
        "within": float(w),
        "between": float(b),
        "contrast": float(w - b),
        "p": float((1 + (null >= w - b).sum()) / (1 + n_perm)),
    }


def mantel(
    A: np.ndarray, B: np.ndarray, n_perm: int = 10000, seed: int = 42
) -> dict[str, float]:
    """Correlation of the upper triangles of two matrices with a Mantel test.

    Parameters
    ----------
    A, B : np.ndarray
        Symmetric matrices of the same shape ``(p, p)``.
    n_perm : int
        Number of simultaneous row/column permutations of ``B``.
    seed : int
        Seed of the permutations.

    Returns
    -------
    dict
        ``r`` (Pearson), ``rho`` (Spearman) and ``p`` (two-sided, on ``r``).

    Examples
    --------
    >>> A = np.array([[1, .5, .1], [.5, 1, .2], [.1, .2, 1]])
    >>> round(mantel(A, A, n_perm=10)["r"], 6)
    1.0
    """
    from scipy.stats import spearmanr

    iu = np.triu_indices(A.shape[0], 1)
    a = A[iu]
    r = np.corrcoef(a, B[iu])[0, 1]
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        q = rng.permutation(A.shape[0])
        null[i] = np.corrcoef(a, B[np.ix_(q, q)][iu])[0, 1]
    return {
        "r": float(r),
        "rho": float(spearmanr(a, B[iu]).statistic),
        "p": float((1 + (np.abs(null) >= abs(r)).sum()) / (1 + n_perm)),
    }


if __name__ == "__main__":
    import doctest

    doctest.testmod()
