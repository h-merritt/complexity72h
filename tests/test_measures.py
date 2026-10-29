"""Tests for insideout.measures."""

import numpy as np

from insideout.measures import (
    align_signs,
    cronbach_alpha,
    leave_one_out_pca,
    loading_group_test,
    mantel,
    measure_fc_correlations,
    pc_group_overlap,
    similarity_contrast,
)


def test_alpha_independent_items_near_zero():
    X = np.random.default_rng(0).normal(size=(2000, 5))
    assert abs(cronbach_alpha(X)["alpha"]) < 0.1


def test_alpha_needs_reverse_scoring():
    rng = np.random.default_rng(0)
    t = rng.normal(size=1000)
    X = np.c_[
        t + rng.normal(size=1000), -t + rng.normal(size=1000), t + rng.normal(size=1000)
    ]
    aligned, signs = align_signs(X)
    assert cronbach_alpha(X)["alpha"] < cronbach_alpha(aligned)["alpha"]
    assert signs[0] == -signs[1]


def test_pc_overlap_perfect_and_null():
    g = np.array(["a"] * 3 + ["b"] * 3)
    L = np.array([[1.0, 1], [1, -1], [1, 1], [-1, -1], [-1, 1], [-1, -1]])
    out = pc_group_overlap(L, g, n_perm=100)
    assert out["ari"][0] == 1.0
    assert out["ari"][1] < 0.5


def test_leave_one_out_shape_and_delta():
    X = np.random.default_rng(0).normal(size=(100, 5))
    out = leave_one_out_pca(X, list("abcde"), n_pcs=2)
    assert out.shape == (10, 5)
    assert np.allclose(out["delta"], out["without"] - out["full"])


def test_measure_fc_correlations_matches_corrcoef():
    rng = np.random.default_rng(1)
    fc = rng.normal(size=(40, 6, 6))
    fc = fc + fc.transpose(0, 2, 1)
    X = rng.normal(size=(40, 2))
    C = measure_fc_correlations(X, fc)
    assert np.allclose(C, C.transpose(0, 2, 1))
    assert np.isclose(C[1, 2, 4], np.corrcoef(X[:, 1], fc[:, 2, 4])[0, 1])


def test_similarity_contrast_and_mantel():
    S = np.kron(np.eye(2), np.ones((3, 3)))
    res = similarity_contrast(S, np.array([0] * 3 + [1] * 3), n_perm=200)
    assert res["contrast"] == 1.0
    assert mantel(S, S, n_perm=50)["r"] > 0.99


def test_loading_group_test_separates_groups():
    L = np.array(
        [[1.0, 0.1], [0.9, -0.1], [0.8, 0.1], [-1.0, 0.1], [-0.9, -0.1], [-0.8, 0.1]]
    )
    g = np.array(["a"] * 3 + ["b"] * 3)
    out = loading_group_test(L, g)
    assert out["mean_a"][0] > 0 > out["mean_b"][0]
    assert out["p_mannwhitney"][0] < out["p_mannwhitney"][1]
