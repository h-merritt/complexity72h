import numpy as np
import pytest

from insideout.analysis.pls.associator import (
    _build_group_masks,
    compare_brain_saliences,
    cross_cov,
    run_behavioral_pls,
    zscore,
)


def test_zscore_columns():
    M = np.random.default_rng(0).normal(5, 3, size=(50, 4))
    Z = zscore(M)
    np.testing.assert_allclose(Z.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(Z.std(0, ddof=1), 1)


def test_zscore_constant_column_is_zero():
    M = np.ones((10, 2))
    np.testing.assert_array_equal(zscore(M), np.zeros((10, 2)))


def test_cross_cov_of_zscored_blocks_is_correlation():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(80, 3)), rng.normal(size=(80, 2))
    R = cross_cov(zscore(X), zscore(Y))
    expected = np.corrcoef(X.T, Y.T)[:3, 3:]
    np.testing.assert_allclose(R, expected)


class TestBuildGroupMasks:
    def test_none_is_single_group(self):
        masks, _ = _build_group_masks(None, 10)
        assert len(masks) == 1 and masks[0].all()

    def test_per_subject_labels(self):
        masks, labels = _build_group_masks(np.array(["b"] * 4 + ["a"] * 6), 10)
        assert list(labels) == ["a", "b"]  # sorted
        assert masks[0].sum() == 6 and masks[1].sum() == 4

    def test_group_sizes(self):
        masks, _ = _build_group_masks([4, 6], 10)
        assert masks[0][:4].all() and not masks[0][4:].any()
        assert masks[1][4:].all()

    def test_bad_length_raises(self):
        with pytest.raises(ValueError):
            _build_group_masks(np.array([0, 1, 0]), 10)

    def test_group_with_one_subject_raises(self):
        with pytest.raises(ValueError):
            _build_group_masks(np.array([0] * 9 + [1]), 10)


@pytest.fixture(scope="module")
def planted():
    """n=95 subjects, 60 brain features, 5 behaviours sharing one latent factor."""
    rng = np.random.default_rng(0)
    n = 95
    z = rng.normal(size=(n, 1))
    X = rng.normal(size=(n, 60)) + 0.6 * z @ rng.normal(size=(1, 60))
    Y = rng.normal(size=(n, 5)) + 0.8 * z @ np.array([[1, 1, 1, -1, -1]])
    return X, Y


def test_pls_detects_planted_latent(planted):
    X, Y = planted
    res = run_behavioral_pls(X, Y, nperm=200, nboot=100, nsplit=10, seed=0)
    assert res["L"] == 5 and res["U"].shape == (60, 5) and res["V"].shape == (5, 5)
    assert res["varexp"].sum() == pytest.approx(1.0)
    assert res["pvals"][0] < 0.05
    assert (res["pvals"][1:] > 0.05).all()
    assert res["behav_loading_ci"].shape == (5, 5, 2)
    # items sharing the latent load with the planted signs on LV1
    load = res["behav_loadings"][:, 0]
    assert np.sign(load[0]) == np.sign(load[1]) == np.sign(load[2]) == -np.sign(load[3])


def test_pls_null_data_is_not_significant():
    rng = np.random.default_rng(5)
    res = run_behavioral_pls(
        rng.normal(size=(95, 40)),
        rng.normal(size=(95, 5)),
        nperm=200,
        nboot=50,
        nsplit=0,
    )
    assert res["pvals"].min() > 0.01


def test_pls_is_reproducible(planted):
    X, Y = planted
    kw = {"nperm": 50, "nboot": 20, "nsplit": 5, "seed": 3}
    a = run_behavioral_pls(X, Y, **kw)
    b = run_behavioral_pls(X, Y, **kw)
    np.testing.assert_array_equal(a["pvals"], b["pvals"])
    np.testing.assert_array_equal(a["U"], b["U"])


def test_pls_without_resampling_has_nan_pvalues(planted):
    X, Y = planted
    res = run_behavioral_pls(X, Y, run_permutations=False)
    assert np.isnan(res["pvals"]).all()


def test_crossval_shapes(planted):
    X, Y = planted
    res = run_behavioral_pls(
        X, Y, run_permutations=False, run_crossval=True, n_test_split=7
    )
    assert res["cv_pearson_r"].shape == (5, 7)
    assert res["cv_r_squared"].shape == (5, 7)
    # the planted signal is predicted better than chance, while R^2 is uncalibrated
    assert np.nanmean(res["cv_pearson_r"]) > 0


def test_compare_brain_saliences_identical_maps(planted):
    X, Y = planted
    res = run_behavioral_pls(X, Y, run_permutations=False)
    out = compare_brain_saliences(res["U"], res["U"], nperm=200)
    assert out["abs_corr"] == pytest.approx(1.0)
    assert out["p_perm"] < 0.05
