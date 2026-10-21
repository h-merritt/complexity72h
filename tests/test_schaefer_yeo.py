import numpy as np
import pytest

from insideout.features.schaefer_yeo import (
    system_block_features,
    system_blocks_from_tensor,
    upper_triangle_tensor,
)


def test_upper_triangle_tensor_excludes_diagonal():
    X = np.arange(4 * 5 * 5).reshape(4, 5, 5)
    vals, (ri, ci) = upper_triangle_tensor(X)
    assert vals.shape == (4, 10)  # 5 * 4 / 2 edges
    assert (ri < ci).all()
    np.testing.assert_array_equal(vals[0], X[0][ri, ci])


def test_block_count_is_n_choose_2_plus_n():
    labels = ["Vis", "Vis", "Som", "Som", "Con", "Con"]
    blocks, names = system_blocks_from_tensor(
        np.random.default_rng(0).normal(size=(3, 6, 6)), labels
    )
    assert blocks.shape == (3, 6)  # 3 networks -> 3 within + 3 between
    assert names == ["Con-Con", "Con-Som", "Con-Vis", "Som-Som", "Som-Vis", "Vis-Vis"]


def test_seven_networks_give_28_blocks():
    nets = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
    labels = [n for n in nets for _ in range(3)]  # 21 parcels
    fc = np.random.default_rng(1).normal(size=(2, 21, 21))
    blocks, names = system_blocks_from_tensor(fc, labels)
    assert blocks.shape == (2, 28)
    assert len(set(names)) == 28


def test_block_means_are_exact():
    labels = ["A", "A", "B", "B"]
    fc = np.zeros((1, 4, 4))
    # A-A edge (0,1) = 2 ; B-B edge (2,3) = 4 ; A-B edges (0,2),(0,3),(1,2),(1,3) = 1,2,3,6
    for (i, j), v in {
        (0, 1): 2,
        (2, 3): 4,
        (0, 2): 1,
        (0, 3): 2,
        (1, 2): 3,
        (1, 3): 6,
    }.items():
        fc[0, i, j] = fc[0, j, i] = v
    blocks, names = system_blocks_from_tensor(fc, labels)
    got = dict(zip(names, blocks[0]))
    assert got == pytest.approx({"A-A": 2.0, "A-B": 3.0, "B-B": 4.0})


def test_label_length_mismatch_raises():
    with pytest.raises(ValueError):
        system_blocks_from_tensor(np.zeros((1, 4, 4)), ["A", "B", "C"])


def test_single_matrix_wrapper_matches_tensor_version():
    fc = np.random.default_rng(2).normal(size=(6, 6))
    fc = fc + fc.T
    labels = ["Vis", "Vis", "Som", "Som", "Con", "Con"]
    vals, names = system_block_features(fc, labels)
    blocks, names_t = system_blocks_from_tensor(fc[None], labels)
    assert names == names_t
    np.testing.assert_allclose(vals, blocks[0])
