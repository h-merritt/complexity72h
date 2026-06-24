# src/features/schaefer_yeo.py
"""
Map Schaefer-2018 parcels to Yeo networks and reduce a full FC matrix to
system-block averages (within- and between-network mean connectivity).

Assumes the FC matrices are stored in Schaefer atlas order (parcel 1..n_rois along both axes).
"""
from typing import List, Sequence, Tuple, Optional

import numpy as np

try:
    from nilearn import datasets
except ImportError:
    datasets = None


def _parse_network_name(raw, detailed: bool = False) -> str:
    """
    Extract the Yeo network name from a Schaefer label.

    Handles bytes or str, and both 7-net and 17-net formats.
    If detailed=False, '7Networks_LH_Default_PFCv' -> 'Default'.
    If detailed=True, '7Networks_LH_Default_PFCv' -> 'Default_PFCv'.
    """
    s = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    parts = s.split("_")
    if parts and parts[-1].isdigit():  # drop trailing within-network index
        parts = parts[:-1]
    if parts and parts[0].endswith("Networks"):  # drop '7Networks' / '17Networks'
        parts = parts[1:]
    if parts and parts[0].upper() in ("LH", "RH", "L", "R"):  # drop hemisphere
        parts = parts[1:]
    if not parts:
        return s
    return "_".join(parts) if detailed else parts[0]

def get_schaefer_networks(
    n_rois: int = 100,
    networks: int = 7,
    data_dir: Optional[str] = None,
    detailed: bool = False
) -> List[str]:
    """Per-parcel Yeo network names, length == n_rois, in atlas order."""
    if datasets is None:
        raise ImportError("nilearn is required for get_schaefer_networks().")
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=networks, data_dir=data_dir
    )
    labels = list(atlas.labels)
    if len(labels) == n_rois + 1:   # some nilearn versions prepend 'Background'
        labels = labels[1:]
    if len(labels) != n_rois:
        raise ValueError(f"Expected {n_rois} parcel labels, got {len(labels)}.")
    return [_parse_network_name(l, detailed=detailed) for l in labels]

def upper_triangle_tensor(fmri_3d: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Vectorize the off-diagonal upper triangle of each (R x R) matrix.

    fmri_3d : (N, R, R) -> returns ((N, n_edges), (row_idx, col_idx)).
    """
    r = fmri_3d.shape[1]
    iu = np.triu_indices(r, k=1)
    return fmri_3d[:, iu[0], iu[1]], iu


def system_blocks_from_tensor(
    fmri_3d: np.ndarray, net_labels: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Collapse an FC tensor to within/between system-block averages.

    fmri_3d   : (N, R, R)
    net_labels: length-R network name per parcel (atlas order).
    Returns (block_tensor (N, n_blocks), block_names). For 7 networks this is
    up to 28 features (7 within + 21 between); empty blocks are dropped.

    The diagonal is excluded (k=1 upper triangle), so within-network blocks are
    means over distinct parcel pairs only -- no self-connections inflating them.
    Block features are deterministic and order-stable (networks sorted
    alphabetically), so saliences line up across runs and across subjects.
    """
    n, r, _ = fmri_3d.shape
    if len(net_labels) != r:
        raise ValueError(f"net_labels length {len(net_labels)} != n_regions {r}.")

    edge_vals, iu = upper_triangle_tensor(fmri_3d)      # (N, n_edges)
    nets = np.asarray(net_labels)

    unique = sorted(set(net_labels))
    code = {name: k for k, name in enumerate(unique)}
    ci = np.array([code[x] for x in nets[iu[0]]])
    cj = np.array([code[x] for x in nets[iu[1]]])
    lo = np.minimum(ci, cj)
    hi = np.maximum(ci, cj)

    observed = sorted(set(zip(lo.tolist(), hi.tolist())))   # only non-empty blocks
    pid_to_idx = {pair: idx for idx, pair in enumerate(observed)}
    edge_block = np.array([pid_to_idx[(a, b)] for a, b in zip(lo, hi)])
    n_blocks = len(observed)

    # One-hot column-averaging matrix: (n_edges x n_blocks), columns sum to 1
    B = np.zeros((edge_vals.shape[1], n_blocks))
    B[np.arange(edge_vals.shape[1]), edge_block] = 1.0
    B /= B.sum(0, keepdims=True)

    block_tensor = edge_vals @ B        # (N, n_blocks)
    block_names = [f"{unique[a]}-{unique[b]}" for a, b in observed]
    return block_tensor, block_names


def system_block_features(fc: np.ndarray, net_labels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """Single-matrix convenience wrapper around system_blocks_from_tensor."""
    vals, names = system_blocks_from_tensor(fc[None, :, :], net_labels)
    return vals[0], names
