# src/features/schaefer_yeo.py
"""
Map Schaefer-2018 parcels to Yeo networks and reduce a full FC matrix to
system-block averages (within- and between-network mean connectivity).

Assumes the FC matrices are stored in Schaefer atlas order (parcel 1..n_rois
along both axes).
"""

from typing import List, Sequence, Tuple, Optional

import numpy as np

try:
    from nilearn import datasets
except ImportError:
    datasets = None


def _parse_network_name(raw, detailed: bool = False) -> str:
    """Extract the Yeo network name from a Schaefer parcel label.

    Handles bytes or str input and both 7-network and 17-network label
    formats used by the Schaefer-2018 atlas.

    Parameters
    ----------
    raw : bytes or str
        Raw parcel label from the atlas, e.g.
        ``"7Networks_LH_Default_PFCv_1"``.
    detailed : bool, optional
        If ``False`` (default), return only the top-level network name
        (e.g. ``"Default"``).  If ``True``, return the full sub-network
        name (e.g. ``"Default_PFCv"``).

    Returns
    -------
    str
        Parsed network name.

    Examples
    --------
    >>> _parse_network_name("7Networks_LH_Default_PFCv_1")
    'Default'
    >>> _parse_network_name("7Networks_LH_Default_PFCv_1", detailed=True)
    'Default_PFCv'
    """
    s = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    parts = s.split("_")
    # Drop trailing within-network index (e.g. "_1")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    # Drop the network-count prefix (e.g. "7Networks" / "17Networks")
    if parts and parts[0].endswith("Networks"):
        parts = parts[1:]
    # Drop hemisphere label (e.g. "LH", "RH")
    if parts and parts[0].upper() in ("LH", "RH", "L", "R"):
        parts = parts[1:]
    if not parts:
        return s
    return "_".join(parts) if detailed else parts[0]


def get_schaefer_networks(
    n_rois: int = 100,
    networks: int = 7,
    data_dir: Optional[str] = None,
    detailed: bool = False,
) -> List[str]:
    """Return per-parcel Yeo network names in Schaefer atlas order.

    Parameters
    ----------
    n_rois : int, optional
        Number of ROIs in the atlas (default ``100``).
    networks : int, optional
        Yeo network granularity — ``7`` or ``17`` (default ``7``).
    data_dir : str or None, optional
        Directory where the atlas has been cached; ``None`` uses nilearn's
        default cache (default ``None``).
    detailed : bool, optional
        Passed through to :func:`_parse_network_name` (default ``False``).

    Returns
    -------
    list of str
        One network label per parcel, length ``n_rois``.

    Raises
    ------
    ImportError
        If ``nilearn`` is not installed.

    Examples
    --------
    >>> nets = get_schaefer_networks(100, 7, detailed=False)  # doctest: +SKIP
    """
    if datasets is None:
        raise ImportError("nilearn is required for get_schaefer_networks().")
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=networks, data_dir=data_dir
    )
    labels = list(atlas.labels)
    # Some nilearn versions prepend 'Background' as the first label
    if len(labels) == n_rois + 1:
        labels = labels[1:]
    if len(labels) != n_rois:
        raise ValueError(f"Expected {n_rois} parcel labels, got {len(labels)}.")
    return [_parse_network_name(lb, detailed=detailed) for lb in labels]


def upper_triangle_tensor(
    fmri_3d: np.ndarray,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Vectorise the off-diagonal upper triangle of each (R x R) matrix.

    Parameters
    ----------
    fmri_3d : np.ndarray of shape (N, R, R)
        Stack of N symmetric connectivity matrices, each of size ``R x R``.

    Returns
    -------
    edge_vals : np.ndarray of shape (N, R*(R-1)//2)
        Vectorised upper-triangle entries for each matrix.
    (row_idx, col_idx) : tuple of np.ndarray
        Row and column indices of the upper-triangle entries, useful for
        mapping back to the full matrix.

    Examples
    --------
    >>> X = np.arange(4 * 3 * 3).reshape(4, 3, 3)
    >>> vals, (ri, ci) = upper_triangle_tensor(X)
    >>> vals.shape
    (4, 3)
    """
    r = fmri_3d.shape[1]
    iu = np.triu_indices(r, k=1)  # (row_idxs, col_idxs) for upper triangle
    return fmri_3d[:, iu[0], iu[1]], iu


def system_blocks_from_tensor(
    fmri_3d: np.ndarray, net_labels: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """Collapse an FC tensor to within- and between-system block averages.

    For each subject, the upper triangle of the FC matrix is partitioned
    according to the Yeo network labels and averaged per block.  The
    diagonal is excluded so within-network blocks contain only cross-parcel
    edges (never self-connections).

    Parameters
    ----------
    fmri_3d : np.ndarray of shape (N, R, R)
        Stack of N symmetric connectivity matrices.
    net_labels : sequence of str of length R
        Yeo network label for each parcel (atlas order).

    Returns
    -------
    block_tensor : np.ndarray of shape (N, n_blocks)
        Mean connectivity per block for each subject.  Rows correspond to
        the ``block_names`` order (alphabetically sorted).
    block_names : list of str
        Names of each block in ``"NetA-NetB"`` format.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> fc = rng.normal(size=(10, 6, 6))
    >>> lbls = ["Vis", "Vis", "Som", "Som", "Con", "Con"]
    >>> blocks, names = system_blocks_from_tensor(fc, lbls)
    >>> blocks.shape
    (10, 3)
    """
    n, r, _ = fmri_3d.shape
    if len(net_labels) != r:
        raise ValueError(f"net_labels length {len(net_labels)} != n_regions {r}.")

    # Vectorise all N matrices into (N, n_edges) edge-value table
    edge_vals, iu = upper_triangle_tensor(fmri_3d)  # (N, n_edges)
    nets = np.asarray(net_labels)

    # Assign each edge to a block based on the networks of its two endpoints.
    # Use canonical ordering so (NetA, NetB) and (NetB, NetA) map to the same block.
    unique = sorted(set(net_labels))
    code = {name: k for k, name in enumerate(unique)}
    ci = np.array([code[x] for x in nets[iu[0]]])  # network code of row endpoint
    cj = np.array([code[x] for x in nets[iu[1]]])  # network code of col endpoint
    lo = np.minimum(ci, cj)
    hi = np.maximum(ci, cj)

    # Build a sparse one-hot averaging matrix B: (n_edges × n_blocks)
    observed = sorted(set(zip(lo.tolist(), hi.tolist())))  # only non-empty blocks
    pid_to_idx = {pair: idx for idx, pair in enumerate(observed)}
    edge_block = np.array([pid_to_idx[(a, b)] for a, b in zip(lo, hi)])
    n_blocks = len(observed)

    # Each column of B averages the edges belonging to that block
    B = np.zeros((edge_vals.shape[1], n_blocks))
    B[np.arange(edge_vals.shape[1]), edge_block] = 1.0
    B /= B.sum(0, keepdims=True)

    block_tensor = edge_vals @ B  # Matrix multiply: (N, n_edges) × (n_edges, n_blocks)
    block_names = [f"{unique[a]}-{unique[b]}" for a, b in observed]
    return block_tensor, block_names


def system_block_features(
    fc: np.ndarray, net_labels: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """Single-matrix convenience wrapper around :func:`system_blocks_from_tensor`.

    Parameters
    ----------
    fc : np.ndarray of shape (R, R)
        A single symmetric connectivity matrix.
    net_labels : sequence of str of length R
        Yeo network label for each parcel.

    Returns
    -------
    vals : np.ndarray of shape (n_blocks,)
        Mean connectivity per block.
    names : list of str
        Block names (same order as *vals*).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> fc = rng.normal(size=(6, 6))
    >>> fc += fc.T  # make symmetric
    >>> lbls = ["Vis", "Vis", "Som", "Som", "Con", "Con"]
    >>> vals, names = system_block_features(fc, lbls)
    >>> len(vals)
    3
    """
    vals, names = system_blocks_from_tensor(fc[None, :, :], net_labels)
    return vals[0], names
