"""I/O utilities for reading tabular and MATLAB data.

Functions
---------
read_csv
    Read a CSV file with sensible defaults.
read_fc_mat
    Load a variable from a ``.mat`` file.
"""

from pathlib import Path

import numpy as np
import polars as pl
import scipy.io


def read_csv(path: str | Path, **kwargs) -> pl.DataFrame:
    """Read a CSV file with sensible defaults.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    **kwargs
        Additional keyword arguments forwarded to ``pl.read_csv``.

    Returns
    -------
    pl.DataFrame
        Parsed data. Null values ``["NA", ""]`` are treated as missing
        and any column named ``""`` is dropped automatically.

    Examples
    --------
    >>> from insideout.io import read_csv
    >>> df = read_csv("data.csv")
    >>> df.shape
    (100, 5)
    """
    defaults = {"null_values": ["NA", ""]}
    defaults.update(kwargs)
    df = pl.read_csv(path, **defaults)
    if "" in df.columns:
        df = df.drop("")
    return df


def read_fc_mat(path: str | Path, key: str = "avg") -> np.ndarray:
    """Load a variable from a MATLAB ``.mat`` file.

    Parameters
    ----------
    path : str | Path
        Path to the ``.mat`` file.
    key : str, optional
        Variable name to extract from the file (default ``"avg"``).

    Returns
    -------
    np.ndarray
        Array stored under *key* in the MAT file.

    Examples
    --------
    >>> from insideout.io import read_fc_mat
    >>> arr = read_fc_mat("functional_connectivity.mat")
    >>> arr.shape
    (116, 116)
    """
    mat = scipy.io.loadmat(str(path))
    return mat[key]


def load_hcp_data(path: str | Path) -> pl.DataFrame:
    return read_csv(path)
