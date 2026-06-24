"""Data loading utilities for survey and functional-connectivity data.

Functions
---------
load_survey_data
    Load a CSV survey file, partition columns by a YAML config.
load_fc_data
    Load fMRI functional-connectivity data from a ``.mat`` file and
    its associated subject IDs from a CSV.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl
import yaml
from scipy.io import loadmat

logger = logging.getLogger(__name__)

DATA_BLOCKS = ("inner", "outer", "mental_health")


def load_survey_data(
    csv_path: str = "data/data_survey.csv",
    yaml_config_path: str = "configs/components/default.yaml",
    drop_na: bool = True,
) -> dict[str, pl.DataFrame | np.ndarray]:
    """Load a survey CSV and partition columns according to a YAML config.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (default ``"data/data_survey.csv"``).
    yaml_config_path : str
        Path to the YAML configuration describing column blocks
        (default ``"configs/components/default.yaml"``).
    drop_na : bool
        If ``True``, drop rows that contain null values in any
        configured column (default ``True``).

    Returns
    -------
    dict of str -> pl.DataFrame or np.ndarray
        A dictionary with keys ``"subjects"`` (an array of subject IDs)
        and one entry per block defined in the YAML config (each a
        ``pl.DataFrame`` containing the block's columns plus the
        subject ID column as its first column).

    Examples
    --------
    >>> data = load_survey_data("data/data_survey.csv", "configs/components/default.yaml")
    >>> data["inner"].columns
    ['Subject', 'Positive_Affect', 'Self_Efficacy', ...]
    >>> data["subjects"].shape
    (100,)
    """
    logger.info(f"Loading YAML config from: {yaml_config_path}")
    with open(yaml_config_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info(f"Loading CSV data from: {csv_path}")
    df = pl.read_csv(csv_path, null_values=["NA", ""])
    if "" in df.columns:
        df = df.drop("")

    subject_colname: str = config["metadata"]["subject_id_col"]
    if subject_colname not in df.columns:
        msg = f"Subject column '{subject_colname}' not found in CSV."
        logger.error(msg)
        raise KeyError(msg)

    relevant_cols = [subject_colname]
    for block_name in DATA_BLOCKS:
        relevant_cols.extend(item["column"] for item in config[block_name])

    existing_relevant_cols = [c for c in relevant_cols if c in df.columns]

    if drop_na:
        n_before = df.height
        df = df.drop_nulls(subset=existing_relevant_cols)
        n_dropped = n_before - df.height
        logger.info(
            f"Cleaned Data: Dropped {n_dropped} subjects out of {n_before} due to NAs."
        )

    subject_spine: np.ndarray = df[subject_colname].to_numpy()

    def _extract_block(cols: list[str]) -> pl.DataFrame:
        existing = [c for c in cols if c in df.columns]
        if not existing:
            logger.warning(
                "None of the configured columns were found in the dataframe."
            )
            return pl.DataFrame()
        block = df.select([subject_colname, *existing])
        return block

    output: dict[str, pl.DataFrame | np.ndarray] = {"subjects": subject_spine}

    for block_name in DATA_BLOCKS:
        logger.debug(f"Extracting block: {block_name}")
        cols = [item["column"] for item in config[block_name]]
        output[block_name] = _extract_block(cols)

    logger.info("Successfully loaded and partitioned survey data.")
    return output


def load_fc_data(
    fc_filepath: str = "data/hcp100_fc.mat",
    fc_key: str = "avg",
    fc_ids_filepath: str = "data/hcp100_fc_ids.csv",
) -> dict[str, np.ndarray]:
    """Load fMRI functional-connectivity data.

    Parameters
    ----------
    fc_filepath : str
        Path to the ``.mat`` file containing the FC tensor
        (default ``"data/hcp100_fc.mat"``).
    fc_key : str
        Variable name to extract from the ``.mat`` file
        (default ``"avg"``).
    fc_ids_filepath : str
        Path to the CSV with subject IDs
        (default ``"data/hcp100_fc_ids.csv"``).

    Returns
    -------
    dict of str -> np.ndarray
        A dictionary with keys ``"data"`` (the FC tensor as a contiguous
        array) and ``"subjects"`` (the subject ID array).

    Examples
    --------
    >>> fc = load_fc_data()
    >>> fc["data"].shape
    (100, 116, 116)
    """
    mat = loadmat(fc_filepath)
    fc_data = np.ascontiguousarray(mat[fc_key])
    logger.info(f"Successfully loaded FC data. Shape: {fc_data.shape}")

    fc_ids = pl.read_csv(fc_ids_filepath)
    fc_subjects = fc_ids["Subject"].to_numpy()
    logger.info(f"Successfully loaded ID data. Shape: {fc_ids.shape}")

    return {
        "data": fc_data,
        "subjects": fc_subjects,
    }
