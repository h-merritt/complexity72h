# src/data/loaders.py
import logging
from typing import Dict, Union, List, Any
import yaml
import numpy as np
import pandas as pd
from scipy.io import loadmat

logger = logging.getLogger(__name__)

def load_survey_data(
    csv_path: str = "data/data_survey.csv", 
    yaml_config_path: str = "configs/components/default.yaml",
    drop_NA: bool = True
) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
    """Load CSV file, optionally drop missing values, and partition data based on config."""

    # Load YAML config
    logger.info(f"Loading YAML config from: {yaml_config_path}")
    with open(yaml_config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Load CSV
    logger.info(f"Loading CSV data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Subject ID
    subject_colname: str = config["metadata"]["subject_id_col"]
    if subject_colname not in df.columns:
        logger.error(f"Subject column '{subject_colname}' not found in CSV.")
        raise KeyError(f"Subject column '{subject_colname}' specified in config not found in CSV.")
    
    ### NA Handling ###
    # Gather all relevant columns to check for NAs
    relevant_cols = [subject_colname]
    for block_name, block_items in config.items():
        if block_name != "metadata":
            relevant_cols.extend([item["column"] for item in block_items])

    # Filter to existing columns to prevent dropna errors
    existing_relevant_cols = [c for c in relevant_cols if c in df.columns]

    # Drop missing values early
    if drop_NA:
        initial_len = len(df)
        df = df.dropna(subset=existing_relevant_cols).reset_index(drop=True)
        dropped_count = initial_len - len(df)
        logger.info(f"Cleaned Data: Dropped {dropped_count} subjects out of {initial_len} due to NAs.")

    ### End of NA Handling ###

    subject_spine: np.ndarray = df[subject_colname].values

    # Separating components
    def _extract_component(cols: List[str]) -> pd.DataFrame:
        existing_cols = [c for c in cols if c in df.columns]
        if not existing_cols:
            logger.warning("None of the configured columns were found in the dataframe.")
            return pd.DataFrame(index=df.index)
        block_df = df[existing_cols].copy()
        block_df.insert(0, subject_colname, subject_spine)
        return block_df

    output: Dict[str, Union[np.ndarray, pd.DataFrame]] = {"subjects": subject_spine}

    # Iterate through the config dictionary
    for block_name, block_items in config.items():
        if block_name == "metadata":
            continue
        
        logger.debug(f"Extracting block: {block_name}")
        cols = [item["column"] for item in block_items]
        output[block_name] = _extract_component(cols)

    logger.info("Successfully loaded and partitioned survey data.")
    return output

def load_fc_data(
    fc_filepath: str = "data/hcp100_fc.mat",
    fc_key: str = "fc",
    fc_ids_filepath: str = "data/hcp100_fc_ids.csv",
) -> Dict[str, np.ndarray]:

    # 1. Load fMRI Tensor
    mat = loadmat(fc_filepath)
    raw_mat = mat[fc_key]
    
    # Dynamically find the subject axis (the one that is NOT 100)
    if raw_mat.ndim == 3:
        # Find which axis has a length different from your ROI count (100)
        roi_count = 100
        subject_axis = [i for i, dim in enumerate(raw_mat.shape) if dim != roi_count]
        
        # If we found a unique axis that isn't 100, move it to the front
        if len(subject_axis) == 1:
            axis_idx = subject_axis[0]
            if axis_idx != 0:
                logger.info(f"Moving subject axis from index {axis_idx} to the front...")
                raw_mat = np.moveaxis(raw_mat, axis_idx, 0)
        elif all(dim == roi_count for dim in raw_mat.shape):
            raise ValueError("All dimensions are 100. Cannot determine subject axis.")

    fc_data = np.ascontiguousarray(raw_mat)
    logger.info(f"Successfully loaded FC data. Shape: {fc_data.shape}")

    # 2. Load fMRI Subject IDs
    fc_ids_df = pd.read_csv(fc_ids_filepath)
    fc_subjects = fc_ids_df['Subject'].values
    logger.info(f"Successfully loaded ID data. Shape: {fc_ids_df.shape}")
    
    return {
        "data": fc_data,
        "subjects": fc_subjects,
    }
