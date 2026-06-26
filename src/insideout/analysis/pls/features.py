# src/analysis/pls/features.py
"""
Block builders for behavioral PLS-correlation.

The core PLS routine is agnostic to what the blocks are; these functions decide:
  * BRAIN block: full edge representation ("edges") or Yeo system blocks ("systems")
  * BEHAVIOR block: any combination of survey component blocks ("inner", "outer",
    "mental_health", ...) selected by name.

Everything is returned aligned to the same subject order, so the two blocks can
be handed straight to run_behavioral_pls.
"""
import logging
from typing import Dict, List, Sequence, Tuple, Union, Optional

import numpy as np
import pandas as pd
import polars as pl

from insideout.features.schaefer_yeo import upper_triangle_tensor, system_blocks_from_tensor

logger = logging.getLogger(__name__)


def build_brain_block(
    fmri_3d: np.ndarray,
    representation: str = "edges",
    net_labels: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Build the 2D brain block (rows = subjects, in the FC tensor's row order).

    representation:
      "edges"   -> (N, n_edges) upper-triangle vector; meta has the (i, j) index.
      "systems" -> (N, n_blocks) Yeo within/between block means; needs net_labels.

    Returns (X_brain, meta). meta["feature_names"] labels the columns so brain
    saliences/BSRs are interpretable regardless of representation.
    """
    if representation == "edges":
        vals, iu = upper_triangle_tensor(fmri_3d)
        names = [f"{i}-{j}" for i, j in zip(iu[0], iu[1])]
        meta = {"type": "edges", "feature_names": names, "edge_index": iu}
        logger.info("Brain block (edges): %s", vals.shape)
        return vals, meta

    if representation == "systems":
        if net_labels is None:
            raise ValueError("representation='systems' requires net_labels.")
        vals, names = system_blocks_from_tensor(fmri_3d, net_labels)
        meta = {"type": "systems", "feature_names": names}
        logger.info("Brain block (systems): %s -> %d blocks", vals.shape, len(names))
        return vals, meta

    raise ValueError(f"Unknown brain representation: {representation!r}")


def build_behavior_block(
    survey_dict: Dict[str, Union[np.ndarray, pd.DataFrame, "pl.DataFrame"]],
    block_names: List[str],
    subject_col: str,
    matched_subjects: np.ndarray,
) -> pd.DataFrame:
    """
    Concatenate the requested survey component blocks and align to subjects.

    survey_dict     : output of load_survey_data (keys: 'subjects', 'inner', ...).
    block_names     : which components to include, e.g. ["inner"], ["outer"],
                      ["inner", "outer"], or any subset defined in the YAML.
    matched_subjects: subject IDs (and order) the brain block is in; the returned
                      DataFrame is reindexed to exactly this order.
    """
    frames = []
    for name in block_names:
        if name not in survey_dict:
            raise KeyError(f"Survey block {name!r} not found. Available: {list(survey_dict)}")
        block = survey_dict[name]
        if hasattr(block, "to_pandas"):
            block = block.to_pandas()
        if not isinstance(block, pd.DataFrame):
            raise TypeError(f"Survey block {name!r} is not a DataFrame.")
        frames.append(block.set_index(subject_col))

    behav = pd.concat(frames, axis=1)

    missing = set(matched_subjects) - set(behav.index)
    if missing:
        raise KeyError(
            f"{len(missing)} matched subjects absent from behavior blocks "
            f"{block_names}, e.g. {list(missing)[:5]}. "
            "Note: load_survey_data drops NA across ALL configured columns, so a "
            "subject with NA only in mental_health is dropped even from an "
            "inner/outer-only run. Load with the blocks you actually need if this bites."
        )

    behav = behav.loc[matched_subjects].copy()
    logger.info("Behavior block %s: %s", block_names, behav.shape)
    assert isinstance(behav, pd.DataFrame)
    return behav