# src/analysis/pls/__main__.py
"""
Config-driven PLS runner -- 3x3 grid.

BEHAVIOR blocks (rows):
    inner       -> the 7 "inner social self" survey items
    outer       -> the 7 "outer social self" survey items
    innerouter  -> inner + outer concatenated

BRAIN representations (cols):
    schaefer_edges -> full 100x100 FC upper triangle (4950 edges)         "100x100"
    yeo_sub        -> Yeo system blocks at the detailed subnetwork level   "~23x23"
    yeo7           -> Yeo system blocks at the 7-network level (28 blocks)  "7x7"

Each of the 9 cells runs behavioral PLS-correlation (brain x behavior). EVERY
variable the routine produces is written to OUTPUT_DIR/<run_name>/ so the
downstream visualizations (visualize.py) and any later analysis read straight
from disk. Inner-vs-outer brain-salience dissociation is computed within each
brain representation.

The two systems representations differ only in the Yeo label granularity handed
to build_brain_block: detailed=True splits each network into its subnetworks,
detailed=False keeps the 7 canonical networks. No change to features.py /
schaefer_yeo.py is needed -- the system-block code already adapts to whatever
label set it is given.

OPTIONAL EXTENSIONS (all opt-in; defaults reproduce the original run exactly):
    SVD_METHOD   -- "full" (scipy, exact) or "randomized" (sklearn, fast/light).
    GROUPS       -- None (single homogeneous sample, original behavior), a
                    per-subject label array aligned to matched_subjects, or a
                    pyls-style list of group sizes. With groups the brain
                    salience is shared and behavior saliences are stacked per
                    group; permutation/bootstrap/split-half become group-aware.
    RUN_CROSSVAL -- out-of-sample R / R^2 via stratified train/test splits.
"""
import json
import logging
import os
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd

from insideout.data.loaders import load_survey_data, load_fc_data
from insideout.features.schaefer_yeo import get_schaefer_networks
from .associator import run_behavioral_pls, compare_brain_saliences
from .features import build_behavior_block, build_brain_block

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 3x3 configuration -- edit here to choose what to run.
# --------------------------------------------------------------------------- #
BEHAV_SPECS: Dict[str, List[str]] = {
    "inner":      ["inner"],
    "outer":      ["outer"],
    "innerouter": ["inner", "outer"],
}

# representation + which net-label granularity build_brain_block should use.
#   "edges"   ignores net labels.
#   "systems" collapses FC to within/between-network block means using the
#             granularity-specific Yeo labels (see get_schaefer_networks).
BRAIN_SPECS: Dict[str, Dict] = {
    "schaefer_edges": {"representation": "edges",   "granularity": None,       "label": "Schaefer edges (4950)"},
    "yeo_sub":        {"representation": "systems", "granularity": "detailed", "label": "Yeo subnetwork blocks (~23)"},
    "yeo7":           {"representation": "systems", "granularity": "coarse",   "label": "Yeo-7 blocks (28)"},
}

# Full cross product -> 9 runs, named "<behav>__<brain>".
RUN_CONFIGS: List[Dict] = [
    {"name": f"{b}__{r}", "behav_blocks": BEHAV_SPECS[b],
     "behav_key": b, "brain_key": r}
    for b, r in product(BEHAV_SPECS, BRAIN_SPECS)
]

# Set False for a fast structural dry-run (skips permutation/bootstrap/split-half;
# BSRs, p-values and split-half corrs come back NaN but every array is still saved
# with the right shape). True produces every statistic -> truly "everything".
#
# NOTE on cost: the heaviest cell is schaefer_edges x innerouter -- the bootstrap
# transiently holds an (nboot, 4950, n_LV) array (~1.1 GB at nboot=2000, n_LV=14).
# If memory/time is tight for the workshop, lower nboot/nperm here.
RUN_PERMUTATIONS = True

# --------------------------------------------------------------------------- #
# Optional extensions -- all defaults reproduce the original single-group,
# full-SVD, no-cross-validation behavior. Flip these to opt in.
# --------------------------------------------------------------------------- #
#   "full"       -> scipy.linalg.svd (exact; original behavior).
#   "randomized" -> sklearn.utils.extmath.randomized_svd (faster / lighter on
#                   the 4950-edge matrices; approximate but very close).
SVD_METHOD = "full"

# None  -> single homogeneous sample (original behavior).
# Otherwise: a per-subject label array aligned to matched_subjects (i.e. to the
# FC subject order used as canonical), or a pyls-style list of group sizes that
# sums to n_subjects. With groups, U is shared and V is stacked per group.
GROUPS = None

# Out-of-sample cross-validation (stratified train/test splits).
RUN_CROSSVAL = True
TEST_SIZE = 0.25
N_TEST_SPLIT = 100

PLS_KWARGS = dict(
    run_permutations=RUN_PERMUTATIONS,
    nperm=2000, nboot=2000, nsplit=100, 
    rotate_perm=True, 
    seed=0,
    # --- extensions ---
    svd_method=SVD_METHOD,
    run_crossval=RUN_CROSSVAL,
    test_size=TEST_SIZE,
    n_test_split=N_TEST_SPLIT,
)

# Inner-vs-outer brain-salience dissociation, computed within each brain repr
# (compare_brain_saliences requires identical feature ordering, which is the
# case here because both runs use the same FC tensor + same Yeo labels).
DISSOCIATION_PAIRS = [(f"inner__{r}", f"outer__{r}") for r in BRAIN_SPECS]
DISSOCIATION_NPERM = 10000

YEO_N_ROIS = 100
YEO_NETWORKS = 7
SUBJECT_COL = "Subject"
OUTPUT_DIR = "results/pls"


# --------------------------------------------------------------------------- #
# Saving helpers
# --------------------------------------------------------------------------- #
def _jsonable(x):
    """Make numpy scalars/arrays/NaN safe for json.dump."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return None if np.isnan(v) else v
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and np.isnan(x):
        return None
    return x


def _save_run(name, cfg, results, brain_meta, behav_cols, out_dir):
    """Persist *everything* produced for one run and return its per-LV summary rows."""
    run_dir = os.path.join(out_dir, name)
    os.makedirs(run_dir, exist_ok=True)

    L = int(results["L"])
    is_edges = brain_meta["type"] == "edges"
    feature_names = list(brain_meta["feature_names"])

    # Group-aware behavior labels: with groups the V matrix is stacked
    # (rows = group x behavior), so the salience rows are labelled "<group>:<col>".
    n_groups = int(results.get("n_groups", 1))
    group_labels = np.asarray(results.get("group_labels", ["all"])).tolist()
    if n_groups > 1:
        behav_row_names = [f"{g}:{c}" for g in group_labels for c in behav_cols]
    else:
        behav_row_names = list(behav_cols)

    # 1) Raw numerical results -> the source of truth for all visualizations.
    #    Includes U/S/V, varexp, brain_scores/behav_scores, pvals, perm_S,
    #    U_bsr/V_bsr, split-half corrs+pvals, L, n, AND the extension outputs
    #    (n_groups, group_labels, svd_method, cv_pearson_r, cv_r_squared, ...).
    np.savez_compressed(os.path.join(run_dir, "pls_raw_results.npz"), **results)

    # 2) Brain-feature metadata so saliences stay interpretable later.
    meta_arrays = {"feature_names": np.asarray(feature_names)}  # plain unicode array (no pickle)
    if is_edges:
        meta_arrays["edge_index_i"] = np.asarray(brain_meta["edge_index"][0])
        meta_arrays["edge_index_j"] = np.asarray(brain_meta["edge_index"][1])
    np.savez_compressed(os.path.join(run_dir, "brain_meta.npz"), **meta_arrays)

    # 3) Human-readable run info (config + shapes + per-LV stats).
    per_lv = [{
        "LV": i + 1,
        "covexp": float(results["varexp"][i]),
        "singular_value": float(results["S"][i]),
        "p_perm": _jsonable(results["pvals"][i]),
        "split_ucorr": _jsonable(results["split_ucorr"][i]),
        "split_vcorr": _jsonable(results["split_vcorr"][i]),
        "split_u_p": _jsonable(results["split_u_p"][i]),
        "split_v_p": _jsonable(results["split_v_p"][i]),
    } for i in range(L)]

    # Cross-validation summary (per behavior column, averaged over splits).
    crossval_ran = bool(np.asarray(results.get("cv_pearson_r", np.empty((0, 0)))).size)
    crossval_summary = None
    if crossval_ran:
        cv_r = np.asarray(results["cv_pearson_r"])
        cv_r2 = np.asarray(results["cv_r_squared"])
        crossval_summary = [{
            "Variable": behav_cols[t],
            "r_mean": _jsonable(np.nanmean(cv_r[t])),
            "r_std": _jsonable(np.nanstd(cv_r[t])),
            "r2_mean": _jsonable(np.nanmean(cv_r2[t])),
            "r2_std": _jsonable(np.nanstd(cv_r2[t])),
        } for t in range(len(behav_cols))]

    info = {
        "name": name,
        "behav_key": cfg["behav_key"],
        "behav_blocks": cfg["behav_blocks"],
        "brain_key": cfg["brain_key"],
        "brain_type": brain_meta["type"],
        "n_subjects": int(results["n"]),
        "n_LVs": L,
        "n_brain_features": int(results["U"].shape[0]),
        "n_behavior_features": int(results["V"].shape[0]),
        "behavior_columns": list(behav_cols),
        "behavior_salience_rows": behav_row_names,
        "feature_names": ("see brain_meta.npz (4950 edges)" if is_edges else feature_names),
        "permutations_run": bool(PLS_KWARGS["run_permutations"]),
        "svd_method": str(results.get("svd_method", PLS_KWARGS["svd_method"])),
        "n_groups": n_groups,
        "group_labels": group_labels,
        "crossval_run": crossval_ran,
        "crossval_summary": crossval_summary,
        "pls_kwargs": {k: _jsonable(v) for k, v in PLS_KWARGS.items()},
        "per_lv": per_lv,
    }
    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # 4) Per-LV salience tables (ALL LVs), behavior + brain, sorted by |BSR|
    #    (falling back to |salience| when permutations were skipped).
    for lv in range(L):
        n = lv + 1
        bdf = pd.DataFrame({
            "Variable": behav_row_names,
            "Salience": results["V"][:, lv],
            "BSR": results["V_bsr"][:, lv],
        })
        if n_groups > 1:
            bdf.insert(0, "Group", [g for g in group_labels for _ in behav_cols])
        bdf["__k"] = bdf["BSR"].abs().where(bdf["BSR"].notna(), bdf["Salience"].abs())
        bdf.sort_values("__k", ascending=False).drop(columns="__k").to_csv(
            os.path.join(run_dir, f"lv{n}_behavior.csv"), index=False)

        brain_df = pd.DataFrame({
            "Feature": feature_names,
            "Salience": results["U"][:, lv],
            "BSR": results["U_bsr"][:, lv],
        })
        if is_edges:
            brain_df.insert(0, "Roi_A", brain_meta["edge_index"][0])
            brain_df.insert(1, "Roi_B", brain_meta["edge_index"][1])
        brain_df["__k"] = brain_df["BSR"].abs().where(
            brain_df["BSR"].notna(), brain_df["Salience"].abs())
        brain_df.sort_values("__k", ascending=False).drop(columns="__k").to_csv(
            os.path.join(run_dir, f"lv{n}_brain.csv"), index=False)

    # 4b) Cross-validation table (per behavior column).
    if crossval_ran:
        pd.DataFrame(crossval_summary).to_csv(
            os.path.join(run_dir, "crossval.csv"), index=False)

    # 5) Console summary.
    print("\n" + "=" * 60)
    print(f"RUN: {name}   (behav={cfg['behav_blocks']}, brain={cfg['brain_key']})")
    print("=" * 60)
    for i in range(L):
        p = results["pvals"][i]
        ve = results["varexp"][i] * 100
        su, sv = results["split_ucorr"][i], results["split_vcorr"][i]
        flag = "*" if (np.isfinite(p) and p < 0.05) else " "
        print(f"  LV{i + 1:<2d} covexp={ve:5.1f}%  p={p:.4f} {flag}  "
              f"split-half u/v={su:.2f}/{sv:.2f}")
    if crossval_ran:
        cv_r = np.asarray(results["cv_pearson_r"])
        cv_r2 = np.asarray(results["cv_r_squared"])
        print(f"  CV (out-of-sample, {cv_r.shape[1]} splits):  "
              f"mean r={np.nanmean(cv_r):.3f}  mean R^2={np.nanmean(cv_r2):.3f}")
    return per_lv


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading survey and fMRI data...")
    survey_dict = load_survey_data(
        csv_path="data/data_survey.csv",
        yaml_config_path="configs/components/default.yaml",
    )
    fc_dict = load_fc_data(
        fc_filepath="data/hcp100_fc.mat",
        fc_ids_filepath="data/hcp100_fc_ids.csv",
    )

    # Match subjects: keep FC order as the canonical subject order.
    survey_subjects = survey_dict["subjects"]
    fc_subjects = fc_dict["subjects"]
    mask = np.isin(fc_subjects, survey_subjects)
    matched_subjects = fc_subjects[mask]
    fmri_matched = fc_dict["data"][mask]
    logger.info("Matched %d subjects with both FC and survey data.", len(matched_subjects))

    # Groups must be aligned to matched_subjects (canonical FC order). A label
    # array is reindexed defensively; a list of sizes is passed through as-is.
    groups_arg = GROUPS
    if GROUPS is not None and not (isinstance(GROUPS, (list, tuple))
                                   and all(isinstance(g, (int, np.integer)) for g in GROUPS)):
        groups_arg = np.asarray(GROUPS)
        if len(groups_arg) != len(matched_subjects):
            raise ValueError(
                f"GROUPS has {len(groups_arg)} labels but there are "
                f"{len(matched_subjects)} matched subjects. Align GROUPS to the "
                "FC subject order (matched_subjects).")

    # Fetch Yeo labels at BOTH granularities, once, if any systems run needs them.
    need_systems = any(
        BRAIN_SPECS[c["brain_key"]]["representation"] == "systems" for c in RUN_CONFIGS)
    net_labels: Dict[str, list] = {}
    if need_systems:
        net_labels["coarse"] = get_schaefer_networks(
            n_rois=YEO_N_ROIS, networks=YEO_NETWORKS, detailed=False)
        net_labels["detailed"] = get_schaefer_networks(
            n_rois=YEO_N_ROIS, networks=YEO_NETWORKS, detailed=True)
        for g, lab in net_labels.items():
            if len(lab) != fmri_matched.shape[1]:
                raise ValueError(
                    f"Schaefer labels ({g}: {len(lab)}) != FC regions "
                    f"({fmri_matched.shape[1]}). Check the parcellation / FC order.")
        logger.info("Yeo granularities: coarse=%d nets, detailed=%d nets",
                    len(set(net_labels["coarse"])), len(set(net_labels["detailed"])))

    # Top-level metadata for the visualizations (per-ROI Yeo labels included so the
    # edges figure can reorder the 100x100 salience matrix by network).
    shared_meta = {
        "matched_subjects": np.asarray(matched_subjects).tolist(),
        "n_subjects": int(len(matched_subjects)),
        "n_rois": int(fmri_matched.shape[1]),
        "yeo_networks": YEO_NETWORKS,
        "behav_specs": BEHAV_SPECS,
        "brain_specs": BRAIN_SPECS,
        "runs": [c["name"] for c in RUN_CONFIGS],
        "net_labels_coarse": net_labels.get("coarse"),
        "net_labels_detailed": net_labels.get("detailed"),
        "svd_method": SVD_METHOD,
        "groups": (None if GROUPS is None else _jsonable(np.asarray(GROUPS))),
        "crossval": {"run": RUN_CROSSVAL, "test_size": TEST_SIZE,
                     "n_test_split": N_TEST_SPLIT},
        "pls_kwargs": {k: _jsonable(v) for k, v in PLS_KWARGS.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "run_index.json"), "w") as f:
        json.dump(shared_meta, f, indent=2)

    grid_rows = []
    stored_U = {}  # run name -> (brain salience matrix U, brain_key) for dissociation

    for cfg in RUN_CONFIGS:
        name = cfg["name"]
        spec = BRAIN_SPECS[cfg["brain_key"]]
        logger.info("=== Running %s (behav=%s, brain=%s) ===",
                    name, cfg["behav_blocks"], cfg["brain_key"])

        behav_df = build_behavior_block(
            survey_dict, cfg["behav_blocks"], SUBJECT_COL, matched_subjects)

        nl = net_labels[spec["granularity"]] if spec["representation"] == "systems" else None
        X_brain, brain_meta = build_brain_block(
            fmri_matched, representation=spec["representation"], net_labels=nl)

        results = run_behavioral_pls(
            X_brain=X_brain, Y_behav=behav_df.values, groups=groups_arg, **PLS_KWARGS)

        per_lv = _save_run(name, cfg, results, brain_meta, list(behav_df.columns), OUTPUT_DIR)

        stored_U[name] = (results["U"], cfg["brain_key"])
        cv_r = np.asarray(results.get("cv_pearson_r", np.empty((0, 0))))
        cv_r2 = np.asarray(results.get("cv_r_squared", np.empty((0, 0))))
        cv_r_mean = float(np.nanmean(cv_r)) if cv_r.size else float("nan")
        cv_r2_mean = float(np.nanmean(cv_r2)) if cv_r2.size else float("nan")
        for d in per_lv:
            grid_rows.append({
                "run": name, "behav": cfg["behav_key"], "brain": cfg["brain_key"],
                "brain_type": brain_meta["type"],
                "n_brain_features": int(results["U"].shape[0]),
                "n_groups": int(results.get("n_groups", 1)),
                "svd_method": str(results.get("svd_method", SVD_METHOD)),
                "cv_r_mean": cv_r_mean, "cv_r2_mean": cv_r2_mean,
                **d,
            })

    # Combined grid summary (one row per run x LV) -- handy for the overview figure.
    pd.DataFrame(grid_rows).to_csv(os.path.join(OUTPUT_DIR, "grid_summary.csv"), index=False)

    # --- Dissociation: inner vs outer brain saliences within each brain repr.
    dissociations = []
    for a, b in DISSOCIATION_PAIRS:
        if a not in stored_U or b not in stored_U:
            logger.warning("Skipping dissociation %s vs %s (run missing).", a, b)
            continue
        (Ua, ra), (Ub, rb) = stored_U[a], stored_U[b]
        if ra != rb:
            logger.warning("Skipping %s vs %s: brain representations differ.", a, b)
            continue
        cmp = compare_brain_saliences(Ua, Ub, lv=0, nperm=DISSOCIATION_NPERM)
        cmp.update({"run_a": a, "run_b": b, "brain": ra})
        dissociations.append(cmp)
        print("\n" + "=" * 60)
        print(f"DISSOCIATION (LV1 brain saliences): {a} vs {b}")
        print(f"  |corr| = {cmp['abs_corr']:.3f}   p_perm = {cmp['p_perm']:.4f}")
        print("  high |corr| -> shared brain axis; low -> dissociable.")

    with open(os.path.join(OUTPUT_DIR, "dissociation.json"), "w") as f:
        json.dump([{k: _jsonable(v) for k, v in d.items()} for d in dissociations], f, indent=2)

    logger.info("All outputs written under %s", OUTPUT_DIR)
    logger.info("Render figures with:  python -m insideout.analysis.pls.visualize")


if __name__ == "__main__":
    main()
