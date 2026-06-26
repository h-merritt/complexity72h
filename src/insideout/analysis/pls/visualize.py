# src/analysis/pls/visualize.py
"""
Visualizations for the 3x3 PLS grid written by `python -m insideout.analysis.pls`.

Reads only what `__main__._save_run` / `main` put on disk:
  <OUTPUT_DIR>/run_index.json
  <OUTPUT_DIR>/dissociation.json
  <OUTPUT_DIR>/<run>/pls_raw_results.npz    (U,S,V,varexp,*_scores,pvals,*_bsr,split_*)
  <OUTPUT_DIR>/<run>/brain_meta.npz         (feature_names [+ edge_index_i/j])
  <OUTPUT_DIR>/<run>/run_info.json

and writes figures to <OUTPUT_DIR>/figures/.
"""

import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

OUTPUT_DIR = "results/pls"

# House style -- inner = blue, outer = orange (matches the YAML component colors).
INNER_C = "#9b59b6"
OUTER_C = "#3d8b3c"
DIVERGE = "RdBu_r"

plt.rcParams.update({
    "figure.figsize": [16, 8],
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 20,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times"],
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "legend.title_fontsize": 20,
    "lines.markersize": 14,
    "lines.linewidth": 3,
})

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _load_run(out_dir: str, run: str):
    run_dir = os.path.join(out_dir, run)
    res = dict(np.load(os.path.join(run_dir, "pls_raw_results.npz")))
    meta = dict(np.load(os.path.join(run_dir, "brain_meta.npz")))
    with open(os.path.join(run_dir, "run_info.json")) as f:
        info = json.load(f)
    return res, meta, info

def _finite(x):
    try:
        return np.isfinite(float(x))
    except (TypeError, ValueError):
        return False

def _blocks_to_matrix(feature_names: List[str], sal: np.ndarray) -> Tuple[List[str], np.ndarray]:
    names = [str(n) for n in feature_names]
    nets = sorted({tok for nm in names for tok in nm.split("-")})
    idx = {n: i for i, n in enumerate(nets)}
    M = np.full((len(nets), len(nets)), np.nan)
    for nm, s in zip(names, np.asarray(sal, float)):
        parts = nm.split("-")
        if len(parts) != 2:
            raise ValueError(f"Unexpected block name {nm!r}")
        a, b = parts
        M[idx[a], idx[b]] = s; M[idx[b], idx[a]] = s
    return nets, M

def _sym_norm(*mats) -> TwoSlopeNorm:
    vmax = max(np.nanmax(np.abs(m)) for m in mats if np.isfinite(np.nanmax(np.abs(m))))
    vmax = vmax if vmax > 0 else 1.0
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

def _draw_matrix(ax, M, labels, norm, title, annotate=None):
    im = ax.imshow(M, cmap=DIVERGE, norm=norm)
    ax.set_title(title)
    if annotate is None:
        annotate = M.shape[0] <= 8
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=10, color="black")
    return im

def _inner_outer_sets(out_dir: str, index: Dict):
    first_brain = next(iter(index["brain_specs"]))
    out = {}
    for key in ("inner", "outer"):
        try:
            with open(os.path.join(out_dir, f"{key}__{first_brain}", "run_info.json")) as f:
                out[key] = set(json.load(f)["behavior_columns"])
        except FileNotFoundError:
            out[key] = set()
    return out

# --------------------------------------------------------------------------- #
# Figure 1: Grid Overview
# --------------------------------------------------------------------------- #
def fig_grid_overview(out_dir: str, index: Dict, figdir: str):
    behav = list(index["behav_specs"].keys())
    brains = list(index["brain_specs"].keys())
    cov = np.full((len(behav), len(brains)), np.nan)
    pmat = np.full_like(cov, np.nan)
    smat = np.full_like(cov, np.nan)
    
    for i, b in enumerate(behav):
        for j, r in enumerate(brains):
            run = f"{b}__{r}"
            try:
                with open(os.path.join(out_dir, run, "run_info.json")) as f:
                    info = json.load(f)
            except FileNotFoundError:
                continue
            lv1 = info["per_lv"][0]
            cov[i, j] = lv1["covexp"] * 100
            pmat[i, j] = lv1["p_perm"] if lv1["p_perm"] is not None else np.nan
            smat[i, j] = lv1["split_ucorr"] if lv1["split_ucorr"] is not None else np.nan

    blabels = [index["brain_specs"][r].get("label", r) for r in brains]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    im = ax.imshow(cov, cmap="viridis", aspect="auto")
    
    ax.set_xticks(range(len(brains)))
    ax.set_xticklabels(blabels, rotation=45, ha="right")
    ax.set_yticks(range(len(behav)))
    ax.set_yticklabels(behav)
    ax.set_title("LV1 covariance explained across the 3x3 grid")
    
    for i in range(len(behav)):
        for j in range(len(brains)):
            if not np.isfinite(cov[i, j]):
                continue
            txt = f"{cov[i, j]:.0f}%"
            if np.isfinite(pmat[i, j]):
                txt += f"\np={pmat[i, j]:.3f}"
            if np.isfinite(smat[i, j]):
                txt += f"\nr½={smat[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    color="white" if cov[i, j] < np.nanmax(cov) * 0.6 else "black")

    fig.colorbar(im, ax=ax, label="LV1 covariance explained (%)", fraction=0.046)
    fig.tight_layout()
    p = os.path.join(figdir, "grid_overview.pdf")
    fig.savefig(p); plt.close(fig)
    return p

# --------------------------------------------------------------------------- #
# Figure 2: Behavior Salience Bars
# --------------------------------------------------------------------------- #
def fig_behavior(out_dir, run, index, figdir, lv=0):
    res, _, info = _load_run(out_dir, run)
    cols = info["behavior_columns"]
    
    if "behav_loadings" in res:
        vals = res["behav_loadings"][:, lv]
        ci   = res.get("behav_loading_ci");  ci  = None if ci  is None or ci.size  == 0 else ci[:, lv, :]
        bsr  = res.get("behav_loading_bsr"); bsr = None if bsr is None or bsr.size == 0 else bsr[:, lv]
        xlabel = "behavior loading (corr with brain score)   [* = 95% CI excludes 0]"
        degenerate = False
    else:
        vals, ci, bsr = res["V"][:, lv], None, res["V_bsr"][:, lv]
        xlabel = "salience (V)   [BSR unreliable: L==n_behav]"
        degenerate = bool(res.get("behav_salience_degenerate", True))

    order = np.argsort(-np.abs(vals))
    vals, cols = vals[order], [cols[i] for i in order]
    if ci is not None: ci = ci[order]
    if bsr is not None: bsr = bsr[order]

    def is_stable(k):
        if ci is not None:
            return (ci[k, 0] > 0) or (ci[k, 1] < 0)
        if bsr is not None and not degenerate:
            return _finite(bsr[k]) and abs(bsr[k]) > 2
        return False

    sets = _inner_outer_sets(out_dir, index)
    if sets["inner"] and sets["outer"]:
        colors = [INNER_C if c in sets["inner"] else (OUTER_C if c in sets["outer"] else "0.5") for c in cols]
    else:
        colors = [INNER_C if v >= 0 else OUTER_C for v in vals]

    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.4 * len(cols) + 1)))
    y = np.arange(len(cols))[::-1]
    ax.barh(y, vals, color=colors, edgecolor="0.2", linewidth=0.4)

    for k, (yi, vi) in enumerate(zip(y, vals)):
        if is_stable(k):
            ax.text(vi + (0.01 if vi >= 0 else -0.01), yi, "*",
                    va="center", ha="left" if vi >= 0 else "right", fontsize=16)

    ax.set_yticks(y); ax.set_yticklabels(cols, fontsize=12)
    ax.axvline(0, color="0.3", lw=0.8)
    
    lv1 = info["per_lv"][lv]
    sub = f"LV{lv + 1}: {lv1['covexp'] * 100:.1f}% cov"
    if lv1["p_perm"] is not None:
        sub += f", p={lv1['p_perm']:.3f}"
        
    ax.set_title(f"{run.replace('_', ' ')} — behavior loadings\n{sub}")
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    
    p = os.path.join(figdir, f"{run}_behavior_lv1.pdf")
    fig.savefig(p); plt.close(fig)
    return p

# --------------------------------------------------------------------------- #
# Figure 3: Scores
# --------------------------------------------------------------------------- #
def fig_scores(out_dir: str, run: str, figdir: str, lv: int = 0):
    res, _, info = _load_run(out_dir, run)
    bs, vs = res["brain_scores"][:, lv], res["behav_scores"][:, lv]
    r = np.corrcoef(bs, vs)[0, 1]
    
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(bs, vs, s=32, color="#444", alpha=0.75, edgecolor="white", linewidth=0.4)
    
    b1, b0 = np.polyfit(bs, vs, 1)
    xs = np.array([bs.min(), bs.max()])
    ax.plot(xs, b1 * xs + b0, color="#c0392b", lw=1.5)
    
    ax.set_xlabel("brain latent score  (X·U)")
    ax.set_ylabel("behavior latent score  (Y·V)")
    
    lv1 = info["per_lv"][lv]
    sub = f"r={r:.2f},  {lv1['covexp'] * 100:.1f}% cov"
    if lv1["p_perm"] is not None:
        sub += f",  p={lv1['p_perm']:.3f}"
        
    ax.set_title(f"{run.replace('_', ' ')} — LV{lv + 1} scores\n{sub}")
    fig.tight_layout()
    
    p = os.path.join(figdir, f"{run}_scores_lv1.pdf")
    fig.savefig(p); plt.close(fig)
    return p

# --------------------------------------------------------------------------- #
# Figure 4: System blocks
# --------------------------------------------------------------------------- #
def fig_systemblocks(out_dir: str, run: str, figdir: str, lv: int = 0):
    res, meta, _ = _load_run(out_dir, run)
    nets, M = _blocks_to_matrix(list(meta["feature_names"]), res["U"][:, lv])
    norm = _sym_norm(M)
    
    fig, ax = plt.subplots(figsize=(0.6 * len(nets) + 3, 0.6 * len(nets) + 2.5))
    im = _draw_matrix(ax, M, nets, norm, f"{run.replace('_', ' ')} — LV{lv + 1} brain saliences (U)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="salience")
    
    fig.tight_layout()
    p = os.path.join(figdir, f"{run}_systemblocks_lv1.pdf")
    fig.savefig(p); plt.close(fig)
    return p

# --------------------------------------------------------------------------- #
# Figure 5: Edge ROI Matrix
# --------------------------------------------------------------------------- #
def fig_edge_roimatrix(out_dir: str, run: str, index: Dict, figdir: str, lv: int = 0):
    res, meta, _ = _load_run(out_dir, run)
    i = np.asarray(meta["edge_index_i"]); j = np.asarray(meta["edge_index_j"])
    sal = res["U"][:, lv]
    
    R = int(index["n_rois"])
    M = np.zeros((R, R)); M[i, j] = sal; M[j, i] = sal
    labels = index.get("net_labels_coarse")
    
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    norm = _sym_norm(M)
    
    if labels and len(labels) == R:
        labels = np.asarray(labels)
        order = np.argsort(labels, kind="stable")
        M = M[np.ix_(order, order)]
        sl = labels[order]
        bounds = np.where(sl[:-1] != sl[1:])[0] + 0.5
        centers, uniq = [], []
        start = 0
        for bpt in list(bounds) + [R - 0.5]:
            end = int(np.ceil(bpt))
            centers.append((start + end - 1) / 2); uniq.append(sl[start])
            start = end
            
        im = ax.imshow(M, cmap=DIVERGE, norm=norm)
        for b in bounds:
            ax.axhline(b, color="k", lw=0.6); ax.axvline(b, color="k", lw=0.6)
        ax.set_xticks(centers); ax.set_xticklabels(uniq, rotation=45, ha="right", fontsize=11)
        ax.set_yticks(centers); ax.set_yticklabels(uniq, fontsize=11)
    else:
        im = ax.imshow(M, cmap=DIVERGE, norm=norm)
        
    ax.set_xlabel("ROI"); ax.set_ylabel("ROI")
    ax.set_title(f"{run.replace('_', ' ')} — LV{lv + 1} edge saliences (ROI x ROI, Yeo-sorted)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="salience")
    
    fig.tight_layout()
    p = os.path.join(figdir, f"{run}_edge_roimatrix_lv1.pdf")
    fig.savefig(p); plt.close(fig)
    return p

# --------------------------------------------------------------------------- #
# Figure 6: Inner vs Outer Dissociation
# --------------------------------------------------------------------------- #
def _dissoc_lookup(out_dir: str) -> Dict[str, Dict]:
    path = os.path.join(out_dir, "dissociation.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {d["brain"]: d for d in json.load(f)}

def fig_dissociation(out_dir: str, brain_key: str, index: Dict, figdir: str, lv: int = 0):
    a, b = f"inner__{brain_key}", f"outer__{brain_key}"
    if not (os.path.isdir(os.path.join(out_dir, a)) and os.path.isdir(os.path.join(out_dir, b))):
        return None
        
    rA, mA, _ = _load_run(out_dir, a)
    rB, mB, infoB = _load_run(out_dir, b)
    
    sa, sb = rA["U"][:, lv], rB["U"][:, lv]
    r_signed = np.corrcoef(sa, sb)[0, 1]
    sign = 1.0 if r_signed >= 0 else -1.0
    sb_al = sb * sign
    
    diss = _dissoc_lookup(out_dir).get(brain_key, {})
    abscorr = diss.get("abs_corr", abs(r_signed))
    pperm = diss.get("p_perm")
    label = index["brain_specs"][brain_key].get("label", brain_key)
    
    if infoB["brain_type"] == "systems":
        nets, Mi = _blocks_to_matrix(list(mA["feature_names"]), sa)
        _, Mo = _blocks_to_matrix(list(mB["feature_names"]), sb_al)
        Md = Mi - Mo
        norm = _sym_norm(Mi, Mo)
        dnorm = _sym_norm(Md)
        
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.6))
        im0 = _draw_matrix(axes[0], Mi, nets, norm, "inner (U, LV1)")
        im1 = _draw_matrix(axes[1], Mo, nets, norm, "outer (U, sign-aligned)")
        im2 = _draw_matrix(axes[2], Md, nets, dnorm, "inner − outer")
        fig.colorbar(im0, ax=axes[:2], fraction=0.025, pad=0.02, label="salience")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Δ salience")
        sc_ax = axes[3]
    else:
        fig, sc_ax = plt.subplots(figsize=(6.2, 6.0))
        
    sc_ax.scatter(sa, sb_al, s=14, alpha=0.55, color="#555", edgecolor="white", linewidth=0.2)
    lim = np.nanmax(np.abs(np.r_[sa, sb_al])) * 1.05
    sc_ax.plot([-lim, lim], [-lim, lim], color="#c0392b", lw=1.0, ls="--")
    sc_ax.set_xlim(-lim, lim); sc_ax.set_ylim(-lim, lim)
    sc_ax.set_aspect("equal")
    sc_ax.axhline(0, color="0.7", lw=0.6); sc_ax.axvline(0, color="0.7", lw=0.6)
    sc_ax.set_xlabel("inner brain salience")
    sc_ax.set_ylabel("outer brain salience (aligned)")
    
    ptxt = "n/a" if pperm is None else f"{pperm:.4f}"
    sc_ax.set_title(f"salience similarity\n|corr|={abscorr:.3f}, p={ptxt}")
    
    verdict = "shared brain axis" if abscorr >= 0.5 else "dissociable"
    fig.suptitle(f"Inner vs outer on {label.replace('_', ' ')} (LV1) — {verdict}", fontsize=18, y=1.02)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
        
    p = os.path.join(figdir, f"inner_vs_outer_{brain_key}_lv1.pdf")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

# --------------------------------------------------------------------------- #
# Figure 7: Combined 2x3 Plot (Inner, Outer, Innerouter) x (Brain, Behavior)
# --------------------------------------------------------------------------- #
def fig_combined_parcellation(out_dir: str, brain_key: str, index: Dict, figdir: str, lv: int = 0):
    a, b, c = f"inner__{brain_key}", f"outer__{brain_key}", f"innerouter__{brain_key}"
    if not (os.path.isdir(os.path.join(out_dir, a)) and os.path.isdir(os.path.join(out_dir, b)) and os.path.isdir(os.path.join(out_dir, c))):
        return None
        
    rA, mA, iA = _load_run(out_dir, a)
    rB, mB, iB = _load_run(out_dir, b)
    rC, mC, iC = _load_run(out_dir, c)
    
    # Brain Maps
    sa, sb, sc = rA["U"][:, lv], rB["U"][:, lv], rC["U"][:, lv]
    r_signed = np.corrcoef(sa, sb)[0, 1]
    sign = 1.0 if r_signed >= 0 else -1.0
    sb_al = sb * sign
    
    r_signed_c = np.corrcoef(sa, sc)[0, 1]
    sign_c = 1.0 if r_signed_c >= 0 else -1.0
    sc_al = sc * sign_c
    
    is_systems = iB["brain_type"] == "systems"
    nets = []
    
    if is_systems:
        nets, Mi = _blocks_to_matrix(list(mA["feature_names"]), sa)
        _, Mo = _blocks_to_matrix(list(mB["feature_names"]), sb_al)
        _, Mc = _blocks_to_matrix(list(mC["feature_names"]), sc_al)
    else:
        R_val = int(index["n_rois"])
        Mi = np.zeros((R_val, R_val))
        i = np.asarray(mA["edge_index_i"]); j = np.asarray(mA["edge_index_j"])
        Mi[i, j] = sa; Mi[j, i] = sa
        
        Mo = np.zeros((R_val, R_val))
        i2 = np.asarray(mB["edge_index_i"]); j2 = np.asarray(mB["edge_index_j"])
        Mo[i2, j2] = sb_al; Mo[j2, i2] = sb_al
        
        Mc = np.zeros((R_val, R_val))
        i3 = np.asarray(mC["edge_index_i"]); j3 = np.asarray(mC["edge_index_j"])
        Mc[i3, j3] = sc_al; Mc[j3, i3] = sc_al
        
    norm_brain = _sym_norm(Mi, Mo, Mc)
    
    # Behavior Loadings and Bootstrap CIs
    def _get_behav_data(res, info):
        cols = info["behavior_columns"]
        if "behav_loadings" in res:
            vals = res["behav_loadings"][:, lv]
            ci   = res.get("behav_loading_ci")
            ci   = None if ci is None or ci.size == 0 else ci[:, lv, :]
            bsr  = res.get("behav_loading_bsr")
            bsr  = None if bsr is None or bsr.size == 0 else bsr[:, lv]
            degenerate = False
        else:
            vals = res["V"][:, lv]
            ci = None
            bsr = res.get("V_bsr")
            bsr = None if bsr is None or bsr.size == 0 else bsr[:, lv]
            degenerate = bool(res.get("behav_salience_degenerate", True))
            
        stable = []
        for k in range(len(cols)):
            is_st = False
            if ci is not None:
                is_st = (ci[k, 0] > 0) or (ci[k, 1] < 0)
            elif bsr is not None and not degenerate:
                is_st = _finite(bsr[k]) and abs(bsr[k]) > 2
            stable.append(is_st)
        return cols, vals, stable
        
    colsA, valsA, stA = _get_behav_data(rA, iA)
    colsB, valsB, stB = _get_behav_data(rB, iB)
    colsC, valsC, stC = _get_behav_data(rC, iC)
    
    valsB = valsB * sign
    valsC = valsC * sign_c
    
    all_cols = list(colsC)
    dictA_vals = dict(zip(colsA, valsA))
    dictB_vals = dict(zip(colsB, valsB))
    dictC_vals = dict(zip(colsC, valsC))
    dictA_st = dict(zip(colsA, stA))
    dictB_st = dict(zip(colsB, stB))
    dictC_st = dict(zip(colsC, stC))
    
    vA = np.array([dictA_vals.get(c, 0.0) for c in all_cols])
    vB = np.array([dictB_vals.get(c, 0.0) for c in all_cols])
    vC = np.array([dictC_vals.get(c, 0.0) for c in all_cols])
    
    sA = [dictA_st.get(c, False) for c in all_cols]
    sB = [dictB_st.get(c, False) for c in all_cols]
    sC = [dictC_st.get(c, False) for c in all_cols]
    
    sets = _inner_outer_sets(out_dir, index)
    if sets["inner"] and sets["outer"]:
        colors_all = [INNER_C if c in sets["inner"] else (OUTER_C if c in sets["outer"] else "0.5") for c in all_cols]
    else:
        colors_all = [INNER_C if v >= 0 else OUTER_C for v in vA]

    # Plot Construction
    fig, axes = plt.subplots(2, 3, figsize=(18, max(11, 0.45 * len(all_cols) + 5)))

    # --- Top Row: Brain Maps ---
    def _draw_brain(ax, M, norm, title):
        if is_systems:
            im = ax.imshow(M, cmap=DIVERGE, norm=norm)
            ax.set_title(title)
            annotate = M.shape[0] <= 8
            ax.set_xticks(range(len(nets)))
            ax.set_yticks(range(len(nets)))
            ax.set_xticklabels(nets, rotation=45, ha="right", fontsize=11)
            ax.set_yticklabels(nets, fontsize=11)
            if annotate:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        if np.isfinite(M[i, j]):
                            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=10, color="black")
            return im
        else:
            labels = index.get("net_labels_coarse")
            R_dim = M.shape[0]
            if labels and len(labels) == R_dim:
                labels = np.asarray(labels)
                order = np.argsort(labels, kind="stable")
                M_sorted = M[np.ix_(order, order)]
                sl = labels[order]
                bounds = np.where(sl[:-1] != sl[1:])[0] + 0.5
                centers, uniq = [], []
                start = 0
                for bpt in list(bounds) + [R_dim - 0.5]:
                    end = int(np.ceil(bpt))
                    centers.append((start + end - 1) / 2)
                    uniq.append(sl[start])
                    start = end
                im = ax.imshow(M_sorted, cmap=DIVERGE, norm=norm)
                for b in bounds:
                    ax.axhline(b, color="k", lw=0.6); ax.axvline(b, color="k", lw=0.6)
                ax.set_xticks(centers)
                ax.set_xticklabels(uniq, rotation=45, ha="right", fontsize=11)
                ax.set_yticks(centers)
                ax.set_yticklabels(uniq, fontsize=11)
            else:
                im = ax.imshow(M, cmap=DIVERGE, norm=norm)
                ax.set_title(title)
            return im

    im0 = _draw_brain(axes[0, 0], Mi, norm_brain, "inner (Brain)")
    im1 = _draw_brain(axes[0, 1], Mo, norm_brain, "outer (Brain, aligned)")
    im2 = _draw_brain(axes[0, 2], Mc, norm_brain, "innerouter (Brain, aligned)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # --- Bottom Row: Behavior Bars ---
    y_pos = np.arange(len(all_cols))[::-1]
    
    def _draw_behav(ax, vals, stable_flags, title, show_y=True):
        ax.barh(y_pos, vals, color=colors_all, edgecolor="0.2", linewidth=0.4)
        
        # Add asterisks for significant/stable variables
        for k, (yi, vi) in enumerate(zip(y_pos, vals)):
            if stable_flags[k]:
                ax.text(vi + (0.01 if vi >= 0 else -0.01), yi, "*",
                        va="center", ha="left" if vi >= 0 else "right", fontsize=16)
        
        ax.set_yticks(y_pos)
        if show_y:
            ax.set_yticklabels(all_cols, fontsize=12)
        else:
            ax.set_yticklabels([])
            
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("loading / salience")

    _draw_behav(axes[1, 0], vA, sA, "inner (Behavior)")
    _draw_behav(axes[1, 1], vB, sB, "outer (Behavior, aligned)", show_y=False)
    _draw_behav(axes[1, 2], vC, sC, "innerouter (Behavior, aligned)", show_y=False)

    label = index["brain_specs"][brain_key].get("label", brain_key)
    fig.suptitle(f"Combined Brain & Behavior Overview: {label.replace('_', ' ')} (LV{lv+1})", fontsize=19, y=0.98)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
    p = os.path.join(figdir, f"combined_{brain_key}_lv1.pdf")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p

# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(out_dir: str = OUTPUT_DIR):
    with open(os.path.join(out_dir, "run_index.json")) as f:
        index = json.load(f)

    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    written = [fig_grid_overview(out_dir, index, figdir)]

    for run in index["runs"]:
        if not os.path.isdir(os.path.join(out_dir, run)):
            continue
        _, _, info = _load_run(out_dir, run)
        written += [fig_behavior(out_dir, run, index, figdir),
                    fig_scores(out_dir, run, figdir)]
        if info["brain_type"] == "systems":
            written.append(fig_systemblocks(out_dir, run, figdir))
        else:
            written.append(fig_edge_roimatrix(out_dir, run, index, figdir))

    for brain_key in index["brain_specs"]:
        p = fig_dissociation(out_dir, brain_key, index, figdir)
        if p:
            written.append(p)
            
        p_comb = fig_combined_parcellation(out_dir, brain_key, index, figdir)
        if p_comb:
            written.append(p_comb)

    print(f"Wrote {len(written)} figures to {figdir}")
    for p in written:
        print("  ", os.path.relpath(p, out_dir))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else OUTPUT_DIR)