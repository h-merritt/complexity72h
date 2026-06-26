"""
Behavioral PLS-correlation (McIntosh/Krishnan; the Misic/Bzdok network-neuro
flavour) between a BRAIN block and a BEHAVIOR block.

This routine is fully agnostic to what the blocks contain:
  - BRAIN block X may be 4950 edges or ~28 Yeo system blocks (or anything 2D).
  - BEHAVIOR block Y may be
    - inner-only
    - outer-only
    - inner + outer
    - etc.

The caller (see __main__) decides the configuration.

- Cross-covariance R = X^T @ Y  (z-scored columns -> R is brain x behav corr)
- SVD: R = U S V^T  -> brain saliences U, behavior saliences V, sing. values S
- Latent scores: brain = X@U, behav = Y@V (their correlation = the LV strength)
- Permutation on singular values (optionally Procrustes-rotated) -> LV significance
- Bootstrap on subjects, Procrustes-aligned -> brain bootstrap ratios; |BSR| > ~2 stable
- Behavior reliability via bootstrapped LOADINGS (behav_loading_bsr / _ci), NOT the
  raw right singular vector V: V is square-orthogonal here (L == n_behav), so its
  bootstrap/split-half collapse to a degenerate |BSR|->inf / corr->1 (see below).
- Split-half resampling -> reproducibility of brain saliences (U) and behavior loadings per LV

For each significant LV, read which BEHAVIOR saliences are large: if inner and
outer load on the SAME LV with the same sign they share a brain axis; if they
split across LVs / flip sign they dissociate. For a cleaner test, run inner-only
and outer-only separately and compare brain saliences with compare_brain_saliences().

ORIENTATION CONVENTION:
    X -> BRAIN block  (n_subjects x n_brain)   -> U / U_bsr = brain saliences
    Y -> BEHAVIOR block (n_subjects x n_behav) -> V / V_bsr = behavior saliences
The SVD is symmetric, so swapping X and Y will not error -- it will silently
swap the meaning of U and V. Be deliberate.

For production, we may prefer pyls.behavioral_pls (rmarkello/pyls; netneurolab/
pypyls), which implements this same algorithm with extra tooling.
But I've checked, and we actually get the same results as the pyls.behavioral_pls algorithm, 
but pyls.behavioral_pls runs ~100 times slower compared to this code.
It is probably because it does more than we need.

-----------------------------------------------------------------------------
OPTIONAL EXTENSIONS (added on top of the original routine; all opt-in, so the
default call is numerically identical to before):

  * svd_method : "full" (scipy.linalg.svd, exact) or "randomized"
        (sklearn.utils.extmath.randomized_svd, faster / lighter on the big
        4950-edge cross-covariance matrices). Used for EVERY decomposition
        (base, permutation, bootstrap, split-half, cross-validation train).

  * groups : multi-group / multi-condition designs. Pass either a per-subject
        label array (length n) or a pyls-style list of group sizes. The brain
        salience pattern U is SHARED across groups; the behavior saliences V are
        STACKED per group (rows = group x behavior). Permutation / bootstrap /
        split-half all become group-aware. groups=None reproduces the original
        single-group behavior exactly.

  * run_crossval : out-of-sample prediction. Repeatedly splits subjects into
        train/test (stratified within groups), fits the SVD on train, rescales
        the test brain block and predicts the behavior block, then scores it.
        Returns per-behavior-feature Pearson r and R^2 across splits. Uses an
        INDEPENDENT random stream, so it never perturbs the permutation /
        bootstrap numerics.
-----------------------------------------------------------------------------
"""
# src/analysis/pls/associator.py

import logging
from typing import Dict, Any, Optional, Sequence, Tuple, List

import numpy as np
from scipy.linalg import svd

try:                                       # randomized SVD is optional
    from sklearn.utils.extmath import randomized_svd
except ImportError:                        # pragma: no cover
    randomized_svd = None

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def zscore(M: np.ndarray) -> np.ndarray:
    """
    Column-wise z-score (ddof=1).

    Re-applied even if survey data was already z-scored on the full N=1200 pool:
    on the analyzed subset the per-column mean/sd are no longer exactly (0, 1),
    and PLS needs each block centered/unit-scaled *within the analyzed sample* so
    that R = X^T @ Y / (n - 1) is the brain-behavior correlation matrix.
    """
    M = np.asarray(M, dtype=float)
    mu = M.mean(0)
    sd = M.std(0, ddof=1)
    sd[sd == 0] = 1.0
    return (M - mu) / sd


def cross_cov(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Cross-covariance (== cross-correlation when X, Y are z-scored)."""
    return X.T @ Y / (X.shape[0] - 1)


def _decompose(
    R: np.ndarray,
    method: str = "full",
    n_components: Optional[int] = None,
    seed: int = 0,
):
    """
    SVD engine wrapper used by EVERY decomposition in this module.

    Parameters
    ----------
    R : (B, T) cross-covariance / cross-correlation matrix to decompose.
    method : "full"        -> scipy.linalg.svd (exact, full_matrices=False).
             "randomized"  -> sklearn.utils.extmath.randomized_svd (fast,
                              memory-light, approximate; great for the
                              4950 x T edge matrices).
    n_components : keep this many LVs. None -> min(R.shape) (full rank of R).
        Always pass the *base* L for resamples so U/S/Vt shapes line up.
    seed : random_state for the randomized solver (ignored by "full").

    Returns
    -------
    U  : (B, L)
    S  : (L,)        singular values, descending
    Vt : (L, T)
    """
    R = np.asarray(R, dtype=float)
    if method == "full":
        U, S, Vt = svd(R, full_matrices=False)
        if n_components is not None:
            U, S, Vt = U[:, :n_components], S[:n_components], Vt[:n_components]
        return U, S, Vt

    if method == "randomized":
        if randomized_svd is None:
            raise ImportError(
                "svd_method='randomized' needs scikit-learn "
                "(sklearn.utils.extmath.randomized_svd)."
            )
        if n_components is None:
            n_components = min(R.shape)
        n_components = min(n_components, min(R.shape))
        U, S, Vt = randomized_svd(R, n_components=n_components, random_state=seed)
        return U, S, Vt

    raise ValueError(f"Unknown svd_method {method!r} (use 'full' or 'randomized').")


def procrustes_rotation(ref: np.ndarray, resampled: np.ndarray) -> np.ndarray:
    """Orthogonal Q (L x L) with `resampled @ Q ~= ref` (orthogonal Procrustes).

    Being orthogonal (det = +/-1), Q absorbs reflections, rotations AND axis
    swaps in one step -- which a manual sign-flip cannot do. Apply the SAME Q to
    both U and V of a resample to keep the pair consistent. (This assumes U and V
    share the resampling rotation, exact for distinct singular values and a good
    approximation within near-degenerate subspaces -- the standard PLS treatment.)

    Always uses the exact (scipy) SVD: the matrix here is tiny (L x L), so there
    is no speed reason to approximate it, and exactness keeps the alignment clean.
    """
    N, _, Pt = svd(resampled.T @ ref)
    return N @ Pt


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return 0.0 if d == 0 else float((a @ b) / d)


def _columnwise_corr(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """Pearson r between matching columns of two (S, T) arrays -> (T,)."""
    yt = Y_true - Y_true.mean(0)
    yp = Y_pred - Y_pred.mean(0)
    num = (yt * yp).sum(0)
    den = np.sqrt((yt ** 2).sum(0) * (yp ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return np.clip(r, -1.0, 1.0)


def _r2_per_column(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """R^2 per column (== sklearn r2_score multioutput='raw_values') -> (T,)."""
    ss_res = ((Y_true - Y_pred) ** 2).sum(0)
    ss_tot = ((Y_true - Y_true.mean(0)) ** 2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
    return r2


def _zmap(test: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Standardize `test` columns using `train`'s mean/sd (scipy.stats.zmap, ddof=1)."""
    mu = train.mean(0)
    sd = train.std(0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return (test - mu) / sd


def _behav_loadings(Xz: np.ndarray, Yz: np.ndarray, Umat: np.ndarray) -> np.ndarray:
    """
    Behavior LOADINGS: Pearson corr of each behavior column with each brain
    latent score (brain score = Xz @ U[:, l]). Returns (T, L).

    This is the correct quantity for behavior-side reliability. Unlike the raw
    right singular vector V -- which is square-orthogonal here (L == n_behav) and
    therefore reconstructed *exactly* by the Procrustes step on every resample,
    collapsing its bootstrap variance to ~0 and inflating |BSR| to ~1e11 -- the
    loadings are not orthonormalized, so they vary across resamples and yield
    meaningful bootstrap ratios / CIs (this is what pyls bootstraps for behavior).
    """
    scores = Xz @ Umat                                  # (n, L)
    sc = scores - scores.mean(0)
    yc = Yz - Yz.mean(0)
    num = yc.T @ sc                                     # (T, L)
    den = np.sqrt((yc ** 2).sum(0)[:, None] * (sc ** 2).sum(0)[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


# --------------------------------------------------------------------------- #
# Group handling
# --------------------------------------------------------------------------- #
def _build_group_masks(groups, n: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Resolve the `groups` argument into boolean per-group masks + ordered labels.

    Accepts:
      * None                      -> single group spanning all n subjects.
      * per-subject labels (len n) -> one group per unique label (sorted).
      * pyls-style group sizes     -> a short list of ints summing to n
                                      (e.g. [40, 35] -> two groups). Subjects are
                                      assumed stacked group-by-group, matching the
                                      pyls "subjects within groups" convention.

    Returns (masks, labels) where labels are sorted/ordered for determinism, so
    stacked behavior saliences line up reproducibly across runs.
    """
    if groups is None:
        return [np.ones(n, dtype=bool)], np.asarray(["all"])

    g = np.asarray(groups)

    # pyls-style list of group SIZES: short integer vector summing to n.
    is_sizes = (
        g.ndim == 1
        and g.size < n
        and np.issubdtype(g.dtype, np.integer)
        and int(g.sum()) == n
    )
    if is_sizes:
        labels_per_subject = np.repeat(np.arange(g.size), g)
    elif g.shape[0] == n:
        labels_per_subject = g
    else:
        raise ValueError(
            f"groups must be per-subject labels of length n={n}, or a list of "
            f"group sizes summing to {n}; got shape {g.shape} summing to "
            f"{g.sum() if np.issubdtype(g.dtype, np.number) else 'NA'}."
        )

    uniq = np.unique(labels_per_subject)            # sorted -> deterministic
    masks = [labels_per_subject == u for u in uniq]
    for u, m in zip(uniq, masks):
        if m.sum() < 2:
            raise ValueError(f"Group {u!r} has <2 subjects; PLS needs >=2 per group.")
    return masks, uniq


def _stacked_crosscov(X: np.ndarray, Y: np.ndarray, masks: List[np.ndarray]) -> np.ndarray:
    """
    Horizontally stack per-group brain-behavior cross-correlations.

    X (n, B), Y (n, T) -> R (B, G*T). Each block is zscore(X_g)^T @ zscore(Y_g)
    / (n_g - 1): the within-group correlation, matching McIntosh behavioral PLS.
    Per-group z-scoring is idempotent w.r.t. any prior global z-scoring, so this
    is identical to z-scoring the raw blocks within each group.

    The brain salience U from SVD(R) is SHARED across groups; the behavior
    salience V is (G*T, L): rows [g*T : (g+1)*T] belong to group g.
    """
    blocks = [cross_cov(zscore(X[m]), zscore(Y[m])) for m in masks]   # each (B, T)
    return np.hstack(blocks)                                          # (B, G*T)


def _boot_stacked_crosscov(X, Y, masks, rng):
    """Bootstrap resample WITHIN each group (preserves group sizes), then stack.
    Also returns the concatenated resampled row indices so behavior loadings can
    be recomputed on the resample."""
    blocks, idxs = [], []
    for m in masks:
        idx = np.where(m)[0]
        bidx = idx[rng.integers(0, len(idx), len(idx))]
        blocks.append(cross_cov(zscore(X[bidx]), zscore(Y[bidx])))
        idxs.append(bidx)
    return np.hstack(blocks), np.concatenate(idxs)


# --------------------------------------------------------------------------- #
# Split-half reliability
# --------------------------------------------------------------------------- #
def _svd_half(Xh, Yh, V_ref, masks_h=None, svd_method="full", seed=0):
    """SVD of one split half, Procrustes-aligned to the full-sample reference.

    Returns (U_aligned, V_aligned, behav_loadings) where behav_loadings is the
    half's behavior-vs-brain-score correlation matrix -- used for the v-side
    reliability, because the aligned V itself is degenerate (see _behav_loadings).
    """
    Xz, Yz = zscore(Xh), zscore(Yh)
    if masks_h is None or len(masks_h) == 1:
        R = cross_cov(Xz, Yz)
    else:
        R = _stacked_crosscov(Xh, Yh, masks_h)
    U, _, Vt = _decompose(R, method=svd_method, n_components=V_ref.shape[1], seed=seed)
    V = Vt.T
    Q = procrustes_rotation(V_ref, V)          # align this half to the full-sample axes
    Ua = U @ Q
    load = _behav_loadings(Xz, Yz, Ua)
    return Ua, V @ Q, load


def _split_half_reliability(X, Y, V_ref, nsplit, rng, masks, svd_method="full", seed=0):
    """
    Per-LV reproducibility of brain (U) and behavior saliences.

    Each iteration splits subjects into two disjoint halves, runs PLS on each,
    Procrustes-aligns both to the full-sample reference, and records |corr| of
    the matched patterns. A null is built by shuffling Y rows within each half
    (breaking brain-behavior correspondence). p is the paired fraction of splits
    where the null reliability matches or beats the observed one. Higher observed
    correlation + small p => the LV's pattern is reproducible.

    BRAIN (u) side: correlation of the aligned brain saliences across halves.
    BEHAVIOR (v) side: correlation of the behavior LOADINGS across halves -- NOT
    the aligned right singular vectors. Because L == n_behav here, the behavior
    salience matrix is square-orthogonal and the Procrustes step reconstructs it
    exactly every split, which would force the v-correlation to a meaningless
    1.00. Loadings do not have that degeneracy.

    With multiple groups the split is stratified WITHIN each group, so both
    halves keep the group structure; otherwise this is the original single-group
    routine (the u-side is unchanged from before).
    """
    n = X.shape[0]
    L = V_ref.shape[1]
    n_groups = len(masks)
    obs_u = np.zeros((nsplit, L)); obs_v = np.zeros((nsplit, L))
    nul_u = np.zeros((nsplit, L)); nul_v = np.zeros((nsplit, L))

    if n_groups == 1:
        # ---- single-group path: u-side identical to before; v-side now loadings ----
        half = n // 2
        for s in range(nsplit):
            perm = rng.permutation(n)
            iA, iB = perm[:half], perm[half:2 * half]

            uA, _, lA = _svd_half(X[iA], Y[iA], V_ref, None, svd_method, seed)
            uB, _, lB = _svd_half(X[iB], Y[iB], V_ref, None, svd_method, seed)
            nuA, _, nlA = _svd_half(X[iA], Y[iA][rng.permutation(half)], V_ref, None, svd_method, seed)
            nuB, _, nlB = _svd_half(X[iB], Y[iB][rng.permutation(half)], V_ref, None, svd_method, seed)

            for l in range(L):
                obs_u[s, l] = abs(_corr(uA[:, l], uB[:, l]))
                obs_v[s, l] = abs(_corr(lA[:, l], lB[:, l]))
                nul_u[s, l] = abs(_corr(nuA[:, l], nuB[:, l]))
                nul_v[s, l] = abs(_corr(nlA[:, l], nlB[:, l]))
    else:
        # ---- grouped path: stratify the split within each group ----
        group_idx = [np.where(m)[0] for m in masks]
        for s in range(nsplit):
            iA, iB, gA, gB = [], [], [], []
            for gi, gidx in enumerate(group_idx):
                perm = rng.permutation(gidx)
                h = len(gidx) // 2
                iA.append(perm[:h]); iB.append(perm[h:2 * h])
                gA.append(np.full(h, gi)); gB.append(np.full(h, gi))
            iA = np.concatenate(iA); iB = np.concatenate(iB)
            gA = np.concatenate(gA); gB = np.concatenate(gB)
            masks_A = [gA == gi for gi in range(n_groups)]
            masks_B = [gB == gi for gi in range(n_groups)]

            # null: permute Y rows within each half, respecting group blocks
            permA = np.concatenate([np.where(masks_A[gi])[0][rng.permutation(masks_A[gi].sum())]
                                    for gi in range(n_groups)])
            permB = np.concatenate([np.where(masks_B[gi])[0][rng.permutation(masks_B[gi].sum())]
                                    for gi in range(n_groups)])

            uA, _, lA = _svd_half(X[iA], Y[iA], V_ref, masks_A, svd_method, seed)
            uB, _, lB = _svd_half(X[iB], Y[iB], V_ref, masks_B, svd_method, seed)
            nuA, _, nlA = _svd_half(X[iA], Y[iA][permA], V_ref, masks_A, svd_method, seed)
            nuB, _, nlB = _svd_half(X[iB], Y[iB][permB], V_ref, masks_B, svd_method, seed)

            for l in range(L):
                obs_u[s, l] = abs(_corr(uA[:, l], uB[:, l]))
                obs_v[s, l] = abs(_corr(lA[:, l], lB[:, l]))
                nul_u[s, l] = abs(_corr(nuA[:, l], nuB[:, l]))
                nul_v[s, l] = abs(_corr(nlA[:, l], nlB[:, l]))

    u_p = (np.sum(nul_u >= obs_u, axis=0) + 1) / (nsplit + 1)
    v_p = (np.sum(nul_v >= obs_v, axis=0) + 1) / (nsplit + 1)
    return obs_u.mean(0), obs_v.mean(0), u_p, v_p


# --------------------------------------------------------------------------- #
# Out-of-sample cross-validation
# --------------------------------------------------------------------------- #
def _crossvalidate(X, Y, masks, V_full, test_size, n_test_split, svd_method, seed):
    """
    Out-of-sample predictive accuracy of the PLS model.

    For each split: stratified train/test partition (within groups), fit the SVD
    on the training brain x behavior cross-covariance, rescale the held-out brain
    block relative to the training block, predict the held-out behavior as
        Y_pred = zmap(X_test | X_train) @ U @ V_g^T + mean(Y_train)
    and score each behavior column with Pearson r and R^2 against the truth.

    Mirrors pyls' BehavioralPLS.crossval / compute.rescale_test, adapted to this
    module's (brain x behavior) orientation. Runs on its OWN random stream so the
    permutation/bootstrap results upstream are untouched.

    Returns (r_scores, r2_scores), each (T, n_test_split) where T = #behavior cols.
    """
    n, B = X.shape
    T = Y.shape[1]
    n_groups = len(masks)
    rng = np.random.default_rng(seed)

    r_scores = np.full((T, n_test_split), np.nan)
    r2_scores = np.full((T, n_test_split), np.nan)

    L_ref = V_full.shape[1]
    group_idx = [np.where(m)[0] for m in masks]

    for s in range(n_test_split):
        train = np.zeros(n, dtype=bool)
        for gidx in group_idx:                       # stratified split within groups
            k = int(round(len(gidx) * (1.0 - test_size)))
            k = max(2, min(k, len(gidx) - 1))         # keep both train & test non-trivial
            sel = rng.permutation(gidx)[:k]
            train[sel] = True
        test = ~train

        masks_tr = [m & train for m in masks]
        masks_te = [m & test for m in masks]

        if n_groups == 1:
            Rtr = cross_cov(zscore(X[train]), zscore(Y[train]))
        else:
            # build training cross-cov from the per-group training blocks
            Rtr = np.hstack([cross_cov(zscore(X[m]), zscore(Y[m])) for m in masks_tr])

        U, _, Vt = _decompose(Rtr, method=svd_method, n_components=L_ref, seed=seed)
        V = Vt.T                                       # (G*T, L)

        Y_pred = np.zeros((test.sum(), T))
        test_pos = np.where(test)[0]
        pos_of = {idx: i for i, idx in enumerate(test_pos)}

        for gi in range(n_groups):
            tr_g = masks_tr[gi]
            te_g = masks_te[gi]
            if tr_g.sum() < 2 or te_g.sum() == 0:
                continue
            Vg = V[gi * T:(gi + 1) * T, :] if n_groups > 1 else V
            X_resc = _zmap(X[te_g], X[tr_g])
            pred_g = X_resc @ U @ Vg.T + Y[tr_g].mean(0, keepdims=True)
            for row, idx in zip(pred_g, np.where(te_g)[0]):
                Y_pred[pos_of[idx]] = row

        Y_true = Y[test]
        r_scores[:, s] = _columnwise_corr(Y_true, Y_pred)
        r2_scores[:, s] = _r2_per_column(Y_true, Y_pred)

    return r_scores, r2_scores


# --------------------------------------------------------------------------- #
# Main routine
# --------------------------------------------------------------------------- #

def run_behavioral_pls(
    X_brain: np.ndarray,
    Y_behav: np.ndarray,
    run_permutations: bool = True,
    nperm: int = 2000,
    nboot: int = 2000,
    nsplit: int = 100,
    rotate_perm: bool = True,
    seed: int = 0,
    # ---- optional extensions (defaults reproduce the original behavior) ----
    svd_method: str = "full",
    groups: Optional[Sequence] = None,
    run_crossval: bool = False,
    test_size: float = 0.25,
    n_test_split: int = 100,
) -> Dict[str, Any]:
    """
    Behavioral PLS-correlation on pre-aligned 2D blocks (same subjects, same order).

    Parameters
    ----------
    X_brain : (n_subjects, n_brain)   -> U / U_bsr
    Y_behav : (n_subjects, n_behav)   -> V / V_bsr
    run_permutations: if False, skips permutation, bootstrap, and split-half testing.
    nsplit  : split-half iterations (0 to skip).
    rotate_perm : Procrustes-rotate each permutation onto the reference axes
        before reading singular values (matches pyls default). This can inflate
        false positives (Kovacevic et al., 2013); set False for the more
        conservative raw-singular-value permutation.

    svd_method : "full" (scipy.linalg.svd, exact) or "randomized"
        (sklearn.utils.extmath.randomized_svd, faster/lighter). Default "full"
        keeps the original numerics exactly.
    groups : None for a single homogeneous sample (original behavior), or a
        per-subject label array (length n), or a pyls-style list of group sizes.
        When groups are given the brain salience U is shared across groups and V
        is stacked (rows = group x behavior); permutation/bootstrap/split-half
        all become group-aware.
    run_crossval : if True, also run out-of-sample cross-validation and return
        cv_pearson_r / cv_r_squared (each (n_behav_cols, n_test_split)).
    test_size : proportion held out per split (0, 1).
    n_test_split : number of train/test splits.

    Returns
    -------
    dict with all original keys (U, S, V, varexp, brain_scores, behav_scores,
    pvals, perm_S, U_bsr, V_bsr, split_ucorr, split_vcorr, split_u_p, split_v_p,
    L, n) PLUS: group_labels, n_groups, behav_per_group, cv_pearson_r,
    cv_r_squared, cv_r_mean, cv_r2_mean, svd_method.
    """
    X = zscore(X_brain)
    Y = zscore(Y_behav)
    n = X.shape[0]
    if n != Y.shape[0]:
        raise ValueError(
            f"Subject mismatch: X has {n} rows, Y has {Y.shape[0]}. "
            "Blocks must be aligned to the same subjects in the same order."
        )

    masks, group_labels = _build_group_masks(groups, n)
    n_groups = len(masks)
    single_group = n_groups == 1
    T = Y.shape[1]

    def build_R(Xb, Yb):
        # single group: identical to original cross_cov(X, Y) (no double z-score).
        return cross_cov(Xb, Yb) if single_group else _stacked_crosscov(Xb, Yb, masks)

    # 1. Base SVD
    R = build_R(X, Y)
    U, S, Vt = _decompose(R, method=svd_method, seed=seed)
    V = Vt.T
    L = len(S)
    varexp = S ** 2 / np.sum(S ** 2)

    # Participant scores
    brain_scores = X @ U                       # shared brain axis projection
    if single_group:
        behav_scores = Y @ V
    else:
        behav_scores = np.empty((n, L))
        for gi, m in enumerate(masks):
            Vg = V[gi * T:(gi + 1) * T, :]
            behav_scores[m] = zscore(Y[m]) @ Vg

    # Behavior LOADINGS (point estimate): corr of each behavior col with the
    # brain latent score. This is the interpretable, non-degenerate behavior
    # quantity (see _behav_loadings). Always available, even without bootstrap.
    behav_loadings = _behav_loadings(X, Y, U)          # (T, L)

    # Flag the V degeneracy: when L == n_behav the right singular vectors are
    # square-orthogonal, so V_bsr / split-half on V are NOT meaningful. Use the
    # loadings (behav_loading_bsr / behav_loading_ci) instead.
    behav_salience_degenerate = bool(V.shape[0] == L and single_group)
    if behav_salience_degenerate:
        logger.warning(
            "Behavior salience matrix is square-orthogonal (L == n_behav = %d): "
            "V_bsr and the V-side split-half are degenerate (|BSR|->inf, corr->1). "
            "Use behav_loading_bsr / behav_loading_ci for behavior reliability.", L)

    rng = np.random.default_rng(seed)

    if run_permutations:
        # 2. Permutation test (shuffle Y rows -> break brain<->behavior correspondence)
        perm_S = np.zeros((nperm, L))
        for i in range(nperm):
            Yp = Y[rng.permutation(n)]
            Rp = build_R(X, Yp)
            Up, Sp, Vtp = _decompose(Rp, method=svd_method, n_components=L, seed=seed)
            if rotate_perm:
                Vp = Vtp.T
                Q = procrustes_rotation(V, Vp)
                perm_S[i] = np.abs(np.diag((Up @ Q).T @ Rp @ (Vp @ Q)))
            else:
                perm_S[i] = Sp[:L]
        pvals = (np.sum(perm_S >= S[None, :], axis=0) + 1) / (nperm + 1)

        # 3. Bootstrap (resample subjects, re-standardize BOTH blocks, Procrustes-align)
        bootU = np.zeros((nboot, X.shape[1], L))
        bootV = np.zeros((nboot, V.shape[0], L))
        bootLoad = np.zeros((nboot, T, L))               # behavior loadings -> reliable BSR/CI
        for b in range(nboot):
            if single_group:
                idx = rng.integers(0, n, n)
                Rb = cross_cov(zscore(X[idx]), zscore(Y[idx]))
            else:
                Rb, idx = _boot_stacked_crosscov(X, Y, masks, rng)
            Ub, _, Vtb = _decompose(Rb, method=svd_method, n_components=L, seed=seed)
            Vb = Vtb.T
            Q = procrustes_rotation(V, Vb)          # one Q, applied to both -> keeps pairing
            UbQ = Ub @ Q
            bootU[b] = UbQ
            bootV[b] = Vb @ Q
            bootLoad[b] = _behav_loadings(zscore(X[idx]), zscore(Y[idx]), UbQ)
        U_bsr = U / (bootU.std(0, ddof=1) + 1e-12)
        V_bsr = V / (bootV.std(0, ddof=1) + 1e-12)      # NOTE: degenerate when L==n_behav
        # Behavior reliability done RIGHT: ratio of loading to its bootstrap SE,
        # plus a percentile CI. |bsr|>2 (or CI excluding 0) flags a stable item.
        behav_loading_bsr = behav_loadings / (bootLoad.std(0, ddof=1) + 1e-12)
        behav_loading_ci = np.stack(
            np.percentile(bootLoad, [2.5, 97.5], axis=0), axis=-1)   # (T, L, 2)

        # 4. Split-half reliability
        if nsplit > 0:
            su, sv, su_p, sv_p = _split_half_reliability(
                X, Y, V, nsplit, rng, masks, svd_method, seed)
        else:
            su = sv = su_p = sv_p = np.full(L, np.nan)

    else:
        # Provide structural defaults if skipping tests to prevent KeyErrors
        pvals = np.full(L, np.nan)
        perm_S = np.zeros((0, L))
        U_bsr = np.full_like(U, np.nan)
        V_bsr = np.full_like(V, np.nan)
        behav_loading_bsr = np.full((T, L), np.nan)
        behav_loading_ci = np.full((T, L, 2), np.nan)
        su = sv = su_p = sv_p = np.full(L, np.nan)

    # 5. Out-of-sample cross-validation (independent RNG -> does not perturb above)
    if run_crossval:
        cv_r, cv_r2 = _crossvalidate(
            X, Y, masks, V, test_size, n_test_split, svd_method, seed + 1)
        cv_r_mean = np.nanmean(cv_r, axis=1)
        cv_r2_mean = np.nanmean(cv_r2, axis=1)
    else:
        cv_r = np.full((T, 0), np.nan)
        cv_r2 = np.full((T, 0), np.nan)
        cv_r_mean = np.full(T, np.nan)
        cv_r2_mean = np.full(T, np.nan)

    logger.info(
        "PLS done. %d LVs; LV1 covexp=%.1f%%, p=%.4f, split-half u/v=%.2f/%.2f%s",
        L, 100 * varexp[0], pvals[0], su[0], sv[0],
        ("" if not run_crossval else f", CV r(LV-indep) mean=%.2f" % np.nanmean(cv_r_mean)),
    )

    return {
        "U": U, "S": S, "V": V,
        "varexp": varexp,
        "brain_scores": brain_scores,
        "behav_scores": behav_scores,
        "pvals": pvals,
        "perm_S": perm_S,
        "U_bsr": U_bsr, "V_bsr": V_bsr,
        "split_ucorr": su, "split_vcorr": sv,
        "split_u_p": su_p, "split_v_p": sv_p,
        "L": L, "n": n,
        # ---- behavior reliability done correctly (use these, not V_bsr) ----
        "behav_loadings": behav_loadings,           # (T, L) corr(behavior, brain score)
        "behav_loading_bsr": behav_loading_bsr,     # (T, L) loading / bootstrap SE
        "behav_loading_ci": behav_loading_ci,       # (T, L, 2) 95% percentile CI
        "behav_salience_degenerate": behav_salience_degenerate,
        # ---- extension outputs ----
        "n_groups": n_groups,
        "group_labels": np.asarray(group_labels),
        "behav_per_group": T,                  # behavior cols per group (V rows = n_groups*T)
        "svd_method": svd_method,
        "cv_pearson_r": cv_r,
        "cv_r_squared": cv_r2,
        "cv_r_mean": cv_r_mean,
        "cv_r2_mean": cv_r2_mean,
    }


# --------------------------------------------------------------------------- #
# Dissociation test: compare brain salience patterns across two runs posthoc
# --------------------------------------------------------------------------- #
def compare_brain_saliences(
    U_a: np.ndarray,
    U_b: np.ndarray,
    lv: int = 0,
    nperm: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Similarity of two brain salience maps for a given LV (e.g. inner vs outer).

    Only meaningful when both runs used the SAME brain representation (same
    feature ordering). Salience signs are arbitrary in PLS, so similarity is the
    absolute correlation, two-sided. The null permutes one map's feature labels.

    Caveat: this label-permutation null ignores spatial/edge autocorrelation. For
    a parcel-level brain map a spin test (e.g. neuromaps) is more rigorous; for
    system blocks the feature count is small, so treat p as descriptive. A
    complementary approach is a bootstrap CI on the similarity itself.
    """
    a, b = U_a[:, lv], U_b[:, lv]
    obs = abs(_corr(a, b))
    rng = np.random.default_rng(seed)
    null = np.array([abs(_corr(a, b[rng.permutation(len(b))])) for _ in range(nperm)])
    p = (np.sum(null >= obs) + 1) / (nperm + 1)
    return {"abs_corr": obs, "p_perm": float(p), "lv": lv}
