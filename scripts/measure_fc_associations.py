"""Association of each inner/outer measure with functional connectivity.

For every measure the Pearson correlation with each of the 4950 Schaefer-100
edges is computed across the subjects with both survey and FC data (one
symmetric 100 x 100 matrix per measure). The measures are then compared
through their edge-wise association profiles:

* similarity matrix (Pearson r between profiles), compared with the
  correlation matrix of the measures themselves. With 95 subjects and 4950
  edges the profile similarity reproduces the score correlations even for
  unrelated FC, so the observed values are tested against a null in which
  the FC matrices are shuffled across subjects;
* contrast of the similarity within versus between the inner/outer groups and
  within versus between the valence groups (valence = sign of the loading on
  PC1 of the measures);
* PCA of the association profiles (measures as observations), compared with
  the PCA of the scores.

Output: ``results/measure_analyses/`` (``measure_fc_corr.npy``,
``similarity_*.csv``, ``fc_association_summary.json``) and
``results/plots/measure_analyses/{png,pdf}/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omegaconf import DictConfig

from insideout.data.loaders import load_fc_data
from insideout.decomposition import run_pca
from insideout.measures import (
    load_measures,
    measure_fc_correlations,
    similarity_contrast,
)
from insideout.viz import configure_plot_style

ROOT_DIR = Path(__file__).resolve().parents[1]
configure_plot_style(ROOT_DIR / "configs" / "plotting" / "default.yaml")


def _save(fig: plt.Figure, plots: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        (plots / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(plots / ext / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def _heatmap(ax, M: np.ndarray, names: list[str], title: str) -> None:
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=14)
    ax.set_yticks(range(len(names)), names, fontsize=14)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)


@hydra.main(config_path="../configs", config_name="measure_analyses", version_base=None)
def main(cfg: DictConfig) -> None:
    m = load_measures(
        str(ROOT_DIR / cfg.io.mh_data),
        str(ROOT_DIR / "configs" / "components" / "default.yaml"),
    )
    fc = load_fc_data(
        str(ROOT_DIR / cfg.io.fc_data), fc_ids_filepath=str(ROOT_DIR / cfg.io.fc_ids)
    )
    order = {s: i for i, s in enumerate(m.subjects)}
    keep = [i for i, s in enumerate(fc["subjects"]) if s in order]
    fc_sub = fc["data"][keep]
    X_sub = m.X[[order[s] for s in fc["subjects"][keep]]]
    print(f"{len(keep)} subjects with survey and FC data")

    # association of each measure with each edge
    C = measure_fc_correlations(X_sub, fc_sub)
    iu = np.triu_indices(C.shape[1], 1)
    profiles = C[:, iu[0], iu[1]]  # (14, 4950)
    out_dir = ROOT_DIR / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "measure_fc_corr.npy", C)
    print(
        f"edge-wise r: mean {profiles.mean():.3f}, sd {profiles.std():.3f}, "
        f"range [{profiles.min():.2f}, {profiles.max():.2f}]"
    )
    mean_by_measure = dict(zip(m.names, profiles.mean(1).round(3)))
    print("mean edge r per measure:", mean_by_measure)

    # similarity of the association profiles vs similarity of the scores
    S_brain = np.corrcoef(profiles)
    S_score_all = np.corrcoef(m.X, rowvar=False)  # Fig. 2b (all subjects)
    S_score_sub = np.corrcoef(X_sub, rowvar=False)  # same subjects as the FC
    for name, S in (("brain", S_brain), ("scores", S_score_all)):
        pl.DataFrame(S, schema=m.names).write_csv(out_dir / f"similarity_{name}.csv")

    summary: dict = {"n_subjects": len(keep)}
    iu_m = np.triu_indices(len(m.names), 1)
    r_obs = float(np.corrcoef(S_score_sub[iu_m], S_brain[iu_m])[0, 1])
    summary["r_scores_fc_subjects_vs_profile_similarity"] = r_obs
    summary["r_scores_all_vs_profile_similarity"] = float(
        np.corrcoef(S_score_all[iu_m], S_brain[iu_m])[0, 1]
    )

    # groupings: inner/outer and valence (sign of PC1 loading)
    pca = run_pca(pl.DataFrame(m.X, schema=m.columns))
    valence = np.where(pca.loadings["PC1"].to_numpy() > 0, "neg", "pos")
    for gname, labels in (("inner_outer", m.group), ("valence", valence)):
        for sname, S in (("scores", S_score_all), ("brain", S_brain)):
            summary[f"contrast_{gname}_{sname}"] = similarity_contrast(
                S, labels, cfg.n_perm, cfg.seed
            )

    # null: FC matrices shuffled across subjects (profiles keep their
    # dependence on the scores through the subject structure of the FC only)
    rng = np.random.default_rng(cfg.seed)
    n_null = min(cfg.n_perm, 2000)
    null_r = np.empty(n_null)
    null_val = np.empty(n_null)
    null_io = np.empty(n_null)
    same_val = (valence[:, None] == valence[None, :])[iu_m]
    same_io = (m.group[:, None] == m.group[None, :])[iu_m]
    for i in range(n_null):
        Cn = measure_fc_correlations(X_sub, fc_sub[rng.permutation(len(keep))])
        Sn = np.corrcoef(Cn[:, iu[0], iu[1]])[iu_m]
        null_r[i] = np.corrcoef(S_score_sub[iu_m], Sn)[0, 1]
        null_val[i] = Sn[same_val].mean() - Sn[~same_val].mean()
        null_io[i] = Sn[same_io].mean() - Sn[~same_io].mean()
    b = S_brain[iu_m]
    obs = {
        "r_with_scores": (r_obs, null_r),
        "valence_contrast": (b[same_val].mean() - b[~same_val].mean(), null_val),
        "inner_outer_contrast": (b[same_io].mean() - b[~same_io].mean(), null_io),
    }
    summary["shuffled_fc_null"] = {
        k: {
            "observed": float(o),
            "null_mean": float(n.mean()),
            "null_95pct": float(np.percentile(n, 95)),
            "p_greater": float((1 + (n >= o).sum()) / (1 + n_null)),
        }
        for k, (o, n) in obs.items()
    }

    # PCA of the association profiles (rows = measures, columns = edges)
    prof_pca = run_pca(pl.DataFrame(profiles.T, schema=m.columns), is_z_score=False)
    summary["profile_pca_explained"] = prof_pca.explained_variance_ratio[:5]
    summary["score_pca_explained"] = pca.explained_variance_ratio[:5]
    # PCA of the profiles: scores = projection of each *edge*; the measure
    # coordinates are the loadings, comparable to the loadings of the score PCA
    for j in (1, 2):
        a = pca.loadings[f"PC{j}"].to_numpy()
        b = prof_pca.loadings[f"PC{j}"].to_numpy()
        summary[f"loading_corr_pc{j}"] = float(abs(np.corrcoef(a, b)[0, 1]))
    summary["profile_pc1_loadings"] = dict(
        zip(m.names, prof_pca.loadings["PC1"].to_numpy().round(3).tolist())
    )
    summary["profile_pc2_loadings"] = dict(
        zip(m.names, prof_pca.loadings["PC2"].to_numpy().round(3).tolist())
    )
    summary["valence_labels"] = dict(zip(m.names, valence.tolist()))
    (out_dir / "fc_association_summary.json").write_text(json.dumps(summary, indent=2))

    # figures
    plots = ROOT_DIR / cfg.plots_dir
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    _heatmap(axes[0], S_score_all, m.names, "Similarity of scores (Fig. 2b)")
    _heatmap(axes[1], S_brain, m.names, "Similarity of FC association profiles")
    fig.tight_layout()
    _save(fig, plots, "similarity_scores_vs_brain")

    nrow, ncol = 2, 7
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    lim = np.abs(C).max()
    for ax, mat, name in zip(axes.ravel(), C, m.names):
        ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(name, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    _save(fig, plots, "measure_fc_corr_matrices")

    for k, v in summary.items():
        print(
            k,
            v
            if not isinstance(v, dict)
            else {
                a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()
            },
        )


if __name__ == "__main__":
    main()
