"""PCA loadings of the measures after reverse scoring.

The measures listed under ``reverse_scored`` in the config (higher = worse) are
multiplied by -1, so that higher means better for all measures. PCA is then
refitted. Explained variance is unchanged (the sign of a variable does not
affect it); the loading signs now show the structure beyond valence.

Output: ``results/measure_analyses/pca_loadings_reverse_scored.csv``
        ``results/plots/measure_analyses/{png,pdf}/pca_loadings_reverse_scored.*``
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omegaconf import DictConfig

from insideout.decomposition import run_pca
from insideout.measures import load_measures
from insideout.viz import configure_plot_style

ROOT_DIR = Path(__file__).resolve().parents[1]
configure_plot_style(ROOT_DIR / "configs" / "plotting" / "default.yaml")


@hydra.main(config_path="../configs", config_name="measure_analyses", version_base=None)
def main(cfg: DictConfig) -> None:
    m = load_measures(
        str(ROOT_DIR / cfg.io.mh_data),
        str(ROOT_DIR / "configs" / "components" / "default.yaml"),
    )
    flip = np.array([n in set(cfg.reverse_scored) for n in m.names])
    X = m.X * np.where(flip, -1.0, 1.0)
    res = run_pca(pl.DataFrame(X, schema=m.names))

    n_pcs = cfg.n_pcs_loadings
    pcs = [f"PC{i + 1}" for i in range(n_pcs)]
    table = pl.DataFrame(
        {
            "measure": m.names,
            "group": m.group.tolist(),
            "reverse_scored": flip.tolist(),
            **{pc: res.loadings[pc].to_numpy().round(3) for pc in pcs},
        }
    )
    out_dir = ROOT_DIR / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_csv(out_dir / "pca_loadings_reverse_scored.csv")

    colors = {
        "inner": cfg.components.colors.inner,
        "outer": cfg.components.colors.outer,
    }
    labels = [f"{n} (rev.)" if f else n for n, f in zip(m.names, flip)]
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, n_pcs, figsize=(5.5 * n_pcs, 8), sharey=True)
    for ax, pc, ev in zip(np.atleast_1d(axes), pcs, res.explained_variance_ratio):
        ax.barh(y, res.loadings[pc].to_numpy(), color=[colors[g] for g in m.group])
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(y, labels)
        ax.set_title(f"{pc} ({ev * 100:.1f}\\,\\%)")
        ax.set_xlabel("Loading")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, ["Inner", "Outer"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("PCA loadings after reverse scoring")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plots = ROOT_DIR / cfg.plots_dir
    for ext in ("png", "pdf"):
        (plots / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(plots / ext / f"pca_loadings_reverse_scored.{ext}", dpi=200)
    plt.close(fig)
    print(table)


if __name__ == "__main__":
    main()
