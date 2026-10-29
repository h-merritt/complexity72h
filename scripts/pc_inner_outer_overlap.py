"""Overlap of each PC of the measures with the inner/outer distinction.

PCA is fitted on the 14 inner and outer measures, in two orientations: the
scores as recorded (``raw``) and with the measures listed under
``reverse_scored`` in the config multiplied by -1 so that higher means better
for all measures (``reverse-scored``). The sign of a loading depends on the
orientation, so the two can give different results.

For each PC the sign of the loadings splits the measures into two
pseudo-clusters, which are compared with the inner/outer grouping by ARI and
AMI (permutation p-value for the ARI). The loading values of the inner and the
outer measures are also compared with a Mann-Whitney test. A high ARI means the
PC separates inner from outer measures; a value near zero means it cuts across
the two groups (for example by valence).

Output: ``results/measure_analyses/pc_inner_outer_overlap.csv``
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from insideout.decomposition import run_pca
from insideout.measures import load_measures, loading_group_test, pc_group_overlap

ROOT_DIR = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="measure_analyses", version_base=None)
def main(cfg: DictConfig) -> None:
    m = load_measures(
        str(ROOT_DIR / cfg.io.mh_data),
        str(ROOT_DIR / "configs" / "components" / "default.yaml"),
    )
    flip = np.array([n in set(cfg.reverse_scored) for n in m.names])
    orientations = {"raw": m.X, "reverse-scored": m.X * np.where(flip, -1.0, 1.0)}

    frames = []
    for name, X in orientations.items():
        res = run_pca(pl.DataFrame(X, schema=m.columns))
        L = res.loadings.select([c for c in res.loadings.columns if c != "variable"])
        L = L.to_numpy()
        overlap = pc_group_overlap(L, m.group, cfg.n_perm, cfg.seed)
        test = loading_group_test(L, m.group).drop("pc")
        out = pl.concat([overlap, test], how="horizontal").with_columns(
            pl.Series("explained_variance", res.explained_variance_ratio[: len(L[0])]),
            pl.lit(name).alias("orientation"),
        )
        frames.append(out)
        print(f"\n--- {name} ---")
        print(out)

    out = pl.concat(frames)
    out_dir = ROOT_DIR / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_dir / "pc_inner_outer_overlap.csv")


if __name__ == "__main__":
    main()
