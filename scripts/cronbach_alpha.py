"""Cronbach's alpha of the inner and the outer measures.

Alpha is computed on the raw scores and after reverse scoring (the measures
listed under ``reverse_scored`` in the config are multiplied by -1, so that
higher means better for all measures), since the sets mix positively and
negatively valenced measures. The same orientation is used in
``pca_loadings_reverse_scored.py`` and ``pc_inner_outer_overlap.py``. The same
is done for all measures together.

Output: ``results/measure_analyses/cronbach_alpha.csv``
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

from insideout.measures import cronbach_alpha, load_measures

ROOT_DIR = Path(__file__).resolve().parents[1]


@hydra.main(config_path="../configs", config_name="measure_analyses", version_base=None)
def main(cfg: DictConfig) -> None:
    m = load_measures(
        str(ROOT_DIR / cfg.io.mh_data),
        str(ROOT_DIR / "configs" / "components" / "default.yaml"),
    )
    sets = {
        "inner": m.group == "inner",
        "outer": m.group == "outer",
        "all": m.group != "",
    }
    flip = np.array([n in set(cfg.reverse_scored) for n in m.names])
    X_rev = m.X * np.where(flip, -1.0, 1.0)
    print(f"reverse-scored items: {m.names_of(flip)}")
    rows = []
    for name, mask in sets.items():
        for scoring, data in (("raw", m.X), ("reverse-scored", X_rev)):
            res = cronbach_alpha(data[:, mask])
            rows.append(
                {"set": name, "scoring": scoring, "n_items": int(mask.sum()), **res}
            )

    out = pl.DataFrame(rows)
    print(out)
    out_dir = ROOT_DIR / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_dir / "cronbach_alpha.csv")


if __name__ == "__main__":
    main()
