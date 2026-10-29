"""Leave-one-measure-out PCA.

PCA is refitted on the combined, inner and outer measures with each measure
removed in turn. The output is the explained variance ratio of the first PCs
with and without the measure, and their difference. A figure shows the
difference for each PC.

Output: ``results/measure_analyses/leave_one_out_pca.csv``
        ``results/plots/measure_analyses/{png,pdf}/leave_one_out_pca.*``
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omegaconf import DictConfig

from insideout.measures import leave_one_out_pca, load_measures
from insideout.viz import configure_plot_style

ROOT_DIR = Path(__file__).resolve().parents[1]
configure_plot_style(ROOT_DIR / "configs" / "plotting" / "default.yaml")


@hydra.main(config_path="../configs", config_name="measure_analyses", version_base=None)
def main(cfg: DictConfig) -> None:
    m = load_measures(
        str(ROOT_DIR / cfg.io.mh_data),
        str(ROOT_DIR / "configs" / "components" / "default.yaml"),
    )
    sets = {
        "combined": np.ones(len(m.names), bool),
        "inner": m.group == "inner",
        "outer": m.group == "outer",
    }
    frames = []
    for name, mask in sets.items():
        res = leave_one_out_pca(m.X[:, mask], m.names_of(mask), cfg.n_pcs)
        frames.append(res.with_columns(pl.lit(name).alias("set")))
    out = pl.concat(frames)

    out_dir = ROOT_DIR / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_csv(out_dir / "leave_one_out_pca.csv")

    colors = {
        "inner": cfg.components.colors.inner,
        "outer": cfg.components.colors.outer,
    }
    group_of = dict(zip(m.names, m.group))
    plots = ROOT_DIR / cfg.plots_dir
    for name in sets:
        sub = out.filter(pl.col("set") == name)
        fig, axes = plt.subplots(1, cfg.n_pcs, figsize=(6 * cfg.n_pcs, 7), sharey=True)
        for ax, pc in zip(
            np.atleast_1d(axes), [f"PC{i + 1}" for i in range(cfg.n_pcs)]
        ):
            d = sub.filter(pl.col("pc") == pc).sort("delta")
            ax.barh(
                d["left_out"].to_list(),
                d["delta"].to_numpy() * 100,
                color=[colors[group_of[n]] for n in d["left_out"]],
            )
            ax.axvline(0, color="k", lw=1)
            ax.set_title(f"{pc} ({d['full'][0] * 100:.1f}\\,\\% with all)")
            ax.set_xlabel("Change in explained variance (pp)")
        fig.suptitle(f"Leave-one-measure-out PCA, {name} measures")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            (plots / ext).mkdir(parents=True, exist_ok=True)
            fig.savefig(plots / ext / f"leave_one_out_pca_{name}.{ext}", dpi=200)
        plt.close(fig)

    wide = out.filter(pl.col("set") == "combined")
    for pc in ("PC1", "PC2"):
        d = wide.filter(pl.col("pc") == pc).sort("delta")
        print(
            f"\n{pc} (combined, full {d['full'][0]:.2%}): largest drops / gains when removed"
        )
        print(d.head(3).select("left_out", "without", "delta"))
        print(d.tail(3).select("left_out", "without", "delta"))
    print("\nInner set, PC1:")
    print(
        out.filter((pl.col("set") == "inner") & (pl.col("pc") == "PC1"))
        .sort("delta")
        .select("left_out", "full", "without", "delta")
    )


if __name__ == "__main__":
    main()
