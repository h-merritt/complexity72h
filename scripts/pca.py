# -*- coding: utf-8 -*-
"""PCA Analysis - Repository Executable Script for complexity72h"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from insideout.data.loaders import load_survey_data
from insideout.decomposition import run_pca
from insideout.viz import (
    configure_plot_style,
    plot_scatter,
    plot_top_loadings,
    plot_variance,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
configure_plot_style(ROOT_DIR / "configs" / "plotting" / "default.yaml")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Loading data with YAML configuration...")
    data = load_survey_data(
        csv_path=str(ROOT_DIR / cfg.io.mh_data),
        yaml_config_path=str(ROOT_DIR / "configs" / "components" / "default.yaml"),
        drop_na=True,
    )

    out_dir = ROOT_DIR / "results" / "plots" / "pca"
    subject_col = cfg.components.metadata.subject_id_col
    meta_cols = list(cfg.components.metadata.values())

    df_inner = data["inner"]
    df_outer = data["outer"]
    inner_vars = [c for c in df_inner.columns if c not in meta_cols]
    outer_vars = [c for c in df_outer.columns if c not in meta_cols]

    df_combined = df_inner.join(df_outer, on=subject_col, how="full").select(
        inner_vars + outer_vars
    )

    groups = [
        (df_combined, "Combined", 10, 6),
        (df_inner.select(inner_vars), "Inner", len(inner_vars), 2),
        (df_outer.select(outer_vars), "Outer", len(outer_vars), 2),
    ]

    for df, label, top_n, n_pcs in groups:
        print(f"\n--- {label} PCA ---")
        result = run_pca(df)
        for i, r in enumerate(result.explained_variance_ratio, 1):
            print(f"PC{i}: {r:.2%}")

        plot_variance(result, label, out_dir)
        plot_scatter(result, label, out_dir)
        for pc in range(1, n_pcs + 1):
            plot_top_loadings(
                result.loadings, pc, inner_vars, outer_vars, top_n, label, out_dir
            )


if __name__ == "__main__":
    main()
