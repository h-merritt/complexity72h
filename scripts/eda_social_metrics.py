"""Exploratory plots of the inner/outer survey variables.

Draws the distributions of all inner and outer variables and their correlation
heatmaps (inner, outer and combined, the latter also reordered by group).

Output: ``results/plots/eda/{png,pdf}/``
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from insideout.data.loaders import load_survey_data
from insideout.graph_models import compute_correlation
from insideout.viz import (
    configure_plot_style,
    plot_combined_distributions,
    plot_combined_heatmap,
    plot_correlation_heatmap,
    save_fig,
)


def _cols_and_names(entries) -> tuple[list[str], list[str]]:
    """Return (column_names, display_names) from a list of {column, name} entries."""
    return [e.column for e in entries], [e.name for e in entries]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    configure_plot_style(ROOT_DIR / "configs" / "plotting" / "default.yaml")
    inner_cols, inner_names = _cols_and_names(cfg.components.inner)
    outer_cols, outer_names = _cols_and_names(cfg.components.outer)
    inner_color: str = cfg.components.colors.inner
    outer_color: str = cfg.components.colors.outer
    out_dir: str = str(ROOT_DIR / cfg.io.eda_plots)

    subject_col: str = cfg.components.metadata.subject_id_col

    data = load_survey_data(
        str(ROOT_DIR / cfg.io.mh_data),
        str(ROOT_DIR / "configs" / "components" / "default.yaml"),
    )
    df_inner = data["inner"]
    df_outer = data["outer"]

    # ── Combined distribution plot (all Inner + Outer metrics) ───────────────
    all_cols = inner_cols + outer_cols
    X = (
        df_inner.join(df_outer, on=subject_col, how="full")
        .select(all_cols)
        .drop_nulls()
        .to_numpy()
    )
    fig = plot_combined_distributions(
        X, inner_names, outer_names, inner_color, outer_color
    )
    save_fig(fig, "distributions_combined", out_dir)
    plt.close(fig)
    print("saved distributions_combined")

    # ── Individual correlation heatmaps ─────────────────────────────────
    X_inner = df_inner.select(inner_cols).drop_nulls().to_numpy()
    X_outer = df_outer.select(outer_cols).drop_nulls().to_numpy()

    fig = plot_correlation_heatmap(
        compute_correlation(X_inner), inner_names, title="Inner"
    )
    save_fig(fig, "correlation_inner", out_dir)
    plt.close(fig)
    print("saved correlation_inner")

    fig = plot_correlation_heatmap(
        compute_correlation(X_outer), outer_names, title="Outer"
    )
    save_fig(fig, "correlation_outer", out_dir)
    plt.close(fig)
    print("saved correlation_outer")

    inner_order = [
        "Openness",
        "Positive Affect",
        "Self Efficacy",
        "Agreeableness",
        "Conscientiousness",
        "Extraversion",
        "Neuroticism",
    ]
    outer_order = [
        "Loneliness",
        "Perceived Hostility",
        "Perceived Rejection",
        "Perceived Stress",
        "Friendship",
        "Emotional Support",
        "Instrumental Support",
    ]

    inner_plot_cols = [inner_cols[inner_names.index(n)] for n in inner_order]
    outer_plot_cols = [outer_cols[outer_names.index(n)] for n in outer_order]
    all_ordered_cols = inner_plot_cols + outer_plot_cols

    X_ordered = (
        df_inner.join(df_outer, on=subject_col, how="full")
        .select(all_ordered_cols)
        .drop_nulls()
        .to_numpy()
    )

    fig = plot_combined_heatmap(
        compute_correlation(X_ordered),
        inner_order,
        outer_order,
        inner_color=inner_color,
        outer_color=outer_color,
        row_cluster=False,
        col_cluster=False,
    )
    save_fig(fig, "correlation_combined2", out_dir)
    plt.close(fig)
    print("saved correlation_combined2")

    print(f"\nAll figures written to {out_dir}/{{png,pdf}}/")


if __name__ == "__main__":
    main()
