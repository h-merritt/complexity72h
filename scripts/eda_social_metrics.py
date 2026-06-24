import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from omegaconf import DictConfig

from insideout.data.loaders import load_survey_data
from insideout.graph_models import compute_correlation
from insideout.viz import (
    configure_plot_style,
    plot_combined_heatmap,
    plot_correlation_heatmap,
    plot_group_distributions,
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
    out_dir: str = cfg.io.eda_plots

    data = load_survey_data(cfg.io.mh_data, "configs/components/default.yaml")
    df_inner = data["inner"]
    df_outer = data["outer"]

    # ── Horizontal distribution plots (violin + box + strip) ──────────────────
    for group_name, cols, names, color in [
        ("Inner", inner_cols, inner_names, inner_color),
        ("Outer", outer_cols, outer_names, outer_color),
    ]:
        X = (
            df_inner.select(cols).drop_nulls().to_numpy()
            if group_name == "Inner"
            else df_outer.select(cols).drop_nulls().to_numpy()
        )
        fig = plot_group_distributions(X, names, color=color, group_name=group_name)
        save_fig(fig, f"distributions_{group_name.lower()}", out_dir)
        plt.close(fig)
        print(f"saved distributions_{group_name.lower()}")

    # ── Correlation clustermaps (no numbers, hierarchical dendrogram) ─────────
    for label, cols, names in [
        ("inner", inner_cols, inner_names),
        ("outer", outer_cols, outer_names),
    ]:
        _df = df_inner if label == "inner" else df_outer
        X = _df.select(cols).drop_nulls().to_numpy()
        fig = plot_correlation_heatmap(compute_correlation(X), names)
        save_fig(fig, f"correlation_{label}", out_dir)
        plt.close(fig)
        print(f"saved correlation_{label}")

    # ── Combined inner + outer (colour-coded labels) ──────────────────────────
    all_cols = inner_cols + outer_cols
    X = (
        df_inner.join(df_outer, on="Subject", how="outer")
        .select(all_cols)
        .drop_nulls()
        .to_numpy()
    )
    fig = plot_combined_heatmap(
        compute_correlation(X),
        inner_names,
        outer_names,
        inner_color=inner_color,
        outer_color=outer_color,
    )
    save_fig(fig, "correlation_combined", out_dir)
    plt.close(fig)
    print("saved correlation_combined")

    print(f"\nAll figures written to {out_dir}/{{png,pdf}}/")


if __name__ == "__main__":
    main()
