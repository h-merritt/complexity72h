import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig


from insideout.io import load_hcp_data
from insideout.graph_models import compute_correlation
from insideout.viz import (
    plot_combined_heatmap,
    plot_correlation_heatmap,
    plot_group_distributions,
    save_fig,
)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    inner: list[str] = list(cfg.components.inner)
    outer: list[str] = list(cfg.components.outer)
    inner_color: str = cfg.components.colors.inner
    outer_color: str = cfg.components.colors.outer
    out_dir: str = cfg.io.eda_plots

    df = load_hcp_data(cfg.io.mh_data)

    # ── Horizontal distribution plots (violin + box + strip) ──────────────────
    for group_name, cols, color in [
        ("Inner", inner, inner_color),
        ("Outer", outer, outer_color),
    ]:
        X = df.select(cols).drop_nulls().to_numpy()
        fig = plot_group_distributions(X, cols, color=color, group_name=group_name)
        save_fig(fig, f"distributions_{group_name.lower()}", out_dir)
        plt.close(fig)
        print(f"saved distributions_{group_name.lower()}")

    # ── Correlation clustermaps (no numbers, hierarchical dendrogram) ─────────
    for label, cols in [("inner", inner), ("outer", outer)]:
        X = df.select(cols).drop_nulls().to_numpy()
        fig = plot_correlation_heatmap(compute_correlation(X), cols)
        save_fig(fig, f"correlation_{label}", out_dir)
        plt.close(fig)
        print(f"saved correlation_{label}")

    # ── Combined inner + outer (colour-coded labels) ──────────────────────────
    cols = inner + outer
    X = df.select(cols).drop_nulls().to_numpy()
    fig = plot_combined_heatmap(
        compute_correlation(X),
        inner,
        outer,
        inner_color=inner_color,
        outer_color=outer_color,
    )
    save_fig(fig, "correlation_combined", out_dir)
    plt.close(fig)
    print("saved correlation_combined")

    print(f"\nAll figures written to {out_dir}/{{png,pdf}}/")


if __name__ == "__main__":
    main()
