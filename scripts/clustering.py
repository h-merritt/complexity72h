import matplotlib

matplotlib.use("Agg")

import json
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

from insideout.data.loaders import load_survey_data
from insideout.viz import (
    configure_plot_style,
    plot_clustering_metrics,
    plot_cluster_heatmap_avg_std,
    plot_cluster_distributions,
    plot_cluster_pca_scatter,
    plot_cluster_means_bars,
    plot_cluster_means_lollipop,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
configure_plot_style(ROOT_DIR / "configs" / "plotting" / "default.yaml")


def _resolve_block_columns(
    components, blocks: list[str]
) -> tuple[list[str], list[str]]:
    cols: list[str] = []
    names: list[str] = []
    for block_name in blocks:
        for entry in components[block_name]:
            cols.append(entry.column)
            names.append(entry.name)
    return cols, names


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    data = load_survey_data(
        csv_path=str(ROOT_DIR / cfg.io.mh_data),
        yaml_config_path=str(ROOT_DIR / "configs" / "components" / "default.yaml"),
        drop_na=True,
    )

    out_dir = ROOT_DIR / "results" / "plots" / "clustering"
    csv_dir = ROOT_DIR / "results" / "clustering"
    csv_dir.mkdir(parents=True, exist_ok=True)

    subject_col = cfg.components.metadata.subject_id_col
    master_df = pl.DataFrame({subject_col: data["subjects"]})

    max_k = cfg.clustering.max_k
    seed = cfg.clustering.seed
    n_init = cfg.clustering.n_init
    heatmap_cmap = cfg.clustering.heatmap_cmap

    subset_names = list(cfg.clustering.subsets.keys())
    results: dict[tuple[str, int], dict] = {}
    mh_results: dict[tuple[str, int], dict] = {}
    metrics_dicts: dict[str, dict] = {}
    all_means_by_k: dict[int, list[np.ndarray]] = {}

    # ---- Phase 1: Compute ----
    for subset_name, subset_cfg in tqdm(
        cfg.clustering.subsets.items(), desc="Subset", unit="subset"
    ):
        blocks = list(subset_cfg.blocks)
        cols, metric_names = _resolve_block_columns(cfg.components, blocks)

        dfs = []
        for block_name in blocks:
            block_cols = [e.column for e in cfg.components[block_name]]
            dfs.append(data[block_name].select(block_cols))
        X = pl.concat(dfs, how="horizontal").drop_nulls().to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        metrics: dict[str, list] = {
            "k": [],
            "Inertia": [],
            "Silhouette": [],
            "CH_Index": [],
            "DB_Index": [],
        }

        for k in tqdm(
            range(2, max_k + 1), desc=f"  {subset_name}", unit="k", leave=False
        ):
            kmeans = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
            labels = kmeans.fit_predict(X_scaled)

            metrics["k"].append(k)
            metrics["Inertia"].append(kmeans.inertia_)
            metrics["Silhouette"].append(silhouette_score(X_scaled, labels))
            metrics["CH_Index"].append(calinski_harabasz_score(X_scaled, labels))
            metrics["DB_Index"].append(davies_bouldin_score(X_scaled, labels))

            unique, counts = np.unique(labels, return_counts=True)
            percentages = counts / len(labels) * 100
            pct_map = {u: p for u, p in zip(unique, percentages)}

            labels_pct = np.array([f"{lb + 1} ({pct_map[lb]:.1f}\\%)" for lb in labels])
            cluster_labels = [f"{u + 1} ({pct_map[u]:.1f}\\%)" for u in unique]

            cluster_means = np.zeros((k, X_scaled.shape[1]))
            cluster_stds = np.zeros((k, X_scaled.shape[1]))
            cluster_sizes = np.zeros(k, dtype=int)
            for cid in range(k):
                mask = labels == cid
                cluster_sizes[cid] = mask.sum()
                if mask.sum() > 0:
                    cluster_means[cid] = X_scaled[mask].mean(axis=0)
                    cluster_stds[cid] = X_scaled[mask].std(axis=0)

            k_out_dir = out_dir / f"k{k}" / subset_name.lower()
            specific_out_dir = k_out_dir / "specific"

            results[(subset_name, k)] = {
                "cluster_means": cluster_means,
                "cluster_stds": cluster_stds,
                "cluster_sizes": cluster_sizes,
                "cluster_labels": cluster_labels,
                "labels": labels,
                "labels_pct": labels_pct,
                "metric_names": metric_names,
                "X_scaled": X_scaled,
                "specific_out_dir": specific_out_dir,
                "k_out_dir": k_out_dir,
            }

            if k not in all_means_by_k:
                all_means_by_k[k] = []
            all_means_by_k[k].append(cluster_means)

            # --- MH block ---
            if "mental_health" not in blocks:
                mh_cols = [e.column for e in cfg.components["mental_health"]]
                mh_names = [e.name for e in cfg.components["mental_health"]]
                mh_data = data["mental_health"].select(mh_cols).to_numpy()

                mh_scaler = StandardScaler()
                mh_scaled = mh_scaler.fit_transform(mh_data)

                mh_cluster_means = np.zeros((k, mh_scaled.shape[1]))
                mh_cluster_stds = np.zeros((k, mh_scaled.shape[1]))
                mh_cluster_sizes = np.zeros(k, dtype=int)
                for cid in range(k):
                    mh_mask = labels == cid
                    mh_cluster_sizes[cid] = mh_mask.sum()
                    if mh_mask.sum() > 0:
                        mh_cluster_means[cid] = mh_scaled[mh_mask].mean(axis=0)
                        mh_cluster_stds[cid] = mh_scaled[mh_mask].std(axis=0)

                mh_out_dir = k_out_dir / "mh_only"

                mh_results[(subset_name, k)] = {
                    "cluster_means": mh_cluster_means,
                    "cluster_stds": mh_cluster_stds,
                    "cluster_sizes": mh_cluster_sizes,
                    "labels": labels,
                    "labels_pct": labels_pct,
                    "cluster_labels": cluster_labels,
                    "metric_names": mh_names,
                    "X_scaled": mh_scaled,
                    "out_dir": mh_out_dir,
                }

                all_means_by_k[k].append(mh_cluster_means)

            # Save membership to parquet
            membership = {}
            for cid in range(k):
                mask = labels == cid
                membership[str(cid)] = data["subjects"][mask].tolist()
            membership_json = json.dumps(membership)
            membership_col = f"Membership_{subset_name}_k{k}"
            master_df = master_df.with_columns(
                pl.Series(membership_col, [membership_json] * master_df.height)
            )

        metrics_dicts[subset_name] = metrics

    # ---- Phase 2: Plot with global vlim per k ----
    for k in tqdm(range(2, max_k + 1), desc="Plot", unit="k"):
        means_list = all_means_by_k[k]
        min_vals = [a.min() for a in means_list]
        max_vals = [a.max() for a in means_list]
        global_vlim = max(abs(min(min_vals)), abs(max(max_vals)))

        for subset_name in subset_names:
            r = results.get((subset_name, k))
            if r is None:
                continue

            plot_cluster_heatmap_avg_std(
                r["cluster_means"],
                r["metric_names"],
                r["cluster_labels"],
                subset_name,
                k,
                r["specific_out_dir"],
                cmap=heatmap_cmap,
                vmin=-global_vlim,
                vmax=global_vlim,
            )
            plot_cluster_means_bars(
                r["cluster_means"],
                r["cluster_stds"],
                r["cluster_sizes"],
                r["metric_names"],
                r["cluster_labels"],
                subset_name,
                k,
                r["specific_out_dir"],
            )
            plot_cluster_means_lollipop(
                r["cluster_means"],
                r["metric_names"],
                r["cluster_labels"],
                subset_name,
                k,
                r["specific_out_dir"],
            )
            plot_cluster_distributions(
                r["X_scaled"],
                r["labels"],
                r["metric_names"],
                subset_name,
                k,
                r["specific_out_dir"],
            )

            pca = PCA(n_components=2, random_state=seed)
            X_pca = pca.fit_transform(r["X_scaled"])
            plot_cluster_pca_scatter(
                X_pca,
                r["labels_pct"],
                subset_name,
                k,
                r["specific_out_dir"],
            )

            # MH-only plots for this subset
            mh_r = mh_results.get((subset_name, k))
            if mh_r is not None:
                pca = PCA(n_components=2, random_state=seed)
                mh_X_pca = pca.fit_transform(mh_r["X_scaled"])

                plot_cluster_heatmap_avg_std(
                    mh_r["cluster_means"],
                    mh_r["metric_names"],
                    mh_r["cluster_labels"],
                    subset_name,
                    k,
                    mh_r["out_dir"],
                    cmap=heatmap_cmap,
                    vmin=-global_vlim,
                    vmax=global_vlim,
                )
                plot_cluster_means_bars(
                    mh_r["cluster_means"],
                    mh_r["cluster_stds"],
                    mh_r["cluster_sizes"],
                    mh_r["metric_names"],
                    mh_r["cluster_labels"],
                    subset_name,
                    k,
                    mh_r["out_dir"],
                )
                plot_cluster_means_lollipop(
                    mh_r["cluster_means"],
                    mh_r["metric_names"],
                    mh_r["cluster_labels"],
                    subset_name,
                    k,
                    mh_r["out_dir"],
                )
                plot_cluster_distributions(
                    mh_r["X_scaled"],
                    mh_r["labels"],
                    mh_r["metric_names"],
                    subset_name,
                    k,
                    mh_r["out_dir"],
                )
                plot_cluster_pca_scatter(
                    mh_X_pca,
                    mh_r["labels_pct"],
                    subset_name,
                    k,
                    mh_r["out_dir"],
                )

    # ---- Phase 3: Evaluation metrics ----
    for subset_name in subset_names:
        plot_clustering_metrics(
            metrics_dicts[subset_name],
            subset_name,
            out_dir / "fitness" / subset_name.lower(),
        )

    parquet_path = csv_dir / "survey_clustered_all_k.parquet"
    csv_dir.mkdir(parents=True, exist_ok=True)
    master_df.write_parquet(parquet_path)
    print(f"Master Parquet saved to {parquet_path}")
    print(f"All clustering figures written to {out_dir}/{{png,pdf}}/")


if __name__ == "__main__":
    main()
