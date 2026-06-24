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

            labels_pct = np.array([f"{lb + 1} ({pct_map[lb]:.1f}%)" for lb in labels])
            cluster_labels = [f"{u + 1} ({pct_map[u]:.1f}%)" for u in unique]

            cluster_means = np.zeros((k, X_scaled.shape[1]))
            cluster_stds = np.zeros((k, X_scaled.shape[1]))
            for cid in range(k):
                mask = labels == cid
                if mask.sum() > 0:
                    cluster_means[cid] = X_scaled[mask].mean(axis=0)
                    cluster_stds[cid] = X_scaled[mask].std(axis=0)

            k_out_dir = out_dir / f"k{k}"

            plot_cluster_heatmap_avg_std(
                cluster_means,
                cluster_stds,
                metric_names,
                cluster_labels,
                subset_name,
                k,
                k_out_dir,
                cmap=heatmap_cmap,
            )

            plot_cluster_distributions(
                X_scaled,
                labels,
                metric_names,
                subset_name,
                k,
                k_out_dir,
            )

            pca = PCA(n_components=2, random_state=seed)
            X_pca = pca.fit_transform(X_scaled)
            plot_cluster_pca_scatter(
                X_pca,
                labels_pct,
                subset_name,
                k,
                k_out_dir,
            )

            membership = {}
            for cid in range(k):
                mask = labels == cid
                membership[str(cid)] = data["subjects"][mask].tolist()
            membership_json = json.dumps(membership)
            membership_col = f"Membership_{subset_name}_k{k}"
            master_df = master_df.with_columns(
                pl.Series(membership_col, [membership_json] * master_df.height)
            )

        plot_clustering_metrics(metrics, subset_name, out_dir)

    parquet_path = csv_dir / "survey_clustered_all_k.parquet"
    csv_dir.mkdir(parents=True, exist_ok=True)
    master_df.write_parquet(parquet_path)
    print(f"Master Parquet saved to {parquet_path}")
    print(f"All clustering figures written to {out_dir}/{{png,pdf}}/")


if __name__ == "__main__":
    main()
