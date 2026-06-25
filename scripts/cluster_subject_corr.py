import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")

import numpy as np
import polars as pl
import yaml
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from insideout.data.loaders import load_survey_data
from insideout.viz import configure_plot_style, plot_cluster_correlation_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]


@hydra.main(
    config_path="../configs",
    config_name="subject_corr",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    configure_plot_style(ROOT_DIR / "configs" / "plotting" / "default.yaml")

    k = cfg.k
    subset_names = cfg.subsets

    parquet_path = (
        ROOT_DIR / "results" / "clustering" / "survey_clustered_all_k.parquet"
    )
    master_df = pl.read_parquet(parquet_path)

    data = load_survey_data(
        csv_path=str(ROOT_DIR / "data" / "hcp_social_mentahealth_data.csv"),
        yaml_config_path=str(ROOT_DIR / "configs" / "components" / "default.yaml"),
        drop_na=True,
    )
    subjects = data["subjects"]

    with open(ROOT_DIR / "configs" / "components" / "default.yaml") as f:
        components_cfg = yaml.safe_load(f)

    out_base = ROOT_DIR / "results" / "plots" / "clustering" / "subject_corr"

    for subset_name in tqdm(subset_names, desc="Subset"):
        membership_col = f"Membership_{subset_name}_k{k}"

        if membership_col not in master_df.columns:
            tqdm.write(f"  Skipping {subset_name}: column {membership_col} not found")
            continue

        membership = json.loads(master_df[membership_col][0])

        subject_to_cluster = {}
        for cid_str, ids in membership.items():
            for sid in ids:
                subject_to_cluster[sid] = int(cid_str)

        labels = np.array([subject_to_cluster[s] for s in subjects])

        subset_blocks = _resolve_subset_blocks(subset_name)

        dfs = []
        for block in subset_blocks:
            cols = [e["column"] for e in components_cfg[block]]
            dfs.append(data[block].select(cols))
        X = pl.concat(dfs, how="horizontal").drop_nulls().to_numpy()
        X_scaled = StandardScaler().fit_transform(X)

        plot_cluster_correlation_matrix(
            X_scaled,
            labels,
            subset_name,
            k,
            out_base / subset_name.lower() / "specific",
        )

        if "mental_health" not in subset_blocks:
            mh_cols = [e["column"] for e in components_cfg["mental_health"]]
            mh_data = data["mental_health"].select(mh_cols).to_numpy()
            mh_scaled = StandardScaler().fit_transform(mh_data)

            plot_cluster_correlation_matrix(
                mh_scaled,
                labels,
                subset_name,
                k,
                out_base / subset_name.lower() / "mh_only",
            )

    print(f"Subject correlation matrices saved to {out_base}/{{png,pdf}}/")


def _resolve_subset_blocks(name: str) -> list[str]:
    mapping = {
        "Inner": ["inner"],
        "Outer": ["outer"],
        "Mental_Health": ["mental_health"],
        "Combined_2": ["inner", "outer"],
        "Combined_3": ["inner", "outer", "mental_health"],
    }
    return mapping[name]


if __name__ == "__main__":
    main()
