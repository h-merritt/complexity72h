"""Statistical tests of cluster differences.

For every clustering subset (``clustering.subsets`` in the config) and every k
in ``2..clustering.max_k``, compares the clusters on the subset's own
variables and on the mental-health variables:

* k = 2: Welch's t-test (``equal_var=False``);
* k > 2: one-way ANOVA.

Bonferroni correction is applied within each (subset, k) over all tests of
that pair (subset variables + mental-health variables). Note that the
subset's own variables were used to build the clusters, so their p-values
are circular and only the mental-health rows are informative.

Output: ``results/clustering/statistical_tests.{parquet,csv}``
Requires: ``scripts/clustering.py`` has been run.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from scipy.stats import f_oneway, ttest_ind

ROOT_DIR = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT_DIR / "results" / "clustering" / "survey_clustered_all_k.parquet"
OUT_PATH = ROOT_DIR / "results" / "clustering" / "statistical_tests.parquet"
CSV_PATH = ROOT_DIR / "results" / "clustering" / "statistical_tests.csv"

MH_BLOCK = "mental_health"


def subset_display_name(blocks: list[str]) -> str:
    """Display name of a subset from its blocks, e.g. ``["inner", "outer"]`` -> ``"Inner & Outer"``."""
    return " & ".join(b.capitalize() for b in blocks)


def _block_measures(components, block: str) -> list[tuple[str, str]]:
    """``(display name, column)`` pairs of a component block."""
    return [(entry.name, entry.column) for entry in components[block]]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    subject_col = cfg.components.metadata.subject_id_col
    max_k = cfg.clustering.max_k

    survey = pl.read_csv(ROOT_DIR / cfg.io.mh_data, null_values=["NA", ""])
    if "" in survey.columns:
        survey = survey.drop("")

    master = pl.read_parquet(PARQUET_PATH)

    rows: list[dict] = []

    for parquet_key, subset_cfg in cfg.clustering.subsets.items():
        blocks = list(subset_cfg.blocks)
        display_name = subset_display_name(blocks)

        measures: list[tuple[str, str]] = []
        for block in blocks:
            measures += _block_measures(cfg.components, block)
        if MH_BLOCK not in blocks:
            measures += _block_measures(cfg.components, MH_BLOCK)

        for k in range(2, max_k + 1):
            col = f"Membership_{parquet_key}_k{k}"
            if col not in master.columns:
                continue

            mapping = json.loads(master[col][0])

            for metric_name, metric_col in measures:
                groups: list[np.ndarray] = []
                for subject_list in mapping.values():
                    vals = (
                        survey.filter(pl.col(subject_col).is_in(subject_list))[
                            metric_col
                        ]
                        .drop_nulls()
                        .to_numpy()
                    )
                    if len(vals) > 0:
                        groups.append(vals)

                if len(groups) < 2 or any(len(g) < 2 for g in groups):
                    continue

                if k == 2:
                    stat, pval = ttest_ind(*groups, equal_var=False)
                else:
                    stat, pval = f_oneway(*groups)

                rows.append(
                    {
                        "subset": display_name,
                        "k": k,
                        "metric": metric_name,
                        "stat": float(stat),
                        "pval": float(pval),
                    }
                )

    result = pl.DataFrame(
        rows,
        schema={
            "subset": pl.String,
            "k": pl.Int64,
            "metric": pl.String,
            "stat": pl.Float64,
            "pval": pl.Float64,
        },
    )

    n_tests = result.group_by("subset", "k").agg(pl.len().alias("n"))
    result = result.join(n_tests, on=["subset", "k"])
    result = result.with_columns(
        pl.min_horizontal(pl.col("pval") * pl.col("n"), pl.lit(1.0)).alias("pval_bonf")
    ).drop("n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(OUT_PATH)
    print(f"Written {OUT_PATH}  ({result.height} rows)")

    result.write_csv(CSV_PATH)
    print(f"Written {CSV_PATH}  ({result.height} rows)")

    print(result.head(12).to_pandas().to_string(index=False))


if __name__ == "__main__":
    main()
