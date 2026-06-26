from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from scipy.stats import f_oneway, ttest_ind

ROOT = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT / "configs" / "components" / "default.yaml"
PARQUET_PATH = ROOT / "results" / "clustering" / "survey_clustered_all_k.parquet"
OUT_PATH = ROOT / "results" / "clustering" / "statistical_tests.parquet"
DATA_PATH = ROOT / "data" / "hcp_social_mentahealth_data.csv"

SUBSET_MAP = {
    "Inner": "Inner",
    "Outer": "Outer",
    "Inner & Outer": "Combined_2",
}

with open(YAML_PATH) as f:
    config = yaml.safe_load(f)

block_measures: dict[str, list[tuple[str, str]]] = {}
for block in ("inner", "outer", "mental_health"):
    block_measures[block] = [
        (entry["name"], entry["column"]) for entry in config[block]
    ]

survey = pl.read_csv(DATA_PATH, null_values=["NA", ""])
if "" in survey.columns:
    survey = survey.drop("")

master = pl.read_parquet(PARQUET_PATH)

rows: list[dict] = []

for display_name, parquet_key in SUBSET_MAP.items():
    if display_name == "Inner & Outer":
        measures = block_measures["inner"] + block_measures["outer"]
    else:
        block = display_name.lower()
        measures = block_measures[block]

    measures = measures + block_measures["mental_health"]
    measure_names = [m[0] for m in measures]
    measure_cols = [m[1] for m in measures]

    for k in range(2, 11):
        col = f"Membership_{parquet_key}_k{k}"
        if col not in master.columns:
            continue

        mapping = json.loads(master[col][0])

        for metric_name, metric_col in zip(measure_names, measure_cols):
            groups: list[np.ndarray] = []
            for subject_list in mapping.values():
                vals = (
                    survey.filter(pl.col("Subject").is_in(subject_list))[metric_col]
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

result.write_parquet(OUT_PATH)
print(f"Written {OUT_PATH}  ({result.height} rows)")

CSV_PATH = ROOT / "results" / "clustering" / "statistical_tests.csv"
result.write_csv(CSV_PATH)
print(f"Written {CSV_PATH}  ({result.height} rows)")

print(result.head(12).to_pandas().to_string(index=False))
