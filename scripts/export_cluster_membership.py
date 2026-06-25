from __future__ import annotations

import json
from pathlib import Path

import polars as pl

PARQUET = "results/clustering/survey_clustered_all_k.parquet"
OUT_DIR = Path("results/clustering/membership")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pl.read_parquet(PARQUET)

for col in df.columns:
    if not col.startswith("Membership_"):
        continue
    rest = col[len("Membership_") :]
    subset, k = rest.rsplit("_k", 1)

    mapping = json.loads(df[col][0])
    rows = [
        {"cluster": int(cid), "subject": subj}
        for cid, subjects in mapping.items()
        for subj in subjects
    ]

    out = pl.DataFrame(rows).sort("cluster", "subject")
    path = OUT_DIR / f"membership_{subset.lower()}_k{k}.csv"
    out.write_csv(path)
    print(f"Written {path}")

print(f"\nAll membership CSVs saved to {OUT_DIR}/")
