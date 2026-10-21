"""Export cluster membership to flat CSV files.

Reads the master parquet written by ``scripts/clustering.py`` (one
``Membership_<subset>_k<k>`` column per subset and k, each holding a JSON
``{cluster: [subject ids]}`` mapping in its first row) and writes one CSV per
subset x k combination with columns ``cluster`` and ``subject``.

Output: ``results/clustering/membership/membership_{subset}_k{k}.csv``
Requires: ``just clustering-plots`` (or ``scripts/clustering.py``) has been run.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[1]
PARQUET = ROOT_DIR / "results" / "clustering" / "survey_clustered_all_k.parquet"
OUT_DIR = ROOT_DIR / "results" / "clustering" / "membership"


def main() -> None:
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


if __name__ == "__main__":
    main()
