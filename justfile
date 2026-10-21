# Setup the repo .venv via uv
setup:
    uv sync

# Open marimo notebooks for interactive exploration
notebooks:
    uv run marimo edit notebooks/

# ── Code quality ─────────────────────────────────────────────────────────────

# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format code according to project style
format:
    uvx ruff format .

# Run formatting and linting (CI-style target)
clean: format check

# ── Analysis Pipeline ────────────────────────────────────────────────────────

# EDA: distribution plots and correlation heatmaps for inner/outer variables
# Output: results/plots/eda/{png,pdf}/
eda-plots:
    uv run python scripts/eda_social_metrics.py

# PCA on inner, outer, and combined variables; variance and loading plots
# Output: results/plots/pca/{png,pdf}/
pca-plots:
    uv run python scripts/pca.py

# K-means clustering over all subsets and k values; saves cluster membership parquet
# Output: results/plots/clustering/  +  results/clustering/survey_clustered_all_k.parquet
clustering-plots:
    uv run python scripts/clustering.py

# Per-cluster subject-level correlation matrices
# Output: results/plots/clustering/subject_corr/{png,pdf}/
# Requires: clustering-plots
cluster-subject-corr-plots: clustering-plots
    uv run python scripts/cluster_subject_corr.py

# Statistical tests (t-test / ANOVA) across clusters, then significance heatmaps
# Output: results/clustering/statistical_tests.{parquet,csv}
#         results/plots/clustering/statistical_tests/{png,pdf}/
# Requires: clustering-plots
cluster-stats: clustering-plots
    uv run python scripts/statistical_tests_clusters.py
    uv run python scripts/plot_cluster_stats_heatmap.py

# Export cluster membership to flat CSVs, one file per subset × k combination
# Output: results/clustering/membership/membership_{subset}_k{k}.csv
# Requires: clustering-plots
export-membership: clustering-plots
    uv run python scripts/export_cluster_membership.py

# Run the complete analysis pipeline end-to-end
analysis: eda-plots pca-plots cluster-subject-corr-plots cluster-stats export-membership

# ── Brain–behaviour PLS ──────────────────────────────────────────────────────

# Behavioural PLS: {inner, outer, inner+outer} x {Schaefer edges, Yeo sub-blocks, Yeo-7 blocks}
# Output: results/pls/<run>/ (+ grid_summary.csv, dissociation.json)
# Requires: data/hcp100_fc.mat (95 subjects x 100 x 100), data/hcp100_fc_ids.csv;
#           nilearn downloads the Schaefer atlas on first use (network access)
pls:
    uv run python -m insideout.analysis.pls

# PLS figures (loadings, scores, edge/system maps, grid overview, dissociation)
# Output: results/pls/figures/
# Requires: pls
pls-plots: pls
    uv run python -m insideout.analysis.pls.visualize

# ── Tests ────────────────────────────────────────────────────────────────────

# Unit tests (PCA, Yeo system blocks, PLS on synthetic data)
test:
    uv run pytest
