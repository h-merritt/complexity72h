# Setup the repo .venv via uv
setup:
    uv sync

# Run static analysis and automatically fix issues where possible
check:
    uvx ruff check . --fix

# Format code according to project style
format:
    uvx ruff format .

# Run formatting and linting (CI-style target)
clean: format check

# Run marimo notebooks
notebooks:
    uv run marimo edit notebooks/

# Generate EDA social metrics figures to results/plots/eda/{png,pdf}/
eda-plots:
    uv run python scripts/eda_social_metrics.py

# Generate PCA figures to results/plots/pca/{png,pdf}/
pca-plots:
    uv run python scripts/pca.py

# Generate clustering figures to results/plots/clustering/{png,pdf}/
clustering-plots:
    uv run python scripts/clustering.py

# Generate cluster subject correlation matrices (k=2) to results/plots/clustering/subject_corr/
cluster-subject-corr-plots:
    uv run python scripts/cluster_subject_corr.py
