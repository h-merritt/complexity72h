# complexity72h
for the inner-outer self project at complexity 72h 2026 in london

<h5 align="center">
    
[![arXiv](https://img.shields.io/badge/Arxiv-ID.HERE-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/CODE.HERE)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/SPAICOM/REPO-NAME-HERE/blob/main/LICENSE)

 <br>

</h5>

> [!TIP]
> How are psychosocial profiles, mental health, and brain functional connectivity related? Studies have been dedicated to unraveling the associations of social support perception and neural functional connectivity. Additionally, personality traits have been explored by examining brain networks. Research on mental health has been developed using a broad range of methods and different approaches. However, little attention has been devoted to understanding how personality traits and social variables are related, and to what extent these components are reflected in brain functional connectivity and mental health outcomes. In this work, we aim to address these complex relations by using data from the Human Connectome Project, both from surveys and resting-state fMRI. The survey data includes personality traits measures and self-reported social support-related variables, which we will refer to as inner- and outer-self, respectively. It also includes data on mental health outcomes. Using z-score standardized measures, we analyze correlation matrices to evaluate the association between the inner- and outer-self domains. Our results show that the social indicators are more evidently grouped by impact on social experience than by the duality of inner-outer selves. Using a $k$-means clustering algorithm, we separate individuals into two groups according to social profiles. When confronting these results with the mental health outcomes, we show that the more socially desirable cluster exhibited a higher score on positive aspects such as life satisfaction and purpose in life. In the functional brain connectivity, we observe that the cluster with a more socially beneficial profile exhibits lower interconnectivity, especially in the default mode network. The pipeline we present uses a combined analysis of both fMRI and psychosocial variables, which could open the path for more extensive analysis. 

## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management and [`just`](https://github.com/casey/just) as the task runner.

### Install prerequisites

Install the required tools:

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- [`just`](https://github.com/casey/just)

Follow the installation instructions from their official documentation.

### Setup the development environment

From the project root, run:

```bash
just setup
```

The `setup` recipe will:

- Create the `.venv` virtual environment (if it does not exist)
- Install all project dependencies using `uv`

After the command completes, the development environment will be ready to use.

## Reproducing the analysis

### One-shot (full pipeline)

```bash
just analysis
```

Runs the entire pipeline in the correct order. The dependency graph is:

```
data/hcp_social_mentahealth_data.csv
├── eda-plots            (independent)
├── pca-plots            (independent)
└── clustering-plots
    ├── cluster-subject-corr-plots
    ├── cluster-stats
    └── export-membership
```

### Step-by-step

Each recipe can also be called individually. Recipes that depend on an earlier step declare that dependency explicitly, so running them standalone will automatically trigger the required prerequisite.

| Recipe | Script(s) | Inputs | Outputs |
|---|---|---|---|
| `just eda-plots` | `scripts/eda_social_metrics.py` | `data/hcp_social_mentahealth_data.csv` | `results/plots/eda/{png,pdf}/` — combined distributions, inner/outer correlation heatmaps |
| `just pca-plots` | `scripts/pca.py` | `data/hcp_social_mentahealth_data.csv` | `results/plots/pca/{png,pdf}/` — explained variance, scatter plots, top-loading bar charts (inner, outer, combined) |
| `just clustering-plots` | `scripts/clustering.py` | `data/hcp_social_mentahealth_data.csv` | `results/plots/clustering/k{k}/{subset}/{png,pdf}/` — heatmaps, bar/lollipop means, distribution plots, PCA scatter; `results/plots/clustering/fitness/` — elbow/silhouette/CH/DB metrics; `results/clustering/survey_clustered_all_k.parquet` — cluster membership for all k |
| `just cluster-subject-corr-plots` | `scripts/cluster_subject_corr.py` | `survey_clustered_all_k.parquet` | `results/plots/clustering/subject_corr/{png,pdf}/` — per-cluster subject correlation matrices |
| `just cluster-stats` | `scripts/statistical_tests_clusters.py` → `scripts/plot_cluster_stats_heatmap.py` | `survey_clustered_all_k.parquet` | `results/clustering/statistical_tests.{parquet,csv}` — t-test/ANOVA results with Bonferroni correction; `results/plots/clustering/statistical_tests/{png,pdf}/` — significance heatmaps |
| `just export-membership` | `scripts/export_cluster_membership.py` | `survey_clustered_all_k.parquet` | `results/clustering/membership/membership_{subset}_k{k}.csv` — flat subject-to-cluster assignment per subset and k |

### Other recipes

| Recipe | Description |
|---|---|
| `just setup` | Create `.venv` and install all dependencies via `uv` |
| `just notebooks` | Open the `notebooks/` directory in marimo |

## Citation

If you find this code useful for your research, please consider citing the following paper:

```
```

## Authors

- [Cosimo Agostinelli](https://orcid.org/0009-0006-5649-0872)
- [Ivan Casanovas](https://orcid.org/0009-0003-6168-7459)
- [Lochan Chaudhari](https://orcid.org/0009-0000-7556-8037)
- [Arda Ergin](https://orcid.org/0009-0008-1006-4688)
- [Pablo Estévez-Gutiérrez](https://orcid.org/0009-0004-5697-5672)
- [Akanksha Gupta](https://orcid.org/0009-0006-3602-1593)
- [Juliane T. Moraes](https://orcid.org/0000-0002-9199-8237)
- [Mario Edoardo Pandolfo](https://orcid.org/0009-0006-6509-4425)
- [Carlos Gershenson](https://orcid.org/0000-0003-0193-3067)
- [Haily Merritt](https://orcid.org/0000-0002-7422-3421)
- [Andreia Sofia Teixeira](https://orcid.org/0000-0002-2758-1891)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
