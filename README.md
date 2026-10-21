<div align='center'>

# Linking the “inner” and “outer” self to mental health and brain networks

[Cosimo Agostinelli](https://orcid.org/0009-0006-5649-0872)<sup>1,2</sup>,
[Ivan Casanovas](https://orcid.org/0009-0003-6168-7459)<sup>3</sup>,
[Lochan Chaudhari](https://orcid.org/0009-0000-7556-8037)<sup>4</sup>,
[Arda Ergin](https://orcid.org/0009-0008-1006-4688)<sup>5</sup>,
[Pablo Estévez-Gutiérrez](https://orcid.org/0009-0004-5697-5672)<sup>6,7</sup>,
[Akanksha Gupta](https://orcid.org/0009-0006-3602-1593)<sup>1</sup>,
[Juliane T. Moraes](https://orcid.org/0000-0002-9199-8237)<sup>8</sup>,
[Mario Edoardo Pandolfo](https://orcid.org/0009-0006-6509-4425)<sup>9</sup>,
[Carlos Gershenson](https://orcid.org/0000-0003-0193-3067)<sup>10</sup>,
[Haily Merritt](https://orcid.org/0000-0002-7422-3421)<sup>11</sup>,
[Andreia Sofia Teixeira](https://orcid.org/0000-0002-2758-1891)<sup>11</sup>

<sup>1</sup>Aix-Marseille University · <sup>2</sup>Université de Toulon · <sup>3</sup>Universitat de Barcelona · <sup>4</sup>Northeastern University Boston · <sup>5</sup>Universiteit van Amsterdam · <sup>6</sup>Institut d’Investigacions Biomèdiques August Pi i Sunyer · <sup>7</sup>Universitat de Barcelona · <sup>8</sup>Universitat Politècnica de Catalunya · <sup>9</sup>Sapienza University of Rome · <sup>10</sup>Binghamton University · <sup>11</sup>Northeastern University London

</div>

<h5 align="center">
    
[![arXiv](https://img.shields.io/badge/Arxiv-2606.27956-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2606.27956)
[![License](https://img.shields.io/badge/Code%20License-MIT-yellow)](https://github.com/h-merritt/complexity72h/blob/main/LICENSE)

 <br>

</h5>

---

> [!NOTE]
> **Abstract**
> How are psychosocial profiles, mental health, and brain functional connectivity related? Studies have been dedicated to unraveling the associations of social support perception and neural functional connectivity. Additionally, personality traits have been explored by examining brain networks. Research on mental health has been developed using a broad range of methods and different approaches. However, little attention has been devoted to understanding how personality traits and social variables are related, and to what extent these components are reflected in brain functional connectivity and mental health outcomes. In this work, we aim to address these complex relations by using data from the Human Connectome Project, both from surveys and resting-state fMRI. The survey data includes personality traits measures and self-reported social support-related variables, which we will refer to as inner- and outer-self, respectively. It also includes data on mental health outcomes. Using z-score standardized measures, we analyze correlation matrices to evaluate the association between the inner- and outer-self domains. Our results show that the social indicators are more evidently grouped by impact on social experience than by the duality of inner-outer selves. Using a $k$-means clustering algorithm, we separate individuals into two groups according to social profiles. When confronting these results with the mental health outcomes, we show that the more socially desirable cluster exhibited a higher score on positive aspects such as life satisfaction and purpose in life. In the functional brain connectivity, we observe that the cluster with a more socially beneficial profile exhibits lower interconnectivity, especially in the default mode network. The pipeline we present uses a combined analysis of both fMRI and psychosocial variables, which could open the path for more extensive analysis. 

## Dependencies

This project uses [`uv`](https://github.com/astral-sh/uv) for Python dependency management and [`just`](https://github.com/casey/just) as the task runner.

### Install prerequisites

Install the required tools:

- [`uv`](https://github.com/astral-sh/uv)
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

After the command completes, the development environment will be ready to use. 🚀

## Data

Place the following files in `data/` (git-ignored):

| File | Content |
|---|---|
| `hcp_social_mentahealth_data.csv` | Survey table: `Subject`, inner/outer measures and mental-health scores (columns defined in `configs/components/default.yaml`) |
| `hcp100_fc.mat` | Variable `avg`: per-subject functional connectivity, array of shape `(95, 100, 100)` (Schaefer-100 parcellation) |
| `hcp100_fc_ids.csv` | Column `Subject`: the 95 subject IDs, in the same order as the first axis of `avg` |

The FC subjects are matched to the survey by ID; the survey rows are listwise-deleted over all configured columns (1206 → 1196 subjects).

Plot scripts use LaTeX for text rendering (`text.usetex: true` in `configs/plotting/default.yaml`), so a LaTeX installation is required to generate figures. The PLS also needs [`nilearn`](https://nilearn.github.io) (installed with `just setup`), which downloads the Schaefer atlas on first use and therefore needs network access.

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

The brain–behaviour PLS (`just pls`, `just pls-plots`) is not part of `just analysis` because it needs the FC data and takes longer.

### Step-by-step

Each recipe can also be called individually; those that depend on `clustering-plots` declare it explicitly and will trigger it automatically if needed.

#### Scripts

- `just eda-plots` — distributions and correlation heatmaps for inner/outer variables → `results/plots/eda/`
- `just pca-plots` — variance, scatter, and loading plots for inner, outer, and combined → `results/plots/pca/`
- `just clustering-plots` — k-means over all subsets and k values → `results/plots/clustering/`, `results/clustering/survey_clustered_all_k.parquet`
- `just cluster-subject-corr-plots` — per-cluster subject correlation matrices → `results/plots/clustering/subject_corr/`
- `just cluster-stats` — t-test/ANOVA with Bonferroni correction and significance heatmaps → `results/clustering/statistical_tests.{parquet,csv}`, `results/plots/clustering/statistical_tests/`
- `just export-membership` — flat CSVs of subject-to-cluster assignments → `results/clustering/membership/`

#### Brain–behaviour PLS

- `just pls` — behavioural PLS between functional connectivity and the inner, outer and inner+outer measures, at three FC resolutions (Schaefer edges, Yeo sub-network blocks, Yeo-7 blocks); permutation, bootstrap, split-half and cross-validation → `results/pls/<run>/`, `results/pls/grid_summary.csv`, `results/pls/dissociation.json`
- `just pls-plots` — loadings, scores, brain maps, grid overview and inner-vs-outer dissociation figures → `results/pls/figures/`

#### Notebooks

- `just notebooks` — open the `notebooks/` directory in marimo for interactive exploration

## Citation

If you find this code useful for your research, please consider citing the following paper:

```bibtex
@misc{agostinelli2026linkinginnerouterself,
      title={Linking the "inner" and "outer" self to mental health and brain networks}, 
      author={Cosimo Agostinelli and Ivan Casanovas and Lochan Chaudhari and Arda Ergin and Pablo Estévez-Gutiérrez and Akanksha Gupta and Juliane T. Moraes and Mario Edoardo Pandolfo and Carlos Gershenson and Haily Merritt and Andreia Sofia Teixeira},
      year={2026},
      eprint={2606.27956},
      archivePrefix={arXiv},
      primaryClass={physics.soc-ph},
      url={https://arxiv.org/abs/2606.27956}, 
}
```

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
