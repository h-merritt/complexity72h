import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import sys
    import pathlib

    _src = str(pathlib.Path(__file__).parent.parent / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    return


@app.cell
def _(mo):
    mo.md("""
    # EDA — HCP Social & Mental Health
    Analysis of `hcp_social_mentahealth_data.csv` using **Polars** + **Seaborn**.
    """)
    return


@app.cell
def _():
    import numpy as np
    import polars as pl
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")
    sns.set_context("notebook", font_scale=1.1)
    return (np, pl, sns)


@app.cell
def _():
    from insideout.io import load_hcp_data
    from insideout.graph_models import (
        compute_covariance,
        compute_correlation,
        compute_precision,
        fit_glasso,
    )
    from insideout.viz import (
        plot_distribution,
        plot_gender_violin,
        plot_group_distributions,
        plot_correlation_heatmap,
        plot_clustermap,
        plot_clustermap_top,
        plot_combined_heatmap,
        plot_covariance_heatmap,
        plot_precision_heatmap,
        plot_precision_graph,
        plot_pairplot,
        save_fig,
    )

    return (
        compute_correlation,
        compute_covariance,
        compute_precision,
        fit_glasso,
        load_hcp_data,
        plot_clustermap,
        plot_clustermap_top,
        plot_combined_heatmap,
        plot_covariance_heatmap,
        plot_distribution,
        plot_gender_violin,
        plot_group_distributions,
        plot_pairplot,
        plot_precision_graph,
        plot_precision_heatmap,
        plot_correlation_heatmap,
        save_fig,
    )


@app.cell(hide_code=True)
def _():
    from omegaconf import OmegaConf

    _cfg = OmegaConf.load("configs/components/default.yaml")
    inner_cols = list(_cfg.inner)
    outer_cols = list(_cfg.outer)
    return inner_cols, outer_cols


@app.cell
def _():
    plots_dir = "results/plots"
    return (plots_dir,)


@app.cell
def _(load_hcp_data):
    df = load_hcp_data("data/hcp_social_mentahealth_data.csv")
    df
    return (df,)


@app.cell
def _(df, mo, pl):
    n_rows, n_cols = df.shape
    numeric_cols = [
        c
        for c, d in zip(df.columns, df.dtypes)
        if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
        and c != "Subject"
    ]

    mo.md(f"""
    ## Dataset overview

    | Metric | Value |
    |--------|-------|
    | Rows | **{n_rows:,}** |
    | Columns | **{n_cols}** |
    | Numeric | **{len(numeric_cols)}** |
    | Categorical / ID | **{n_cols - len(numeric_cols)}** |
    """)
    return (numeric_cols,)


@app.cell
def _(df, mo):
    schema_rows = [
        {
            "column": c,
            "dtype": str(d),
            "nulls": df[c].null_count(),
            "null_%": round(df[c].null_count() / df.height * 100, 1),
        }
        for c, d in zip(df.columns, df.dtypes)
    ]
    mo.ui.table(schema_rows, label="Column schema")
    return


@app.cell
def _(df, mo, numeric_cols):
    stats = df.select(numeric_cols).describe()
    mo.ui.table(stats, label="Descriptive statistics (numeric)")
    return


@app.cell
def _(mo, numeric_cols):
    mo.md("## Distribution explorer")
    col_picker = mo.ui.dropdown(
        options=numeric_cols,
        value=numeric_cols[0] if numeric_cols else None,
        label="Column",
    )
    col_picker
    return (col_picker,)


@app.cell
def _(col_picker, df, mo, plot_distribution, plots_dir, save_fig):
    _data = df[col_picker.value].drop_nulls().to_numpy()
    _fig = plot_distribution(_data, col_picker.value)
    save_fig(_fig, f"distribution_{col_picker.value}", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


@app.cell
def _(df, mo, numeric_cols):
    mo.md("## Gender breakdown")
    gender_col = "Gender" if "Gender" in df.columns else None
    gender_picker = mo.ui.dropdown(
        options=numeric_cols,
        value=numeric_cols[0],
        label="Metric",
    )
    _out = (
        mo.vstack([mo.md("Compare a metric across **Gender**:"), gender_picker])
        if gender_col
        else None
    )
    _out
    return gender_col, gender_picker


@app.cell
def _(df, gender_col, gender_picker, mo, plot_gender_violin, plots_dir, save_fig):
    if gender_col:
        _sel = df.select([gender_col, gender_picker.value]).drop_nulls()
        _fig = plot_gender_violin(
            _sel[gender_picker.value].to_numpy(),
            _sel[gender_col].to_numpy(),
            gender_picker.value,
            gender_col,
        )
        save_fig(_fig, f"violin_{gender_picker.value}", plots_dir)
        _out = mo.center(mo.mpl.interactive(_fig))
    else:
        _out = mo.md("")
    _out
    return


@app.cell(column=0)
def _(mo, numeric_cols):
    mo.md("## Correlation heatmap")
    heatmap_max = mo.ui.slider(
        5, len(numeric_cols), value=len(numeric_cols), label="Max columns"
    )
    heatmap_max
    return (heatmap_max,)


# ── Column 1 ─────────────────────────────────────────────────────────────────


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md("## Correlation clustering")
    n_clusters = mo.ui.slider(2, 10, value=3, label="N clusters")
    n_clusters
    return (n_clusters,)


@app.cell(column=1, hide_code=True)
def _(
    df,
    heatmap_max,
    mo,
    n_clusters,
    np,
    numeric_cols,
    plot_clustermap,
    plots_dir,
    save_fig,
):
    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _fig = plot_clustermap(np.corrcoef(_arr.T), _cols, n_clusters.value)
    save_fig(_fig, "clustermap", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


@app.cell(column=1, hide_code=True)
def _(df, heatmap_max, mo, np, numeric_cols, plot_clustermap_top, plots_dir, save_fig):
    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _fig = plot_clustermap_top(np.corrcoef(_arr.T), _cols)
    save_fig(_fig, "clustermap_top", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


@app.cell(column=1, hide_code=True)
def _(
    compute_covariance,
    df,
    heatmap_max,
    mo,
    numeric_cols,
    plot_covariance_heatmap,
    plots_dir,
    save_fig,
):
    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _fig = plot_covariance_heatmap(compute_covariance(_arr), _cols)
    save_fig(_fig, "covariance_heatmap", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


@app.cell(column=1, hide_code=True)
def _(df, inner_cols, mo, plot_group_distributions, plots_dir, save_fig):
    mo.md("## Inner metrics")
    _arr = df.select(inner_cols).drop_nulls().to_numpy()
    _fig = plot_group_distributions(
        _arr, inner_cols, color="#3498db", group_name="Inner"
    )
    save_fig(_fig, "distributions_inner", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


@app.cell(column=1, hide_code=True)
def _(df, mo, outer_cols, plot_group_distributions, plots_dir, save_fig):
    mo.md("## Outer metrics")
    _arr = df.select(outer_cols).drop_nulls().to_numpy()
    _fig = plot_group_distributions(
        _arr, outer_cols, color="#e67e22", group_name="Outer"
    )
    save_fig(_fig, "distributions_outer", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


# ── Column 2 ─────────────────────────────────────────────────────────────────


@app.cell(column=2, hide_code=True)
def _(
    compute_covariance,
    df,
    heatmap_max,
    mo,
    numeric_cols,
    plot_precision_graph,
    plots_dir,
    save_fig,
    threshold_slider,
):
    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _fig = plot_precision_graph(compute_covariance(_arr), _cols, threshold_slider.value)
    save_fig(_fig, "graph_covariance", plots_dir)
    mo.vstack(
        [
            mo.md(r"""### Covariance Graph
Edge $(i,j)$ if $|\Sigma_{ij}| > \tau$, where

$$\Sigma_{ij} = \mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)]$$

| ✓ | ✗ |
|---|---|
| Captures total co-variation | Confounded by indirect paths $X \to Z \to Y$ |
| Intuitive scale | Scale-dependent; high-variance vars dominate |"""),
            mo.center(mo.mpl.interactive(_fig)),
        ]
    )
    return


@app.cell(column=2, hide_code=True)
def _(
    compute_correlation,
    df,
    heatmap_max,
    mo,
    numeric_cols,
    plot_precision_graph,
    plots_dir,
    save_fig,
    threshold_slider,
):
    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _fig = plot_precision_graph(
        compute_correlation(_arr), _cols, threshold_slider.value
    )
    save_fig(_fig, "graph_correlation", plots_dir)
    mo.vstack(
        [
            mo.md(r"""### Correlation Graph
Edge $(i,j)$ if $|\rho_{ij}| > \tau$, where

$$\rho_{ij} = \frac{\Sigma_{ij}}{\sqrt{\Sigma_{ii}\,\Sigma_{jj}}} \in [-1,\,1]$$

| ✓ | ✗ |
|---|---|
| Scale-free, values in $[-1,1]$ | Still confounded by indirect paths |
| Comparable across variables | Cannot separate direct from mediated effects |"""),
            mo.center(mo.mpl.interactive(_fig)),
        ]
    )
    return


@app.cell(column=2, hide_code=True)
def _(
    compute_precision,
    df,
    heatmap_max,
    mo,
    numeric_cols,
    plot_precision_heatmap,
    plots_dir,
    save_fig,
):
    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _fig = plot_precision_heatmap(compute_precision(_arr), _cols)
    save_fig(_fig, "precision_heatmap", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    threshold_slider = mo.ui.slider(
        0.0, 2.0, step=0.05, value=0.1, label="Edge threshold"
    )
    threshold_slider
    return (threshold_slider,)


@app.cell(column=2, hide_code=True)
def _(
    compute_precision,
    df,
    heatmap_max,
    mo,
    numeric_cols,
    plot_precision_graph,
    plots_dir,
    save_fig,
    threshold_slider,
):
    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _precision = compute_precision(_arr)
    _fig = plot_precision_graph(_precision, _cols, threshold_slider.value)
    save_fig(_fig, "graph_precision", plots_dir)
    mo.vstack(
        [
            mo.md(r"""### Precision Graph (pinv)
Edge $(i,j)$ if $|\Theta_{ij}| > \tau$, encoding the partial correlation

$$\Theta = \Sigma^{-1}, \qquad \rho_{ij \mid \text{rest}} = -\frac{\Theta_{ij}}{\sqrt{\Theta_{ii}\,\Theta_{jj}}}$$

| ✓ | ✗ |
|---|---|
| Removes indirect paths; direct dependencies only | Unstable when $p \approx n$ |
| Sparse if variables are conditionally independent | No sparsity guarantee: all $\Theta_{ij}$ may be non-zero |"""),
            mo.center(mo.mpl.interactive(_fig)),
        ]
    )
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md("### Graphical Lasso")
    glasso_alpha = mo.ui.slider(0.01, 0.5, step=0.01, value=0.1, label="Alpha (L1)")
    glasso_alpha
    return (glasso_alpha,)


@app.cell(column=2, hide_code=True)
def _(
    df,
    fit_glasso,
    glasso_alpha,
    heatmap_max,
    mo,
    numeric_cols,
    plot_precision_graph,
    plots_dir,
    save_fig,
    threshold_slider,
):
    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _glasso_prec, _alpha_used = fit_glasso(_arr, glasso_alpha.value)
    _fig = plot_precision_graph(_glasso_prec, _cols, threshold_slider.value)
    save_fig(_fig, "graph_glasso", plots_dir)
    mo.vstack(
        [
            mo.md(r"""### Graphical Lasso
Solves the $\ell_1$-penalised log-likelihood:

$$\hat{\Theta} = \arg\max_{\Theta \succ 0}\bigl[\log\det\Theta - \operatorname{tr}(S\Theta) - \alpha\|\Theta\|_1\bigr]$$

| ✓ | ✗ |
|---|---|
| Explicit sparsity: weak edges are exactly zero | Requires tuning $\alpha$ |
| Stable when $p \approx n$ | $\ell_1$ penalty biases edge weights toward zero |
| Yields a true GGM | May over-shrink strong edges at high $\alpha$ |"""),
            mo.center(mo.mpl.interactive(_fig)),
            mo.callout(mo.md(f"α used = **{_alpha_used:.3f}**"), kind="info"),
        ]
    )
    return


@app.cell(column=2, hide_code=True)
def _(
    compute_correlation,
    compute_covariance,
    compute_precision,
    df,
    fit_glasso,
    glasso_alpha,
    heatmap_max,
    mo,
    numeric_cols,
    threshold_slider,
):
    def _n_edges(mat, tau):
        n = mat.shape[0]
        return sum(1 for i in range(n) for j in range(i + 1, n) if abs(mat[i, j]) > tau)

    _cols = numeric_cols[: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _tau = threshold_slider.value
    _al = glasso_alpha.value

    _mats = {
        "Covariance": compute_covariance(_arr),
        "Correlation": compute_correlation(_arr),
        "Precision (pinv)": compute_precision(_arr),
        "Glasso": fit_glasso(_arr, _al)[0],
    }
    _rows = "\n".join(f"| {name} | {_n_edges(m, _tau)} |" for name, m in _mats.items())
    mo.callout(
        mo.md(f"""**Graph comparison** — threshold τ = {_tau:.2f}, Glasso α = {_al:.3f}, columns = {len(_cols)}

Both marginal graphs (Covariance, Correlation) tend to be denser because they do not
condition out shared causes. The Precision graph is conditional but can still be dense
when regularisation is absent. Glasso is the sparsest and most interpretable structure
at the cost of shrinkage bias.

| Graph | Edges above τ |
|-------|--------------|
{_rows}
"""),
        kind="neutral",
    )
    return


# ── Column 3 ─────────────────────────────────────────────────────────────────


@app.cell(column=3)
def _(mo, numeric_cols):
    mo.md("## Pairplot (subset)")
    pair_cols = mo.ui.multiselect(
        options=numeric_cols,
        value=numeric_cols[:4],
        label="Columns to include (keep ≤ 6 for readability)",
    )
    pair_cols
    return (pair_cols,)


@app.cell(column=3)
def _(df, mo, pair_cols, plot_pairplot, plots_dir, save_fig):
    if len(pair_cols.value) >= 2:
        _hue_col = "Gender" if "Gender" in df.columns else None
        _sel = pair_cols.value + ([_hue_col] if _hue_col else [])
        _data = df.select(_sel).drop_nulls()
        _arr = _data.select(pair_cols.value).to_numpy()
        _hue = _data[_hue_col].to_numpy() if _hue_col else None
        _fig = plot_pairplot(_arr, pair_cols.value, hue=_hue, hue_label=_hue_col)
        save_fig(_fig, "pairplot", plots_dir)
        _out = mo.center(mo.mpl.interactive(_fig))
    else:
        _out = mo.callout(mo.md("Select at least 2 columns."), kind="warn")
    _out
    return


# ── Column 4 ─────────────────────────────────────────────────────────────────


@app.cell(column=4, hide_code=True)
def _(
    compute_correlation,
    df,
    inner_cols,
    mo,
    plot_correlation_heatmap,
    plots_dir,
    save_fig,
):
    mo.md("## Inner correlation")
    _arr = df.select(inner_cols).drop_nulls().to_numpy()
    _fig = plot_correlation_heatmap(compute_correlation(_arr), inner_cols)
    save_fig(_fig, "correlation_inner", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


@app.cell(column=4, hide_code=True)
def _(
    compute_correlation,
    df,
    mo,
    outer_cols,
    plot_correlation_heatmap,
    plots_dir,
    save_fig,
):
    mo.md("## Outer correlation")
    _arr = df.select(outer_cols).drop_nulls().to_numpy()
    _fig = plot_correlation_heatmap(compute_correlation(_arr), outer_cols)
    save_fig(_fig, "correlation_outer", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


@app.cell(column=4, hide_code=True)
def _(
    compute_correlation,
    df,
    inner_cols,
    mo,
    outer_cols,
    plot_combined_heatmap,
    plots_dir,
    save_fig,
):
    mo.md("## Inner + Outer correlation")
    _cols = inner_cols + outer_cols
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _fig = plot_combined_heatmap(compute_correlation(_arr), inner_cols, outer_cols)
    save_fig(_fig, "correlation_combined", plots_dir)
    mo.center(mo.mpl.interactive(_fig))
    return


if __name__ == "__main__":
    app.run()
