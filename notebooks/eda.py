import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # EDA — HCP Social & Mental Health
    Analysis of `hcp_social_mentahealth_data.csv` using **Polars** + **Seaborn**.
    """)
    return


@app.cell
def _():
    import polars as pl
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    sns.set_theme(style="whitegrid", palette="muted")
    sns.set_context("notebook", font_scale=1.1)
    return pd, pl, plt, sns, np


@app.cell
def _(pl):
    df = pl.read_csv(
        "data/hcp_social_mentahealth_data.csv", null_values=["NA", ""]
    ).drop("")
    df
    return (df,)


@app.cell
def _(df, mo, pl):
    n_rows, n_cols = df.shape
    numeric_cols = [
        c
        for c, d in zip(df.columns, df.dtypes)
        if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
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
def _(col_picker, df, mo, plt, sns):
    _col = col_picker.value
    _data = df[_col].drop_nulls().to_list()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(_data, kde=True, ax=axes[0])
    axes[0].set_title(f"Histogram — {_col}")
    axes[0].set_xlabel(_col)

    sns.boxplot(y=_data, ax=axes[1])
    axes[1].set_title(f"Boxplot — {_col}")
    axes[1].set_ylabel(_col)

    plt.tight_layout()
    mo.center(mo.mpl.interactive(fig))
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
def _(df, gender_col, gender_picker, mo, pd, plt, sns):
    if gender_col:
        _data = df.select([gender_col, gender_picker.value]).drop_nulls()
        _pdf = pd.DataFrame(
            {
                gender_col: _data[gender_col].to_list(),
                gender_picker.value: _data[gender_picker.value].to_list(),
            }
        )
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.violinplot(
            data=_pdf,
            y=gender_picker.value,
            hue=gender_col,
            split=True,
            gap=0.05,
            ax=ax2,
            inner="box",
            palette={"F": "#e74c3c", "M": "#3498db"},
        )
        ax2.set_title(f"{gender_picker.value} — split by Gender")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        _out = mo.center(mo.mpl.interactive(fig2))
    else:
        _out = mo.md("")
    _out
    return


@app.cell
def _(mo, numeric_cols):
    mo.md("## Correlation heatmap")
    _corr_cols = [c for c in numeric_cols if c != "Subject"]
    heatmap_max = mo.ui.slider(
        5, len(_corr_cols), value=len(_corr_cols), label="Max columns"
    )
    heatmap_max
    return (heatmap_max,)


@app.cell
def _(df, heatmap_max, mo, np, numeric_cols, plt, sns):
    _cols = [c for c in numeric_cols if c != "Subject"][: heatmap_max.value]
    _arr = df.select(_cols).drop_nulls().to_numpy()
    _corr = np.corrcoef(_arr.T)

    _size = max(8, len(_cols) * 0.55)
    fig3, ax3 = plt.subplots(figsize=(_size, _size * 0.85))
    sns.heatmap(
        _corr,
        ax=ax3,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=len(_cols) <= 15,
        fmt=".1f",
        linewidths=0.3,
        square=True,
        xticklabels=_cols,
        yticklabels=_cols,
    )
    ax3.set_title(f"Pearson correlation — first {len(_cols)} numeric columns")
    plt.tight_layout()
    mo.center(mo.mpl.interactive(fig3))
    return


@app.cell
def _(mo, numeric_cols):
    mo.md("## Pairplot (subset)")
    pair_cols = mo.ui.multiselect(
        options=numeric_cols,
        value=numeric_cols[:4],
        label="Columns to include (keep ≤ 6 for readability)",
    )
    pair_cols
    return (pair_cols,)


@app.cell
def _(df, mo, pair_cols, pd, plt, sns):
    if len(pair_cols.value) >= 2:
        _hue = "Gender" if "Gender" in df.columns else None
        _sel = pair_cols.value + (["Gender"] if _hue else [])
        _data = df.select(_sel).drop_nulls()
        _pdf = pd.DataFrame({col: _data[col].to_list() for col in _sel})
        fig4 = sns.pairplot(_pdf, hue=_hue, diag_kind="kde", plot_kws={"alpha": 0.4})
        fig4.fig.suptitle("Pairplot", y=1.01)
        plt.tight_layout()
        _out = mo.center(mo.mpl.interactive(fig4.fig))
    else:
        _out = mo.callout(mo.md("Select at least 2 columns."), kind="warn")
    _out
    return


if __name__ == "__main__":
    app.run()
