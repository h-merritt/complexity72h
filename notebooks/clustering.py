import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import yaml
    from pathlib import Path
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )
    from sklearn.preprocessing import StandardScaler
    from insideout.data.loaders import load_survey_data
    from insideout.viz.utils import configure_plot_style

    return (
        KMeans,
        PCA,
        Path,
        StandardScaler,
        calinski_harabasz_score,
        configure_plot_style,
        davies_bouldin_score,
        load_survey_data,
        mo,
        np,
        pd,
        pl,
        plt,
        silhouette_score,
        sns,
        yaml,
    )


@app.cell
def _(Path, configure_plot_style, load_survey_data, yaml):
    ROOT_DIR = Path(__file__).resolve().parents[1]
    configure_plot_style(ROOT_DIR / "configs" / "plotting" / "default.yaml")

    with open(ROOT_DIR / "configs" / "clustering" / "default.yaml") as _f:
        clustering_cfg = yaml.safe_load(_f)

    with open(ROOT_DIR / "configs" / "components" / "default.yaml") as _f:
        components_cfg = yaml.safe_load(_f)

    data = load_survey_data(
        csv_path=str(ROOT_DIR / "data" / "hcp_social_mentahealth_data.csv"),
        yaml_config_path=str(ROOT_DIR / "configs" / "components" / "default.yaml"),
        drop_na=True,
    )

    seed = clustering_cfg["seed"]
    n_init = clustering_cfg["n_init"]
    max_k = clustering_cfg["max_k"]
    heatmap_cmap = clustering_cfg["heatmap_cmap"]
    return (
        clustering_cfg,
        components_cfg,
        data,
        heatmap_cmap,
        max_k,
        n_init,
        seed,
    )


@app.cell
def _(clustering_cfg, max_k, mo):
    subset_names = list(clustering_cfg["subsets"].keys())

    subset_dropdown = mo.ui.dropdown(
        options=subset_names,
        value=subset_names[0],
        label="Subset",
    )
    k_slider = mo.ui.slider(
        start=2,
        stop=max_k,
        value=3,
        step=1,
        label="k",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md("# Clustering Explorer"),
            mo.hstack([subset_dropdown, k_slider], justify="start", gap=2),
        ]
    )
    return k_slider, subset_dropdown


@app.cell
def _(
    KMeans,
    PCA,
    StandardScaler,
    calinski_harabasz_score,
    clustering_cfg,
    components_cfg,
    data,
    davies_bouldin_score,
    k_slider,
    n_init,
    np,
    pl,
    seed,
    silhouette_score,
    subset_dropdown,
):
    subset_name = subset_dropdown.value
    k = k_slider.value

    subset_cfg = clustering_cfg["subsets"][subset_name]
    blocks = list(subset_cfg["blocks"])

    metric_names = []
    for _block in blocks:
        for _entry in components_cfg[_block]:
            metric_names.append(_entry["name"])

    dfs = []
    for _block in blocks:
        _cols = [e["column"] for e in components_cfg[_block]]
        dfs.append(data[_block].select(_cols))
    X = pl.concat(dfs, how="horizontal").drop_nulls().to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    labels = kmeans.fit_predict(X_scaled)

    unique, counts = np.unique(labels, return_counts=True)
    pct_map = {u: p for u, p in zip(unique, counts / len(labels) * 100)}
    labels_pct = np.array([f"{lb + 1} ({pct_map[lb]:.1f}\%)" for lb in labels])
    cluster_labels = [f"{u + 1} ({pct_map[u]:.1f}\%)" for u in unique]

    cluster_means = np.zeros((k, X_scaled.shape[1]))
    cluster_stds = np.zeros((k, X_scaled.shape[1]))
    for _cid in range(k):
        _mask = labels == _cid
        if _mask.sum() > 0:
            cluster_means[_cid] = X_scaled[_mask].mean(axis=0)
            cluster_stds[_cid] = X_scaled[_mask].std(axis=0)

    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    return (
        X_pca,
        X_scaled,
        ch,
        cluster_labels,
        cluster_means,
        cluster_stds,
        db,
        k,
        labels,
        labels_pct,
        metric_names,
        sil,
        subset_name,
    )


@app.cell(hide_code=True)
def _(
    KMeans,
    X_scaled,
    calinski_harabasz_score,
    davies_bouldin_score,
    k,
    max_k,
    mo,
    n_init,
    plt,
    seed,
    silhouette_score,
    subset_name,
):
    _metrics = {"k": [], "Silhouette": [], "CH Index": [], "Davies-Bouldin": []}
    for _ki in range(2, max_k + 1):
        _km = KMeans(n_clusters=_ki, random_state=seed, n_init=n_init)
        _lb = _km.fit_predict(X_scaled)
        _metrics["k"].append(_ki)
        _metrics["Silhouette"].append(silhouette_score(X_scaled, _lb))
        _metrics["CH Index"].append(calinski_harabasz_score(X_scaled, _lb))
        _metrics["Davies-Bouldin"].append(davies_bouldin_score(X_scaled, _lb))

    import base64 as _b64mod
    from io import BytesIO as _BytesIO

    _colors = {
        "Silhouette": "tab:blue",
        "CH Index": "tab:green",
        "Davies-Bouldin": "tab:orange",
    }
    _arrows = {
        "Silhouette": r"$\uparrow$",
        "CH Index": r"$\uparrow$",
        "Davies-Bouldin": r"$\downarrow$",
    }

    fig_metrics, _axes = plt.subplots(1, 3, figsize=(26, 8))
    for _ax, _metric in zip(_axes, ["Silhouette", "CH Index", "Davies-Bouldin"]):
        _ax.plot(
            _metrics["k"],
            _metrics[_metric],
            marker="o",
            color=_colors[_metric],
            linewidth=2,
        )
        _ax.axvline(k, color="red", linestyle="--", linewidth=1.5, label=f"k={k}")
        _ax.set_title(f"{_metric} {_arrows[_metric]}", fontsize=22)
        _ax.set_xlabel("k", fontsize=18)
        _ax.tick_params(axis="both", labelsize=16)
        _ax.set_xticks(_metrics["k"])
        _ax.legend(fontsize=16)
    plt.tight_layout()

    _buf2 = _BytesIO()
    fig_metrics.savefig(_buf2, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig_metrics)
    _buf2.seek(0)
    _img64 = _b64mod.b64encode(_buf2.read()).decode()
    mo.Html(
        f'<img src="data:image/png;base64,{_img64}" style="max-width:100%;height:auto">'
    )
    return


@app.cell
def _(ch, db, k, mo, sil, subset_name):
    mo.vstack(
        [
            mo.md(f"### {subset_name} — k = {k}"),
            mo.hstack(
                [
                    mo.callout(
                        mo.md(f"**Silhouette Score ↑**\n\n## {sil:.4f}"), kind="info"
                    ),
                    mo.callout(
                        mo.md(f"**Calinski-Harabasz Index ↑**\n\n## {ch:.2f}"),
                        kind="success",
                    ),
                    mo.callout(
                        mo.md(f"**Davies-Bouldin Index ↓**\n\n## {db:.4f}"), kind="warn"
                    ),
                ],
                justify="space-between",
            ),
        ]
    )
    return


@app.cell
def _(
    cluster_labels,
    cluster_means,
    cluster_stds,
    heatmap_cmap,
    k,
    metric_names,
    mo,
    np,
    plt,
    sns,
    subset_name,
):
    import base64
    from io import BytesIO

    _annot = np.empty_like(cluster_means, dtype=object)
    for _i in range(cluster_means.shape[0]):
        for _j in range(cluster_means.shape[1]):
            _annot[_i, _j] = f"{cluster_means[_i, _j]:.2f}\n±{cluster_stds[_i, _j]:.2f}"

    fig_heatmap, _ax = plt.subplots(
        figsize=(max(18, len(metric_names) * 1.2), max(8, k * 1.5))
    )
    sns.heatmap(
        cluster_means,
        annot=_annot,
        fmt="",
        annot_kws={"size": 20},
        cmap=heatmap_cmap,
        center=0,
        cbar_kws={"label": "Mean Z-Score"},
        linewidths=0.5,
        xticklabels=metric_names,
        yticklabels=cluster_labels,
        ax=_ax,
    )
    _ax.set_title(f"{subset_name} Clusters (k={k}): Average Z-Scores ± SD")
    _ax.set_ylabel("Cluster")
    _ax.set_xlabel("Survey Measure")
    _ax.tick_params(axis="y", labelsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    _buf = BytesIO()
    fig_heatmap.savefig(_buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig_heatmap)
    _buf.seek(0)
    _b64 = base64.b64encode(_buf.read()).decode()
    mo.Html(
        f'<img src="data:image/png;base64,{_b64}" style="max-width:100%;height:auto">'
    )
    return


@app.cell
def _(X_scaled, k, labels, metric_names, np, pd, plt, sns, subset_name):
    _n_clusters = len(np.unique(labels))
    _n_cols = min(3, _n_clusters)
    _n_rows = (_n_clusters + _n_cols - 1) // _n_cols
    fig_dist, _axes = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(10 * _n_cols, 8 * _n_rows),
        squeeze=False,
    )
    _axes_flat = _axes.flatten()

    for _cid in range(_n_clusters):
        _X_c = X_scaled[labels == _cid]
        _long = pd.DataFrame(_X_c, columns=metric_names).melt(
            var_name="Metric", value_name="Value"
        )
        _ax = _axes_flat[_cid]
        sns.violinplot(
            data=_long,
            x="Value",
            y="Metric",
            order=metric_names,
            ax=_ax,
            inner=None,
            alpha=0.45,
            color="tab:blue",
        )
        sns.boxplot(
            data=_long,
            x="Value",
            y="Metric",
            order=metric_names,
            ax=_ax,
            width=0.3,
            fliersize=0,
            linewidth=1.5,
            color="tab:blue",
        )
        sns.stripplot(
            data=_long,
            x="Value",
            y="Metric",
            order=metric_names,
            ax=_ax,
            alpha=0.25,
            jitter=True,
            size=2,
            color="tab:blue",
        )
        _ax.tick_params(axis="y", labelsize=7)
        _ax.set_title(f"Cluster {_cid + 1} (n={_X_c.shape[0]})")
        _ax.set_xlabel("Z-Score")
        _ax.set_ylabel("")

    for _idx in range(_n_clusters, len(_axes_flat)):
        _axes_flat[_idx].set_visible(False)

    plt.tight_layout()
    fig_dist
    return


@app.cell(hide_code=True)
def _(X_scaled, k, labels, metric_names, mo, pd, plt, sns, subset_name):
    import base64 as _b64met
    from io import BytesIO as _BytesIOmet

    _order = [f"Cluster {_ci + 1}" for _ci in range(k)]
    _df_wide = pd.DataFrame(X_scaled, columns=metric_names)
    _df_wide["Cluster"] = [f"Cluster {_lb + 1}" for _lb in labels]
    _df_long = _df_wide.melt(id_vars="Cluster", var_name="Metric", value_name="Value")

    _palette = dict(zip(_order, sns.color_palette("Set2", k)))
    _ncols_met = 4
    _nrows_met = -(-len(metric_names) // _ncols_met)

    fig_met, _axes_met = plt.subplots(
        _nrows_met,
        _ncols_met,
        figsize=(9 * _ncols_met, 7 * _nrows_met),
        squeeze=False,
    )
    _axes_met_flat = _axes_met.flatten()

    for _mi, _metric in enumerate(metric_names):
        _ax = _axes_met_flat[_mi]
        _df_m = _df_long[_df_long["Metric"] == _metric]
        sns.violinplot(
            data=_df_m,
            x="Cluster",
            y="Value",
            hue="Cluster",
            order=_order,
            palette=_palette,
            inner=None,
            alpha=0.45,
            ax=_ax,
            legend=False,
        )
        sns.boxplot(
            data=_df_m,
            x="Cluster",
            y="Value",
            hue="Cluster",
            order=_order,
            palette=_palette,
            width=0.25,
            fliersize=0,
            linewidth=1.5,
            ax=_ax,
            legend=False,
        )
        sns.stripplot(
            data=_df_m,
            x="Cluster",
            y="Value",
            hue="Cluster",
            order=_order,
            palette=_palette,
            alpha=0.3,
            jitter=True,
            size=3,
            ax=_ax,
            legend=False,
        )
        _ax.set_title(_metric, fontsize=14)
        _ax.set_xlabel("")
        _ax.set_ylabel("Z-Score", fontsize=11)
        _ax.tick_params(axis="x", labelsize=11)

    for _mi in range(len(metric_names), len(_axes_met_flat)):
        _axes_met_flat[_mi].set_visible(False)

    plt.tight_layout()

    _buf_met = _BytesIOmet()
    fig_met.savefig(_buf_met, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig_met)
    _buf_met.seek(0)
    _img64_met = _b64met.b64encode(_buf_met.read()).decode()
    mo.Html(
        f'<img src="data:image/png;base64,{_img64_met}" style="max-width:100%;height:auto">'
    )
    return


@app.cell
def _(X_pca, k, labels_pct, pd, plt, sns, subset_name):
    _plot_df = pd.DataFrame(
        {"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": labels_pct}
    )
    fig_pca, _ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=_plot_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        alpha=0.8,
        ax=_ax,
    )
    _ax.set_title(f"{subset_name} Clusters (k={k}) on Principal Components")
    _ax.legend(title="Cluster (\%)")
    plt.tight_layout()
    fig_pca
    return


@app.cell(hide_code=True)
def _(mo):
    show_jaccard_values = mo.ui.switch(value=False, label="Show Jaccard values")
    mo.vstack(
        [
            mo.md("## Jaccard Similarity between Cluster Assignments"),
            mo.hstack([show_jaccard_values], justify="start"),
        ]
    )
    return (show_jaccard_values,)


@app.cell(hide_code=True)
def _(
    KMeans,
    StandardScaler,
    clustering_cfg,
    components_cfg,
    data,
    k,
    mo,
    n_init,
    np,
    pl,
    plt,
    seed,
    show_jaccard_values,
    sns,
):
    import base64 as _b64jac
    from io import BytesIO as _BytesIO2
    import matplotlib.gridspec as _mgs

    # Cluster assignments for every subset at the selected k
    _subset_labels = {}
    for _sname, _scfg in clustering_cfg["subsets"].items():
        _dfs = []
        for _b in list(_scfg["blocks"]):
            _cols = [e["column"] for e in components_cfg[_b]]
            _dfs.append(data[_b].select(_cols))
        _Xs = StandardScaler().fit_transform(
            pl.concat(_dfs, how="horizontal").drop_nulls().to_numpy()
        )
        _subset_labels[_sname] = KMeans(
            n_clusters=k, random_state=seed, n_init=n_init
        ).fit_predict(_Xs)

    # Group pairs by first subset -> triangular layout
    _snames = list(clustering_cfg["subsets"].keys())
    _row_groups = [
        [(_sA, _sB) for _sB in _snames[_i + 1 :]] for _i, _sA in enumerate(_snames[:-1])
    ]
    _max_cols = max(len(_r) for _r in _row_groups)
    _n_rows = len(_row_groups)
    _cell_size = 7

    fig_jaccard = plt.figure(figsize=(_cell_size * _max_cols, _cell_size * _n_rows))
    _gs = _mgs.GridSpec(
        _n_rows, _max_cols, figure=fig_jaccard, hspace=0.15, wspace=0.15
    )

    _annot_on = show_jaccard_values.value
    _jac_cmap = "Blues" if _annot_on else "magma"

    def _jac_matrix(_labA, _labB):
        _sA2 = {_ci: set(np.where(_labA == _ci)[0]) for _ci in range(k)}
        _sB2 = {_cj: set(np.where(_labB == _cj)[0]) for _cj in range(k)}
        _J = np.zeros((k, k))
        for _ci in range(k):
            for _cj in range(k):
                _i2 = len(_sA2[_ci] & _sB2[_cj])
                _u2 = len(_sA2[_ci] | _sB2[_cj])
                _J[_ci, _cj] = _i2 / _u2 if _u2 > 0 else 0.0
        return _J

    for _ri, _row_pairs in enumerate(_row_groups):
        for _col_i, (_sA, _sB) in enumerate(_row_pairs):
            _ax = fig_jaccard.add_subplot(_gs[_ri, _col_i])
            _J = _jac_matrix(_subset_labels[_sA], _subset_labels[_sB])
            sns.heatmap(
                _J,
                annot=_annot_on,
                fmt=".2f",
                cmap=_jac_cmap,
                vmin=0,
                vmax=1,
                ax=_ax,
                annot_kws={"size": 18},
                square=True,
                xticklabels=[str(_kj + 1) for _kj in range(k)],
                yticklabels=[str(_ki + 1) for _ki in range(k)],
                linewidths=0.5,
            )
            _ax.set_title(f"{_sA} vs {_sB}", fontsize=20)
            _ax.set_xlabel(_sB, fontsize=16)
            _ax.set_ylabel(_sA, fontsize=16)
            _ax.tick_params(axis="both", labelsize=15)

    _buf_jac = _BytesIO2()
    fig_jaccard.savefig(_buf_jac, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig_jaccard)
    _buf_jac.seek(0)
    _img64_jac = _b64jac.b64encode(_buf_jac.read()).decode()
    mo.Html(
        f'<img src="data:image/png;base64,{_img64_jac}" style="max-width:100%;height:auto">'
    )
    return


if __name__ == "__main__":
    app.run()
