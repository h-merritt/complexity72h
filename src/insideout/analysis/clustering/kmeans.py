import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import List
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(yaml_path: str | Path) -> dict:
    with open(yaml_path, "r") as file:
        return yaml.safe_load(file)

config_path = "configs/components/default.yaml"
config = load_config(config_path)
INNER_COLS = [item["column"] for item in config.get("inner", [])]
OUTER_COLS = [item["column"] for item in config.get("outer", [])]
MENTAL_HEALTH_COLS = [item["column"] for item in config.get("mental_health", [])]

COMBINED_2 = INNER_COLS + OUTER_COLS 
COMBINED_3 = INNER_COLS + OUTER_COLS + MENTAL_HEALTH_COLS

SUBSETS = {
    "Inner": INNER_COLS,
    "Outer": OUTER_COLS,
    "Mental_Health": MENTAL_HEALTH_COLS,
    "Combined_2": COMBINED_2,
    "Combined_3": COMBINED_3
}

def load_and_clean_data(filepath: str | Path) -> pd.DataFrame:
    """Loads survey data and drops rows with missing social/mental health variables."""
    path = Path(filepath)
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"Missing dataset at {path}")
        
    df = pd.read_csv(path)
    initial_len = len(df)
    
    df_clean = df.dropna(subset=COMBINED_3).copy()
    logger.info(f"Loaded data. Dropped {initial_len - len(df_clean)} rows due to NaNs. Final N={len(df_clean)}")
    return df_clean

def evaluate_and_plot_metrics(X: pd.DataFrame, subset_name: str, max_k: int = 5):
    """Calculates and plots Inertia, Silhouette, CH, and DB indices for all k."""
    metrics = {'k': [], 'Inertia': [], 'Silhouette': [], 'CH_Index': [], 'DB_Index': []}
    
    logger.info(f"[{subset_name}] Evaluating metrics for k=2 to {max_k}...")
    
    # k=1 for Inertia baseline
    kmeans_1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(X)
    metrics['k'].append(1)
    metrics['Inertia'].append(kmeans_1.inertia_)
    metrics['Silhouette'].append(np.nan)
    metrics['CH_Index'].append(np.nan)
    metrics['DB_Index'].append(np.nan)

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertia = kmeans.inertia_
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        
        metrics['k'].append(k)
        metrics['Inertia'].append(inertia)
        metrics['Silhouette'].append(sil)
        metrics['CH_Index'].append(ch)
        metrics['DB_Index'].append(db)
        
        logger.info(f"[{subset_name}] k={k} | Inertia: {inertia:.2f} | Sil: {sil:.3f} | CH: {ch:.2f} | DB: {db:.3f}")

    # Plot a 2x2 grid for the metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{subset_name} Variables: Clustering Evaluation Metrics', fontsize=14)
    
    k_range_all = metrics['k']
    k_range_valid = metrics['k'][1:] # k=2 and above for cluster separation metrics

    # Inertia
    axes[0, 0].plot(k_range_all, metrics['Inertia'], marker='o', color='tab:red')
    axes[0, 0].set_title('Inertia (WCSS)')
    axes[0, 0].set_xticks(k_range_all)

    # Silhouette
    axes[0, 1].plot(k_range_valid, metrics['Silhouette'][1:], marker='s', color='tab:blue')
    axes[0, 1].set_title('Silhouette Score')
    axes[0, 1].set_xticks(k_range_valid)

    # Calinski-Harabasz
    axes[1, 0].plot(k_range_valid, metrics['CH_Index'][1:], marker='^', color='tab:green')
    axes[1, 0].set_title('Calinski-Harabasz Index')
    axes[1, 0].set_xticks(k_range_valid)

    # Davies-Bouldin
    axes[1, 1].plot(k_range_valid, metrics['DB_Index'][1:], marker='d', color='tab:purple')
    axes[1, 1].set_title('Davies-Bouldin Index')
    axes[1, 1].set_xticks(k_range_valid)

    plt.tight_layout()
    Path("figs").mkdir(exist_ok=True)
    plt.savefig(f'figs/{subset_name.lower()}_evaluation_metrics.png', dpi=300)
    plt.close()

def apply_and_visualize_all_k(df: pd.DataFrame, X: pd.DataFrame, cols: List[str], subset_name: str, max_k: int = 4) -> pd.DataFrame:
    """Fits KMeans for each k, appending labels to DF, and generating Heatmaps and PCA plots."""
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate cluster percentages
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        percentages = (counts / len(X)) * 100
        cluster_pct_map = {label: pct for label, pct in zip(unique_labels, percentages)}
        
        # Log the percentages for this k
        pct_log_str = " | ".join([f"Cluster {label+1}: {pct:.1f}%" for label, pct in cluster_pct_map.items()])
        logger.info(f"[{subset_name}] k={k} sizes: {pct_log_str}")
        
        # Save labels explicitly for this subset and this k
        cluster_col = f'Cluster_{subset_name}_k{k}'
        df[cluster_col] = cluster_labels
        
        # Apply the format: "1 (15.2%)"
        df[f'{cluster_col}_Name'] = df[cluster_col].apply(lambda x: f"{x+1} ({cluster_pct_map[x]:.1f}%)")
        
        # Heatmap
        cluster_means = df.groupby(f'{cluster_col}_Name')[cols].mean()
        
        plt.figure(figsize=(max(10, len(cols)*0.6), max(4, k*0.8)))
        # Using a diverging colormap centered at 0 (Z-score mean)
        sns.heatmap(
            cluster_means, 
            cmap='coolwarm', 
            center=0, annot=True, fmt=".2f", 
            cbar_kws={'label': 'Mean Z-Score'}, linewidths=.5
        )
        plt.title(f'{subset_name} Clusters (k={k}): Average Z-Scores')
        plt.ylabel('Cluster') 
        plt.xlabel('Survey Measure')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0) # y-axis horizontal
        plt.tight_layout()
        plt.savefig(f'figs/{subset_name.lower()}_k{k}_heatmap.png', dpi=300)
        plt.close()

        # PCA Scatter
        plot_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Cluster': df[f'{cluster_col}_Name']})
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', alpha=0.8)
        plt.title(f'{subset_name} Clusters (k={k}) on Principal Components')
        plt.legend(title='Cluster (%)')
        plt.tight_layout()
        plt.savefig(f'figs/{subset_name.lower()}_k{k}_pca_scatter.png', dpi=300)
        plt.close()

    return df

def main():
    data_path = "data/data_survey.csv"
    
    try:
        df_final = load_and_clean_data(data_path)
        
        # Iterate through the subsets
        for subset_name, cols in SUBSETS.items():
            X = df_final[cols]
            evaluate_and_plot_metrics(X, subset_name, max_k=5)
            df_final = apply_and_visualize_all_k(df_final, X, cols, subset_name, max_k=5)
            
        # Export master dataset containing all subset/k cluster assignments
        out_path = Path("figs/survey_clustered_all_k.csv")
        out_path.parent.mkdir(exist_ok=True)
        df_final.to_csv(out_path, index=False)
        logger.info(f"Pipeline complete. Master dataset saved to {out_path.name}.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()