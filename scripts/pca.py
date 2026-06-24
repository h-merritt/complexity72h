# -*- coding: utf-8 -*-
"""
PCA Analysis - Repository Executable Script for complexity72h
"""

import os
import sys
from pathlib import Path

# 1. Repository path configuration
# __file__ is 'complexity72h/scripts/pca_analysis.py'
# parent[0] is 'scripts/' | parent[1] is 'complexity72h' (Repository Root)
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Target directory for the output plots (complexity72h/results)
RESULTS_DIR = ROOT_DIR / "results"

# 2. Local imports from the src directory
from src.data.loaders import load_survey_data

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA


def perform_pca(data_frame, title_prefix, is_z_score=True):
    print(f"\n{'-'*40}")
    print(f"Performing PCA for variables: {title_prefix}")
    print(f"{'-'*40}")

    # 1. Missing values imputation using mean
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_frame)
    df_imputed = pd.DataFrame(data_imputed, columns=data_frame.columns, index=data_frame.index)

    # 2. Scaling (StandardScaler ensures unit variance if they are not pre-scaled)
    if not is_z_score:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns, index=df_imputed.index)
    else:
        df_scaled = df_imputed

    # 3. PCA Execution
    pca = PCA(n_components=None)
    principal_components = pca.fit_transform(df_scaled)

    # DataFrame with principal components
    pca_columns = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    df_pca = pd.DataFrame(data=principal_components, columns=pca_columns, index=df_scaled.index)
    print(f"PCA analysis for {title_prefix} completed.\n")

    # 4. Variance Results Output
    explained_variance_ratio = pca.explained_variance_ratio_
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"PC{i+1}: {ratio:.2%}")

    cumulative_variance = explained_variance_ratio.cumsum()
    
    # Plot 1: Accumulated explained variance
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.title(f'Accumulated variance explained by ({title_prefix})')
    plt.xlabel('Number of PCs')
    plt.ylabel('Accumulated explained variance')
    plt.grid(True)
    
    # Save cumulative variance plot to results/
    cum_variance_path = RESULTS_DIR / f"{title_prefix.lower()}_cumulative_variance.png"
    plt.savefig(cum_variance_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Plot 2: Scatterplot of the first two PCs
    if df_pca.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', data=df_pca)
        plt.title(f'PCA - First two PCs ({title_prefix})')
        plt.xlabel(f'PC 1 ({explained_variance_ratio[0]:.2%})')
        plt.ylabel(f'PC 2 ({explained_variance_ratio[1]:.2%})')
        plt.grid(True)
        
        # Save scatterplot to results/
        scatterplot_path = RESULTS_DIR / f"{title_prefix.lower()}_pc1_pc2_scatter.png"
        plt.savefig(scatterplot_path, bbox_inches='tight', dpi=300)
        plt.close()

    print(f"\nPC loadings for {title_prefix}:")
    loadings = pd.DataFrame(pca.components_.T, columns=pca_columns, index=df_scaled.columns)
    print(loadings.head()) 
    return loadings


def plot_top_loadings(loadings_df, pc_num, inner_vars, outer_vars, top_n=5, title_suffix=""):
    pc_col = f'PC{pc_num}'
    if pc_col not in loadings_df.columns:
        print(f"Principal Component {pc_num} does not exist.")
        return

    # Sort values to extract top loadings
    top_loadings = loadings_df[pc_col].sort_values(ascending=False).head(top_n)

    # Plot 3: Top loadings barplot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=top_loadings.values, y=top_loadings.index, color='purple', alpha=0.5)
    plt.title(f'Top {top_n} most relevant variables in {pc_col} ({title_suffix})')
    plt.xlabel('Loading')
    plt.ylabel('Variable')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Color labels dynamically depending on variable block type
    for label in ax.get_yticklabels():
        if label.get_text() in inner_vars:
            label.set_color('steelblue') 
        elif label.get_text() in outer_vars:
            label.set_color('tomato') 

    # Save top loadings plot to results/
    loadings_plot_path = RESULTS_DIR / f"{title_suffix.lower()}_pc{pc_num}_top_loadings.png"
    plt.savefig(loadings_plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nComposition of top {top_n} variables in {pc_col} ({title_suffix}):")
    for var_name, loading_val in top_loadings.items():
        if var_name in inner_vars:
            print(f"  - {var_name} (Inner): {loading_val:.2f}")
        elif var_name in outer_vars:
            print(f"  - {var_name} (Outer): {loading_val:.2f}")
        else:
            print(f"  - {var_name} (Other): {loading_val:.2f}")


def main():
    # Absolute path setup using the root directory reference (complexity72h/)
    CSV_PATH = str(ROOT_DIR / "data" / "data_survey.csv")
    YAML_PATH = str(ROOT_DIR / "configs" / "components" / "default.yaml")
    
    print("Loading data with YAML configuration...")
    data_dict = load_survey_data(csv_path=CSV_PATH, yaml_config_path=YAML_PATH, drop_NA=True)
    
    subject_col = "Subject"
    
    df_inner = data_dict["inner"].set_index(subject_col)
    df_outer = data_dict["outer"].set_index(subject_col)
    
    inner_variables = df_inner.columns.tolist()
    outer_variables = df_outer.columns.tolist()
    
    df_combined = pd.concat([df_inner, df_outer], axis=1)

    # --- Analysis Pipeline Execution ---
    
    # 1. Combined PCA
    loadings_combined = perform_pca(df_combined, 'Combined', is_z_score=True)
    for i in range(1, 7):
        if f'PC{i}' in loadings_combined.columns:
            plot_top_loadings(loadings_combined, i, inner_variables, outer_variables, top_n=10, title_suffix="Combined")

    # 2. Inner Variables PCA
    loadings_inner = perform_pca(df_inner, 'Inner', is_z_score=True)
    print("\n--- Main loadings (Inner) ---")
    plot_top_loadings(loadings_inner, 1, inner_variables, outer_variables, top_n=len(inner_variables), title_suffix="Inner")
    plot_top_loadings(loadings_inner, 2, inner_variables, outer_variables, top_n=len(inner_variables), title_suffix="Inner")

    # 3. Outer Variables PCA
    loadings_outer = perform_pca(df_outer, 'Outer', is_z_score=True)
    print("\n--- Main loadings (Outer) ---")
    plot_top_loadings(loadings_outer, 1, inner_variables, outer_variables, top_n=len(outer_variables), title_suffix="Outer")
    plot_top_loadings(loadings_outer, 2, inner_variables, outer_variables, top_n=len(outer_variables), title_suffix="Outer")


if __name__ == "__main__":
    main()
