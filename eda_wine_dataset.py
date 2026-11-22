"""
Comprehensive Exploratory Data Analysis (EDA) on Wine Dataset
Author: Antigravity Assistant
Description:
    Performs detailed EDA on 'Wine_dataset_cleaned.csv':
    1. Statistical summary & distribution metrics (Mean, Std, Skewness, Kurtosis).
    2. Class-stratified feature analysis.
    3. Pairwise correlation analysis and multicollinearity checks.
    4. ANOVA F-test feature ranking for class separability.
    5. Principal Component Analysis (PCA) for unsupervised dimensionality reduction.
    6. Generates high-resolution visualization plots saved to 'eda_plots/'.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set aesthetic visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = 'eda_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda(csv_path='Wine_dataset_cleaned.csv'):
    print("=" * 70)
    print("        EXPLORATORY DATA ANALYSIS (EDA) - WINE DATASET")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != 'class']
    n_samples, n_features = len(df), len(feature_cols)
    print(f"\n[1] DATASET STRUCTURE:")
    print(f"    - Total Samples: {n_samples}")
    print(f"    - Target Variable: 'class' (Classes: {sorted(df['class'].unique())})")
    print(f"    - Chemical Features: {n_features}")

    # Class balance
    class_counts = df['class'].value_counts().sort_index()
    print(f"\n[2] TARGET CLASS DISTRIBUTION:")
    for cls, cnt in class_counts.items():
        print(f"    - Class {cls}: {cnt} samples ({cnt / n_samples * 100:.2f}%)")

    # 1. Descriptive Statistics & Skewness
    print(f"\n[3] DESCRIPTIVE STATISTICS & SKEWNESS:")
    desc = df[feature_cols].describe().T[['mean', 'std', 'min', '50%', 'max']]
    desc['skewness'] = df[feature_cols].skew()
    desc['kurtosis'] = df[feature_cols].kurtosis()
    print(desc.round(3).to_string())

    # 2. ANOVA F-Score Ranking (Class Separability Power)
    print(f"\n[4] ANOVA F-STATISTIC (Feature Discriminative Power):")
    f_scores, p_values = f_classif(df[feature_cols], df['class'])
    anova_df = pd.DataFrame({
        'Feature': feature_cols,
        'F_Score': f_scores,
        'p_value': p_values
    }).sort_values('F_Score', ascending=False).reset_index(drop=True)
    for idx, row in anova_df.iterrows():
        print(f"    {idx+1:2d}. {row['Feature']:30s} F = {row['F_Score']:8.2f} (p = {row['p_value']:.2e})")

    # 3. Correlation Analysis
    corr_matrix = df[feature_cols].corr()
    corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            corr_pairs.append({
                'Feature 1': feature_cols[i],
                'Feature 2': feature_cols[j],
                'Correlation': corr_matrix.iloc[i, j],
                'Abs_Correlation': abs(corr_matrix.iloc[i, j])
            })
    corr_df = pd.DataFrame(corr_pairs).sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
    
    print(f"\n[5] TOP 5 HIGHEST POSITIVE CORRELATIONS:")
    top_pos = corr_df.sort_values('Correlation', ascending=False).head(5)
    for idx, r in top_pos.iterrows():
        print(f"    - {r['Feature 1']} & {r['Feature 2']}: r = {r['Correlation']:+.3f}")

    print(f"\n[6] TOP 5 STRONGEST NEGATIVE CORRELATIONS:")
    top_neg = corr_df.sort_values('Correlation', ascending=True).head(5)
    for idx, r in top_neg.iterrows():
        print(f"    - {r['Feature 1']} & {r['Feature 2']}: r = {r['Correlation']:+.3f}")

    # 4. Principal Component Analysis (PCA)
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(df[feature_cols])
    pca = PCA(n_components=5)
    pca_transformed = pca.fit_transform(scaled_X)
    explained_var = pca.explained_variance_ratio_
    print(f"\n[7] PCA VARIANCE EXPLAINED:")
    for i, var in enumerate(explained_var):
        print(f"    - PC{i+1}: {var*100:.2f}% (Cumulative: {np.sum(explained_var[:i+1])*100:.2f}%)")

    # ==========================================
    # GENERATING VISUALIZATION PLOTS
    # ==========================================
    palette = ['#2b5c8f', '#e26d5c', '#38b000']
    class_labels = ['Class 1', 'Class 2', 'Class 3']

    # Plot 1: Target Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0], palette=palette, hue=class_counts.index, legend=False)
    axes[0].set_title('Target Class Frequency Count', fontsize=13, weight='bold')
    axes[0].set_xlabel('Wine Class')
    axes[0].set_ylabel('Sample Count')
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())} ({p.get_height()/n_samples*100:.1f}%)",
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 7), textcoords='offset points', weight='bold')

    axes[1].pie(class_counts.values, labels=[f"Class {c}" for c in class_counts.index],
                autopct='%1.1f%%', startangle=140, colors=palette,
                wedgeprops=dict(width=0.4, edgecolor='w'))
    axes[1].set_title('Target Class Proportion', fontsize=13, weight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_class_distribution.png'), dpi=300)
    plt.close()

    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Wine Chemical Features', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_correlation_heatmap.png'), dpi=300)
    plt.close()

    # Plot 3: Feature Distributions (Histograms & KDE)
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#1d3557', bins=20)
        axes[i].set_title(f'{col}\n(Skew: {df[col].skew():.2f})', fontsize=11, weight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Density')
    # Hide unused subplots
    for j in range(len(feature_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle('Univariate Distributions and Skewness of Wine Features', fontsize=16, weight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_feature_distributions.png'), dpi=300)
    plt.close()

    # Plot 4: Feature Boxplots Stratified by Class
    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.boxplot(x='class', y=col, data=df, ax=axes[i], palette=palette, hue='class', legend=False)
        axes[i].set_title(col, fontsize=11, weight='bold')
        axes[i].set_xlabel('Wine Class')
        axes[i].set_ylabel('')
    for j in range(len(feature_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle('Class-Stratified Feature Distributions (Boxplots)', fontsize=16, weight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_feature_boxplots_by_class.png'), dpi=300)
    plt.close()

    # Plot 5: Scatter Comparison of Top Discriminative Features
    # Top features: Flavanoids, Color intensity, Proline, OD280/OD315
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x='Flavanoids', y='Color intensity', hue='class', data=df,
                    palette=palette, style='class', s=70, ax=axes[0])
    axes[0].set_title('Flavanoids vs. Color Intensity', fontsize=13, weight='bold')
    axes[0].set_xlabel('Flavanoids')
    axes[0].set_ylabel('Color Intensity')
    axes[0].legend(title='Wine Class')

    sns.scatterplot(x='Proline', y='OD280/OD315 of diluted wines', hue='class', data=df,
                    palette=palette, style='class', s=70, ax=axes[1])
    axes[1].set_title('Proline vs. OD280/OD315 of Diluted Wines', fontsize=13, weight='bold')
    axes[1].set_xlabel('Proline')
    axes[1].set_ylabel('OD280/OD315')
    axes[1].legend(title='Wine Class')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_top_discriminative_features.png'), dpi=300)
    plt.close()

    # Plot 6: 2D PCA Projection
    pca_df = pd.DataFrame(pca_transformed[:, :2], columns=['PC1', 'PC2'])
    pca_df['class'] = df['class'].values
    plt.figure(figsize=(9, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='class', style='class', data=pca_df,
                    palette=palette, s=80, alpha=0.9)
    plt.title(f'PCA 2D Projection of Wine Dataset\n(Total Explained Variance: {(explained_var[0]+explained_var[1])*100:.1f}%)',
              fontsize=14, weight='bold', pad=12)
    plt.xlabel(f'Principal Component 1 ({explained_var[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({explained_var[1]*100:.1f}% variance)')
    plt.legend(title='Wine Class', labels=['Class 1', 'Class 2', 'Class 3'])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_pca_2d_projection.png'), dpi=300)
    plt.close()

    # Plot 7: ANOVA F-Score Feature Ranking
    plt.figure(figsize=(10, 6))
    sns.barplot(x='F_Score', y='Feature', data=anova_df, palette='viridis', hue='Feature', legend=False)
    plt.title('Feature Discriminative Power (ANOVA F-Statistic)', fontsize=14, weight='bold', pad=12)
    plt.xlabel('ANOVA F-Statistic (Higher = Greater Class Separability)')
    plt.ylabel('Feature')
    for index, value in enumerate(anova_df['F_Score']):
        plt.text(value + 3, index, f"{value:.1f}", va='center', fontsize=10, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_feature_importance_anova.png'), dpi=300)
    plt.close()

    print(f"\n[8] VISUALIZATIONS GENERATED:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {OUTPUT_DIR}/{f}")

    print("\n" + "=" * 70)
    print("                EDA PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 70)

if __name__ == '__main__':
    run_eda()

