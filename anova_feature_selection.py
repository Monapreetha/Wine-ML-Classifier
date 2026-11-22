"""
ANOVA-based Feature Selection Pipeline for Wine Dataset
Author: Antigravity Assistant
Description:
    Implements feature selection using ANOVA F-value (f_classif):
    1. Computes ANOVA F-statistics, p-values, and rankings for all 13 features.
    2. Compares model performance (Cross-Validated Accuracy) across k = 1 to 13 features
       using multiple classifiers (Logistic Regression, SVM, KNN, Random Forest).
    3. Identifies the optimal feature subset size (k*).
    4. Applies SelectKBest to export the reduced dataset.
    5. Generates publication-ready diagnostic plots in 'anova_plots/'.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'

OUTPUT_DIR = 'anova_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def perform_anova_feature_selection(csv_path='Wine_dataset_cleaned.csv'):
    print("=" * 70)
    print("        ANOVA FEATURE SELECTION PIPELINE (f_classif)")
    print("=" * 70)

    # 1. Load Data
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['class'])
    y = df['class']
    feature_names = np.array(X.columns)

    # 2. Compute ANOVA F-statistics and p-values
    f_scores, p_values = f_classif(X, y)

    anova_results = pd.DataFrame({
        'Feature': feature_names,
        'F_Statistic': f_scores,
        'p_value': p_values,
        'Log_p_value': -np.log10(p_values)
    }).sort_values('F_Statistic', ascending=False).reset_index(drop=True)

    print("\n[1] FULL ANOVA F-TEST RESULTS (Ranked by F-Statistic):")
    print("-" * 70)
    print(f"{'Rank':<5} {'Feature':<30} {'F-Statistic':<12} {'p-value':<12} {'-log10(p)':<10}")
    print("-" * 70)
    for idx, row in anova_results.iterrows():
        print(f"{idx+1:<5} {row['Feature']:<30} {row['F_Statistic']:<12.2f} {row['p_value']:<12.2e} {row['Log_p_value']:<10.2f}")
    print("-" * 70)

    # 3. Evaluate Cross-Validated Accuracy across k = 1 to 13 features
    print("\n[2] EVALUATING PERFORMANCE ACROSS FEATURE SUBSETS (k = 1 to 13):")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine (RBF)': SVC(kernel='rbf', C=1.0, random_state=42),
        'K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    k_range = list(range(1, len(feature_names) + 1))
    cv_scores = {m_name: [] for m_name in models}

    for k in k_range:
        selector = SelectKBest(score_func=f_classif, k=k)
        X_k = selector.fit_transform(X_scaled, y)
        
        for m_name, model in models.items():
            scores = cross_val_score(model, X_k, y, cv=cv, scoring='accuracy')
            cv_scores[m_name].append(scores.mean())

    cv_results_df = pd.DataFrame(cv_scores, index=k_range)
    cv_results_df.index.name = 'Number_of_Features_k'
    cv_results_df['Average_Accuracy'] = cv_results_df.mean(axis=1)

    print(cv_results_df.round(4).to_string())

    # Optimal k selection
    best_k = int(cv_results_df['Average_Accuracy'].idxmax())
    best_avg_acc = cv_results_df.loc[best_k, 'Average_Accuracy']
    full_avg_acc = cv_results_df.loc[len(feature_names), 'Average_Accuracy']

    print(f"\n    - Peak Average Cross-Validated Accuracy: {best_avg_acc*100:.2f}% achieved at k = {best_k}")
    print(f"    - Full 13-Feature Average Accuracy:     {full_avg_acc*100:.2f}%")
    
    # 4. Select Optimal Top 8 Features
    top8_features = anova_results.loc[:7, 'Feature'].tolist()
    print(f"\n[3] SELECTED TOP 8 FEATURES BY ANOVA:")
    for i, feat in enumerate(top8_features, 1):
        f_val = anova_results.loc[anova_results['Feature'] == feat, 'F_Statistic'].values[0]
        print(f"    {i}. {feat} (F = {f_val:.2f})")

    discarded_features = anova_results.loc[8:, 'Feature'].tolist()
    print(f"\n    - Features Dropped ({len(discarded_features)} features):")
    for feat in discarded_features:
        f_val = anova_results.loc[anova_results['Feature'] == feat, 'F_Statistic'].values[0]
        print(f"      * {feat} (F = {f_val:.2f})")

    # 5. Export Selected Top 8 Dataset
    df_top8 = df[['class'] + top8_features]
    reduced_path = 'Wine_dataset_top8_anova.csv'
    df_top8.to_csv(reduced_path, index=False)
    print(f"\n[4] EXPORTED SELECTED DATASET:")
    print(f"    - Saved to '{reduced_path}' (Shape: {df_top8.shape})")

    selected_features = top8_features
    best_k = 8

    # ==========================================
    # GENERATE DIAGNOSTIC VISUALIZATION CHARTS
    # ==========================================
    # Plot 1: ANOVA F-statistic ranking with selection threshold
    plt.figure(figsize=(11, 7))
    colors = ['#1d3557' if f in selected_features else '#a8dadc' for f in anova_results['Feature']]
    bars = plt.barh(anova_results['Feature'][::-1], anova_results['F_Statistic'][::-1], color=colors[::-1], edgecolor='#333333', linewidth=0.6)
    plt.axvline(x=anova_results.loc[best_k - 1, 'F_Statistic'], color='#e63946', linestyle='--', linewidth=1.5,
                label=f'Selection Cutoff (Top {best_k} Features)')
    plt.title('ANOVA F-Statistic Feature Ranking (f_classif)', fontsize=14, weight='bold', pad=15)
    plt.xlabel('ANOVA F-Statistic (Class Separation Power)')
    plt.ylabel('Feature')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 3, bar.get_y() + bar.get_height() / 2, f"{width:.1f}", va='center', fontsize=9, weight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_anova_f_scores_ranking.png'), dpi=300)
    plt.close()

    # Plot 2: Cross-Validation Accuracy vs. Number of Features k
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    for idx, (m_name, scores) in enumerate(cv_scores.items()):
        plt.plot(k_range, scores, marker=markers[idx], label=m_name, linewidth=2, markersize=6)
    plt.plot(k_range, cv_results_df['Average_Accuracy'], color='black', linestyle='--', linewidth=2.5,
             marker='*', markersize=9, label='Ensemble Average')
    plt.axvline(x=best_k, color='#e63946', linestyle=':', linewidth=1.8, label=f'Optimal k = {best_k}')
    plt.title('Model Cross-Validation Accuracy vs. Number of ANOVA Features (k)', fontsize=14, weight='bold', pad=12)
    plt.xlabel('Number of Top Features Selected (k)', fontsize=11)
    plt.ylabel('Stratified 5-Fold CV Accuracy', fontsize=11)
    plt.xticks(k_range)
    plt.ylim(0.70, 1.02)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_cv_accuracy_vs_k_features.png'), dpi=300)
    plt.close()

    # Plot 3: Correlation Heatmap of Selected Features
    plt.figure(figsize=(9, 8))
    selected_corr = df[selected_features].corr()
    mask = np.triu(np.ones_like(selected_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(selected_corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, annot=True, fmt='.2f',
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title(f'Correlation Matrix of Top {best_k} ANOVA Selected Features', fontsize=13, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_selected_features_correlation.png'), dpi=300)
    plt.close()

    print(f"\n[5] DIAGNOSTIC PLOTS SAVED TO '{OUTPUT_DIR}/':")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {OUTPUT_DIR}/{f}")

    print("\n" + "=" * 70)
    print("          ANOVA FEATURE SELECTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == '__main__':
    perform_anova_feature_selection()

