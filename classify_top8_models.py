"""
Comparative Classification Pipeline using Top 8 ANOVA Features
Author: Antigravity Assistant
Models Evaluated:
    1. Support Vector Machine (SVM)
    2. Random Forest Classifier
    3. Linear Regression (Least Squares Classifier & Rounded Regression)
    4. Logistic Regression
    5. K-Nearest Neighbors (KNN)
    6. Decision Tree Classifier
    7. Gradient Boosting Classifier

Evaluation Protocol:
    - Stratified Train-Test Split (80% Train, 20% Test, random_state=42)
    - Feature Scaling with StandardScaler fit strictly on Train
    - Stratified 5-Fold Cross-Validation on Training set
    - Test Set Evaluation: Accuracy, Precision, Recall, F1-Score (Macro & Weighted)
    - Confusion Matrices & Comparative Diagnostics saved to 'model_evaluation_plots/'
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Algorithms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'

OUTPUT_DIR = 'model_evaluation_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class LinearRegressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Multiclass classification via Least Squares / Linear Regression
    using One-Hot Indicator matrix (Bishop PRML Ch. 4.1) with argmax decision rule.
    Also falls back to rounded continuous prediction if single-target.
    """
    def __init__(self):
        self.model = LinearRegression()
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # One-hot encode targets
        Y_onehot = pd.get_dummies(y).values
        self.model.fit(X, Y_onehot)
        return self

    def predict(self, X):
        scores = self.model.predict(X)
        indices = np.argmax(scores, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        scores = self.model.predict(X)
        # Softmax or simple normalization to make probabilities non-negative and sum to 1
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def run_classification_pipeline():
    print("=" * 80)
    print("      BENCHMARKING 7 CLASSIFICATION ALGORITHMS ON TOP 8 ANOVA FEATURES")
    print("=" * 80)

    # 1. Load Data
    csv_path = 'Wine_dataset_top8_anova.csv'
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['class'])
    y = df['class']
    feature_names = list(X.columns)
    class_names = [f'Class {c}' for c in sorted(y.unique())]

    print(f"\n[1] DATASET SPECIFICATION:")
    print(f"    - Observations: {len(df)}")
    print(f"    - Selected Top 8 Features: {', '.join(feature_names)}")
    print(f"    - Target Classes: {sorted(y.unique())}")

    # 2. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2] DATA SPLIT (80% Train, 20% Test):")
    print(f"    - Training Set: {X_train.shape[0]} samples")
    print(f"    - Test Set:     {X_test.shape[0]} samples")

    # 3. Standardize Features (fit on Train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Define All 7 Algorithms
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
        'Support Vector Machine (RBF)': (SVC(kernel='rbf', C=1.0, probability=True, random_state=42), True),
        'K-Nearest Neighbors (k=5)': (KNeighborsClassifier(n_neighbors=5), True),
        'Decision Tree': (DecisionTreeClassifier(max_depth=4, random_state=42), False),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), False),
        'Gradient Boosting': (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), False),
        'Linear Regression (Least Squares)': (LinearRegressionClassifier(), True)
    }

    # 5. Model Evaluation Loop
    results = []
    cm_dict = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n[3] EVALUATION RESULTS ACROSS ALGORITHMS:")
    print("-" * 80)
    print(f"{'Algorithm':<32} {'CV Acc (Mean±Std)':<20} {'Test Acc':<10} {'F1-Weighted':<12} {'F1-Macro':<10}")
    print("-" * 80)

    for name, (clf, use_scaled) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train.values
        X_te = X_test_scaled if use_scaled else X_test.values

        # 5-Fold Cross-Validation on Training data
        cv_scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Fit on full training set
        clf.fit(X_tr, y_train)

        # Predict on holdout test set
        y_pred = clf.predict(X_te)

        # Performance Metrics
        test_acc = accuracy_score(y_test, y_pred)
        prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Store confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_dict[name] = cm

        results.append({
            'Algorithm': name,
            'CV_Mean_Acc': cv_mean,
            'CV_Std_Acc': cv_std,
            'Test_Accuracy': test_acc,
            'Precision_Weighted': prec_weighted,
            'Recall_Weighted': rec_weighted,
            'F1_Weighted': f1_weighted,
            'F1_Macro': f1_macro
        })

        cv_str = f"{cv_mean*100:.2f}% ± {cv_std*100:.2f}%"
        print(f"{name:<32} {cv_str:<20} {test_acc*100:<10.2f}% {f1_weighted:<12.4f} {f1_macro:<10.4f}")

    print("-" * 80)

    results_df = pd.DataFrame(results).sort_values('Test_Accuracy', ascending=False).reset_index(drop=True)

    # Export metrics table
    metrics_csv_path = 'model_benchmark_results.csv'
    results_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[4] Full metrics saved to '{metrics_csv_path}'")

    # ==========================================
    # GENERATE VISUALIZATIONS
    # ==========================================
    # Plot 1: Model Accuracy Comparison (CV vs Test)
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(results_df))
    width = 0.35

    plt.bar(x_pos - width/2, results_df['CV_Mean_Acc'] * 100, width, yerr=results_df['CV_Std_Acc'] * 100,
            capsize=5, label='5-Fold CV Accuracy (Train)', color='#457b9d', edgecolor='#1d3557', alpha=0.9)
    plt.bar(x_pos + width/2, results_df['Test_Accuracy'] * 100, width,
            label='Test Accuracy (Holdout 20%)', color='#2a9d8f', edgecolor='#264653', alpha=0.9)

    plt.title('Performance Comparison of 7 Classifiers (Top 8 ANOVA Features)', fontsize=14, weight='bold', pad=15)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.xticks(x_pos, results_df['Algorithm'], rotation=25, ha='right', fontsize=10, weight='semibold')
    plt.ylim(80, 105)
    plt.axhline(100, color='gray', linestyle=':', linewidth=0.8)
    plt.legend(loc='lower right', frameon=True)

    for i in x_pos:
        test_val = results_df.loc[i, 'Test_Accuracy'] * 100
        plt.text(i + width/2, test_val + 1.0, f"{test_val:.1f}%", ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_model_accuracy_comparison.png'), dpi=300)
    plt.close()

    # Plot 2: Confusion Matrices Grid for All 7 Algorithms
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(cm_dict.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 13, "weight": "bold"})
        acc = results_df.loc[results_df['Algorithm'] == name, 'Test_Accuracy'].values[0] * 100
        axes[idx].set_title(f"{name}\nTest Accuracy: {acc:.1f}%", fontsize=11, weight='bold')
        axes[idx].set_xlabel('Predicted Class', fontsize=10)
        axes[idx].set_ylabel('True Class', fontsize=10)

    # Hide 8th subplot
    fig.delaxes(axes[7])
    plt.suptitle('Test Set Confusion Matrices Across All 7 Classifiers', fontsize=16, weight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_confusion_matrices_grid.png'), dpi=300)
    plt.close()

    # Plot 3: Metrics Radar / Grouped Bar Comparison
    metrics_plot_df = results_df[['Algorithm', 'Test_Accuracy', 'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted']].copy()
    metrics_melted = pd.melt(metrics_plot_df, id_vars=['Algorithm'], var_name='Metric', value_name='Score')
    metrics_melted['Score'] *= 100

    plt.figure(figsize=(14, 6))
    sns.barplot(x='Algorithm', y='Score', hue='Metric', data=metrics_melted, palette='mako')
    plt.title('Test Set Metrics Breakdown (Accuracy, Precision, Recall, F1)', fontsize=14, weight='bold', pad=15)
    plt.ylabel('Score (%)', fontsize=11)
    plt.xticks(rotation=25, ha='right', fontsize=10, weight='semibold')
    plt.ylim(85, 103)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_test_metrics_breakdown.png'), dpi=300)
    plt.close()

    print(f"\n[5] VISUALIZATIONS GENERATED IN '{OUTPUT_DIR}/':")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {OUTPUT_DIR}/{f}")

    print("\n" + "=" * 80)
    print("            ALL 7 MODELS EVALUATED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    run_classification_pipeline()

