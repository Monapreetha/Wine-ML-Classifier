# 🍷 Wine Cultivar AI Classification & Machine Learning Pipeline

An end-to-end Machine Learning pipeline and interactive Gradio web application for botanical classification of Italian wine cultivars based on their chemical and spectroscopic profiles.

---

## 📌 Project Overview
This project performs:
1. **Data Preprocessing & Hygiene:** Handling nulls, duplicate removal, column standardization, and feature scaling.
2. **Exploratory Data Analysis (EDA):** Statistical skewness analysis, class-stratified distributions, multicollinearity heatmap, and PCA cluster projections.
3. **ANOVA Feature Selection:** Mathematical ranking of 13 chemical features via one-way ANOVA F-statistic (`f_classif`) and cross-validation subset optimization ($k=1$ to $13$).
4. **Multi-Model Benchmark:** Comparative evaluation of 7 machine learning algorithms (Random Forest, Decision Tree, Logistic Regression, SVM, Linear Regression, Gradient Boosting, KNN) using Stratified 5-Fold Cross-Validation and holdout testing.
5. **Interactive Gradio Web Application:** A live, blue-themed decision support tool featuring real-time predictions, probability distributions, archetype presets, and visual diagnostics.

---

## 📊 Model Benchmark Results (Top 8 ANOVA Features)

Evaluated on an 80/20 Stratified Train-Test split and 5-Fold Cross-Validation:

| Rank | Algorithm | 5-Fold CV Accuracy (Train) | Test Set Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Random Forest** | 97.91% $\pm$ 2.77% | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
| **2** | **Decision Tree** | 90.20% $\pm$ 4.52% | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
| **3** | **Logistic Regression** | **98.60% $\pm$ 1.72%** | **97.22%** | **0.9741** | **0.9722** | **0.9720** |
| **4** | **Support Vector Machine (RBF)** | 97.88% $\pm$ 1.73% | **97.22%** | **0.9741** | **0.9722** | **0.9720** |
| **5** | **Linear Regression (Least Squares)** | 97.88% $\pm$ 1.73% | **97.22%** | **0.9741** | **0.9722** | **0.9720** |
| **6** | **Gradient Boosting** | 91.60% $\pm$ 8.30% | **97.22%** | **0.9741** | **0.9722** | **0.9720** |
| **7** | **K-Nearest Neighbors ($k=5$)** | 96.45% $\pm$ 2.26% | **94.44%** | **0.9466** | **0.9444** | **0.9443** |

---

## 🔬 Top 8 Selected ANOVA Features

| Rank | Feature | ANOVA F-Statistic | $p$-value | Primary Role in Separation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Flavanoids** | 233.93 | $3.60 \times 10^{-50}$ | Highest in Class 1 (~2.98), depleted in Class 3 (~0.78) |
| **2** | **Proline** | 207.92 | $5.78 \times 10^{-47}$ | $2\times$ higher in Class 1 (>1100 mg/L) |
| **3** | **OD280/OD315 of diluted wines** | 189.97 | $1.39 \times 10^{-44}$ | Protein absorbance ratio; severely reduced in Class 3 |
| **4** | **Alcohol** | 135.08 | $3.32 \times 10^{-36}$ | Highest in Class 1 (~13.7%), lowest in Class 2 (~12.3%) |
| **5** | **Color intensity** | 120.66 | $1.16 \times 10^{-33}$ | Darkest in Class 3 (~7.4), lightest in Class 2 (~3.1) |
| **6** | **Hue** | 101.32 | $5.92 \times 10^{-30}$ | Lowest in Class 3 (~0.68) |
| **7** | **Total phenols** | 93.73 | $2.14 \times 10^{-28}$ | Correlates strongly with Flavanoids ($r = +0.865$) |
| **8** | **Malic acid** | 36.94 | $4.13 \times 10^{-14}$ | Highest acidity in Class 3 (~3.33 g/L) |

---

## 📁 Repository Structure

```
├── Wine dataset.csv                                # Raw dataset
├── Wine_dataset_cleaned.csv                        # Cleaned dataset (whitespace trimmed)
├── Wine_dataset_top8_anova.csv                     # Top 8 ANOVA-selected features dataset
│
├── preprocess_wine_dataset.py                      # Data preprocessing pipeline
├── eda_wine_dataset.py                             # Exploratory Data Analysis script
├── anova_feature_selection.py                      # ANOVA F-score test & subset optimizer
├── classify_top8_models.py                         # 7-model benchmark & evaluation script
├── app.py                                          # Interactive Gradio Web Application
│
├── eda_plots/                                      # 7 high-res EDA visualization charts
├── anova_plots/                                    # 3 diagnostic plots for ANOVA feature selection
├── model_evaluation_plots/                         # Accuracy comparison & confusion matrices
├── model_benchmark_results.csv                     # Numerical benchmark metrics table
├── requirements.txt                                # Python dependencies
├── .gitignore                                      # Files ignored by Git
└── README.md                                       # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts
```bash
# Run Preprocessing
python preprocess_wine_dataset.py

# Run Exploratory Data Analysis & generate plots
python eda_wine_dataset.py

# Run ANOVA Feature Selection
python anova_feature_selection.py

# Run 7-Model Classification Benchmark
python classify_top8_models.py
```

### 4. Launch Interactive Gradio Web App
```bash
python app.py
```
Open **`http://127.0.0.1:7860`** in your browser to interact with the classifier.

