"""
Interactive Wine Cultivar Classification & Analysis Web App
Built with Gradio 6.x & Scikit-Learn
Author: Antigravity Assistant
"""

import os
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. Custom Least Squares Classifier for Linear Regression
class LeastSquaresClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = LinearRegression()
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y_onehot = pd.get_dummies(y).values
        self.model.fit(X, Y_onehot)
        return self

    def predict(self, X):
        scores = self.model.predict(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self.model.predict(X)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# 2. Load Data and Train All 7 Models
DATA_PATH = 'Wine_dataset_top8_anova.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing '{DATA_PATH}'. Please run feature selection first.")

df = pd.read_csv(DATA_PATH)
FEATURES = ['Flavanoids', 'Proline', 'OD280/OD315 of diluted wines', 'Alcohol',
            'Color intensity', 'Hue', 'Total phenols', 'Malic acid']
X = df[FEATURES]
y = df['class']

# Fit Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train all models
MODELS = {
    'Random Forest (Recommended - 100% Test Acc)': (
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X, y),
        False
    ),
    'Logistic Regression (Highest CV - 98.6% Acc)': (
        LogisticRegression(max_iter=1000, random_state=42).fit(X_scaled, y),
        True
    ),
    'Support Vector Machine (SVC - RBF)': (
        CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, random_state=42), ensemble=False).fit(X_scaled, y),
        True
    ),
    'Gradient Boosting Classifier': (
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42).fit(X, y),
        False
    ),
    'K-Nearest Neighbors (k=5)': (
        KNeighborsClassifier(n_neighbors=5).fit(X_scaled, y),
        True
    ),
    'Decision Tree Classifier': (
        DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y),
        False
    ),
    'Linear Regression (Least Squares Classifier)': (
        LeastSquaresClassifier().fit(X_scaled, y),
        True
    )
}

CLASS_PROFILES = {
    1: {
        'name': 'Cultivar 1 (Full-Bodied Barolo / Nebbiolo Type)',
        'description': (
            "**Chemical Profile:** High Alcohol (mean ~13.7%), exceptionally high Proline (>1100 mg/L), "
            "and rich Flavanoid content (~3.0 g/L).\n\n"
            "**Tasting & Sensory Notes:** Deep structure, firm tannins, complex tertiary bouquet, "
            "and excellent aging potential."
        ),
        'badge': '🔵 Cultivar 1 (Premium Reserve)'
    },
    2: {
        'name': 'Cultivar 2 (Light, Fresh & Vibrant Type)',
        'description': (
            "**Chemical Profile:** Lower Alcohol (mean ~12.3%), delicate color intensity (~3.1), "
            "and moderate Proline (~520 mg/L).\n\n"
            "**Tasting & Sensory Notes:** Light-to-medium body, vibrant fruity aromas, bright acidity, "
            "and approachable style for early consumption."
        ),
        'badge': '🔷 Cultivar 2 (Aromatic / Everyday Selection)'
    },
    3: {
        'name': 'Cultivar 3 (High-Acid, Intensely Colored Type)',
        'description': (
            "**Chemical Profile:** Distinctly elevated Malic Acid (mean ~3.3 g/L), intense deep color (~7.4), "
            "lower Hue (~0.68), and depleted Flavanoids (<0.8 g/L).\n\n"
            "**Tasting & Sensory Notes:** Sharp crisp acidity, dark hue, astringent rustic finish, "
            "and pronounced tartness."
        ),
        'badge': '🔹 Cultivar 3 (Rustic / High-Acidity Style)'
    }
}


def predict_wine_cultivar(model_name, flavanoids, proline, od280, alcohol, color_intensity, hue, total_phenols, malic_acid):
    model, needs_scaling = MODELS[model_name]
    input_df = pd.DataFrame([{
        'Flavanoids': flavanoids,
        'Proline': proline,
        'OD280/OD315 of diluted wines': od280,
        'Alcohol': alcohol,
        'Color intensity': color_intensity,
        'Hue': hue,
        'Total phenols': total_phenols,
        'Malic acid': malic_acid
    }])[FEATURES]

    if needs_scaling:
        X_input = scaler.transform(input_df)
    else:
        X_input = input_df

    # Predict class
    pred_class = int(model.predict(X_input)[0])

    # Predict probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_input)[0]
        prob_dict = {f"Class {cls}": float(p) for cls, p in zip([1, 2, 3], probs)}
    else:
        prob_dict = {f"Class {pred_class}": 1.0}

    profile = CLASS_PROFILES[pred_class]
    confidence = prob_dict.get(f"Class {pred_class}", 1.0) * 100

    report_md = f"""
### {profile['badge']}
**Predicted Cultivar:** {profile['name']}  
**Model Selected:** `{model_name}`  
**Model Confidence:** **`{confidence:.1f}%`**  

---
#### Sensory & Chemical Insights:
{profile['description']}
"""
    return prob_dict, report_md


# 3. Build Gradio UI with Blue Theme
custom_css = """
h1 { color: #1e40af !important; }
h3 { color: #1d4ed8 !important; }
"""

blue_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate"
)

with gr.Blocks(title="Wine Cultivar AI Classifier") as demo:
    gr.Markdown(
        """
        # 🍇 Wine Cultivar AI Classification System
        ### Machine Learning Decision Support Powered by the Top 8 ANOVA-Selected Features
        Predict the botanical cultivar (Class 1, 2, or 3) of wines from their chemical fingerprint using 7 state-of-the-art ML models.
        """
    )

    with gr.Tab("🧪 Live Cultivar Predictor"):
        with gr.Row():
            with gr.Column(scale=5):
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value='Random Forest (Recommended - 100% Test Acc)',
                    label="Select Machine Learning Algorithm"
                )

                gr.Markdown("#### Chemical Feature Inputs (Top 8 ANOVA Selected):")
                with gr.Row():
                    flavanoids = gr.Slider(0.3, 5.2, value=2.98, step=0.01, label="Flavanoids (Phenolic compound)")
                    proline = gr.Slider(250, 1700, value=1115, step=10, label="Proline (mg/L)")
                with gr.Row():
                    od280 = gr.Slider(1.2, 4.0, value=3.16, step=0.01, label="OD280/OD315 (Protein dilution ratio)")
                    alcohol = gr.Slider(11.0, 15.0, value=13.74, step=0.05, label="Alcohol (% vol)")
                with gr.Row():
                    color_intensity = gr.Slider(1.2, 13.0, value=5.53, step=0.1, label="Color Intensity")
                    hue = gr.Slider(0.45, 1.75, value=1.06, step=0.01, label="Hue (Color shade ratio)")
                with gr.Row():
                    total_phenols = gr.Slider(0.9, 4.0, value=2.84, step=0.05, label="Total Phenols (g/L)")
                    malic_acid = gr.Slider(0.7, 6.0, value=2.01, step=0.05, label="Malic Acid (g/L)")

                predict_btn = gr.Button("🔮 Predict Wine Cultivar", variant="primary", size="lg")

            with gr.Column(scale=4):
                gr.Markdown("### 📊 Classification Output")
                output_label = gr.Label(label="Class Probability Distribution", num_top_classes=3)
                output_markdown = gr.Markdown()

        # Preset Archetype Examples
        gr.Markdown("### 📌 Test with Benchmark Archetype Samples:")
        gr.Examples(
            examples=[
                ['Random Forest (Recommended - 100% Test Acc)', 2.98, 1115, 3.16, 13.74, 5.53, 1.06, 2.84, 2.01],
                ['Random Forest (Recommended - 100% Test Acc)', 2.08, 519, 2.79, 12.28, 3.09, 1.06, 2.26, 1.93],
                ['Random Forest (Recommended - 100% Test Acc)', 0.78, 630, 1.68, 13.15, 7.40, 0.68, 1.68, 3.33],
                ['Logistic Regression (Highest CV - 98.6% Acc)', 3.49, 1480, 3.45, 14.37, 7.80, 0.86, 3.85, 1.95],
                ['Support Vector Machine (SVC - RBF)', 1.09, 680, 1.67, 12.33, 3.27, 1.25, 2.05, 1.10]
            ],
            inputs=[model_dropdown, flavanoids, proline, od280, alcohol, color_intensity, hue, total_phenols, malic_acid],
            outputs=[output_label, output_markdown],
            fn=predict_wine_cultivar,
            cache_examples=False,
            label="Click an example below to auto-populate features and test"
        )

        predict_btn.click(
            fn=predict_wine_cultivar,
            inputs=[model_dropdown, flavanoids, proline, od280, alcohol, color_intensity, hue, total_phenols, malic_acid],
            outputs=[output_label, output_markdown]
        )

    with gr.Tab("📈 Model Performance Benchmark"):
        gr.Markdown(
            r"""
            ### Benchmark Results on 80/20 Stratified Split & 5-Fold Cross-Validation
            | Algorithm | 5-Fold CV Accuracy | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score |
            | :--- | :---: | :---: | :---: | :---: | :---: |
            | **Random Forest** | 97.91% $\pm$ 2.77% | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
            | **Decision Tree** | 90.20% $\pm$ 4.52% | **100.00%** | **1.0000** | **1.0000** | **1.0000** |
            | **Logistic Regression** | **98.60% $\pm$ 1.72%** | **97.22%** | **0.9741** | **0.9722** | **0.9720** |
            | **Support Vector Machine (RBF)** | 97.88% $\pm$ 1.73% | **97.22%** | **0.9741** | **0.9722** | **0.9720** |
            | **Linear Regression (Least Squares)** | 97.88% $\pm$ 1.73% | **97.22%** | **0.9741** | **0.9722** | **0.9720** |
            | **Gradient Boosting** | 91.60% $\pm$ 8.30% | **97.22%** | **0.9741** | **0.9722** | **0.9720** |
            | **K-Nearest Neighbors (k=5)** | 96.45% $\pm$ 2.26% | **94.44%** | **0.9466** | **0.9444** | **0.9443** |
            """
        )
        with gr.Row():
            gr.Image("model_evaluation_plots/01_model_accuracy_comparison.png", label="Accuracy Comparison", show_label=True)
            gr.Image("model_evaluation_plots/02_confusion_matrices_grid.png", label="Confusion Matrices Grid", show_label=True)

    with gr.Tab("🔬 ANOVA Feature Selection Insights"):
        gr.Markdown(
            r"""
            ### Why These Top 8 Features?
            Using ANOVA One-Way F-test ($F = \frac{\text{MSB}}{\text{MSW}}$), all 13 features were statistically ranked.
            The Top 8 features retained over **99%** of the full classification signal while eliminating noisy and redundant dimensions:
            
            1. **Flavanoids ($F = 233.93$):** Primary phenolic indicator; clearly separates Class 1 (>2.5), Class 2 (1.5-2.5), and Class 3 (<1.0).
            2. **Proline ($F = 207.92$):** Amino acid fermentation product; over $2\times$ higher in Class 1 (>1100 mg/L) than Classes 2 and 3 (~500-600 mg/L).
            3. **OD280/OD315 of diluted wines ($F = 189.97$):** Spectrophotometric protein absorbance ratio; severely depleted in Class 3 (<1.8).
            4. **Alcohol ($F = 135.08$):** Higher in Class 1 (~13.7%) and lowest in Class 2 (~12.3%).
            5. **Color intensity ($F = 120.66$):** Highest in Class 3 (~7.4) and lowest in Class 2 (~3.1).
            6. **Hue ($F = 101.32$):** Red/yellow color ratio; sharply depressed in Class 3 (~0.68).
            7. **Total phenols ($F = 93.73$):** Correlates strongly with flavanoids ($r = +0.865$).
            8. **Malic acid ($F = 36.94$):** Organic acid imparting tartness; significantly higher in Class 3 (~3.3 g/L).
            """
        )
        with gr.Row():
            gr.Image("anova_plots/01_anova_f_scores_ranking.png", label="ANOVA F-Score Ranking", show_label=True)
            gr.Image("anova_plots/02_cv_accuracy_vs_k_features.png", label="Cross-Validation Accuracy vs. k", show_label=True)

if __name__ == '__main__':
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        share=False,
        theme=blue_theme,
        css=custom_css
    )
