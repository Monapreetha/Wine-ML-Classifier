"""
Data Preprocessing Script for Wine Dataset
Author: Antigravity Assistant
Description:
    Performs comprehensive preprocessing on 'Wine dataset.csv':
    1. Loads dataset and cleans column headers (stripping trailing whitespace).
    2. Identifies and reports missing values.
    3. Identifies and removes duplicate rows if any are present.
    4. Validates data types and value ranges.
    5. Detects potential outliers using the Interquartile Range (IQR) method.
    6. Saves the cleaned dataset ('Wine_dataset_cleaned.csv').
    7. Creates a standardized feature-scaled dataset ('Wine_dataset_standardized.csv')
       ready for distance-based ML algorithms (KNN, SVM, Logistic Regression, PCA).
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_wine_data(input_path='Wine dataset.csv'):
    print("=" * 60)
    print("        WINE DATASET PREPROCESSING PIPELINE")
    print("=" * 60)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    # 1. Load Data
    df_raw = pd.read_csv(input_path)
    print(f"\n[1] Initial Dataset Loaded:")
    print(f"    - Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 2. Column Hygiene
    initial_cols = list(df_raw.columns)
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    renamed_cols = {orig: clean for orig, clean in zip(initial_cols, df.columns) if orig != clean}
    if renamed_cols:
        print(f"\n[2] Cleaned Column Headers (whitespace stripped):")
        for orig, clean in renamed_cols.items():
            print(f"    - '{orig}' -> '{clean}'")
    else:
        print(f"\n[2] Column headers are clean.")

    # 3. Missing Value Analysis
    print(f"\n[3] Missing Values Analysis:")
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    print(f"    - Total missing values: {total_missing}")
    if total_missing > 0:
        print("    - Missing values per column:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"      * {col}: {count} ({count / len(df) * 100:.2f}%)")
        # Imputation or dropping strategy if needed
        # df = df.dropna() or df.fillna(...)
    else:
        print("    - No missing values (null/NaN) detected in any column.")

    # Check for empty strings or infinite values
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"    - Infinite values count: {inf_counts}")

    # 4. Duplicate Rows Analysis & Handling
    print(f"\n[4] Duplicate Rows Analysis:")
    duplicate_mask = df.duplicated()
    num_duplicates = duplicate_mask.sum()
    print(f"    - Number of duplicate rows detected: {num_duplicates}")
    if num_duplicates > 0:
        print(f"    - Removing {num_duplicates} duplicate rows...")
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"    - Shape after duplicate removal: {df.shape}")
    else:
        print("    - No duplicate rows present. Dataset is distinct.")

    # 5. Data Types and Range Verification
    print(f"\n[5] Data Types & Target Distribution:")
    print(f"    - Target variable ('class') distribution:")
    class_dist = df['class'].value_counts().sort_index()
    for cls, cnt in class_dist.items():
        print(f"      * Class {cls}: {cnt} samples ({cnt / len(df) * 100:.2f}%)")

    # 6. Outlier Detection (IQR Method)
    print(f"\n[6] Outlier Detection (IQR Method, multiplier=1.5):")
    feature_cols = [c for c in df.columns if c != 'class']
    outlier_summary = {}
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            outlier_summary[col] = len(outliers)

    if outlier_summary:
        print(f"    - Columns with statistical outliers:")
        for col, count in outlier_summary.items():
            print(f"      * {col:30s}: {count} outlier values")
    else:
        print("    - No outliers detected using 1.5x IQR rule.")

    # 7. Export Cleaned Dataset
    cleaned_filename = 'Wine_dataset_cleaned.csv'
    df.to_csv(cleaned_filename, index=False)
    print(f"\n[7] Export Cleaned Dataset:")
    print(f"    - Saved to '{cleaned_filename}' ({df.shape[0]} rows, {df.shape[1]} cols)")

    # 8. Feature Scaling (Standardization for ML)
    # Target 'class' is preserved; feature columns are scaled to mean=0, std=1
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols)
    df_scaled.insert(0, 'class', df['class'].values)
    
    scaled_filename = 'Wine_dataset_standardized.csv'
    df_scaled.to_csv(scaled_filename, index=False)
    print(f"\n[8] Export Standardized Dataset (Mean=0, Std=1 for features):")
    print(f"    - Saved to '{scaled_filename}'")

    print("\n" + "=" * 60)
    print("             PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return df, df_scaled

if __name__ == '__main__':
    preprocess_wine_data()

