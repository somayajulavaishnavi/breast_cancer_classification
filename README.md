# Breast Cancer Classification
End to End ML classification project on UCI breast cancer dataset - trains and compares 5 models and deploys an interactive streamlit app for live model evaluation with confusion matrix and metrics.

## Problem Statement :
Breast cancer is a kind of cancer that begins as a growth of cells in the breast tissue. Breast Cancer is the most common cancers that affect women worldwide.
This project aims to build and compare multiple machine learning classification models to predict whether a breast mass is malignant (cancerous) or benign (non-cancerous), based on quantitative features taken from digitized images of fine needle aspirate (FNA) biopsies. Using the UCI Breast Cancer Wisconsin (Diagnostic) dataset, I implement five classification algorithms — Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Random Forest — and evaluate their performance using standard classification metrics (Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient).

All the models are deployed into as an interactive Streamlit web application, allowing users to upload test data, select a model, and view real-time predictions along with evaluation metrics and a confusion matrix.

## Dataset Description

The dataset used is the UCI Breast Cancer Wisconsin (Diagnostic) dataset, sourced from the UCI Machine Learning Repository.

**Instances**: 569 patient samples
**Features**: 30 numeric features (plus an ID column)
**Target variable**: Diagnosis — binary classification (M = Malignant, B = Benign)
**Class distribution**: 212 Malignant, 357 Benign
**Missing values**: None

Each biopsy image records measurements from multiple cell nuclei present in the sample. For each of 10 base features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension), three summary statistics are computed across all nuclei in that image:

**Mean** — the average value across all nuclei
**Standard Error (SE)** — how much the values varied between nuclei
**Worst** — the average of the three largest (most extreme) values

This gives 10 × 3 = 30 total features

Since the 30 features span very different numeric ranges (e.g. area in the hundreds vs. smoothness as a small decimal), all features were standardized using StandardScaler before training.

## GitHub Repo Link :
https://github.com/somayajulavaishnavi/breast_cancer_classification

## Live Streamlit App :
https://somayajulavaishnavi-breast-cancer-classification-app-80uzvq.streamlit.app/#breast-cancer-diagnosis-model-comparison-app

## Models Used

All 5 models were trained and evaluated on the same train/test split (80/20, stratified) of the dataset, using standardized features.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9970 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9035 | 0.9087 | 0.8298 | 0.9286 | 0.8764 | 0.8011 |
| K-Nearest Neighbors | 0.9561 | 0.9830 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9878 | 0.8667 | 0.9286 | 0.8966 | 0.8341 |
| Random Forest (Ensemble) | 0.9649 | 0.9975 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best overall; dataset is near-linearly separable, so a linear boundary works well. |
| Decision Tree | Weakest performer; a single unpruned tree overfits and generalizes less reliably. |
| K-Nearest Neighbors | Strong results, aided by feature scaling; high precision, slightly lower recall. |
| Naive Bayes | Lower precision than others, likely due to correlated features violating its independence assumption. |
| Random Forest (Ensemble) | Best overall; averaging many trees reduces the overfitting seen in the single Decision Tree. |
| **Overall Winner** | **Random Forest (Ensemble)** — highest or near-highest score across nearly every metric. |
