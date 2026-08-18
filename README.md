# Breast Cancer Classification
End to End ML classification project on UCI breast cancer dataset - trains and compares 5 models and deploys an interactive streamlit app for live model evaluation with confusion matrix and metrics.

Problem Statement :
Breast cancer is a kind of cancer that begins as a growth of cells in the breast tissue. Breast Cancer is the most common cancers that affect women worldwide.
This project aims to build and compare multiple machine learning classification models to predict whether a breast mass is malignant (cancerous) or benign (non-cancerous), based on quantitative features extracted from digitized images of fine needle aspirate (FNA) biopsies. Using the UCI Breast Cancer Wisconsin (Diagnostic) dataset, we implement five classification algorithms — Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Random Forest — and evaluate their performance using standard classification metrics (Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient).

The best-performing model is deployed as an interactive Streamlit web application, allowing users to upload test data, select a model, and view real-time predictions along with evaluation metrics and a confusion matrix — demonstrating a complete, end-to-end machine learning deployment workflow from data preprocessing to production-ready inference.

Dataset Description

The dataset used is the UCI Breast Cancer Wisconsin (Diagnostic) dataset, sourced from the UCI Machine Learning Repository.

Instances: 569 patient samples
Features: 30 numeric features (plus an ID column, which is dropped before modeling)
Target variable: Diagnosis — binary classification (M = Malignant, B = Benign)
Class distribution: 212 Malignant (~37%), 357 Benign (~63%)
Missing values: None

Each biopsy image records measurements from multiple cell nuclei present in the sample. For each of 10 base characteristics (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension), three summary statistics are computed across all nuclei in that image:

Mean — the average value across all nuclei
Standard Error (SE) — how much the values varied between nuclei
Worst — the average of the three largest (most extreme) values

This gives 10 × 3 = 30 total features. Features on the "worst" statistic (e.g. radius3, concave_points3) tend to be the strongest predictors of malignancy, since cancerous tumors often contain a subset of highly irregular, enlarged cells even when the average cell appears normal.

Since the 30 features span very different numeric ranges (e.g. area in the hundreds vs. smoothness as a small decimal), all features were standardized using StandardScaler before training.

GitHub Repo Link :
https://github.com/somayajulavaishnavi/breast_cancer_classification

Live Streamlit App :
https://somayajulavaishnavi-breast-cancer-classification-app-80uzvq.streamlit.app/#breast-cancer-diagnosis-model-comparison-app
