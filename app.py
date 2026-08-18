#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")
st.title("Breast Cancer Diagnosis — Model Comparison App")
st.write(
    "Upload a test CSV (30 numeric features, optionally with a 'Diagnosis' column) "
    "to get predictions from a selected model, along with evaluation metrics and a confusion matrix."
)

# ---- Load models and scaler (cached so this only runs once) ----
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load('model/logistic_regression.pkl'),
        'Decision Tree': joblib.load('model/decision_tree.pkl'),
        'K Neighbors': joblib.load('model/knneighbors.pkl'),
        'Naive Bayes': joblib.load('model/gaussianNB.pkl'),
        'Random Forest': joblib.load('model/random_forest.pkl'),
    }
    scaler = joblib.load('model/scaler.pkl')
    return models, scaler

models, scaler = load_models()

# ---- Sidebar controls ----
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=['csv'])
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))

#Uploading the CSV
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

   #Removing the target value from the test csv
    if 'Diagnosis' in data.columns:
        y_true = data['Diagnosis']
        if y_true.dtype == object:
            y_true = y_true.map({'M': 1, 'B': 0})
        X = data.drop(columns=['Diagnosis'])
    else:
        y_true = None
        X = data

    #Dropping ID column if its present
    if 'ID' in X.columns:
        X = X.drop(columns=['ID'])

    # Scaling the features for an accurate prediction
    X_scaled = scaler.transform(X)

    # Predict using the selected model
    model = models[selected_model_name]
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # Predcitions
    st.subheader(f"Predictions using {selected_model_name}")
    results = X.copy()
    results['Predicted'] = np.where(y_pred == 1, 'Malignant', 'Benign')
    results['Probability (Malignant)'] = np.round(y_proba, 4)
    st.dataframe(results)

    #If target values available, Check the metrics
    if y_true is not None:
        st.subheader("Evaluation Metrics")

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_proba)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col1.metric("AUC", f"{auc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col2.metric("Recall", f"{rec:.4f}")
        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'],
            ax=ax
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Classfication Report
        st.subheader("Classification Report")
        report = classification_report(
            y_true, y_pred,
            target_names=['Benign', 'Malignant'],
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

    else:
        st.info("No 'Diagnosis' column found in uploaded file — showing predictions only, metrics unavailable.")

else:
    st.info("Upload a CSV file from the sidebar to get started.")
    st.write(
        "Expected format: the 30 numeric features from the Breast Cancer Wisconsin "
        "dataset, optionally with a 'Diagnosis' column (M/B or 1/0) for evaluation."
    )

