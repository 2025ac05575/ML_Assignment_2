"""
Streamlit app for ML Assignment 2
Breast Cancer Wisconsin (Diagnostic) — binary classification demo.

Features:
  a) CSV upload of test data
  b) Model selection dropdown (5 models)
  c) Display of evaluation metrics
  d) Confusion matrix + classification report
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")

# Resolve all paths relative to THIS script's location, not the current
# working directory. Streamlit Cloud can launch the app from different
# working directories depending on deployment settings, so relative paths
# like "model/..." or "../model/..." are unreliable and cause
# FileNotFoundError. Anchoring to __file__ always works.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
    "Decision Tree": MODEL_DIR / "decision_tree.pkl",
    "kNN": MODEL_DIR / "knn.pkl",
    "Naive Bayes": MODEL_DIR / "naive_bayes.pkl",
    "Random Forest": MODEL_DIR / "random_forest.pkl",
}


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    with open(MODEL_DIR / "target_names.json") as f:
        target_names = json.load(f)
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}
    return scaler, feature_names, target_names, models


scaler, feature_names, target_names, models = load_artifacts()

st.title("🩺 Breast Cancer Classification — Model Comparison App")
st.markdown(
    """
This app demonstrates **5 classification models** trained on the
[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
dataset (569 instances, 30 features, binary classification).

**How to use:** Upload a CSV with the same 30 feature columns plus a `target`
column (0 = malignant, 1 = benign) — e.g. the provided `test_data.csv` — then
pick a model from the dropdown to see its predictions and evaluation metrics.
"""
)

# ---------------------------------------------------------------------
# a) Dataset upload
# ---------------------------------------------------------------------
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (test data)", type=["csv"])

# ---------------------------------------------------------------------
# b) Model selection dropdown
# ---------------------------------------------------------------------
st.sidebar.header("2. Select Model")
selected_model_name = st.sidebar.selectbox("Choose a classification model", list(models.keys()))
selected_model = models[selected_model_name]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must include a 'target' column with true labels.")
        st.stop()

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    y_true = df["target"]

    X_scaled = scaler.transform(X)
    y_pred = selected_model.predict(X_scaled)

    if hasattr(selected_model, "predict_proba"):
        y_score = selected_model.predict_proba(X_scaled)[:, 1]
    else:
        y_score = selected_model.decision_function(X_scaled)

    st.subheader(f"Results — {selected_model_name}")

    # ---------------------------------------------------------------
    # c) Evaluation metrics
    # ---------------------------------------------------------------
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    col2.metric("AUC", f"{roc_auc_score(y_true, y_score):.4f}")
    col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
    col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
    col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
    col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

    # ---------------------------------------------------------------
    # d) Confusion matrix + classification report
    # ---------------------------------------------------------------
    left, right = st.columns(2)

    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with right:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3))

    st.markdown("**Predictions Preview**")
    preview = df.copy()
    preview["predicted"] = y_pred
    st.dataframe(preview.head(20))

    # ---------------------------------------------------------------
    # Bonus: compare all 5 models on the uploaded data at once
    # ---------------------------------------------------------------
    st.subheader("Compare All Models on This Data")
    all_results = []
    for name, mdl in models.items():
        pred = mdl.predict(X_scaled)
        score = (
            mdl.predict_proba(X_scaled)[:, 1]
            if hasattr(mdl, "predict_proba")
            else mdl.decision_function(X_scaled)
        )
        all_results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_true, pred), 4),
            "AUC": round(roc_auc_score(y_true, score), 4),
            "Precision": round(precision_score(y_true, pred), 4),
            "Recall": round(recall_score(y_true, pred), 4),
            "F1": round(f1_score(y_true, pred), 4),
            "MCC": round(matthews_corrcoef(y_true, pred), 4),
        })
    st.dataframe(pd.DataFrame(all_results))

else:
    st.info("👈 Upload a CSV file (e.g. `test_data.csv` from this repo) to see predictions and metrics.")
