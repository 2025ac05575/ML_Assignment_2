"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates them with 6 metrics, saves the trained models + scaler
to disk, and writes out:
    - test_data.csv        (held-out test set, features + true label, used
                             by the Streamlit app for the "upload CSV" demo)
    - metrics_summary.csv  (comparison table used in the README)

Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set
Source : UCI Machine Learning Repository / scikit-learn built-in loader
         https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Task   : Binary classification (malignant vs benign)
Size   : 569 instances, 30 numeric features  (>= 500 instances, >= 12 features)
"""

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()  # 0 = malignant, 1 = benign

print(f"Dataset shape: {X.shape[0]} instances, {X.shape[1]} features")
print(f"Class balance:\n{y.value_counts()}")

# ---------------------------------------------------------------------
# 2. Train / test split (stratified)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------
# 3. Scale features (fit on train only, then apply to test)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(list(X.columns), "model/feature_names.pkl")

# ---------------------------------------------------------------------
# 4. Define models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # AUC needs probability / decision scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_score = model.decision_function(X_test_scaled)

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_score), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # Save each trained model
    fname = "model/" + name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, fname)

# ---------------------------------------------------------------------
# 5. Save comparison table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("metrics_summary.csv", index=False)
print("\nComparison table:\n", results_df)

# ---------------------------------------------------------------------
# 6. Save test data (features + true target) for the Streamlit app demo
#    This is what gets uploaded to the app / committed as test_data.csv
# ---------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"\nSaved test_data.csv with {test_df.shape[0]} rows.")

# ---------------------------------------------------------------------
# 7. Save target class names for the app
# ---------------------------------------------------------------------
with open("model/target_names.json", "w") as f:
    json.dump(list(data.target_names), f)

print("\nAll models and artifacts saved successfully.")
