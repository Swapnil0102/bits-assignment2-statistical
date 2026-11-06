from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

MODEL_DIR = Path(__file__).parent / "model"
TARGET = "diagnosis"
MODELS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
LABELS = ["Benign (0)", "Malignant (1)"]

st.set_page_config(page_title="Breast Cancer Classifier Lab", layout="wide")


@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)


@st.cache_resource
def load_features() -> list:
    return joblib.load(MODEL_DIR / "feature_columns.joblib")


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba) if y_true.nunique() > 1 else float("nan"),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("Breast Cancer Diagnosis - Model Evaluation Dashboard")
st.caption("Upload held-out test data and evaluate pre-trained classifiers (no retraining).")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    model_name = st.selectbox("Choose a model", list(MODELS))
    compare_all = st.checkbox("Also compare all models", value=True)

if uploaded is None:
    st.info(f"Upload a CSV containing the 30 feature columns and the `{TARGET}` target column.")
    st.stop()

df = pd.read_csv(uploaded)
features = load_features()
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}")
    st.stop()

X, y = df[features], df[TARGET].astype(int)
st.subheader("Data preview")
st.write(f"{len(df)} rows - class balance: {y.value_counts().to_dict()}")
st.dataframe(df.head(), use_container_width=True)

model = load_model(MODELS[model_name])
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]
metrics = compute_metrics(y, y_pred, y_proba)

st.subheader(f"Metrics - {model_name}")
for col, (k, v) in zip(st.columns(len(metrics)), metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(confusion_matrix(y, y_pred, labels=[0, 1]), annot=True, fmt="d",
                cmap="Purples", xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, labels=[0, 1], target_names=LABELS,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

if compare_all:
    st.subheader("All models on uploaded data")
    rows = []
    for name, fname in MODELS.items():
        m = load_model(fname)
        rows.append({"Model": name, **compute_metrics(y, m.predict(X), m.predict_proba(X)[:, 1])})
    table = pd.DataFrame(rows).set_index("Model").round(4)
    st.dataframe(table.style.highlight_max(axis=0, color="#d4f4dd"), use_container_width=True)
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
