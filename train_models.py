"""Train 5 classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset,
save them to model/, export the held-out test split to test_data.csv and
write the metric comparison table to model/metrics.csv."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TARGET = "diagnosis"
SEED = 42


def load_data() -> pd.DataFrame:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.rename(columns={"target": TARGET})
    # sklearn encodes 0 = malignant, 1 = benign; flip so malignant is the positive class
    df[TARGET] = 1 - df[TARGET]
    return df


def build_models() -> dict:
    def scaled(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=5000, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": scaled(SVC(kernel="rbf", probability=True, random_state=SEED)),
    }


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }


def model_filename(name: str) -> str:
    return name.lower().replace(" ", "_") + ".joblib"


def main():
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    MODEL_DIR.mkdir(exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv(ROOT / "test_data.csv", index=False)
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.joblib")

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / model_filename(name))
        rows.append({"ML Model Name": name, **evaluate(model, X_test, y_test)})

    results = pd.DataFrame(rows).round(4)
    results.to_csv(MODEL_DIR / "metrics.csv", index=False)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
