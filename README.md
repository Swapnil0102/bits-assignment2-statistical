# Breast Cancer Diagnosis – ML Classification & Streamlit Deployment

## a. Problem Statement
Predict whether a breast tumour is **malignant (1)** or **benign (0)** from 30 numeric features computed on digitized images of fine-needle aspirates (FNA) of breast masses. Five classification models are trained, compared on six evaluation metrics, and served through an interactive Streamlit app that evaluates the saved models on uploaded test data (no retraining).

## b. Dataset Description
- **Source:** UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic) (loaded via `sklearn.datasets.load_breast_cancer`)
- **Instances:** 569 (212 malignant, 357 benign)
- **Features:** 30 real-valued features – mean, standard error and "worst" values of 10 cell-nucleus properties (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** `diagnosis` – 1 = malignant (positive class), 0 = benign
- **Split:** 80 / 20 stratified train/test split (`random_state=42`) → 455 train, 114 test rows. The test split is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside a pipeline for Logistic Regression, KNN and SVM; Decision Tree and Gaussian NB use raw features.

## c. GitHub Repository Link
https://github.com/Swapnil0102/bits-assignment2-statistical

**Live Streamlit App:** <ADD YOUR STREAMLIT APP LINK HERE>

### Repository structure
```
ml_assignment2/
│-- app.py               # Streamlit app
│-- train_models.py      # Training + evaluation script
│-- requirements.txt
│-- README.md
│-- test_data.csv        # Held-out test split (features + target)
│-- model/               # Saved .joblib models, feature list, metrics.csv
```

## d. Models Used & Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN (k=7) | 0.9561 | 0.9825 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes (Gaussian) | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine (RBF) | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <WRITE YOUR OWN OBSERVATION> |
| Decision Tree | <WRITE YOUR OWN OBSERVATION> |
| KNN | <WRITE YOUR OWN OBSERVATION> |
| Naive Bayes | <WRITE YOUR OWN OBSERVATION> |
| Support Vector Machine | <WRITE YOUR OWN OBSERVATION> |
| **Overall winner** | <WRITE YOUR OWN CONCLUSION> |

## How to Run Locally
```bash
pip install -r requirements.txt
python train_models.py      # retrains, regenerates model/ and test_data.csv
streamlit run app.py
```
