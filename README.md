# Customer Churn Prediction Pipeline

## Objective
Build a production‑ready machine learning pipeline that predicts whether a telecom customer will churn, using the Telco Churn dataset.

## Methodology
- **Data preprocessing**: Missing values imputed (median for numeric, constant for categorical); numeric features scaled with `StandardScaler`; categorical features one‑hot encoded.
- **Models compared**: Logistic Regression vs Random Forest.
- **Hyperparameter tuning**: `GridSearchCV` with 5‑fold cross‑validation.
- **Best model**: Random Forest (cross‑val accuracy = 0.8032, test accuracy = 0.7903).
- **Pipeline export**: Saved using `joblib` and uploaded to Hugging Face Hub.

## Results
| Model | Cross‑val Accuracy | Test Accuracy |
|-------|--------------------|---------------|
| Logistic Regression | 0.8025 | 0.7881 |
| Random Forest | **0.8032** | **0.7903** |

## How to Use the Pipeline

### Option 1: Download from Hugging Face Hub
```python
from huggingface_hub import hf_hub_download
import joblib

model_path = hf_hub_download(
    repo_id="narmeenbilal/churn-pipeline",
    filename="churn_pipeline.pkl"
)
pipeline = joblib.load(model_path)


## Link of churn-pipeline.pkl
https://huggingface.co/narmeenbilal/churn-pipeline/tree/main

# Predict on new data (must have same feature columns)
prediction = pipeline.predict(X_new)
