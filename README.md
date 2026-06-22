# Credit Risk Scoring Engine

End-to-end credit risk model on LendingClub loan data (2007-2018).

## What it does
- Predicts probability of loan default using XGBoost
- Converts default probability into a FICO-style credit score (300-850)
- Explains score via key risk factors (DTI, FICO, employment length)
- Interactive Streamlit demo for real-time scoring

## Model Performance
| Model | AUC-ROC | Gini | KS |
|---|---|---|---|
| Logistic Regression | 0.716 | 0.432 | 0.317 |
| XGBoost | 0.730 | 0.460 | 0.335 |
| LightGBM | 0.730 | 0.460 | 0.335 |

## Key findings from EDA
- Default rate increases monotonically from Grade A (6%) to Grade G (50%)
- Higher interest rate and DTI strongly correlate with default
- FICO score alone is a weaker separator than interest rate

## How to run
```bash
conda create -n creditrisk python=3.11
conda activate creditrisk
conda install -c conda-forge xgboost lightgbm scikit-learn pandas numpy joblib shap streamlit
cd app
streamlit run app.py
```

## Stack
Python | XGBoost | LightGBM | SHAP | Streamlit | Pandas | Scikit-learn

