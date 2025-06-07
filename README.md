
# 🏦 Loan Default Prediction Using Ensemble Models

## 📌 Table of Contents

- [Introduction](#introduction)
- [Objective](#objective)
- [Tools Used](#tools-used)
- [Dataset Description](#dataset-description)
- [Data Cleaning](#data-cleaning)
- [EDA and Feature Engineering](#eda-and-feature-engineering)
- [Model Building](#model-building)
- [Model Comparison](#model-comparison)
- [Conclusion](#conclusion)
- [Future Scope](#future-scope)

---

## 📖 Introduction

Predicting whether a borrower will default on a loan helps lenders mitigate financial risk. This project implements **advanced ensemble learning techniques** to build a robust loan default prediction model.

---

## 🎯 Objective

- Identify high-risk loan applicants.
- Use ensemble algorithms to improve predictive performance.
- Deploy insights for proactive loan portfolio management.

---

## 🛠️ Tools Used

- Python, Pandas, NumPy
- XGBoost, LightGBM, Random Forest
- Scikit-learn
- SHAP for model explainability

---

## 📂 Dataset Description

- Columns: `loan_amount`, `term`, `employment_length`, `credit_history_length`, `loan_status`, `purpose`, `interest_rate`, `default_flag`
- Target: `default_flag` (0 = No Default, 1 = Default)

---

## 🧹 Data Cleaning

```python
df.dropna(inplace=True)
df = df[df['loan_amount'] > 0]
df['emp_length'] = df['employment_length'].str.extract('(\d+)').fillna(0).astype(int)
```

---

## 📊 EDA and Feature Engineering

- Created `debt_to_income = loan_amount / income`
- Encoded categorical features using OneHotEncoding
- Extracted duration from term field

---

## 🤖 Model Building

```python
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

X = df.drop('default_flag', axis=1)
y = df['default_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print("ROC AUC Score:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))
```

---

## 📈 Model Comparison

| Model         | ROC AUC |
|---------------|---------|
| Random Forest | 0.78    |
| XGBoost       | 0.83    |
| LightGBM      | 0.82    |

---

## ✅ Conclusion

XGBoost outperformed other models and can be used as the production model for real-time scoring. Use of SHAP further enables interpretability for regulators and business teams.

---

## 🔮 Future Scope

- Add alternative data (social score, e-wallet usage)
- Build API for real-time predictions
