Improving bank loan approval decisions through predictive analytics: A comparative analysis of random forest and XGBoost models

Overview
This project builds a machine learning based loan approval prediction system by comparing between Rnadom Forest and XGBoost to see which model is better at predicting loan approval. The goal is to identify high-risk borroweres in an imbalanced dataset and support credit risk decision making. 

Dataset
- 9578 loan records
- Target: loan default (binary)

Approach
- CRISP-DM framework
- Data cleaning and feature engineering
- Log transformation for skewed variables
- One-hot and label encoding
- Standard scaling
- SMOTE for class imbalance
- Threshold optimization (Threshold = 0.3)

Models
- Random Forest
- XGBoost 

Evalaution Metrics
- Recall
- Precision
- F1-Score
- Accuracy
- AUC-ROC
- Confusion Matrix

Key results
- Random Forest acheives higher recall and better AUC-ROC
- XGBoost achieves higher accruacy but misses more defaulters
- Random Forest is more suitable for predicting loan approval

Explainability
- SHAP is used for feature importance
- Key drivers: Recent credit inquiries, loan purpose, FICO score, interest rate

Technical
- Python, pandas, scikit-learn, XGBoost, imbalanced-learn, SHAP, matplotlib

