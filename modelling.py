import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# LOAD SPLITS
X_train = joblib.load("artifacts/X_train.pkl")
X_test  = joblib.load("artifacts/X_test.pkl")
y_train = joblib.load("artifacts/y_train.pkl")
y_test  = joblib.load("artifacts/y_test.pkl")

# OUTPUT FOLDERS
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# MODELS
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

# TRAIN + EVAL + SAVE
for name, model in models.items():
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    THRESHOLD = 0.30
    y_pred = (y_proba >= THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model: {name}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-----------------------------")

    # save trained model
    joblib.dump(model, MODELS_DIR / f"{name}.pkl")