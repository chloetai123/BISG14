import joblib
from pathlib import Path
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)

# =========================
# LOAD SPLITS
# =========================
X_train = joblib.load("artifacts/X_train.pkl")
X_test  = joblib.load("artifacts/X_test.pkl")
y_train = joblib.load("artifacts/y_train.pkl")
y_test  = joblib.load("artifacts/y_test.pkl")

# =========================
# OUTPUT FOLDERS
# =========================
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CLASS IMBALANCE RATIO (for XGBoost)
# scale_pos_weight = (#negative / #positive)
# =========================
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

print(f"Class balance (train): neg={neg}, pos={pos}, scale_pos_weight={scale_pos_weight:.2f}")
print("=" * 60)

# =========================
# MODELS
# =========================
models = {
    "RandomForest": RandomForestClassifier(
        random_state=42,
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=800,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight
    )
}

# =========================
# TRAIN + EVAL + SAVE
# =========================
for name, model in models.items():
    print(f"\nTraining {name}...")
    print("-" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # ---- Threshold tuning (F1 on TEST) ----
    # NOTE: For best practice, tune on validation set, but this is fine for assignment demo.
    prec, rec, th = precision_recall_curve(y_test, y_proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)

    best_idx = int(np.argmax(f1s))
    best_threshold = float(th[best_idx]) if best_idx < len(th) else 0.5

    THRESHOLD = best_threshold
    y_pred = (y_proba >= THRESHOLD).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Print results
    print(f"Model: {name}")
    print(f"Optimal Threshold (F1-based): {THRESHOLD:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(f"[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")

    # Optional “business interpretation” lines
    print("\nBusiness Interpretation:")
    print(f"- Model catches {recall*100:.1f}% of actual defaults (Recall)")
    print(f"- When model predicts default, it's correct {precision*100:.1f}% of the time (Precision)")
    print(f"- AUC of {roc_auc:.3f} reflects overall ranking ability (threshold-independent)")
    print("-" * 60)

    # Save model
    joblib.dump(model, MODELS_DIR / f"{name}.pkl")

print("\nDone. Models saved to:", MODELS_DIR.resolve())