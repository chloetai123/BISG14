import joblib
from pathlib import Path
import pandas as pd
import shap
import matplotlib.pyplot as plt

#  CONFIG 
MODEL_PATH = Path("models/XGBoost.pkl")   # change to models/RandomForest.pkl if you want RF
OUT_DIR = Path("shap_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#  LOAD 
X_train = joblib.load("artifacts/X_train.pkl")
X_test  = joblib.load("artifacts/X_test.pkl")
model   = joblib.load(MODEL_PATH)

# Make sure we have feature names (important for SHAP plots)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)
if not isinstance(X_train, pd.DataFrame):
    X_train = pd.DataFrame(X_train)

#  SHAP EXPLAINER 
# TreeExplainer works for XGBoost + RandomForest
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, SHAP sometimes returns list [class0, class1]
if isinstance(shap_values, list):
    shap_values_to_use = shap_values[1]
else:
    shap_values_to_use = shap_values

#  1) GLOBAL IMPORTANCE (summary) 
plt.figure()
shap.summary_plot(shap_values_to_use, X_test, show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary.png", dpi=300)
plt.close()

#  2) BAR SUMMARY 
plt.figure()
shap.summary_plot(shap_values_to_use, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary_bar.png", dpi=300)
plt.close()

#  3) LOCAL EXPLANATION (one sample) 
# pick one row index
idx = 0
plt.figure()
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_to_use[idx],
        base_values=explainer.expected_value if not isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value[1],
        data=X_test.iloc[idx],
        feature_names=X_test.columns
    ),
    show=False
)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_waterfall_idx0.png", dpi=300)
plt.close()

print(f"Saved SHAP plots to: {OUT_DIR.resolve()}")