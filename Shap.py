import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
MODELS = {
    "XGBoost": Path("models/XGBoost.pkl"),
    "RandomForest": Path("models/RandomForest.pkl"),
}
ARTIFACTS_DIR = Path("artifacts")
OUT_DIR = Path("shap_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD DATA ----------------
X_test = joblib.load(ARTIFACTS_DIR / "X_test.pkl")

# Ensure DataFrame (feature names help a lot)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test)

def ensure_2d_shap_for_positive_class(shap_result):
    """
    Returns SHAP values shaped (n_samples, n_features) for the POSITIVE class (class 1).
    Handles common SHAP return formats across versions/models.
    """
    # Newer SHAP: explainer(X) returns Explanation
    if isinstance(shap_result, shap._explanation.Explanation):
        vals = shap_result.values
        # vals can be (n, f) or (n, f, k)
        if vals.ndim == 3:
            return vals[:, :, 1]  # class 1
        return vals

    # Older SHAP: explainer.shap_values(X) can return list
    if isinstance(shap_result, list):
        return shap_result[1]  # class 1

    # Or ndarray
    if isinstance(shap_result, np.ndarray):
        if shap_result.ndim == 3:
            return shap_result[:, :, 1]
        return shap_result

    raise ValueError(f"Unsupported SHAP result type: {type(shap_result)}")

def expected_value_for_positive_class(explainer):
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)) and len(ev) > 1:
        return ev[1]
    return ev

def run_shap(model_name: str, model_path: Path):
    print(f"\n=== Running SHAP for {model_name} ===")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    model = joblib.load(model_path)

    model_out = OUT_DIR / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    # TreeExplainer works for both XGBoost + RandomForest
    explainer = shap.TreeExplainer(model)

    # Prefer new API if available; fallback to shap_values
    try:
        shap_exp = explainer(X_test)
        shap_vals = ensure_2d_shap_for_positive_class(shap_exp)
        base_val = shap_exp.base_values
        # base_values can be (n,) or (n, k)
        if isinstance(base_val, np.ndarray) and base_val.ndim == 2:
            base_val_pos = base_val[:, 1]
        else:
            base_val_pos = base_val
    except Exception:
        shap_raw = explainer.shap_values(X_test)
        shap_vals = ensure_2d_shap_for_positive_class(shap_raw)
        base_val_pos = expected_value_for_positive_class(explainer)

    # 1) GLOBAL SUMMARY
    shap.summary_plot(shap_vals, X_test, show=False)
    plt.tight_layout()
    plt.savefig(model_out / "shap_summary.png", dpi=300)
    plt.close()

    # 2) BAR SUMMARY
    shap.summary_plot(shap_vals, X_test, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)
    plt.subplots_adjust(bottom=0.22, left=0.30, right=0.98)
    plt.savefig(model_out / "shap_summary_bar.png", dpi=300)
    plt.close()

    # 3) LOCAL WATERFALL (single sample)
    idx = 0
    exp = shap.Explanation(
        values=shap_vals[idx],                 # MUST be 1D (n_features,)
        base_values=base_val_pos[idx] if isinstance(base_val_pos, (np.ndarray, list)) else base_val_pos,
        data=X_test.iloc[idx].values,
        feature_names=list(X_test.columns),
    )
    shap.plots.waterfall(exp, show=False)
    plt.tight_layout()
    plt.savefig(model_out / "shap_waterfall_idx0.png", dpi=300)
    plt.close()

    print(f"Saved to: {model_out.resolve()}")

if __name__ == "__main__":
    for name, path in MODELS.items():
        run_shap(name, path)

    print("\nDone. SHAP generated for all models.")