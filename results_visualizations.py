import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Create DataFrame
# -------------------------------
data = {
    "Model": ["Random Forest", "XGBoost"],
    "Optimal Threshold": [0.32, 0.30],
    "Accuracy": [0.6242, 0.6801],
    "Precision": [0.2422, 0.2390],
    "Recall": [0.6393, 0.4623],
    "F1 Score": [0.3514, 0.3151],
    "ROC AUC Score": [0.6672, 0.6273],
}

df = pd.DataFrame(data)
df_metrics = df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"]]

# -------------------------------
# Plotting
# -------------------------------
metrics = df_metrics.columns
x = np.arange(len(metrics))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Bars for each model
ax.bar(x - width/2, df_metrics.loc["Random Forest"], width, label="Random Forest", color="#1f77b4")
ax.bar(x + width/2, df_metrics.loc["XGBoost"], width, label="XGBoost", color="#ff7f0e")

# Labels and title
ax.set_ylabel("Score")
ax.set_title("Comparison of Model Performance Metrics")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Annotate bar values with 4 decimal places
for i in range(len(metrics)):
    ax.text(i - width/2, df_metrics.loc["Random Forest"][i] + 0.02,
            f"{df_metrics.loc['Random Forest'][i]:.4f}", ha='center', fontsize=9)
    ax.text(i + width/2, df_metrics.loc["XGBoost"][i] + 0.02,
            f"{df_metrics.loc['XGBoost'][i]:.4f}", ha='center', fontsize=9)

# -------------------------------
# Save and show figure
# -------------------------------
plt.tight_layout()
plt.savefig("model_performance_comparison.png", dpi=300)
plt.close()
