import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def evaluate_model(model_name, y_true, y_pred, prices_actual=None):
    """Phase 7: Evaluate model with ML and Business metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.01))) * 100
    r2 = r2_score(y_true, y_pred)
    
    median_abs_error = np.median(np.abs(y_true - y_pred))
    within_5pct = np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 0.01) <= 0.05) * 100
    within_10pct = np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 0.01) <= 0.10) * 100
    within_20pct = np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 0.01) <= 0.20) * 100
    
    metrics = {
        "model": model_name,
        "mae_gbp": round(float(mae), 4),
        "rmse_gbp": round(float(rmse), 4),
        "mape_pct": round(float(mape), 2),
        "r2_score": round(float(r2), 4),
        "median_abs_error": round(float(median_abs_error), 4),
        "within_5pct": round(float(within_5pct), 1),
        "within_10pct": round(float(within_10pct), 1),
        "within_20pct": round(float(within_20pct), 1)
    }
    
    print(f"\n--- {model_name} Results ---")
    # Pass/Fail benchmarks
    print(f"{'MAE < £0.50':25}: {'PASS' if mae < 0.5 else 'FAIL'} ({mae:.4f})")
    print(f"{'MAPE < 5%':25}: {'PASS' if mape < 5 else 'FAIL'} ({mape:.2f}%)")
    print(f"{'R2 > 0.90':25}: {'PASS' if r2 > 0.9 else 'FAIL'} ({r2:.4f})")
    print(f"{'Within 10% acc > 85%':25}: {'PASS' if within_10pct > 85 else 'FAIL'} ({within_10pct:.1f}%)")
    
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.METRICS_DIR / f"{model_name}_metrics.json"
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def compare_all_models(metrics_list):
    """Print formatted comparison table."""
    df = pd.DataFrame(metrics_list)
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(df.to_string(index=False))
    
    save_path = config.METRICS_DIR / "comparison.json"
    df.to_json(save_path, orient="records", indent=4)
    print(f"\nSaved comparison to {save_path}")

def plot_diagnostics(y_true, y_pred, model_name, importance_df=None):
    """4-panel diagnostic plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Diagnostics: {model_name}", fontsize=16)
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.3, color='teal')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel("Actual Price")
    axes[0, 0].set_ylabel("Predicted Price")
    axes[0, 0].set_title("Actual vs Predicted")
    
    # 2. Residuals Histogram
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, ax=axes[0, 1], color='purple')
    axes[0, 1].set_title("Residuals Distribution")
    
    # 3. Residuals vs Predicted
    axes[1, 0].scatter(y_pred, residuals, alpha=0.3, color='orange')
    axes[1, 0].axhline(0, color='r', linestyle='--')
    axes[1, 0].set_xlabel("Predicted Price")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Residuals vs Predicted")
    
    # 4. Feature Importance
    if importance_df is not None:
        top_20 = importance_df.head(20)
        sns.barplot(x='importance', y='feature', data=top_20, ax=axes[1, 1], palette='viridis')
        axes[1, 1].set_title("Top 20 Features")
    else:
        axes[1, 1].text(0.5, 0.5, "No importance data", ha='center')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = config.METRICS_DIR / f"{model_name}_diagnostics.png"
    plt.savefig(save_path)
    print(f"Saved diagnostics plot to {save_path}")
    plt.close()

if __name__ == "__main__":
    # Example usage (testing)
    y_true = np.random.normal(10, 2, 100)
    y_pred = y_true + np.random.normal(0, 0.5, 100)
    evaluate_model("test_model", y_true, y_pred)
