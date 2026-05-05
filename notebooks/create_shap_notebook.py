import nbformat as nbf
import os

def create_shap_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    # Title markdown
    cells.append(nbf.v4.new_markdown_cell("# SHAP Analysis for Dynamic Pricing Model\n\nThis notebook loads the trained XGBoost model and the test set, then uses SHAP (SHapley Additive exPlanations) to interpret the model globally and locally. All visualizations are accompanied by business‑focused interpretations."))
    # Imports code cell
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Plot style
sns.set_theme(style='whitegrid')
"""))
    # Load model and data
    cells.append(nbf.v4.new_code_cell("""# Paths (relative to notebook location)
model_path = '../models/xgboost_pricing_v1.pkl'
xgb_model = joblib.load(model_path)
features_path = '../data/processed/features.parquet'
df = pd.read_parquet(features_path)
# Ensure chronological order
df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
"""))
    # Define feature columns and split
    cells.append(nbf.v4.new_code_cell("""feature_cols = [
    'price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
    'demand_score_7d', 'demand_score_30d', 'demand_velocity',
    'inventory_ratio', 'price_percentile_in_category',
    'competitor_delta', 'review_elasticity'
]
X = df[feature_cols]
y = df['optimal_price']
# Chronological 80/20 split
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f'Loaded test set: {X_test.shape[0]} rows, {X_test.shape[1]} features')
"""))
    # Global SHAP summary (beeswarm)
    cells.append(nbf.v4.new_markdown_cell("## 1. Global SHAP Summary (Beeswarm)\n\nThe summary plot shows the distribution of SHAP values for each feature across the test set. Positive values push the price up, negative values push it down. We display the top 12 most influential features."))
    cells.append(nbf.v4.new_code_cell("""explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
# Beeswarm summary – limit to top 12 features
shap.summary_plot(shap_values, X_test, max_display=12, plot_size=(10,6), show=False)
plt.title('SHAP Summary (Beeswarm) – Top 12 Features')
plt.tight_layout()
plt.show()
"""))
    # SHAP bar plot
    cells.append(nbf.v4.new_markdown_cell("## 2. SHAP Bar Plot (Mean Absolute)\n\nBar plot of mean absolute SHAP values, again limited to the top 12 features, provides an easy ranking of feature importance."))
    cells.append(nbf.v4.new_code_cell("""shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=12, show=False)
plt.title('Mean Absolute SHAP Values – Top 12 Features')
plt.tight_layout()
plt.show()
"""))
    # Dependence plots for top 3 features
    cells.append(nbf.v4.new_markdown_cell("## 3. SHAP Dependence Plots for Top 3 Features\n\nThese plots show how a feature’s value interacts with another feature (color) to affect the prediction. We use the three most important features from the bar plot."))
    cells.append(nbf.v4.new_code_cell("""# Identify top 3 features by mean absolute SHAP
mean_abs = np.mean(np.abs(shap_values), axis=0)
top3_idx = np.argsort(mean_abs)[-3:][::-1]
top3_features = [X_test.columns[i] for i in top3_idx]
for feat in top3_features:
    shap.dependence_plot(feat, shap_values, X_test, show=False)
    plt.title(f'SHAP Dependence – {feat}')
    plt.tight_layout()
    plt.show()
"""))
    # Local explanations for 3 SKUs
    cells.append(nbf.v4.new_markdown_cell("## 4. Local Explanations – Waterfall Plots\n\nWe select three representative SKUs:\n- High price (95th percentile)\n- Low price (5th percentile)\n- Outlier price (beyond 3 std devs)\nThe waterfall plot shows how each feature contributes to that SKU’s predicted price."))
    cells.append(nbf.v4.new_code_cell("""# Select indices
high_idx = y_test.quantile(0.95)
low_idx = y_test.quantile(0.05)
outlier_idx = y_test[y_test > (y_test.mean() + 3*y_test.std())].index[0]
# Get the nearest row indices in X_test
high_row = X_test.iloc[(y_test - high_idx).abs().argsort().iloc[0]]
low_row = X_test.iloc[(y_test - low_idx).abs().argsort().iloc[0]]
outlier_row = X_test.loc[outlier_idx]
for name, row in [('High Price', high_row), ('Low Price', low_row), ('Outlier', outlier_row)]:
    shap_vals = explainer.shap_values(row)
    shap.plots.waterfall(shap.Explanation(values=shap_vals, base_values=explainer.expected_value, data=row), show=False)
    plt.title(f'Waterfall – {name} SKU')
    plt.tight_layout()
    plt.show()
"""))
    # Stability analysis
    cells.append(nbf.v4.new_markdown_cell("## 5. Feature Stability Across Subsamples\n\nWe repeatedly sample 20% of the test set (5 random seeds) and compute SHAP rankings. The heat‑map shows how often each feature appears in the top‑5, indicating stability of explanations."))
    cells.append(nbf.v4.new_code_cell("""import random
from collections import Counter
rankings = []
for seed in range(5):
    sample = X_test.sample(frac=0.2, random_state=seed)
    expl = shap.TreeExplainer(xgb_model)
    sv = expl.shap_values(sample)
    mean_abs = np.mean(np.abs(sv), axis=0)
    top5 = np.array(sample.columns)[np.argsort(mean_abs)[-5:]][::-1]
    rankings.append(list(top5))
# Frequency of each feature in top‑5
flat = [f for sub in rankings for f in sub]
freq = Counter(flat)
freq_df = pd.DataFrame.from_dict(freq, orient='index', columns=['count']).sort_values('count', ascending=False)
# Plot frequency
plt.figure(figsize=(8,4))
freq_df.plot(kind='bar', legend=False)
plt.title('Feature Frequency in Top‑5 Across 5 Subsamples')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
"""))
    # Save notebook
    nb['cells'] = cells
    output_path = '04_shap_analysis.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Notebook saved to {output_path}')

if __name__ == '__main__':
    create_shap_notebook()
