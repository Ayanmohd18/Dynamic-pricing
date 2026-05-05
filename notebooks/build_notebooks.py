import nbformat as nbf
import os

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# Exploratory Data Analysis: Olist Dataset\nThis notebook loads the 9 Olist datasets, explores their structures, merges them, and performs visual and statistical analyses."))
    
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
# %matplotlib inline"""))

    cells.append(nbf.v4.new_markdown_cell("## 1. Load Data\nWe load all 9 CSV files and display their shapes, dtypes, and null counts."))

    cells.append(nbf.v4.new_code_cell("""data_dir = '../dataset olist/'

files = {
    'customers': 'olist_customers_dataset.csv',
    'geolocation': 'olist_geolocation_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'order_payments': 'olist_order_payments_dataset.csv',
    'order_reviews': 'olist_order_reviews_dataset.csv',
    'orders': 'olist_orders_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'translations': 'product_category_name_translation.csv'
}

dfs = {}
for name, file in files.items():
    dfs[name] = pd.read_csv(os.path.join(data_dir, file))
    print(f"--- {name.upper()} ---")
    print(f"Shape: {dfs[name].shape}")
    print(f"Nulls: {dfs[name].isnull().sum().sum()}")
    print("-" * 30)"""))

    cells.append(nbf.v4.new_markdown_cell("## 2. Merge Data\nMerging datasets into a single master dataframe."))

    cells.append(nbf.v4.new_code_cell("""# Merging strategy
df = dfs['order_items'].merge(dfs['orders'], on='order_id', how='left')
df = df.merge(dfs['products'], on='product_id', how='left')
df = df.merge(dfs['translations'], on='product_category_name', how='left')
df = df.merge(dfs['order_reviews'].drop_duplicates(subset=['order_id']), on='order_id', how='left')
df = df.merge(dfs['customers'], on='customer_id', how='left')
df = df.merge(dfs['sellers'], on='seller_id', how='left')
df = df.merge(dfs['order_payments'].groupby('order_id')['payment_value'].sum().reset_index(), on='order_id', how='left')

# Use english category names
df['category'] = df['product_category_name_english'].fillna(df['product_category_name'])

# Convert timestamps
date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

print(f"Master dataframe shape: {df.shape}")"""))

    cells.append(nbf.v4.new_markdown_cell("## 3. Visualizations\n- Order volume by month (2016-2018)\n- Top 20 product categories by revenue\n- Price distribution (log scale)\n- Review score distribution"))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Order volume by month
df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M')
monthly_orders = df.groupby('year_month')['order_id'].nunique()
monthly_orders.plot(kind='bar', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Order Volume by Month (2016-2018)')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Number of Orders')

# Top 20 categories by revenue
category_revenue = df.groupby('category')['price'].sum().sort_values(ascending=False).head(20)
category_revenue.plot(kind='bar', ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Top 20 Categories by Revenue')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Revenue')

# Price distribution (log scale)
np.log1p(df['price']).plot(kind='hist', bins=50, ax=axes[1, 0], color='green', edgecolor='black')
axes[1, 0].set_title('Price Distribution (Log Scale)')
axes[1, 0].set_xlabel('Log(1 + Price)')

# Review score distribution
df['review_score'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Review Score Distribution')
axes[1, 1].set_xlabel('Review Score')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("## 4. Metrics & Correlation\n- Avg price per category\n- Demand velocity (orders per day per SKU)\n- Price-review correlation heatmap"))

    cells.append(nbf.v4.new_code_cell("""# Avg price per category
avg_price_cat = df.groupby('category')['price'].mean().sort_values(ascending=False)
print("Top 5 Categories by Avg Price:")
print(avg_price_cat.head())

# Demand velocity (orders per day per SKU)
total_days = (df['order_purchase_timestamp'].max() - df['order_purchase_timestamp'].min()).days
sku_orders = df.groupby('product_id')['order_id'].nunique()
demand_velocity = sku_orders / total_days
print("\\nTop 5 SKUs by Demand Velocity (Orders/Day):")
print(demand_velocity.sort_values(ascending=False).head())

# Price-review correlation
corr_df = df[['price', 'review_score', 'freight_value']].dropna()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Price vs Review Score Correlation')
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("## 5. Data Quality Issues\n- Duplicate order_ids\n- SKUs with < 5 orders\n- Price outliers (> 3 std devs)"))

    cells.append(nbf.v4.new_code_cell("""# Duplicates
dup_orders = df['order_id'].duplicated().sum()
print(f"Duplicate order_items rows (expected for multi-item orders): {dup_orders}")

# SKUs with < 5 orders
sku_counts = df['product_id'].value_counts()
low_volume_skus = (sku_counts < 5).sum()
print(f"SKUs with < 5 orders: {low_volume_skus} out of {len(sku_counts)} ({low_volume_skus/len(sku_counts):.1%})")

# Price outliers (> 3 std devs)
price_mean = df['price'].mean()
price_std = df['price'].std()
outliers = df[df['price'] > (price_mean + 3 * price_std)]
print(f"Price outliers (> 3 std devs): {len(outliers)} items")"""))

    nb['cells'] = cells
    with open('01_eda_olist.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def create_feature_engineering_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("# Feature Engineering for Dynamic Pricing\nEngineer ALL features needed for an XGBoost dynamic pricing model from the merged dataframe."))
    
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Load and prepare merged data (Simulation of ETL)
data_dir = '../dataset olist/'
order_items = pd.read_csv(os.path.join(data_dir, 'olist_order_items_dataset.csv'))
orders = pd.read_csv(os.path.join(data_dir, 'olist_orders_dataset.csv'))
products = pd.read_csv(os.path.join(data_dir, 'olist_products_dataset.csv'))
translations = pd.read_csv(os.path.join(data_dir, 'product_category_name_translation.csv'))
reviews = pd.read_csv(os.path.join(data_dir, 'olist_order_reviews_dataset.csv'))

df = order_items.merge(orders[['order_id', 'order_status', 'order_purchase_timestamp']], on='order_id', how='left')
df = df.merge(products[['product_id', 'product_category_name']], on='product_id', how='left')
df = df.merge(translations, on='product_category_name', how='left')
df['category'] = df['product_category_name_english'].fillna(df['product_category_name'])
df.drop(['product_category_name', 'product_category_name_english'], axis=1, inplace=True)
reviews_dedup = reviews.drop_duplicates(subset=['order_id'], keep='first')
df = df.merge(reviews_dedup[['order_id', 'review_score']], on='order_id', how='left')

df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
df['review_score'] = df['review_score'].fillna(df['review_score'].median())
df['category'] = df['category'].fillna('unknown')

print(f"Base dataframe shape: {df.shape}")"""))

    cells.append(nbf.v4.new_markdown_cell("## Temporal Features\n- hour_of_day, day_of_week encoded as sin/cos\n- is_weekend, is_month_end, is_holiday\n- days_since_last_order per SKU"))

    cells.append(nbf.v4.new_code_cell("""df['hour_of_day'] = df['order_purchase_timestamp'].dt.hour
df['day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek

df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_month_end'] = df['order_purchase_timestamp'].dt.is_month_end.astype(int)

br_holidays = holidays.Brazil(years=df['order_purchase_timestamp'].dt.year.unique())
df['is_holiday'] = df['order_purchase_timestamp'].dt.date.apply(lambda x: 1 if x in br_holidays else 0)

df['prev_order_time'] = df.groupby('product_id')['order_purchase_timestamp'].shift(1)
df['days_since_last_order'] = (df['order_purchase_timestamp'] - df['prev_order_time']).dt.total_seconds() / (24 * 3600)
df['days_since_last_order'] = df['days_since_last_order'].fillna(-1)
df.drop('prev_order_time', axis=1, inplace=True)"""))

    cells.append(nbf.v4.new_markdown_cell("## Demand Features\n- demand_score_7d\n- demand_score_30d\n- demand_velocity"))

    cells.append(nbf.v4.new_code_cell("""temp_df = df[['product_id', 'order_purchase_timestamp', 'order_item_id']].copy()
temp_df.set_index('order_purchase_timestamp', inplace=True)
temp_df.sort_index(inplace=True)

demand_7d = temp_df.groupby('product_id').rolling('7D').count()['order_item_id'].reset_index()
demand_30d = temp_df.groupby('product_id').rolling('30D').count()['order_item_id'].reset_index()

df['demand_score_7d'] = demand_7d['order_item_id'].values
df['demand_score_30d'] = demand_30d['order_item_id'].values

df['demand_velocity'] = (df['demand_score_7d'] - (df['demand_score_30d'] / 4)) / (df['demand_score_30d'].replace(0, 1))
df['demand_velocity'] = df['demand_velocity'].fillna(0)"""))

    cells.append(nbf.v4.new_markdown_cell("## Inventory Proxy\n- inventory_ratio: simulate as 1 / (demand_score_7d + 1), clipped to [0.1, 1.0]"))

    cells.append(nbf.v4.new_code_cell("""df['inventory_ratio'] = 1 / (df['demand_score_7d'] + 1)
df['inventory_ratio'] = df['inventory_ratio'].clip(lower=0.1, upper=1.0)"""))

    cells.append(nbf.v4.new_markdown_cell("## Price Features\n- price_percentile_in_category\n- competitor_delta\n- review_elasticity"))

    cells.append(nbf.v4.new_code_cell("""df['price_percentile_in_category'] = df.groupby('category')['price'].rank(pct=True)
category_median = df.groupby('category')['price'].transform('median')
df['competitor_delta'] = (df['price'] - category_median) / category_median

df['review_elasticity'] = 1.0 + (df['review_score'] - 3.0) * 0.05
df['review_elasticity'] = df['review_elasticity'].clip(lower=0.85, upper=1.15)"""))

    cells.append(nbf.v4.new_markdown_cell("## Target Variable\n- optimal_price = price * (1 + demand_velocity * 0.1) * review_elasticity, clipped to [price * 0.85, price * 1.35]"))

    cells.append(nbf.v4.new_code_cell("""df['optimal_price'] = df['price'] * (1 + df['demand_velocity'] * 0.1) * df['review_elasticity']
df['optimal_price'] = df['optimal_price'].clip(
    lower=df['price'] * 0.85, 
    upper=df['price'] * 1.35
)"""))

    cells.append(nbf.v4.new_markdown_cell("## Output and Visualizations"))

    cells.append(nbf.v4.new_code_cell("""feature_cols = [
    'price', 'freight_value',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
    'demand_score_7d', 'demand_score_30d', 'demand_velocity',
    'inventory_ratio', 'price_percentile_in_category',
    'competitor_delta', 'review_elasticity', 'optimal_price'
]

features_df = df[['order_id', 'product_id', 'category', 'order_purchase_timestamp'] + feature_cols].copy()

# Correlation Matrix
plt.figure(figsize=(14, 10))
corr = features_df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.show()

# Distributions
dist_cols = ['days_since_last_order', 'demand_score_7d', 'demand_velocity', 
             'inventory_ratio', 'competitor_delta', 'review_elasticity', 'optimal_price']

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

for i, col in enumerate(dist_cols):
    sns.histplot(features_df[col], bins=50, kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    
fig.delaxes(axes[-1])
plt.tight_layout()
plt.show()

# Export
output_dir = '../data/processed'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'features.parquet')

features_df.to_parquet(output_path, index=False)
print(f"Successfully saved {features_df.shape[0]} rows and {features_df.shape[1]} columns to {output_path}")"""))

    nb['cells'] = cells
    with open('02_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


def create_model_experimentation_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("# Model Experimentation\nTrain pricing models on `features.parquet` with chronological train/test split. Target variable is `optimal_price`."))

    # ── Cell 1: imports + data load merged so every cell is runnable standalone ──
    cells.append(nbf.v4.new_markdown_cell("## 1. Imports & Setup"))

    cells.append(nbf.v4.new_code_cell("""# ── Imports ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import time
import warnings

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
print("Libraries loaded ✓")"""))

    cells.append(nbf.v4.new_markdown_cell("## 2. Load Data & Chronological Split\nLoad `features.parquet`, sort by time, and split 80/20."))

    cells.append(nbf.v4.new_code_cell("""features_path = '../data/processed/features.parquet'
df = pd.read_parquet(features_path)

# Sort by timestamp
df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)

# Select features
features = [
    'price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
    'demand_score_7d', 'demand_score_30d', 'demand_velocity',
    'inventory_ratio', 'price_percentile_in_category',
    'competitor_delta', 'review_elasticity'
]
target = 'optimal_price'

X = df[features]
y = df[target]

# 80/20 chronological split — NOT random, no data leakage
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Dataset shape:  {df.shape}")
print(f"Train size:     {X_train.shape}")
print(f"Test size:      {X_test.shape}")
print(f"Date range:     {df['order_purchase_timestamp'].min().date()} → {df['order_purchase_timestamp'].max().date()}")
print(f"Train cutoff:   {df['order_purchase_timestamp'].iloc[split_idx].date()}")"""))

    cells.append(nbf.v4.new_markdown_cell("## 3. Baseline: Linear Regression"))

    cells.append(nbf.v4.new_code_cell("""lr_model = LinearRegression()
start_time = time.time()
lr_model.fit(X_train, y_train)
lr_time = time.time() - start_time

lr_preds = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)

print(f"Linear Regression | MAE: ₹{lr_mae:.2f} | RMSE: {lr_rmse:.2f} | R²: {lr_r2:.4f}")"""))

    cells.append(nbf.v4.new_markdown_cell("## 4. Random Forest Regressor"))

    cells.append(nbf.v4.new_code_cell("""rf_model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=42)

start_time = time.time()
rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

rf_preds = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)

print(f"Random Forest | MAE: ₹{rf_mae:.2f} | RMSE: {rf_rmse:.2f} | R²: {rf_r2:.4f}")

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
importances.plot(kind='bar', ax=axes[0], color='indigo')
axes[0].set_title('RF Feature Importances')

# Predicted vs Actual
axes[1].scatter(y_test, rf_preds, alpha=0.3, color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[1].set_title('RF Predicted vs Actual')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')

# Residuals
residuals = y_test - rf_preds
sns.histplot(residuals, bins=50, ax=axes[2], color='red')
axes[2].set_title('RF Residual Distribution')

plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("## 5. XGBoost Regressor\nUsing TimeSeriesSplit for CV. Early stopping on a held-out 20% validation slice."))

    cells.append(nbf.v4.new_code_cell("""xgb_model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=7,
    subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50,
    random_state=42
)

# Use last 10% of training data as validation set for early stopping
tscv = TimeSeriesSplit(n_splits=5)
# In standard practice we fit with an eval set
eval_set = [(X_train.iloc[int(len(X_train)*0.8):], y_train.iloc[int(len(y_train)*0.8):])]
X_train_sub = X_train.iloc[:int(len(X_train)*0.8)]
y_train_sub = y_train.iloc[:int(len(y_train)*0.8)]

start_time = time.time()
xgb_model.fit(X_train_sub, y_train_sub, eval_set=eval_set, verbose=False)
xgb_time = time.time() - start_time

xgb_preds = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2 = r2_score(y_test, xgb_preds)

print(f"XGBoost | MAE: ₹{xgb_mae:.2f} | RMSE: {xgb_rmse:.2f} | R²: {xgb_r2:.4f}")

if xgb_mae < 8:
    print("\\n✅ Target Met: XGBoost MAE is under ₹8")
else:
    print(f"\\n❌ Target Not Met: XGBoost MAE is ₹{xgb_mae:.2f} (Target < ₹8)")

# Plot learning curve
results = xgb_model.evals_result()
plt.figure(figsize=(8, 4))
plt.plot(results['validation_0']['rmse'], label='Eval RMSE')
plt.title('XGBoost Learning Curve')
plt.xlabel('Boosting Rounds')
plt.ylabel('RMSE')
plt.legend()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("## 6. Comparison Table & Save Model"))

    cells.append(nbf.v4.new_code_cell("""metrics = {
    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
    'MAE': [lr_mae, rf_mae, xgb_mae],
    'RMSE': [lr_rmse, rf_rmse, xgb_rmse],
    'R²': [lr_r2, rf_r2, xgb_r2],
    'Inference Time (s)': [lr_time, rf_time, xgb_time]
}

metrics_df = pd.DataFrame(metrics).round(4)
display(metrics_df)

# Save best model
models_dir = '../models'
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'xgboost_pricing_v1.pkl')
joblib.dump(xgb_model, model_path)
print(f"Saved XGBoost model to {model_path}")"""))

    nb['cells'] = cells
    with open('03_model_experimentation.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    import sys
    # Pass 'all' or specific notebook names as args; default rebuilds only notebook 03
    targets = sys.argv[1:] if len(sys.argv) > 1 else ['03']
    if 'all' in targets or '01' in targets:
        create_eda_notebook()
        print("01_eda_olist.ipynb rebuilt")
    if 'all' in targets or '02' in targets:
        create_feature_engineering_notebook()
        print("02_feature_engineering.ipynb rebuilt")
    if 'all' in targets or '03' in targets:
        create_model_experimentation_notebook()
        print("03_model_experimentation.ipynb rebuilt")
    print("Done!")
