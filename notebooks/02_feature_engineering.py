#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering for Dynamic Pricing
# 
# This notebook focuses on feature engineering. We start from the merged Olist dataset 
# and compute temporal, demand, inventory, and price features needed for an XGBoost model. 
# Finally, we define the target variable `optimal_price` and save the dataset to a Parquet file.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
# get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Load Data
# 
# First, we reproduce the merging logic from `01_eda_olist.ipynb` to get our starting dataframe. In a production environment, this would be handled by a dedicated ETL script.

# In[ ]:


# Re-run merging logic quickly
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

print(f"Base dataframe shape: {df.shape}")


# ## 2. Temporal Features
# 
# - `hour_of_day`, `day_of_week` with cyclic encoding (sin/cos)
# - `is_weekend`, `is_month_end`, `is_holiday`
# - `days_since_last_order` per SKU

# In[ ]:


df['hour_of_day'] = df['order_purchase_timestamp'].dt.hour
df['day_of_week'] = df['order_purchase_timestamp'].dt.dayofweek

# Cyclic encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

# Weekend and month end
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_month_end'] = df['order_purchase_timestamp'].dt.is_month_end.astype(int)

# Brazilian holidays
br_holidays = holidays.Brazil(years=df['order_purchase_timestamp'].dt.year.unique())
df['is_holiday'] = df['order_purchase_timestamp'].dt.date.apply(lambda x: 1 if x in br_holidays else 0)

# Days since last order per SKU
df['prev_order_time'] = df.groupby('product_id')['order_purchase_timestamp'].shift(1)
df['days_since_last_order'] = (df['order_purchase_timestamp'] - df['prev_order_time']).dt.total_seconds() / (24 * 3600)
df['days_since_last_order'] = df['days_since_last_order'].fillna(-1) # -1 for first order
df.drop('prev_order_time', axis=1, inplace=True)


# ## 3. Demand Features
# 
# Using rolling windows to compute 7-day and 30-day demand scores per product.

# In[ ]:


# Create an index on timestamp for rolling calculations
temp_df = df[['product_id', 'order_purchase_timestamp', 'order_item_id']].copy()
temp_df.set_index('order_purchase_timestamp', inplace=True)
temp_df.sort_index(inplace=True)

# Rolling 7-day and 30-day counts
# We group by product_id and count order items
demand_7d = temp_df.groupby('product_id').rolling('7D').count()['order_item_id'].reset_index()
demand_30d = temp_df.groupby('product_id').rolling('30D').count()['order_item_id'].reset_index()

demand_7d.rename(columns={'order_item_id': 'demand_score_7d'}, inplace=True)
demand_30d.rename(columns={'order_item_id': 'demand_score_30d'}, inplace=True)

# Since multiple orders can happen at the exact same timestamp, 
# we merge back by row index to ensure perfect alignment.
df['demand_score_7d'] = demand_7d['demand_score_7d'].values
df['demand_score_30d'] = demand_30d['demand_score_30d'].values

# Demand Velocity
# Add small epsilon to denominator to avoid division by zero
df['demand_velocity'] = (df['demand_score_7d'] - (df['demand_score_30d'] / 4)) / (df['demand_score_30d'].replace(0, 1))
df['demand_velocity'] = df['demand_velocity'].fillna(0)


# ## 4. Inventory Proxy
# 
# Simulating `inventory_ratio` based on recent demand.

# In[ ]:


df['inventory_ratio'] = 1 / (df['demand_score_7d'] + 1)
df['inventory_ratio'] = df['inventory_ratio'].clip(lower=0.1, upper=1.0)


# ## 5. Price Features
# 
# Comparing SKU price with its category and incorporating review elasticity.

# In[ ]:


# Calculate category-level statistics
df['price_percentile_in_category'] = df.groupby('category')['price'].rank(pct=True)
category_median = df.groupby('category')['price'].transform('median')
df['competitor_delta'] = (df['price'] - category_median) / category_median

# Review elasticity
df['review_elasticity'] = 1.0 + (df['review_score'] - 3.0) * 0.05
df['review_elasticity'] = df['review_elasticity'].clip(lower=0.85, upper=1.15)


# ## 6. Target Variable: Optimal Price
# 
# Defining the dynamic target `optimal_price` using demand velocity and elasticity factors.

# In[ ]:


df['optimal_price'] = df['price'] * (1 + df['demand_velocity'] * 0.1) * df['review_elasticity']

# Clamp between 85% and 135% of original price
df['optimal_price'] = df['optimal_price'].clip(
    lower=df['price'] * 0.85, 
    upper=df['price'] * 1.35
)


# ## 7. Analysis & Visualizations
# 
# We drop unneeded raw columns and plot the feature correlation matrix and distribution plots.

# In[ ]:


feature_cols = [
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


# ### Feature Distributions

# In[ ]:


dist_cols = [
    'days_since_last_order', 'demand_score_7d', 'demand_velocity', 
    'inventory_ratio', 'competitor_delta', 'review_elasticity', 'optimal_price'
]

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

for i, col in enumerate(dist_cols):
    sns.histplot(features_df[col], bins=50, kde=True, ax=axes[i], color='teal')
    axes[i].set_title(f'Distribution of {col}', fontsize=12)
    
fig.delaxes(axes[-1]) # Remove empty subplot
plt.tight_layout()
plt.show()


# ## 8. Export to Parquet
# 
# Saving the finalized features dataframe.

# In[ ]:


output_dir = '../data/processed'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'features.parquet')

features_df.to_parquet(output_path, index=False)
print(f"Successfully saved {features_df.shape[0]} rows and {features_df.shape[1]} columns to {output_path}")

