import pandas as pd
import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import sys
import os
from tqdm import tqdm

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- Helper functions for multiprocessing and pickling ---
def get_days_to_xmas(dt):
    xmas = pd.Timestamp(year=dt.year, month=12, day=25)
    if dt > xmas:
        xmas = pd.Timestamp(year=dt.year+1, month=12, day=25)
    return (xmas - dt).days

def get_category(desc):
    category_rules = {
      "HOME_DECOR": ["DECORATION","DECOR","ORNAMENT","CANDLE","HOLDER","FRAME","MIRROR"],
      "KITCHEN": ["MUG","CUP","PLATE","BOWL","JAR","BOTTLE","KITCHEN","STORAGE"],
      "STATIONERY": ["CARD","WRAP","PAPER","PEN","NOTEBOOK","LABEL","STAMP"],
      "GIFT": ["GIFT","BOX","PACK","BAG","WRAP","RIBBON","TAG"],
      "SEASONAL": ["CHRISTMAS","XMAS","EASTER","HALLOWEEN","HEART","VALENTINE"],
      "TEXTILE": ["CUSHION","BLANKET","TOWEL","FABRIC","CLOTH","BAG"],
      "GARDEN": ["GARDEN","PLANT","OUTDOOR","BIRD"]
    }
    if not isinstance(desc, str): return "OTHER"
    desc_upper = desc.upper()
    for cat, keywords in category_rules.items():
        if any(k in desc_upper for k in keywords):
            return cat
    return "OTHER"

def calculate_slope(data):
    if len(data) < 2: return 0
    try:
        slope, _, _, _, _ = stats.linregress(np.arange(len(data)), data)
        return slope
    except:
        return 0

def get_slope_for_stock(args):
    stock_code, stock_df = args
    last_30_days = stock_df.tail(30)["daily_qty"].values
    return stock_code, calculate_slope(last_30_days)

def simulate_prices(group):
    seed = hash(str(group.name)) % 10000
    np.random.seed(seed)
    group['competitor_price_1'] = group['product_avg_price_global'] * np.random.normal(0.95, 0.08, size=len(group))
    np.random.seed(seed + 1)
    group['competitor_price_2'] = group['product_avg_price_global'] * np.random.normal(1.02, 0.06, size=len(group))
    return group

# --- Feature Groups ---

def add_temporal_features(df):
    """GROUP 1: TEMPORAL FEATURES"""
    print("[*] Adding temporal features...")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["hour"] = df.invoice_date.dt.hour
    df["day_of_week"] = df.invoice_date.dt.dayofweek
    df["day_of_month"] = df.invoice_date.dt.day
    df["week_of_year"] = df.invoice_date.dt.isocalendar().week.astype(int)
    df["month"] = df.invoice_date.dt.month
    df["quarter"] = df.invoice_date.dt.quarter
    df["year"] = df.invoice_date.dt.year
    
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["is_month_start"] = (df.day_of_month <= 3).astype(int)
    df["is_month_end"] = (df.day_of_month >= 28).astype(int)
    
    df["is_christmas_season"] = ((df.month == 11) | (df.month == 12)).astype(int)
    df["is_valentine_week"] = ((df.month == 2) & (df.week_of_year == 7)).astype(int)
    df["is_easter_approx"] = ((df.month == 4) & (df.week_of_year.isin([14, 15]))).astype(int)
    df["is_summer"] = df.month.isin([6, 7, 8]).astype(int)
    df["is_new_year"] = ((df.month == 1) & (df.day_of_month <= 7)).astype(int)
    
    # Optimize with unique dates
    unique_dates = pd.Series(df["invoice_date"].unique())
    date_to_xmas = {d: get_days_to_xmas(d) for d in unique_dates}
    df["days_to_christmas"] = df["invoice_date"].map(date_to_xmas).clip(0, 365)
    
    # Time of day: morning (6-12), afternoon (12-17), evening (17-21), night (21-6)
    df["time_of_day"] = pd.cut(df.hour, bins=[0, 6, 12, 17, 21, 24], labels=[3, 0, 1, 2, 3], ordered=False, include_lowest=True).astype(int)
    
    return df

def add_product_features(df):
    """GROUP 2: PRODUCT-LEVEL FEATURES"""
    print("[*] Adding product-level features...")
    df = df.sort_values(["stock_code", "invoice_date"])
    
    gp = df.groupby("stock_code")
    df["product_avg_price_global"] = gp["unit_price"].transform("mean")
    df["product_price_std"] = gp["unit_price"].transform("std").fillna(0)
    df["product_price_min"] = gp["unit_price"].transform("min")
    df["product_price_max"] = gp["unit_price"].transform("max")
    df["product_price_range"] = df["product_price_max"] - df["product_price_min"]
    df["price_vs_product_mean"] = df["unit_price"] / df["product_avg_price_global"]
    
    df["product_total_revenue"] = gp["revenue"].transform("sum")
    df["product_transaction_count"] = gp["invoice_no"].transform("count")
    
    # Popularity rank
    counts = df.groupby("stock_code")["invoice_no"].count()
    labels = ["tail", "low", "mid", "top20", "top5"]
    try:
        ranks = pd.qcut(counts, 5, labels=labels)
    except:
        ranks = pd.cut(counts, 5, labels=labels)
    rank_map = {"tail": 1, "low": 2, "mid": 3, "top20": 4, "top5": 5}
    df["product_popularity_rank"] = df["stock_code"].map(ranks).map(rank_map).fillna(1).astype(int)
    
    df["product_avg_quantity"] = gp["quantity"].transform("mean")
    
    unique_descs = df["description"].unique()
    desc_cat_map = {d: get_category(d) for d in unique_descs}
    df["product_category"] = df["description"].map(desc_cat_map)
    
    # Label encode
    cats = sorted(df["product_category"].unique())
    cat_to_id = {cat: i for i, cat in enumerate(cats)}
    df["product_category_id"] = df["product_category"].map(cat_to_id)
    
    df["product_name_length"] = df["description"].fillna("").str.split().str.len()
    return df

def add_demand_features(df):
    """GROUP 3: DEMAND & SALES VELOCITY FEATURES"""
    print("[*] Adding demand features...")
    df = df.sort_values(["stock_code", "invoice_date"])
    
    for w in config.DEMAND_WINDOWS:
        window_str = f"{w}D"
        gp = df.groupby("stock_code")
        
        print(f"    Computing rolling {w}d features...")
        # Rolling features with shift(1) to avoid leakage
        df[f"rolling_demand_{w}d"] = gp.apply(lambda x: x.rolling(window_str, on="invoice_date")["quantity"].sum().shift(1)).reset_index(level=0, drop=True).fillna(0)
        df[f"rolling_revenue_{w}d"] = gp.apply(lambda x: x.rolling(window_str, on="invoice_date")["revenue"].sum().shift(1)).reset_index(level=0, drop=True).fillna(0)
        df[f"rolling_transactions_{w}d"] = gp.apply(lambda x: x.rolling(window_str, on="invoice_date")["invoice_no"].count().shift(1)).reset_index(level=0, drop=True).fillna(0)
        df[f"rolling_avg_price_{w}d"] = gp.apply(lambda x: x.rolling(window_str, on="invoice_date")["unit_price"].mean().shift(1)).reset_index(level=0, drop=True).fillna(method='ffill').fillna(df["product_avg_price_global"])
        
        if w != 7:
            df[f"demand_acceleration_{w}d"] = (df["rolling_demand_7d"] / df[f"rolling_demand_{w}d"]).replace([np.inf, -np.inf], 0).fillna(0).clip(0, 10)
            
    # Lags
    print("    Computing lag features...")
    for l in config.LAG_DAYS:
        df[f"price_lag_{l}d"] = df.groupby("stock_code")["unit_price"].shift(l).fillna(method='ffill')
        df[f"demand_lag_{l}d"] = df.groupby("stock_code")["quantity"].shift(l).fillna(0)

    # Demand trend slope (HEAVY)
    print("    Computing demand trend slope (multiprocessing)...")
    daily_demand = df.groupby(["stock_code", df.invoice_date.dt.date])["quantity"].sum().reset_index()
    daily_demand.columns = ["stock_code", "date", "daily_qty"]
    
    unique_stocks = list(daily_demand.groupby("stock_code"))
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(get_slope_for_stock, unique_stocks), total=len(unique_stocks)))
    
    slope_map = dict(results)
    df["demand_trend_slope"] = df["stock_code"].map(slope_map).fillna(0)
    
    return df

def add_elasticity_features(df):
    """GROUP 4: PRICE ELASTICITY FEATURES"""
    print("[*] Adding elasticity features...")
    df["pct_price_change"] = (df["unit_price"] - df["rolling_avg_price_30d"]) / df["rolling_avg_price_30d"]
    df["pct_demand_change"] = (df["rolling_demand_7d"] - df["rolling_demand_30d"]/4) / (df["rolling_demand_30d"]/4)
    
    df["price_elasticity_30d"] = df["pct_demand_change"] / df["pct_price_change"]
    df["price_elasticity_30d"] = df["price_elasticity_30d"].replace([np.inf, -np.inf], np.nan).clip(-10, 0)
    
    cat_median = df.groupby("product_category")["price_elasticity_30d"].transform("median")
    df["price_elasticity_30d"] = df["price_elasticity_30d"].fillna(cat_median).fillna(-1.5)
    
    df["is_price_elastic"] = (df["price_elasticity_30d"] < -1).astype(int)
    df["optimal_price_elasticity_estimate"] = df["unit_price"] * (df["price_elasticity_30d"] / (df["price_elasticity_30d"] + 1))
    df["optimal_price_elasticity_estimate"] = df["optimal_price_elasticity_estimate"].clip(df["product_price_min"], df["product_price_max"] * 1.5)
    
    return df

def add_customer_features(df):
    """GROUP 5: CUSTOMER & GEOGRAPHY FEATURES"""
    print("[*] Adding customer features...")
    df["country_avg_price"] = df.groupby("country")["unit_price"].transform("mean")
    df["country_demand_share"] = df.groupby("country")["invoice_no"].transform("count") / len(df)
    df["is_uk"] = (df.country == "United Kingdom").astype(int)
    
    df["customer_lifetime_value"] = df.groupby("customer_id")["revenue"].transform("sum")
    df["customer_purchase_frequency"] = df.groupby("customer_id")["invoice_date"].transform("nunique")
    df["customer_avg_basket_size"] = df.groupby("customer_id")["quantity"].transform("mean")
    df["customer_price_sensitivity"] = df.groupby("customer_id")["unit_price"].transform("std").fillna(0)
    
    df["is_wholesale_customer"] = ((df["is_bulk"] == 1) | (df["customer_avg_basket_size"] > 20)).astype(int)
    
    return df

def add_inventory_features(df):
    """GROUP 6: INVENTORY PROXY FEATURES"""
    print("[*] Adding inventory proxy features...")
    df = df.sort_values(["stock_code", "invoice_date"])
    gp = df.groupby("stock_code")
    
    df["cumulative_units_sold"] = gp["quantity"].transform("cumsum")
    df["units_sold_today"] = df.groupby(["stock_code", df.invoice_date.dt.date])["quantity"].transform("sum")
    df["units_sold_vs_30d_avg"] = df["units_sold_today"] / (df["rolling_demand_30d"] / 30).replace(0, 1)
    
    df["days_since_last_sale"] = gp["invoice_date"].diff().dt.days.fillna(0).clip(0, 90)
    
    df["stockout_risk_score"] = (df["demand_acceleration_30d"] * 0.5) + (df["units_sold_vs_30d_avg"].clip(0, 5) / 5 * 0.3) + ((1 / (df["days_since_last_sale"] + 1)) * 0.2)
    df["stockout_risk_score"] = (df["stockout_risk_score"] - df["stockout_risk_score"].min()) / (df["stockout_risk_score"].max() - df["stockout_risk_score"].min())
    
    return df

def add_competitor_features(df):
    """GROUP 7: COMPETITOR PRICE FEATURES"""
    print("[*] Adding competitor features...")
    df = df.groupby('stock_code', group_keys=False).apply(simulate_prices)
    df['competitor_price_1'] = df['competitor_price_1'].clip(config.MIN_UNIT_PRICE, config.MAX_UNIT_PRICE)
    df['competitor_price_2'] = df['competitor_price_2'].clip(config.MIN_UNIT_PRICE, config.MAX_UNIT_PRICE)
    df['competitor_avg_price'] = (df['competitor_price_1'] + df['competitor_price_2']) / 2
    df['price_vs_competitor'] = df['unit_price'] / df['competitor_avg_price']
    df['competitor_price_gap'] = df['unit_price'] - df['competitor_avg_price']
    df['is_price_competitive'] = (df['price_vs_competitor'] <= 1.05).astype(int)
    return df

def add_flash_sale_features(df):
    """GROUP 8: FLASH SALE / ANOMALY FEATURES"""
    print("[*] Adding flash sale features...")
    hourly_stats = df.groupby(["stock_code", "hour"])["quantity"].agg(["mean", "std"]).reset_index()
    hourly_stats.columns = ["stock_code", "hour", "h_mean", "h_std"]
    df = df.merge(hourly_stats, on=["stock_code", "hour"], how="left")
    
    df["hourly_demand_zscore"] = (df["quantity"] - df["h_mean"]) / df["h_std"].replace(0, 1)
    df["is_flash_sale_window"] = (df["hourly_demand_zscore"] > 2.5).astype(int)
    
    df["price_discount_pct"] = (df["product_avg_price_global"] - df["unit_price"]) / df["product_avg_price_global"] * 100
    df["is_discounted"] = (df["price_discount_pct"] > 5).astype(int)
    
    return df

def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to build all features."""
    print(f"[*] Starting feature engineering on {len(df):,} rows")
    
    df = add_temporal_features(df)
    df = add_product_features(df)
    df = add_demand_features(df)
    df = add_elasticity_features(df)
    df = add_customer_features(df)
    df = add_inventory_features(df)
    df = add_competitor_features(df)
    df = add_flash_sale_features(df)
    
    print("[*] Finalizing features...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    demand_cols = [c for c in df.columns if "rolling" in c or "lag" in c]
    df[demand_cols] = df[demand_cols].fillna(0)
    df = df.fillna(0)
    
    print(f"[*] Total features built: {len(df.columns)}")
    
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.FEATURES_DIR / "features.parquet"
    df.to_parquet(save_path, index=False)
    print(f"[*] Saved features to {save_path}")
    
    return df

if __name__ == "__main__":
    from clean import clean
    from ingest import load_or_merge
    df = load_or_merge()
    df = clean(df)
    build_all_features(df)
