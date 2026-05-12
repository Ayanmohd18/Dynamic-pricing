import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 5: Create pricing target variables based on future revenue optimization."""
    print("[*] Creating pricing target variables...")
    
    # Sort for time-series operations
    df = df.sort_values(["stock_code", "invoice_date"])
    
    # STEP 1 — Compute forward revenue & demand (NEXT 7 days)
    print("    Computing forward revenue/demand (7-day horizon)...")
    # Trick: Sort in reverse, compute rolling sum (past in reversed = future in normal), then reverse back.
    df_rev = df.iloc[::-1].copy()
    gp_rev = df_rev.groupby("stock_code")
    
    # We use a 7-day time window
    df["forward_revenue_7d"] = gp_rev.apply(
        lambda x: x.rolling("7D", on="invoice_date")["revenue"].sum().shift(1)
    ).reset_index(level=0, drop=True).iloc[::-1].fillna(0)
    
    df["forward_demand_7d"] = gp_rev.apply(
        lambda x: x.rolling("7D", on="invoice_date")["quantity"].sum().shift(1)
    ).reset_index(level=0, drop=True).iloc[::-1].fillna(0)
    
    # STEP 2 — Identify historical revenue-maximizing price
    print("    Identifying historically optimal prices...")
    # Group by stock_code and week to find which prices led to highest revenue
    df["week_year"] = df.invoice_date.dt.isocalendar().week.astype(str) + "_" + df.invoice_date.dt.year.astype(str)
    weekly_stats = df.groupby(["stock_code", "week_year"]).agg({
        "revenue": "sum",
        "unit_price": "mean"
    }).reset_index()
    
    def get_hist_optimal(group):
        if len(group) == 0: return np.nan
        q75 = group["revenue"].quantile(0.75)
        top_weeks = group[group["revenue"] >= q75]
        return top_weeks["unit_price"].mean()

    hist_opt_map = weekly_stats.groupby("stock_code").apply(get_hist_optimal).to_dict()
    df["optimal_price_historical"] = df["stock_code"].map(hist_opt_map).fillna(df["product_avg_price_global"])
    
    # STEP 3 — Blend signals for final target_price
    print("    Blending price signals...")
    df["target_price"] = (
        0.40 * df["unit_price"] + 
        0.35 * df["optimal_price_historical"] + 
        0.25 * df["optimal_price_elasticity_estimate"]
    )
    
    # Clip to reasonable bounds
    df["target_price"] = df.apply(
        lambda row: np.clip(row["target_price"], row["product_price_min"] * 0.5, row["product_price_max"] * 1.5),
        axis=1
    )
    
    # STEP 4 — Auxiliary targets
    df["target_demand_7d"] = df["forward_demand_7d"]
    df["target_revenue_7d"] = df["forward_revenue_7d"]
    
    # STEP 5 — Clean up
    initial_len = len(df)
    # Remove rows where forward data is not available (the last 7 days of the dataset)
    # We detect this if forward_demand_7d is 0 AND date is near the max date
    max_date = df.invoice_date.max()
    threshold_date = max_date - pd.Timedelta(days=7)
    df = df[df.invoice_date < threshold_date]
    
    print(f"[*] Valid rows for training: {len(df):,} (Removed {initial_len - len(df):,} tail rows)")
    
    return df

if __name__ == "__main__":
    from features import build_all_features
    from clean import clean
    from ingest import load_or_merge
    df = load_or_merge()
    df = clean(df)
    df = build_all_features(df)
    df = create_target(df)
    df.to_parquet(config.FEATURES_DIR / "features_with_targets.parquet", index=False)
