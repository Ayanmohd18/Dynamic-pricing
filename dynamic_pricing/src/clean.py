import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning pipeline as specified in Phase 3."""
    print(f"[*] Starting cleaning pipeline. Initial rows: {len(df):,}")
    
    # STEP 1 — Remove cancellations
    initial_len = len(df)
    df = df[~df.invoice_no.astype(str).str.startswith('C')]
    df = df[df.quantity > 0]
    print(f"    Step 1: Removed {initial_len - len(df):,} cancellations/negative quantities.")
    
    # STEP 2 — Remove invalid prices
    initial_len = len(df)
    df = df[(df.unit_price >= config.MIN_UNIT_PRICE) & (df.unit_price <= config.MAX_UNIT_PRICE)]
    print(f"    Step 2: Removed {initial_len - len(df):,} rows with invalid prices.")
    
    # STEP 3 — Remove invalid quantities
    initial_len = len(df)
    df = df[(df.quantity >= config.MIN_QUANTITY) & (df.quantity <= config.MAX_QUANTITY)]
    print(f"    Step 3: Removed {initial_len - len(df):,} rows with invalid quantities.")
    
    # STEP 4 — Remove bad descriptions
    initial_len = len(df)
    df = df.dropna(subset=["description"])
    df = df[df.description.str.strip().str.len() >= config.MIN_DESCRIPTION_LEN]
    bad_descriptions = ["POSTAGE", "DOTCOM POSTAGE", "BANK CHARGES", "MANUAL", "SAMPLES", "AMAZON FEE"]
    df = df[~df.description.str.upper().str.contains('|'.join(bad_descriptions))]
    print(f"    Step 4: Removed {initial_len - len(df):,} rows with bad descriptions.")
    
    # STEP 5 — Handle missing CustomerID
    initial_len = len(df)
    # Use vectorized fillna instead of apply for performance
    df["customer_id"] = df["customer_id"].astype(str).replace("nan", np.nan)
    df["customer_id"] = df["customer_id"].fillna("GUEST_" + df["invoice_no"].astype(str))
    print(f"    Step 5: Handled missing customer_id (filled guest IDs).")
    
    # STEP 6 — Standardize StockCode
    initial_len = len(df)
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    # Remove rows where stock_code is purely alphabetic with len < 3
    df = df[~(df.stock_code.str.isalpha() & (df.stock_code.str.len() < 3))]
    bad_codes = ["POST", "D", "M", "BANK CHARGES", "PADS", "DOT"]
    df = df[~df.stock_code.isin(bad_codes)]
    print(f"    Step 6: Removed {initial_len - len(df):,} rows with invalid stock codes.")
    
    # STEP 7 — Create revenue column
    df["revenue"] = df.quantity * df.unit_price
    print(f"    Step 7: Created revenue column.")
    
    # STEP 8 — Create is_bulk flag
    df["is_bulk"] = (df.quantity >= 12).astype(int)
    print(f"    Step 8: Created is_bulk flag.")
    
    # STEP 9 — Remove duplicate transactions
    initial_len = len(df)
    df = df.drop_duplicates(subset=["invoice_no", "stock_code", "quantity", "unit_price"], keep="first")
    print(f"    Step 9: Removed {initial_len - len(df):,} duplicate transactions.")
    
    # STEP 10 — Final type casting
    df["quantity"] = df["quantity"].astype(np.int32)
    df["unit_price"] = df["unit_price"].astype(np.float32)
    df["revenue"] = df["revenue"].astype(np.float32)
    df["customer_id"] = df["customer_id"].astype(str)
    df["stock_code"] = df["stock_code"].astype(str)
    print(f"    Step 10: Final type casting complete.")
    
    print(f"[*] Final shape: {df.shape}")
    print(df.describe())
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.PROCESSED_DIR / "cleaned.parquet"
    df.to_parquet(save_path, index=False)
    print(f"[*] Saved cleaned data to {save_path}")
    
    return df

if __name__ == "__main__":
    from ingest import load_or_merge
    df = load_or_merge()
    clean(df)
