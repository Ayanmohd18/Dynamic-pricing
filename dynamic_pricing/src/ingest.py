import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_retail_i() -> pd.DataFrame:
    """Read RETAIL_I_FILE and standardize."""
    print(f"[*] Loading Retail I: {config.RETAIL_I_FILE}")
    df = pd.read_excel(config.RETAIL_I_FILE, engine="openpyxl")
    df["source"] = "retail_i"
    df["dataset_year_range"] = "2010-2011"
    
    # Standardize column names
    rename_map = {
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country"
    }
    df = df.rename(columns=rename_map)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    
    print(f"    Shape: {df.shape} | Dates: {df.invoice_date.min()} to {df.invoice_date.max()}")
    return df

def load_retail_ii() -> pd.DataFrame:
    """Read RETAIL_II_FILE (both sheets) and standardize."""
    print(f"[*] Loading Retail II: {config.RETAIL_II_FILE}")
    
    sheets_dict = pd.read_excel(config.RETAIL_II_FILE, sheet_name=config.RETAIL_II_SHEETS, engine="openpyxl")
    df = pd.concat(sheets_dict.values(), ignore_index=True)
    
    # Note: Online Retail II often uses "Invoice", "Price", "Customer ID"
    # But we follow the prompt's mapping. We'll handle both common naming conventions.
    rename_map = {
        "InvoiceNo": "invoice_no",
        "Invoice": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "Price": "unit_price",
        "CustomerID": "customer_id",
        "Customer ID": "customer_id",
        "Country": "country"
    }
    
    df = df.rename(columns=rename_map)
    df["source"] = "retail_ii"
    df["dataset_year_range"] = "2009-2011"
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    
    # Ensure customer_id exists if it was named something else
    if "customer_id" not in df.columns and "Customer ID" in df.columns:
        df = df.rename(columns={"Customer ID": "customer_id"})
        
    print(f"    Shape: {df.shape} | Dates: {df.invoice_date.min()} to {df.invoice_date.max()}")
    return df

def merge_datasets() -> pd.DataFrame:
    """Merge both loaders and save as parquet."""
    df1 = load_retail_i()
    df2 = load_retail_ii()
    
    print("[*] Merging datasets...")
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Cast mixed-type columns to string to prevent Arrow conversion errors
    for col in ["invoice_no", "stock_code", "customer_id", "description", "country"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    df = df.sort_values("invoice_date").reset_index(drop=True)
    
    print(f"    Combined: {len(df):,} rows")
    print(f"    Date range: {df.invoice_date.min()} to {df.invoice_date.max()}")
    print(f"    Unique products: {df.stock_code.nunique()}")
    print(f"    Unique customers: {df.customer_id.nunique()}")
    
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = config.PROCESSED_DIR / "raw_combined.parquet"
    df.to_parquet(save_path, index=False)
    print(f"[*] Saved combined raw data to {save_path}")
    
    return df

def load_or_merge() -> pd.DataFrame:
    """Load from parquet if exists, else merge."""
    save_path = config.PROCESSED_DIR / "raw_combined.parquet"
    if save_path.exists():
        print(f"[*] Loading combined data from cache: {save_path}")
        return pd.read_parquet(save_path)
    return merge_datasets()

if __name__ == "__main__":
    load_or_merge()
