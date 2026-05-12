import time
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

def main():
    total_start = time.time()
    
    print("="*60)
    print("DYNAMIC PRICING ENGINE — TRAINING PIPELINE")
    print("="*60)
    
    # Create directories
    for d in ["data/processed", "data/features", "models/saved", "models/metrics"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    print("\n[1/7] Loading datasets...")
    from ingest import load_or_merge
    df = load_or_merge()
    print(f"  Loaded {len(df):,} rows")
    
    print("\n[2/7] Cleaning data...")
    from clean import clean
    df = clean(df)
    print(f"  After cleaning: {len(df):,} rows")
    
    print("\n[3/7] Engineering features...")
    from features import build_all_features
    df = build_all_features(df)
    print(f"  Features built: {len(df.columns)}")
    
    print("\n[4/7] Creating pricing targets...")
    from pricing_target import create_target
    df = create_target(df)
    print(f"  Valid training rows: {len(df):,}")
    
    print("\n[5/7] Training models...")
    from train import run_full_training_pipeline
    run_full_training_pipeline()
    
    print("\n[6/7] Pipeline complete!")
    elapsed = time.time() - total_start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print("\nNext steps:")
    print("  API:       uvicorn api.main:app --reload")
    print("  Dashboard: streamlit run dashboard/app.py")
    print("  Docker:    docker-compose up")

if __name__ == "__main__":
    main()
