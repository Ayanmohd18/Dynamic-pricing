import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import time
import sys
import os

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def prepare_train_test(df: pd.DataFrame):
    """Phase 6: Time-series split and feature selection."""
    print("[*] Preparing train/valid/test splits (time-series)...")
    
    # Split by date to avoid leakage
    train_df = df[df.invoice_date < "2011-09-01"]
    valid_df = df[(df.invoice_date >= "2011-09-01") & (df.invoice_date < "2011-11-01")]
    test_df = df[df.invoice_date >= "2011-11-01"]
    
    target = "target_price"
    # Drop non-predictive or leaky columns
    drop_cols = [
        "invoice_no", "invoice_date", "customer_id", "description", "stock_code", 
        "source", "dataset_year_range", "target_price", "target_demand_7d", 
        "target_revenue_7d", "revenue", "unit_price", "week_year", 
        "product_category", "h_mean", "h_std", "pct_price_change", "pct_demand_change",
        "country"
    ]
    
    # Ensure only existing columns are dropped
    drop_cols = [c for c in drop_cols if c in train_df.columns]
    
    X_train = train_df.drop(columns=drop_cols)
    X_valid = valid_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)
    
    y_train = train_df[target]
    y_valid = valid_df[target]
    y_test = test_df[target]
    
    feature_names = X_train.columns.tolist()
    print(f"    Features: {len(feature_names)} | Train: {len(X_train)} | Valid: {len(X_valid)} | Test: {len(X_test)}")
    
    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)
    
    return (X_train_scaled, X_valid_scaled, X_test_scaled, 
            y_train, y_valid, y_test, 
            feature_names, scaler, test_df)

def train_xgboost(X_train, X_valid, y_train, y_valid, feature_names):
    print("[*] Training XGBoost...")
    start = time.time()
    model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"    XGBoost done in {time.time()-start:.1f}s | Best MAE: {model.best_score:.4f}")
    return model, importance

def train_lightgbm(X_train, X_valid, y_train, y_valid, feature_names):
    print("[*] Training LightGBM...")
    start = time.time()
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    model.fit(
        X_train, y_train, 
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"    LightGBM done in {time.time()-start:.1f}s")
    return model, importance

def train_random_forest(X_train, y_train, feature_names):
    print("[*] Training Random Forest...")
    start = time.time()
    model = RandomForestRegressor(**config.RF_PARAMS)
    model.fit(X_train, y_train)
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"    Random Forest done in {time.time()-start:.1f}s")
    return model, importance

def train_catboost(X_train, X_valid, y_train, y_valid, feature_names):
    print("[*] Training CatBoost...")
    start = time.time()
    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.05, depth=8,
        eval_metric="MAE", early_stopping_rounds=50,
        random_seed=42, verbose=0
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.get_feature_importance()
    }).sort_values('importance', ascending=False)
    
    print(f"    CatBoost done in {time.time()-start:.1f}s")
    return model, importance

def train_stacking_ensemble(models, X_valid, X_test, y_valid, y_test):
    print("[*] Training Stacking Ensemble (Meta-learner)...")
    
    # Level 1: Predictions on validation set
    oof_preds = pd.DataFrame({
        "xgb": models["xgb"].predict(X_valid),
        "lgbm": models["lgbm"].predict(X_valid),
        "rf": models["rf"].predict(X_valid),
        "catboost": models["catboost"].predict(X_valid)
    })
    
    # Level 2: Ridge meta-learner
    meta = Ridge(alpha=1.0, positive=True)
    meta.fit(oof_preds, y_valid)
    
    weights = meta.coef_ / np.sum(meta.coef_)
    weights_dict = {
        "xgb": weights[0], "lgbm": weights[1],
        "rf": weights[2], "catboost": weights[3]
    }
    
    print(f"    Blend weights: {weights_dict}")
    return meta, weights_dict

def run_full_training_pipeline():
    """Main training orchestration."""
    total_start = time.time()
    
    # Load features
    df = pd.read_parquet(config.FEATURES_DIR / "features.parquet")
    
    # Create target
    from pricing_target import create_target
    df = create_target(df)
    
    # Prepare data
    (X_train, X_valid, X_test, y_train, y_valid, y_test, 
     feature_names, scaler, test_df) = prepare_train_test(df)
    
    # Train models
    print("[*] Starting model training phase...")
    xgb_model, xgb_imp = train_xgboost(X_train, X_valid, y_train, y_valid, feature_names)
    lgbm_model, lgbm_imp = train_lightgbm(X_train, X_valid, y_train, y_valid, feature_names)
    rf_model, rf_imp = train_random_forest(X_train, y_train, feature_names)
    cat_model, cat_imp = train_catboost(X_train, X_valid, y_train, y_valid, feature_names)
    
    models = {
        "xgb": xgb_model, "lgbm": lgbm_model, 
        "rf": rf_model, "catboost": cat_model
    }
    
    # Stacking
    meta_model, weights = train_stacking_ensemble(models, X_valid, X_test, y_valid, y_test)
    
    # Save artifacts
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, config.MODELS_DIR / "xgb_model.pkl")
    joblib.dump(lgbm_model, config.MODELS_DIR / "lgbm_model.pkl")
    joblib.dump(rf_model, config.MODELS_DIR / "rf_model.pkl")
    joblib.dump(cat_model, config.MODELS_DIR / "catboost_model.pkl")
    joblib.dump(meta_model, config.MODELS_DIR / "meta_model.pkl")
    joblib.dump(feature_names, config.MODELS_DIR / "feature_names.pkl")
    joblib.dump(scaler, config.MODELS_DIR / "scaler.pkl")
    
    print(f"[*] Pipeline complete in {(time.time()-total_start)/60:.1f} minutes")
    
    # Evaluate (Simplified call for now, full evaluation in evaluate.py)
    test_preds = meta_model.predict(pd.DataFrame({
        "xgb": xgb_model.predict(X_test),
        "lgbm": lgbm_model.predict(X_test),
        "rf": rf_model.predict(X_test),
        "catboost": cat_model.predict(X_test)
    }))
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)
    print(f"[*] Ensemble Performance - MAE: {mae:.4f} | R2: {r2:.4f}")

if __name__ == "__main__":
    run_full_training_pipeline()
