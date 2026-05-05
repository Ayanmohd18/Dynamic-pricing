import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import mlflow
import mlflow.xgboost
import joblib
import argparse
import json
import os
import time
import hashlib
import structlog
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Structured Logging ---
logger = structlog.get_logger()

# --- Configuration ---
FEATURE_COLS = [
    'price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
    'demand_score_7d', 'demand_score_30d', 'demand_velocity',
    'inventory_ratio', 'price_percentile_in_category',
    'competitor_delta', 'review_elasticity'
]
TARGET_COL = 'optimal_price'

def get_data_hash(df):
    """Generates a hash of the feature names and data shape for reproducibility."""
    hash_input = f"{list(df.columns)}_{df.shape}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def validate_schema(df):
    """Ensures all required columns exist in the dataframe."""
    missing = [c for c in FEATURE_COLS + [TARGET_COL, 'category', 'order_purchase_timestamp'] if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")
    logger.info("Schema validation successful.")

def objective(trial, X, y):
    """Optuna objective function for XGBoost tuning."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_cv, y_train_cv)
        preds = model.predict(X_val_cv)
        maes.append(mean_absolute_error(y_val_cv, preds))
        
    return np.mean(maes)

def run_training(args):
    """Main training pipeline."""
    start_time = time.time()
    logger.info("Starting model training pipeline", args=vars(args))
    
    # 1. Load & Validate
    df = pd.read_parquet(args.data_path)
    df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
    validate_schema(df)
    data_hash = get_data_hash(df)
    
    # 2. Chronological Split (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]
    
    # 3. Optuna Tuning
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="Hyperparameter_Tuning"):
        logger.info("Starting hyperparameter tuning with Optuna (50 trials)...")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
        
        best_params = study.best_params
        logger.info("Tuning complete", best_params=best_params, best_mae=study.best_value)
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_best_mae", study.best_value)

    # 4. Final Train
    with mlflow.start_run(run_name="Final_Model_Train"):
        logger.info("Training final model on full train set...")
        final_model = xgb.XGBRegressor(**best_params, random_state=42)
        final_model.fit(X_train, y_train)
        
        # 5. Evaluate
        preds = final_model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        logger.info("Evaluation results", mae=mae, rmse=rmse, r2=r2)
        mlflow.log_metrics({"test_mae": mae, "test_rmse": rmse, "test_r2": r2})
        
        # Per-category breakdown
        test_df['pred'] = preds
        test_df['abs_error'] = (test_df[TARGET_COL] - test_df['pred']).abs()
        cat_mae = test_df.groupby('category')['abs_error'].mean().sort_values(ascending=False)
        
        # 6. Threshold Check
        if mae > args.retrain_threshold:
            logger.error("Model performance failed threshold check!", mae=mae, threshold=args.retrain_threshold)
            exit(1)
            
        # 7. Save Model & Metadata
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        # Use native XGBoost JSON format for better compatibility
        json_output = args.model_output.replace('.pkl', '.json')
        final_model.save_model(json_output)
        
        metadata = {
            "training_date": datetime.now().isoformat(),
            "best_params": best_params,
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
            "data_hash": data_hash,
            "feature_list": FEATURE_COLS,
            "model_format": "xgboost_json"
        }
        with open(json_output.replace('.json', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # 8. Log Artifacts
        mlflow.xgboost.log_model(final_model, "model")
        
        # Feature Importance Plot
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(final_model, max_num_features=15)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        # Print Summary Table
        print("\n" + "="*40)
        print("TRAINING RUN SUMMARY")
        print("="*40)
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")
        print("-"*40)
        print("Top 5 Category MAEs:")
        print(cat_mae.head(5))
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Olist Dynamic Pricing Training Pipeline")
    parser.add_argument("--data-path", type=str, required=True, help="Path to features.parquet")
    parser.add_argument("--model-output", type=str, required=True, help="Path to save .pkl")
    parser.add_argument("--experiment-name", type=str, default="Dynamic_Pricing_v1", help="MLflow Experiment")
    parser.add_argument("--retrain-threshold", type=float, default=8.0, help="MAE exit threshold")
    
    args = parser.parse_args()
    run_training(args)
