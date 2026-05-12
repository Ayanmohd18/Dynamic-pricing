import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import os

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_models():
    """Phase 8: Load all saved model artifacts."""
    print("[*] Loading models from", config.MODELS_DIR)
    try:
        return {
            "xgb": joblib.load(config.MODELS_DIR / "xgb_model.pkl"),
            "lgbm": joblib.load(config.MODELS_DIR / "lgbm_model.pkl"),
            "rf": joblib.load(config.MODELS_DIR / "rf_model.pkl"),
            "catboost": joblib.load(config.MODELS_DIR / "catboost_model.pkl"),
            "meta": joblib.load(config.MODELS_DIR / "meta_model.pkl"),
            "feature_names": joblib.load(config.MODELS_DIR / "feature_names.pkl"),
            "scaler": joblib.load(config.MODELS_DIR / "scaler.pkl")
        }
    except Exception as e:
        print(f"[!] Error loading models: {e}")
        return None

# Global state for cached stats
_models = None
_product_stats = None

def get_product_stats(stock_code):
    global _product_stats
    if _product_stats is None:
        # Load precomputed stats if available, else return defaults
        path = config.PROCESSED_DIR / "product_stats.parquet"
        if path.exists():
            _product_stats = pd.read_parquet(path).set_index("stock_code")
        else:
            return None
    
    if stock_code in _product_stats.index:
        return _product_stats.loc[stock_code].to_dict()
    return None

def build_inference_features(stock_code, current_price, quantity, country, timestamp, models):
    """Build a single-row feature vector for inference."""
    if timestamp is None:
        timestamp = datetime.now()
        
    stats = get_product_stats(stock_code)
    if stats is None:
        # Fallback to global defaults if product is new
        stats = {
            "product_avg_price_global": current_price,
            "product_price_std": 0.1,
            "product_popularity_rank": 1,
            "product_price_min": current_price * 0.8,
            "product_price_max": current_price * 1.2
        }

    # Temporal features
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    day_of_month = timestamp.day
    week_of_year = timestamp.isocalendar()[1]
    month = timestamp.month
    
    # Assembly
    data = {
        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "week_of_year": week_of_year,
        "month": month,
        "quarter": (month-1)//3 + 1,
        "year": timestamp.year,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "is_month_start": 1 if day_of_month <= 3 else 0,
        "is_month_end": 1 if day_of_month >= 28 else 0,
        "is_christmas_season": 1 if month in [11, 12] else 0,
        "is_summer": 1 if month in [6, 7, 8] else 0,
        "product_avg_price_global": stats.get("product_avg_price_global"),
        "product_price_std": stats.get("product_price_std", 0),
        "product_popularity_rank": stats.get("product_popularity_rank", 1),
        "price_vs_product_mean": current_price / stats.get("product_avg_price_global", current_price),
        "is_bulk": 1 if quantity >= 12 else 0,
        "is_uk": 1 if country == "United Kingdom" else 0,
        # Add more features as needed to match feature_names
    }
    
    # Ensure all feature_names are present, fill missing with 0
    full_row = {f: data.get(f, 0) for f in models["feature_names"]}
    df = pd.DataFrame([full_row])
    
    # Scale
    scaled = models["scaler"].transform(df)
    return scaled

def predict_optimal_price(stock_code, current_price, quantity=1, country="United Kingdom", timestamp=None, adjustment_factors=None):
    global _models
    if _models is None:
        _models = load_models()
    
    if _models is None:
        return {"error": "Models not loaded"}

    # 1. Build features
    X = build_inference_features(stock_code, current_price, quantity, country, timestamp, _models)
    
    # 2. Get base predictions
    xgb_p = _models["xgb"].predict(X)[0]
    lgbm_p = _models["lgbm"].predict(X)[0]
    rf_p = _models["rf"].predict(X)[0]
    cat_p = _models["catboost"].predict(X)[0]
    
    # 3. Ensemble prediction
    oof = pd.DataFrame({"xgb": [xgb_p], "lgbm": [lgbm_p], "rf": [rf_p], "catboost": [cat_p]})
    ensemble_p = _models["meta"].predict(oof)[0]
    
    final_price = ensemble_p
    adjustments = []
    reasoning = "Base ML recommendation based on ensemble of 4 models."
    
    # 4. Business adjustments
    if adjustment_factors:
        if adjustment_factors.get("stockout_risk", 0) > 0.7:
            risk = adjustment_factors["stockout_risk"]
            factor = 1 + (risk * 0.2)
            final_price *= factor
            adjustments.append(f"Stockout Risk (+{risk*20:.1f}%)")
            reasoning += " Price increased due to high stockout risk."
            
        if adjustment_factors.get("flash_sale"):
            final_price *= 0.85
            adjustments.append("Flash Sale (-15%)")
            reasoning += " Applied discount to drive volume during flash sale."
            
        if adjustment_factors.get("inventory_excess", 0) > 0.7:
            excess = adjustment_factors["inventory_excess"]
            final_price *= (1 - excess * 0.15)
            adjustments.append(f"Inventory Excess (-{excess*15:.1f}%)")
            reasoning += " Price lowered to clear excess inventory."

    # 5. Bounds and Rounding
    stats = get_product_stats(stock_code)
    p_min = stats.get("product_price_min", current_price * 0.5) if stats else current_price * 0.5
    p_max = stats.get("product_price_max", current_price * 1.5) if stats else current_price * 1.5
    
    final_price = np.clip(final_price, p_min * 0.5, p_max * 1.5)
    final_price = round(float(final_price) * 20) / 20  # Round to nearest 0.05
    
    # 6. Confidence (Standard Deviation of base models)
    pred_std = np.std([xgb_p, lgbm_p, rf_p, cat_p])
    
    return {
        "stock_code": stock_code,
        "current_price": round(current_price, 2),
        "recommended_price": final_price,
        "price_change_pct": round(((final_price - current_price) / current_price) * 100, 2),
        "confidence_low": round(final_price - 1.96 * pred_std, 2),
        "confidence_high": round(final_price + 1.96 * pred_std, 2),
        "model_predictions": {
            "xgb": round(float(xgb_p), 2),
            "lgbm": round(float(lgbm_p), 2),
            "rf": round(float(rf_p), 2),
            "catboost": round(float(cat_p), 2)
        },
        "ensemble_prediction": round(float(ensemble_p), 2),
        "adjustments_applied": adjustments,
        "reasoning": reasoning
    }

if __name__ == "__main__":
    # Test
    res = predict_optimal_price("22423", 12.75, quantity=5)
    print(res)
