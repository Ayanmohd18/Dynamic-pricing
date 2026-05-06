import joblib
import pandas as pd
import numpy as np
import os

class DynamicPricingEngine:
    def __init__(self, model_path, tf_model_path=None, tf_scaler_path=None):
        import xgboost as xgb
        # If it's a pickle, use joblib; if JSON, use native loader
        if model_path.endswith('.pkl'):
            self.model = joblib.load(model_path)
        else:
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            
        # Optional: TensorFlow Baseline Integration
        self.tf_model = None
        self.tf_scaler = None
        if tf_model_path and os.path.exists(tf_model_path) and tf_scaler_path and os.path.exists(tf_scaler_path):
            import tensorflow as tf
            self.tf_model = tf.keras.models.load_model(tf_model_path)
            self.tf_scaler = joblib.load(tf_scaler_path)
            
        # In a real app, cost and MSRP would come from a DB. 
        # For this implementation, we'll simulate them.
        self.margin_floor = 1.05
        self.msrp_ceiling = 1.50

    def predict_base_price(self, features_df):
        """Returns the ML recommended price from XGBoost (blended with TF baseline if loaded)."""
        xgb_price = self.model.predict(features_df)[0]
        
        if self.tf_model is not None and self.tf_scaler is not None:
            scaled_features = self.tf_scaler.transform(features_df.values)
            tf_price = self.tf_model.predict(scaled_features, verbose=0)[0][0]
            # Blend 70% XGBoost with 30% TensorFlow Baseline
            return (0.7 * xgb_price) + (0.3 * tf_price)
            
        return xgb_price

    def apply_guardrails(self, price, cost, msrp):
        """Ensures price stays within [cost * 1.05, msrp * 1.50]."""
        lower_bound = cost * self.margin_floor
        upper_bound = msrp * self.msrp_ceiling
        return np.clip(price, lower_bound, upper_bound)

    def blend_with_competitor(self, ml_price, competitor_avg):
        """Applies 70/30 blending logic."""
        if competitor_avg is None or np.isnan(competitor_avg):
            return ml_price
        return (0.70 * ml_price) + (0.30 * competitor_avg)

    def get_final_price(self, features_dict, competitor_avg=None, cost=None, msrp=None):
        """
        Complete pipeline: 
        Predict -> Blend -> Guardrails
        """
        # Prepare features for model
        # Note: Order must match the training feature list
        feature_cols = [
            'price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
            'demand_score_7d', 'demand_score_30d', 'demand_velocity',
            'inventory_ratio', 'price_percentile_in_category',
            'competitor_delta', 'review_elasticity'
        ]
        
        input_df = pd.DataFrame([features_dict])[feature_cols]
        
        # 1. ML Prediction
        ml_price = self.predict_base_price(input_df)
        
        # 2. Competitor Blending
        blended_price = self.blend_with_competitor(ml_price, competitor_avg)
        
        # 3. Apply Guardrails
        # If cost/msrp aren't provided, use base price as fallback for simulation
        cost = cost if cost is not None else features_dict['price'] * 0.8
        msrp = msrp if msrp is not None else features_dict['price'] * 1.2
        
        final_price = self.apply_guardrails(blended_price, cost, msrp)
        
        return {
            "ml_price": float(ml_price),
            "blended_price": float(blended_price),
            "final_price": float(final_price),
            "applied_guardrail": final_price != blended_price
        }

if __name__ == "__main__":
    # Test block
    # Prefer JSON format if it exists
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models/xgboost_pricing_v1.json')
    pkl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models/xgboost_pricing_v1.pkl')
    model_file = json_path if os.path.exists(json_path) else pkl_path
    
    engine = DynamicPricingEngine(model_file)
    sample_features = {
        'price': 100.0, 'freight_value': 15.0, 'hour_sin': 0.5, 'hour_cos': 0.8,
        'day_sin': 0.7, 'day_cos': 0.7, 'is_weekend': 0, 'is_month_end': 0,
        'is_holiday': 0, 'days_since_last_order': 2.5, 'demand_score_7d': 10,
        'demand_score_30d': 50, 'demand_velocity': 0.1, 'inventory_ratio': 0.5,
        'price_percentile_in_category': 0.5, 'competitor_delta': 0.05,
        'review_elasticity': 1.0
    }
    result = engine.get_final_price(sample_features, competitor_avg=105.0)
    print(f"Pricing Result: {result}")
