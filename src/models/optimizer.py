import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import math

@dataclass
class PriceDecision:
    sku_id: str
    base_price: float
    final_price: float
    multipliers_applied: Dict[str, float]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)

class PriceOptimizer:
    """
    Core pricing logic module. Orchestrates ML predictions, 
    business rule multipliers, and market guardrails.
    """

    def __init__(self, model, config: Dict, feature_store):
        """
        Initializes the optimizer with model, config and a feature store.
        """
        self.model = model
        self.config = config
        self.feature_store = feature_store
        self.logger = logging.getLogger(__name__)
        
        # Rule constants from config
        self.rules = config.get('pricing_rules', {})
        self.blend_ratio = self.rules.get('competitor_blend_ratio', 0.3)
        self.floor_margin = self.rules.get('price_floor_margin', 1.05)
        self.ceiling_msrp = self.rules.get('price_ceiling_msrp', 1.50)

    def predict_base_price(self, features: Dict) -> float:
        """
        Generates raw ML prediction using XGBoost.
        """
        # Ensure feature order matches model training
        feature_cols = [
            'price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
            'demand_score_7d', 'demand_score_30d', 'demand_velocity',
            'inventory_ratio', 'price_percentile_in_category',
            'competitor_delta', 'review_elasticity'
        ]
        
        # Convert dict to DataFrame for model
        input_df = pd.DataFrame([features])[feature_cols]
        pred = self.model.predict(input_df)[0]
        return float(pred)

    def apply_inventory_multiplier(self, price: float, inventory_ratio: float) -> Tuple[float, float]:
        """
        Applies scarcity markup or excess stock markdown.
        """
        multiplier = 1.0
        if inventory_ratio < 0.1:
            multiplier = 1.20 # Scarcity markup
        elif inventory_ratio < 0.3:
            multiplier = 1.10
        elif inventory_ratio > 0.9:
            multiplier = 0.85 # Excess stock markdown
            
        return price * multiplier, multiplier

    def apply_demand_multiplier(self, price: float, demand_score_7d: float) -> Tuple[float, float]:
        """
        Scales price using a sigmoid function based on demand intensity.
        Maps demand_score_7d to a range between 0.90 and 1.35.
        """
        # Sigmoid: L + (H - L) / (1 + exp(-k * (x - x0)))
        L, H = 0.90, 1.35
        x0 = 500 # Midpoint of demand
        k = 0.005 # Steepness
        
        multiplier = L + (H - L) / (1 + math.exp(-k * (demand_score_7d - x0)))
        return price * multiplier, round(multiplier, 3)

    def apply_competitor_blend(self, ml_price: float, competitor_price: Optional[float]) -> float:
        """
        Blends ML price with market average (70/30 split).
        """
        if competitor_price is None or np.isnan(competitor_price):
            return ml_price
            
        return (1 - self.blend_ratio) * ml_price + (self.blend_ratio * competitor_price)

    def apply_flash_sale_multiplier(self, price: float, is_flash_sale: bool) -> float:
        """
        Applies flat flash sale boost if active.
        """
        multiplier = self.rules.get('flash_sale_multiplier', 1.25)
        return price * multiplier if is_flash_sale else price

    def apply_segment_multiplier(self, price: float, segment: int) -> Tuple[float, float]:
        """
        Applies segment-based multiplier [0.95-1.10 by customer segment].
        """
        multiplier_map = {0: 0.95, 1: 0.98, 2: 1.00, 3: 1.05, 4: 1.10}
        multiplier = multiplier_map.get(segment, 1.0)
        return price * multiplier, multiplier

    def apply_price_guardrails(self, price: float, cost: float, msrp: float) -> float:
        """
        Clamps price within safe business margins.
        """
        lower_bound = cost * self.floor_margin
        upper_bound = msrp * self.ceiling_msrp
        return float(np.clip(price, lower_bound, upper_bound))

    def get_optimal_price(self, sku_id: str, context: Dict) -> PriceDecision:
        """
        Orchestrates the entire pricing pipeline for a single SKU.
        """
        self.logger.info(f"Optimizing price for SKU: {sku_id}")
        
        # 1. Fetch live features from store
        live_features = self.feature_store.get_features(sku_id)
        
        # Merge context (static) with live features
        merged_features = {**context, **live_features}
        
        # 2. ML Base Prediction
        base_price = self.predict_base_price(merged_features)
        current_price = base_price
        
        # 3. Apply Multipliers
        multipliers = {}
        
        # Inventory
        current_price, inv_m = self.apply_inventory_multiplier(current_price, live_features['inventory_ratio'])
        multipliers['inventory'] = inv_m
        
        # Demand
        current_price, dem_m = self.apply_demand_multiplier(current_price, live_features['demand_score_7d'])
        multipliers['demand'] = dem_m
        
        # Competitor Blend
        current_price = self.apply_competitor_blend(current_price, live_features['competitor_price'])
        
        # Flash Sale
        is_flash = context.get('is_flash_sale', False)
        if is_flash:
            current_price = self.apply_flash_sale_multiplier(current_price, True)
            multipliers['flash_sale'] = self.rules.get('flash_sale_multiplier', 1.25)
            
        # Segment
        segment = context.get('customer_segment', 2)
        current_price, seg_m = self.apply_segment_multiplier(current_price, segment)
        multipliers['segment'] = seg_m
            
        # 4. Final Guardrails
        cost = context.get('cost', base_price * 0.7)
        msrp = context.get('msrp', base_price * 1.5)
        final_price = self.apply_price_guardrails(current_price, cost, msrp)
        
        # 5. Reasoning & Decision
        reasoning = f"Base ML: INR {base_price:.2f}. "
        if inv_m != 1.0: reasoning += f"Inventory adjustment: {inv_m}x. "
        if dem_m != 1.0: reasoning += f"Demand adjustment: {dem_m}x. "
        if is_flash: reasoning += f"Flash sale boost applied. "
        if seg_m != 1.0: reasoning += f"Segment adjustment: {seg_m}x. "
        if final_price != current_price: reasoning += "Guardrails applied."
        
        # Simulated confidence interval (can be improved with quantile regression)
        ci = (final_price * 0.95, final_price * 1.05)
        
        return PriceDecision(
            sku_id=sku_id,
            base_price=base_price,
            final_price=final_price,
            multipliers_applied=multipliers,
            reasoning=reasoning,
            confidence_interval=ci
        )

if __name__ == "__main__":
    # Mocking for standalone test
    import joblib
    import os
    from unittest.mock import MagicMock
    
    print("Testing PriceOptimizer...")
    
    # Mock FeatureStore
    mock_fs = MagicMock()
    mock_fs.get_features.return_value = {
        "demand_score_7d": 800, # High demand
        "inventory_ratio": 0.05, # Scarcity
        "competitor_price": 105.0
    }
    
    # Try loading real model (Prefer JSON)
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models/xgboost_pricing_v1.json")
    pkl_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models/xgboost_pricing_v1.pkl")
    
    if os.path.exists(json_path):
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.load_model(json_path)
    elif os.path.exists(pkl_path):
        model = joblib.load(pkl_path)
    else:
        model = MagicMock()
        model.predict.return_value = [100.0]
        
    config = {
        "pricing_rules": {
            "competitor_blend_ratio": 0.3,
            "price_floor_margin": 1.05,
            "price_ceiling_msrp": 1.50,
            "flash_sale_multiplier": 1.25
        }
    }
    
    optimizer = PriceOptimizer(model, config, mock_fs)
    
    context = {
        "cost": 50.0,
        "msrp": 150.0,
        "is_flash_sale": False,
        # Dummy ML features
        "price": 100.0, "freight_value": 15.0, "hour_sin": 0.5, "hour_cos": 0.8,
        "day_sin": 0.7, "day_cos": 0.7, "is_weekend": 0, "is_month_end": 0, "is_holiday": 0,
        "days_since_last_order": 2.0, "demand_score_30d": 2400, "demand_velocity": 0.1,
        "price_percentile_in_category": 0.5, "competitor_delta": 0.0, "review_elasticity": 1.0
    }
    
    decision = optimizer.get_optimal_price("TEST_SKU_001", context)
    
    print(f"\nSKU: {decision.sku_id}")
    print(f"Base ML Price: INR {decision.base_price:.2f}")
    print(f"Final Optimized Price: INR {decision.final_price:.2f}")
    print(f"Multipliers Applied: {decision.multipliers_applied}")
    print(f"Reasoning: {decision.reasoning}")

