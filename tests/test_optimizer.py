
import pytest
from models.optimizer import PriceOptimizer, PriceDecision

def test_base_price_prediction(mock_model, mock_config, mock_feature_store):
    optimizer = PriceOptimizer(mock_model, mock_config, mock_feature_store)
    features = {k: 0.0 for k in [
        'price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
        'demand_score_7d', 'demand_score_30d', 'demand_velocity',
        'inventory_ratio', 'price_percentile_in_category',
        'competitor_delta', 'review_elasticity'
    ]}
    price = optimizer.predict_base_price(features)
    assert isinstance(price, float)
    assert price == 100.0

def test_inventory_multiplier_scarcity(mock_model, mock_config, mock_feature_store):
    optimizer = PriceOptimizer(mock_model, mock_config, mock_feature_store)
    # Scarcity threshold < 0.1 should give 1.20x
    price, mult = optimizer.apply_inventory_multiplier(100.0, 0.05)
    assert mult == 1.20
    assert price == 120.0

def test_inventory_multiplier_excess(mock_model, mock_config, mock_feature_store):
    optimizer = PriceOptimizer(mock_model, mock_config, mock_feature_store)
    # Excess threshold > 0.9 should give 0.90x
    price, mult = optimizer.apply_inventory_multiplier(100.0, 0.95)
    assert mult == 0.90
    assert price == 90.0

def test_competitor_blend(mock_model, mock_config, mock_feature_store):
    optimizer = PriceOptimizer(mock_model, mock_config, mock_feature_store)
    # 70/30 blend of 100 and 200 = 70 + 60 = 130
    blended = optimizer.apply_competitor_blend(100.0, 200.0)
    assert blended == 130.0

def test_price_guardrails_floor(mock_model, mock_config, mock_feature_store):
    optimizer = PriceOptimizer(mock_model, mock_config, mock_feature_store)
    # Cost 100, floor margin 1.05 -> price must be >= 105
    clamped = optimizer.apply_price_guardrails(90.0, 100.0, 200.0)
    assert clamped == 105.0

def test_price_guardrails_ceiling(mock_model, mock_config, mock_feature_store):
    optimizer = PriceOptimizer(mock_model, mock_config, mock_feature_store)
    # MSRP 100, ceiling 1.50 -> price must be <= 150
    clamped = optimizer.apply_price_guardrails(200.0, 50.0, 100.0)
    assert clamped == 150.0

def test_flash_sale_multiplier(mock_model, mock_config, mock_feature_store):
    optimizer = PriceOptimizer(mock_model, mock_config, mock_feature_store)
    # Flash sale mult is 1.25
    price = optimizer.apply_flash_sale_multiplier(100.0, True)
    assert price == 125.0

def test_full_pipeline_integration(mock_model, mock_config, mock_feature_store):
    optimizer = PriceOptimizer(mock_model, mock_config, mock_feature_store)
    # Mocking FeatureStore responses for SKU
    mock_feature_store.redis.set("feat:SKU_1:demand", 100) # Low demand
    mock_feature_store.redis.set("feat:SKU_1:inventory", 0.05) # Extreme scarcity (1.2x)
    mock_feature_store.redis.set("feat:SKU_1:comp_price", 100.0)
    
    # context with static values
    context = {"cost": 50.0, "msrp": 200.0, "is_flash_sale": False}
    # Add dummy ML features
    for f in ['price', 'freight_value', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
              'is_weekend', 'is_month_end', 'is_holiday', 'days_since_last_order',
              'demand_score_30d', 'demand_score_7d', 'demand_velocity', 'price_percentile_in_category',
              'competitor_delta', 'review_elasticity']:
        context[f] = 0.0

    decision = optimizer.get_optimal_price("SKU_1", context)
    assert isinstance(decision, PriceDecision)
    assert decision.multipliers_applied['inventory'] == 1.20
    assert decision.final_price >= 100.0 # Base 100 * 1.2 * demand_mult * blend
