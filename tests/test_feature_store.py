import pytest
from data.feature_store import FeatureStore

def test_get_features_cache_miss_returns_defaults(mock_redis, mock_config):
    fs = FeatureStore(mock_redis, mock_config)
    features = fs.get_features("NON_EXISTENT_SKU")
    # Defaults: demand=0.0, inventory=0.5, comp=None
    assert features["demand_score_7d"] == 0.0
    assert features["inventory_ratio"] == 0.5
    assert features["competitor_price"] is None

def test_set_and_get_features_roundtrip(mock_redis, mock_config):
    fs = FeatureStore(mock_redis, mock_config)
    fs.set_features("SKU_123", {"demand_score_7d": 450, "inventory_ratio": 0.25})
    
    features = fs.get_features("SKU_123")
    assert features["demand_score_7d"] == 450.0
    assert features["inventory_ratio"] == 0.25

def test_batch_get_uses_pipeline(mock_redis, mock_config):
    fs = FeatureStore(mock_redis, mock_config)
    fs.set_features("A", {"demand_score_7d": 100})
    fs.set_features("B", {"demand_score_7d": 200})
    
    batch = fs.batch_get_features(["A", "B"])
    assert batch["A"]["demand_score_7d"] == 100.0
    assert batch["B"]["demand_score_7d"] == 200.0

def test_invalidate(mock_redis, mock_config):
    fs = FeatureStore(mock_redis, mock_config)
    fs.set_features("SKU_X", {"demand_score_7d": 500})
    fs.invalidate("SKU_X")
    
    features = fs.get_features("SKU_X")
    assert features["demand_score_7d"] == 0.0 # Back to default
