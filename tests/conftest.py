import pytest
import fakeredis
import yaml
import os
from unittest.mock import MagicMock
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

@pytest.fixture
def mock_config():
    return {
        "model": {"path": "mock.pkl", "version": "1.0.0"},
        "redis": {"host": "localhost", "port": 6379, "ttl": {"price_cache": 30, "competitor": 300, "inventory": 60}},
        "pricing_rules": {
            "inventory_multiplier": {"min": 0.9, "max": 1.15},
            "demand_multiplier": {"min": 0.95, "max": 1.20},
            "flash_sale_multiplier": 1.25,
            "competitor_blend_ratio": 0.3,
            "price_floor_margin": 1.05,
            "price_ceiling_msrp": 1.50
        }
    }

@pytest.fixture
def mock_redis():
    return fakeredis.FakeRedis(decode_responses=True)

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [100.0] # Returns 100.0 for any input
    return model

@pytest.fixture
def mock_feature_store(mock_redis, mock_config):
    from data.feature_store import FeatureStore
    return FeatureStore(mock_redis, mock_config)
