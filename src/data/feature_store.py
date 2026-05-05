import redis
import yaml
import json
from typing import Dict, List, Optional
import logging

class FeatureStore:
    """
    Production-grade Feature Store interface for Redis.
    Handles real-time feature retrieval, caching, and invalidation for the pricing engine.
    """

    def __init__(self, redis_client: redis.Redis, config: Dict):
        """
        Initializes the FeatureStore with a Redis connection and configuration.

        Args:
            redis_client (redis.Redis): An active Redis connection.
            config (Dict): Configuration dictionary loaded from config.yaml.
        """
        self.redis = redis_client
        self.config = config
        self.ttl_price = config['redis']['ttl']['price_cache']
        self.ttl_comp = config['redis']['ttl']['competitor']
        self.ttl_inv = config['redis']['ttl']['inventory']
        self.logger = logging.getLogger(__name__)

    def get_features(self, sku_id: str) -> Dict:
        """
        Fetches all live features for a specific SKU from Redis.
        Falls back to default values if keys are missing.

        Args:
            sku_id (str): The unique identifier for the SKU.

        Returns:
            Dict: A dictionary containing demand_score, inventory_ratio, and competitor_price.
        """
        try:
            # Batch fetch using Redis MGET for performance
            keys = [f"feat:{sku_id}:demand", f"feat:{sku_id}:inventory", f"feat:{sku_id}:comp_price"]
            values = self.redis.mget(keys)
            
            return {
                "demand_score_7d": float(values[0]) if values[0] else 0.0,
                "inventory_ratio": float(values[1]) if values[1] else 0.5,
                "competitor_price": float(values[2]) if values[2] else None
            }
        except Exception as e:
            self.logger.error(f"Error fetching features for {sku_id}: {e}")
            return {"demand_score_7d": 0.0, "inventory_ratio": 0.5, "competitor_price": None}

    def set_features(self, sku_id: str, features: Dict):
        """
        Writes feature updates to Redis with their respective TTLs.

        Args:
            sku_id (str): The unique identifier for the SKU.
            features (Dict): Dictionary of features to update (demand_score, inventory_ratio).
        """
        pipeline = self.redis.pipeline()
        if 'demand_score_7d' in features:
            pipeline.setex(f"feat:{sku_id}:demand", self.ttl_price, features['demand_score_7d'])
        if 'inventory_ratio' in features:
            pipeline.setex(f"feat:{sku_id}:inventory", self.ttl_inv, features['inventory_ratio'])
        pipeline.execute()

    def get_competitor_price(self, sku_id: str) -> Optional[float]:
        """
        Retrieves the competitor price for a SKU with a 5-minute cache (TTL).

        Args:
            sku_id (str): The unique identifier for the SKU.

        Returns:
            Optional[float]: The competitor price if found, else None.
        """
        val = self.redis.get(f"feat:{sku_id}:comp_price")
        return float(val) if val else None

    def invalidate(self, sku_id: str):
        """
        Clears all cached features and prices for a specific SKU.

        Args:
            sku_id (str): The unique identifier for the SKU.
        """
        keys = [f"feat:{sku_id}:demand", f"feat:{sku_id}:inventory", f"feat:{sku_id}:comp_price"]
        self.redis.delete(*keys)

    def batch_get_features(self, sku_ids: List[str]) -> Dict[str, Dict]:
        """
        Efficiently fetches features for multiple SKUs using a Redis pipeline.

        Args:
            sku_ids (List[str]): List of SKU identifiers.

        Returns:
            Dict[str, Dict]: A mapping of SKU IDs to their feature dictionaries.
        """
        pipeline = self.redis.pipeline()
        for sku_id in sku_ids:
            pipeline.mget([f"feat:{sku_id}:demand", f"feat:{sku_id}:inventory", f"feat:{sku_id}:comp_price"])
        
        results = pipeline.execute()
        
        batch_output = {}
        for idx, sku_id in enumerate(sku_ids):
            values = results[idx]
            batch_output[sku_id] = {
                "demand_score_7d": float(values[0]) if values[0] else 0.0,
                "inventory_ratio": float(values[1]) if values[1] else 0.5,
                "competitor_price": float(values[2]) if values[2] else None
            }
        return batch_output

if __name__ == "__main__":
    # Test stub (Assuming local redis is running)
    try:
        with open("../../configs/config.yaml", "r") as f:
            test_config = yaml.safe_load(f)
        
        r = redis.Redis(host='localhost', port=6379, db=0)
        fs = FeatureStore(r, test_config)
        print("FeatureStore initialized successfully.")
    except Exception as e:
        print(f"Initialization skipped (Redis likely not running): {e}")
