import boto3
import json
import numpy as np
import pandas as pd
import structlog
import threading
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from io import StringIO

# --- Logging ---
logger = structlog.get_logger()

class PricingMonitor:
    """
    Production monitoring module for Dynamic Pricing Engine.
    Handles CloudWatch metrics, Data Drift (PSI), S3 Audit Logging, and Grafana exports.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.aws_config = config.get('aws', {})
        self.region = self.aws_config.get('region', 'us-east-1')
        self.bucket = self.aws_config.get('s3_bucket', 'olist-pricing-monitor')
        self.namespace = self.aws_config.get('cloudwatch_namespace', 'PricingEngine/Production')
        
        self.cw = boto3.client('cloudwatch', region_name=self.region)
        self.s3 = boto3.client('s3', region_name=self.region)
        
        self.metrics_buffer = []
        self.lock = threading.Lock()

    # --- 1. CloudWatch Metrics ---

    def publish_metric(self, name: str, value: float, unit: str = 'None', dimensions: List[Dict] = None):
        """Adds a metric datum to the buffer for batch publishing."""
        with self.lock:
            self.metrics_buffer.append({
                'MetricName': name,
                'Value': value,
                'Unit': unit,
                'Timestamp': datetime.now(),
                'Dimensions': dimensions or []
            })
            
            # Batch publish if buffer >= 20 (CloudWatch limit per call)
            if len(self.metrics_buffer) >= 20:
                threading.Thread(target=self._flush_metrics).start()

    def _flush_metrics(self):
        with self.lock:
            if not self.metrics_buffer: return
            to_publish = self.metrics_buffer[:20]
            self.metrics_buffer = self.metrics_buffer[20:]
            
        try:
            self.cw.put_metric_data(Namespace=self.namespace, MetricData=to_publish)
            logger.info("Published batch metrics to CloudWatch", count=len(to_publish))
        except Exception as e:
            logger.error("Failed to publish CloudWatch metrics", error=str(e))

    # --- 2. Feature Drift Detector (PSI) ---

    def compute_psi(self, expected: np.array, actual: np.array, buckets: int = 10) -> float:
        """Computes the Population Stability Index."""
        def scale_range(data, min_val, max_val):
            return (data - min_val) / (max_val - min_val)

        # Create buckets based on expected data
        breakpoints = np.linspace(0, 1, buckets + 1)
        
        def get_counts(data):
            # Clip and normalize for bucketing
            normalized = np.clip(scale_range(data, np.min(expected), np.max(expected)), 0, 1)
            counts, _ = np.histogram(normalized, bins=breakpoints)
            return counts / len(data)

        expected_percents = get_counts(expected) + 1e-6 # Avoid div by zero
        actual_percents = get_counts(actual) + 1e-6
        
        psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return float(psi)

    def check_all_features_drift(self, current_df: pd.DataFrame, baseline_stats: Dict[str, np.array]) -> Dict[str, float]:
        """Checks PSI for all features against baseline distributions."""
        drifts = {}
        for feat in baseline_stats.keys():
            if feat in current_df.columns:
                drifts[feat] = self.compute_psi(baseline_stats[feat], current_df[feat].values)
        return drifts

    # --- 3. Model Performance & S3 Audit ---

    def log_prediction(self, sku_id: str, predicted_price: float, actual_price: float, features: Dict):
        """Asynchronously writes a pricing decision to the S3 Audit Log."""
        def _write():
            date_str = datetime.now().strftime('%Y-%m-%d')
            key = f"audit/date={date_str}/{sku_id}_{datetime.now().timestamp()}.json"
            payload = {
                "sku_id": sku_id,
                "predicted": predicted_price,
                "actual": actual_price,
                "features": features,
                "timestamp": datetime.now().isoformat()
            }
            try:
                self.s3.put_object(Bucket=self.bucket, Key=key, Body=json.dumps(payload))
            except Exception as e:
                logger.error("S3 Audit log failed", error=str(e))

        threading.Thread(target=_write).start()

    # --- 4. Grafana Dashboard Config ---

    def generate_grafana_dashboard(self) -> Dict:
        """Generates the JSON configuration for the production Grafana dashboard."""
        dashboard = {
            "title": "Olist Dynamic Pricing Control Center",
            "panels": [
                {"title": "Total Revenue (Real-time)", "type": "graph", "id": 1},
                {"title": "Average Margin %", "type": "gauge", "id": 2},
                {"title": "Pricing Decisions / Min", "type": "stat", "id": 3},
                {"title": "Flash Sales Active", "type": "stat", "id": 4, "color": "red"},
                {"title": "Model MAE (Last 1H)", "type": "graph", "id": 5},
                {"title": "Cache Hit Rate %", "type": "gauge", "id": 6}
            ]
        }
        return dashboard

    def save_dashboard_json(self, path: str = "configs/grafana_dashboard.json"):
        """Saves the generated dashboard to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.generate_grafana_dashboard(), f, indent=4)
        logger.info("Grafana dashboard config saved", path=path)

if __name__ == "__main__":
    # Test block
    mock_config = {'aws': {'region': 'us-east-1', 's3_bucket': 'test-bucket'}}
    monitor = PricingMonitor(mock_config)
    
    # Simulate drift test
    base = np.random.normal(100, 10, 1000)
    curr = np.random.normal(110, 10, 1000) # Shifted
    psi = monitor.compute_psi(base, curr)
    print(f"Computed PSI (should be > 0): {psi:.4f}")
    
    monitor.save_dashboard_json()
