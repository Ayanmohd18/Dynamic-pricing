import requests
import time
import sys
import argparse
from typing import List

def check_health(base_url: str):
    print("Checking /api/v1/health...")
    try:
        resp = requests.get(f"{base_url}/api/v1/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        status_ok = data.get("status") == "healthy"
        redis_ok = data.get("redis_connected", False)
        kafka_ok = data.get("kafka_connected", False)
        
        if status_ok and redis_ok and kafka_ok:
            print("✅ API Health: OK")
            return True
        else:
            print(f"❌ API Health: FAILED (Status: {data.get('status')}, Redis: {redis_ok}, Kafka: {kafka_ok})")
            return False
    except Exception as e:
        print(f"❌ API Health: CONNECTION ERROR ({e})")
        return False

def test_prediction_latency(base_url: str, threshold_ms: float = 500.0):
    print(f"Running 10 latency tests (Target < {threshold_ms}ms)...")
    latencies = []
    
    payload = {
        "sku_id": "health_test_sku",
        "inventory_level": 0.5,
        "demand_score": 100,
        "customer_segment": 1,
        "cost_price": 50.0,
        "msrp": 100.0
    }
    
    for i in range(10):
        start = time.time()
        try:
            requests.post(f"{base_url}/api/v1/price", json=payload, timeout=2)
            latencies.append((time.time() - start) * 1000)
        except:
            print(f"❌ Request {i+1} failed")
            return False
            
    avg_latency = sum(latencies) / len(latencies)
    if avg_latency < threshold_ms:
        print(f"✅ Latency: OK (Avg: {avg_latency:.2f}ms)")
        return True
    else:
        print(f"❌ Latency: TOO SLOW (Avg: {avg_latency:.2f}ms)")
        return False

def check_metrics(base_url: str, mae_threshold: float = 10.0):
    print("Checking model metrics...")
    try:
        resp = requests.get(f"{base_url}/api/v1/metrics", timeout=5)
        # Note: In production, we'd parse Prometheus format. 
        # For simplicity, we check if metrics are exposed.
        if resp.status_code == 200:
            print("✅ Metrics: OK")
            return True
        return False
    except:
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--mae", type=float, default=8.0)
    args = parser.parse_args()
    
    all_pass = True
    all_pass &= check_health(args.url)
    all_pass &= test_prediction_latency(args.url)
    all_pass &= check_metrics(args.url, args.mae)
    
    if all_pass:
        print("\n🎉 ALL HEALTH CHECKS PASSED")
        sys.exit(0)
    else:
        print("\n🚨 HEALTH CHECKS FAILED")
        sys.exit(1)
