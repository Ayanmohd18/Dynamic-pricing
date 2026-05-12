import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to sys.path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_simulator(test_df: pd.DataFrame):
    """Phase 9: Simulator for real-time order events."""
    print("[*] Initializing event simulator...")
    test_df = test_df.sort_values("invoice_date")
    events = test_df.to_dict('records')
    
    print(f"[*] Simulating {len(events)} events with 1000x time compression...")
    
    for i in range(len(events)):
        event = events[i]
        
        # Format event
        yield {
            "event_type": "order",
            "stock_code": str(event["stock_code"]),
            "invoice_no": str(event["invoice_no"]),
            "quantity": int(event["quantity"]),
            "unit_price": float(event["unit_price"]),
            "customer_id": str(event["customer_id"]),
            "country": str(event["country"]),
            "timestamp": event["invoice_date"].isoformat() if isinstance(event["invoice_date"], datetime) else str(event["invoice_date"])
        }
        
        # Simulate inter-arrival time compressed by 1000x
        if i < len(events) - 1:
            next_date = events[i+1]["invoice_date"]
            curr_date = event["invoice_date"]
            
            if isinstance(next_date, datetime) and isinstance(curr_date, datetime):
                # 1 real minute = 1000x compressed = 0.06 seconds
                wait_seconds = (next_date - curr_date).total_seconds() / 1000.0
                if wait_seconds > 0:
                    time.sleep(min(wait_seconds, 2.0)) # Cap wait at 2 seconds

def simulate_flash_sale(stock_code: str, duration_minutes: int = 30):
    """Generate a burst of synthetic events."""
    print(f"[*] Simulating flash sale for {stock_code}...")
    events = []
    start_time = datetime.now()
    
    # Volume: 10x normal rate (assume normal is 1 order per 5 mins)
    orders_per_min = 2.0 
    total_orders = int(orders_per_min * duration_minutes)
    
    for i in range(total_orders):
        # Poisson process inter-arrival
        wait = np.random.exponential(scale=60.0 / orders_per_min)
        timestamp = start_time + timedelta(seconds=i * (60.0 / orders_per_min))
        
        events.append({
            "event_type": "order",
            "stock_code": stock_code,
            "invoice_no": f"SIM_{int(time.time())}_{i}",
            "quantity": np.random.randint(1, 10),
            "unit_price": 15.0, # Placeholder
            "customer_id": f"GUEST_SIM_{i}",
            "country": "United Kingdom",
            "timestamp": timestamp.isoformat()
        })
    
    return events

if __name__ == "__main__":
    # Example usage
    try:
        df = pd.read_parquet(config.PROCESSED_DIR / "cleaned.parquet").tail(100)
        sim = build_simulator(df)
        for event in sim:
            print(event)
    except Exception as e:
        print(f"Error in simulator: {e}")
