from collections import defaultdict, deque
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

class FlashSaleDetector:
    """Phase 10: Complex Event Processing for demand spike detection."""
    
    def __init__(self, window_seconds: int = 300, spike_multiplier: float = 1.5):
        self.window_seconds = window_seconds
        self.spike_multiplier = spike_multiplier
        # Per-product: deque of (timestamp, quantity) events
        self.event_windows = defaultdict(deque)
        # Historical baseline demand rates (units per window_seconds)
        self.baseline_rates = {}
        # Active alerts: stock_code -> alert_dict
        self.active_alerts = {}

    def load_baselines(self, df: pd.DataFrame):
        """
        Compute baseline events-per-5-minutes per stock_code from historical data.
        """
        print("[*] Computing demand baselines for flash sale detection...")
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        
        # Floor dates to the window size (e.g. 5 min)
        window_label = f"{self.window_seconds}s"
        df['interval'] = df['invoice_date'].dt.floor(window_label)
        
        # Sum quantities per interval
        interval_demand = df.groupby(['stock_code', 'interval'])['quantity'].sum().reset_index()
        
        # Median demand per interval per product
        self.baseline_rates = interval_demand.groupby('stock_code')['quantity'].median().to_dict()
        print(f"[*] Loaded baselines for {len(self.baseline_rates)} products.")

    def ingest_event(self, event: dict) -> dict or None:
        """
        Ingest one order event and detect spikes.
        """
        stock_code = event.get("stock_code")
        if not stock_code:
            return None
            
        qty = event.get("quantity", 1)
        ts_str = event.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if isinstance(ts_str, str) else ts_str
        except:
            ts = datetime.now()

        # 1. Add event to window
        window = self.event_windows[stock_code]
        window.append((ts, qty))

        # 2. Purge old events
        cutoff = ts - timedelta(seconds=self.window_seconds)
        while window and window[0][0] < cutoff:
            window.popleft()

        # 3. Compute current rate
        current_rate = sum(item[1] for item in window)
        baseline = self.baseline_rates.get(stock_code, 1.0) # Fallback to 1
        
        # 4. Detect Spike
        spike_ratio = current_rate / max(baseline, 0.1)
        
        if spike_ratio > self.spike_multiplier:
            # 5. Generate Alert
            severity = "LOW"
            if spike_ratio >= 5.0: severity = "CRITICAL"
            elif spike_ratio >= 3.0: severity = "HIGH"
            elif spike_ratio >= 2.0: severity = "MEDIUM"
            
            # recommended_price_multiplier: 1 + (0.1 * log(spike_ratio))
            # Actually, for flash sales we might want to DISCOUNT, 
            # but usually Surge Pricing means INCREASING.
            # The prompt says: "recommended_price_multiplier = 1 + (0.1 * log(current_rate / baseline))"
            multiplier = 1 + (0.1 * math.log(spike_ratio))
            multiplier = min(multiplier, 1.40) # Cap at 1.4x
            
            alert = {
                "type": "FLASH_SALE_DETECTED",
                "stock_code": stock_code,
                "detected_at": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                "current_rate_units_per_window": float(current_rate),
                "baseline_rate": float(baseline),
                "spike_ratio": round(float(spike_ratio), 2),
                "recommended_price_multiplier": round(multiplier, 2),
                "severity": severity
            }
            self.active_alerts[stock_code] = alert
            return alert
        else:
            # If rate falls below threshold, clear alert
            if stock_code in self.active_alerts:
                del self.active_alerts[stock_code]
            return None

    def get_alert(self, stock_code: str) -> dict or None:
        return self.active_alerts.get(stock_code)

    def get_active_alerts(self) -> list:
        return list(self.active_alerts.values())

if __name__ == "__main__":
    # Test
    detector = FlashSaleDetector()
    detector.baseline_rates = {"TEST": 5.0}
    
    now = datetime.now()
    # Normal volume
    detector.ingest_event({"stock_code": "TEST", "quantity": 2, "timestamp": now.isoformat()})
    # Spike!
    alert = detector.ingest_event({"stock_code": "TEST", "quantity": 10, "timestamp": (now + timedelta(seconds=10)).isoformat()})
    print(alert)
