import random
import time
import pandas as pd
import os

class CompetitorMonitor:
    """
    Simulates polling competitor price APIs or scraping market data.
    """
    def __init__(self, data_path):
        # We'll use the existing Olist categories as a base
        self.df = pd.read_parquet(data_path)
        self.categories = self.df['category'].unique()

    def fetch_market_prices(self, category):
        """
        Simulates an API call to a competitor aggregator.
        Returns the average market price for a category with some random variance.
        """
        # Get baseline median price for this category from our historical data
        cat_median = self.df[self.df['category'] == category]['price'].median()
        
        # Simulate market volatility (+/- 10%)
        volatility = random.uniform(0.9, 1.1)
        return round(cat_median * volatility, 2)

    def run_polling_cycle(self, interval_seconds=30):
        """
        Simulates a background polling task.
        """
        print(f"Starting Competitor Polling (Interval: {interval_seconds}s)...")
        try:
            while True:
                # Pick a random category to refresh
                cat = random.choice(self.categories)
                avg_price = self.fetch_market_prices(cat)
                print(f"Market Sync [Category: {cat}]: Avg Price ₹{avg_price}")
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("Polling stopped.")

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "features.parquet")
    if os.path.exists(DATA_PATH):
        monitor = CompetitorMonitor(DATA_PATH)
        # Running a single simulation check
        test_cat = "health_beauty"
        print(f"Simulated Market Fetch for '{test_cat}': ₹{monitor.fetch_market_prices(test_cat)}")
    else:
        print("Error: Features data not found. Run Notebook 02 first.")
