from pathlib import Path

# ── Paths ─────────────────────────────────────────────
RAW_DIR = Path("D:/DP/online+retail+ii")
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = Path("models/saved")
METRICS_DIR = Path("models/metrics")

# ── Dataset filenames ─────────────────────────────────
RETAIL_I_FILE = RAW_DIR / "Online Retail.xlsx"
RETAIL_II_FILE = RAW_DIR / "online_retail_II.xlsx"
RETAIL_II_SHEETS = ["Year 2009-2010", "Year 2010-2011"]

# ── Cleaning thresholds ───────────────────────────────
MIN_UNIT_PRICE = 0.01       # filter out free/error rows
MAX_UNIT_PRICE = 5000.0     # filter outlier prices
MIN_QUANTITY = 1            # remove returns/cancellations
MAX_QUANTITY = 10000        # filter bulk anomalies
MIN_DESCRIPTION_LEN = 3

# ── Feature engineering ───────────────────────────────
DEMAND_WINDOWS = [7, 14, 30]     # rolling days
LAG_DAYS = [1, 7, 14, 30]        # lag features
PRICE_ELASTICITY_WINDOW = 30     # days for elasticity calc
FLASH_SALE_MULTIPLIER_MIN = 1.5  # spike detection threshold
COMPETITOR_BLEND_RATIO = 0.30    # 30% competitor, 70% ML

# ── Model hyperparameters ─────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1
}
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "n_jobs": -1,
    "random_state": 42
}
LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1
}

# ── API ───────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
REDIS_URL = "redis://localhost:6379"
CACHE_TTL_SECONDS = 60

# ── Kafka ─────────────────────────────────────────────
KAFKA_BROKER = "localhost:9092"
KAFKA_ORDER_TOPIC = "order_events"
KAFKA_PRICE_TOPIC = "price_updates"
KAFKA_ALERT_TOPIC = "flash_sale_alerts"
