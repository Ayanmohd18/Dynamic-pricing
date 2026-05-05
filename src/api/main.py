import time
import os
import yaml
import json
import asyncio
import joblib
import structlog
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from redis import asyncio as aioredis
from aiokafka import AIOKafkaProducer
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.optimizer import PriceOptimizer, PriceDecision
from data.feature_store import FeatureStore

# --- Structured Logging Setup ---
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter("pricing_request_total", "Total Request Count", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("pricing_request_latency_seconds", "Request Latency", ["endpoint"])
INFERENCE_LATENCY = Histogram("pricing_inference_latency_seconds", "Inference Latency")

# --- Rate Limiter ---
limiter = Limiter(key_func=get_remote_address)

# --- Configuration ---
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "config.yaml")

# --- Pydantic Models ---
class PriceRequest(BaseModel):
    sku_id: str
    inventory_level: float = Field(..., ge=0, le=1)
    demand_score: float = Field(..., ge=0, le=1000)
    customer_segment: int = Field(..., ge=0, le=4)
    is_flash_sale: bool = False
    cost_price: float
    msrp: float

class PriceResponse(BaseModel):
    sku_id: str
    recommended_price: float
    confidence_interval: Tuple[float, float]
    reasoning: str
    multipliers: Dict[str, float]
    latency_ms: float
    cached: bool
    timestamp: datetime

class BatchPriceRequest(BaseModel):
    items: List[PriceRequest]

class HealthResponse(BaseModel):
    status: str
    model_version: str
    redis_connected: bool
    kafka_connected: bool
    uptime_seconds: float

# --- Global State ---
state = {
    "model": None,
    "config": None,
    "redis": None,
    "kafka": None,
    "optimizer": None,
    "feature_store": None,
    "start_time": time.time()
}

# --- Background Tasks ---
async def publish_price_update(sku_id: str, price: float, decision: dict):
    """Publishes the pricing decision to Kafka."""
    if state["kafka"]:
        payload = {
            "sku_id": sku_id,
            "price": price,
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        }
        await state["kafka"].send_and_wait(
            state["config"]["kafka"]["topics"]["price_updates"],
            json.dumps(payload).encode("utf-8")
        )

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Pricing API Services...")
    
    # 1. Load Config
    with open(CONFIG_PATH, "r") as f:
        state["config"] = yaml.safe_load(f)
    
    # 2. Load Model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "models", "xgboost_pricing_v1.pkl")
    if os.path.exists(model_path):
        state["model"] = joblib.load(model_path)
        logger.info("XGBoost Model Loaded", path=model_path)
    else:
        logger.error("Model file not found!", path=model_path)
    
    # 3. Connect Redis
    state["redis"] = aioredis.from_url(
        f"redis://{state['config']['redis']['host']}:{state['config']['redis']['port']}",
        decode_responses=True
    )
    
    # 4. Connect Kafka
    state["kafka"] = AIOKafkaProducer(bootstrap_servers=state["config"]["kafka"]["bootstrap_servers"])
    await state["kafka"].start()
    
    # 5. Initialize Optimizer & Store
    state["feature_store"] = FeatureStore(state["redis"], state["config"])
    state["optimizer"] = PriceOptimizer(state["model"], state["config"], state["feature_store"])
    
    logger.info("All services started.")
    yield
    
    # Shutdown
    logger.info("Shutting down Pricing API Services...")
    await state["kafka"].stop()
    await state["redis"].close()

# --- App Setup ---
app = FastAPI(title="Dynamic Pricing Engine", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    return {
        "status": "healthy" if state["model"] else "degraded",
        "model_version": state["config"]["model"]["version"],
        "redis_connected": state["redis"] is not None,
        "kafka_connected": state["kafka"] is not None,
        "uptime_seconds": time.time() - state["start_time"]
    }

@app.get("/api/v1/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/api/v1/price", response_model=PriceResponse)
@limiter.limit("100/minute")
async def get_price(request: Request, price_req: PriceRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    sku_id = price_req.sku_id
    
    # 1. Check Cache
    cache_key = f"price:cache:{sku_id}"
    cached_val = await state["redis"].get(cache_key)
    if cached_val:
        decision = json.loads(cached_val)
        latency = (time.time() - start_time) * 1000
        REQUEST_COUNT.labels("POST", "/price", "200").inc()
        return {**decision, "latency_ms": latency, "cached": True}

    # 2. Inference
    if not state["optimizer"]:
        raise HTTPException(status_code=503, detail="Model/Optimizer not initialized")

    with INFERENCE_LATENCY.time():
        # Mapping API request to internal context format
        context = {
            "inventory_ratio": price_req.inventory_level,
            "demand_score_7d": price_req.demand_score,
            "is_flash_sale": price_req.is_flash_sale,
            "customer_segment": price_req.customer_segment,
            "cost": price_req.cost_price,
            "msrp": price_req.msrp,
            # Placeholder ML features (In production these would come from FeatureStore or lookup)
            "price": price_req.msrp * 0.8,
            "freight_value": 15.0, "hour_sin": 0.5, "hour_cos": 0.8,
            "day_sin": 0.7, "day_cos": 0.7, "is_weekend": 0, "is_month_end": 0, "is_holiday": 0,
            "days_since_last_order": 2.0, "demand_score_30d": price_req.demand_score * 3,
            "demand_velocity": 0.05, "price_percentile_in_category": 0.5,
            "competitor_delta": 0.0, "review_elasticity": 1.0
        }
        
        decision = state["optimizer"].get_optimal_price(sku_id, context)

    # 3. Cache Result (TTL from config)
    res_dict = {
        "sku_id": sku_id,
        "recommended_price": decision.final_price,
        "confidence_interval": decision.confidence_interval,
        "reasoning": decision.reasoning,
        "multipliers": decision.multipliers_applied,
        "timestamp": decision.timestamp.isoformat()
    }
    await state["redis"].setex(cache_key, state["config"]["redis"]["ttl"]["price_cache"], json.dumps(res_dict))

    # 4. Background Task (Kafka)
    background_tasks.add_task(publish_price_update, sku_id, decision.final_price, res_dict)

    latency = (time.time() - start_time) * 1000
    REQUEST_COUNT.labels("POST", "/price", "200").inc()
    REQUEST_LATENCY.labels("/price").observe(time.time() - start_time)
    
    return {**res_dict, "latency_ms": latency, "cached": False}

@app.post("/api/v1/price/batch")
async def get_price_batch(batch_req: BatchPriceRequest, background_tasks: BackgroundTasks):
    """
    Processes multiple pricing requests in parallel.
    """
    if not state["optimizer"]:
        raise HTTPException(status_code=503, detail="Optimizer not initialized")
    
    results = []
    start_time = time.time()
    
    # Use asyncio.gather for concurrent processing
    async def process_item(item: PriceRequest):
        # 1. Check Cache
        cache_key = f"price:cache:{item.sku_id}"
        cached_val = await state["redis"].get(cache_key)
        if cached_val:
            res = json.loads(cached_val)
            return {**res, "cached": True, "latency_ms": 0}

        # 2. Inference
        context = {
            "inventory_ratio": item.inventory_level,
            "demand_score_7d": item.demand_score,
            "is_flash_sale": item.is_flash_sale,
            "customer_segment": item.customer_segment,
            "cost": item.cost_price,
            "msrp": item.msrp,
            # Placeholder ML features
            "price": item.msrp * 0.8,
            "freight_value": 15.0, "hour_sin": 0.5, "hour_cos": 0.8,
            "day_sin": 0.7, "day_cos": 0.7, "is_weekend": 0, "is_month_end": 0, "is_holiday": 0,
            "days_since_last_order": 2.0, "demand_score_30d": item.demand_score * 3,
            "demand_velocity": 0.05, "price_percentile_in_category": 0.5,
            "competitor_delta": 0.0, "review_elasticity": 1.0
        }
        
        decision = state["optimizer"].get_optimal_price(item.sku_id, context)
        
        res_dict = {
            "sku_id": item.sku_id,
            "recommended_price": decision.final_price,
            "confidence_interval": decision.confidence_interval,
            "reasoning": decision.reasoning,
            "multipliers": decision.multipliers_applied,
            "timestamp": decision.timestamp.isoformat(),
            "cached": False
        }
        
        # Cache and background tasks
        await state["redis"].setex(cache_key, state["config"]["redis"]["ttl"]["price_cache"], json.dumps(res_dict))
        background_tasks.add_task(publish_price_update, item.sku_id, decision.final_price, res_dict)
        
        return res_dict

    results = await asyncio.gather(*[process_item(item) for item in batch_req.items])
    
    latency = (time.time() - start_time) * 1000
    return {"results": results, "batch_latency_ms": latency}

@app.get("/api/v1/price/{sku_id}")
async def get_cached_price(sku_id: str):
    val = await state["redis"].get(f"price:cache:{sku_id}")
    if not val:
        raise HTTPException(status_code=404, detail="SKU price not in cache")
    return json.loads(val)

@app.delete("/api/v1/price/{sku_id}/cache")
async def invalidate_cache(sku_id: str):
    await state["redis"].delete(f"price:cache:{sku_id}")
    return {"status": "success", "message": f"Cache invalidated for {sku_id}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
