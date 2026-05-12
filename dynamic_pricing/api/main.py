from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import redis.asyncio as aioredis
import time
import json
import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.predict import predict_optimal_price, load_models
from api.schemas import PriceRequest, PriceResponse, BulkPriceRequest

app = FastAPI(
    title="Dynamic Pricing Engine",
    description="Real-time price optimization API for Online Retail datasets",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
redis = None
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    global redis
    print("[*] API Starting up...")
    # 1. Initialize Redis
    try:
        redis = aioredis.from_url(config.REDIS_URL, decode_responses=True)
        await redis.ping()
        print("[*] Connected to Redis")
    except Exception as e:
        print(f"[!] Redis connection failed: {e}. Caching disabled.")
        redis = None

    # 2. Pre-load models (This is done in predict.py on first call, but we can trigger it here)
    load_models()
    print("[*] Models pre-loaded")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - start_time,
        "redis_connected": redis is not None
    }

@app.post("/price/recommend", response_model=PriceResponse)
async def recommend_price(request: PriceRequest):
    start = time.perf_counter()
    
    # 1. Check Cache
    cache_key = f"price:{request.stock_code}:{hash(str(request.dict()))}"
    if redis:
        try:
            cached_res = await redis.get(cache_key)
            if cached_res:
                res = json.loads(cached_res)
                res["cached"] = True
                res["latency_ms"] = (time.perf_counter() - start) * 1000
                return res
        except:
            pass

    # 2. Inference
    try:
        result = predict_optimal_price(
            stock_code=request.stock_code,
            current_price=request.current_price,
            quantity=request.quantity,
            country=request.country,
            timestamp=request.timestamp,
            adjustment_factors=request.adjustment_factors
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        result["latency_ms"] = (time.perf_counter() - start) * 1000
        result["cached"] = False
        
        # 3. Store in Cache
        if redis:
            try:
                await redis.setex(cache_key, config.CACHE_TTL_SECONDS, json.dumps(result))
            except:
                pass
                
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/price/bulk")
async def bulk_recommend(request: BulkPriceRequest):
    start = time.perf_counter()
    
    # Process in parallel using asyncio
    async def process_item(item):
        return await recommend_price(item)
    
    tasks = [process_item(item) for item in request.requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter exceptions
    final_results = []
    for r in results:
        if isinstance(r, Exception):
            final_results.append({"error": str(r)})
        else:
            final_results.append(r)
            
    return {
        "count": len(final_results),
        "latency_ms": (time.perf_counter() - start) * 1000,
        "results": final_results
    }

@app.get("/products/search")
async def search_products(q: str):
    # This would typically query a database. For demo, we return placeholders or query product_stats.
    # In a real app, we'd use a search engine or SQL.
    return {"query": q, "results": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
