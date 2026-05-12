from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional

class PriceRequest(BaseModel):
    stock_code: str
    current_price: float = Field(gt=0, lt=10000)
    quantity: int = Field(default=1, ge=1)
    country: str = Field(default="United Kingdom")
    timestamp: Optional[datetime] = None
    adjustment_factors: Optional[Dict] = None

class PriceResponse(BaseModel):
    model_config = {'protected_namespaces': ()}
    
    stock_code: str
    current_price: float
    recommended_price: float
    price_change_pct: float
    confidence_low: float
    confidence_high: float
    model_predictions: Dict
    adjustments_applied: List[str]
    reasoning: str
    latency_ms: float
    cached: bool


class BulkPriceRequest(BaseModel):
    requests: List[PriceRequest] = Field(..., max_items=500)

class FlashSaleAlert(BaseModel):
    stock_code: str
    spike_ratio: float
    recommended_price: float
    severity: str
