from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")


class PredictionData(BaseModel):
    price_target: float = Field(..., example=155.00)
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.75)
    direction: Literal["bullish", "bearish"] = Field(..., example="bullish")


class PredictionResponse(BaseModel):
    ticker: str = Field(..., example="AAPL")
    current_price: float = Field(..., example=150.00)
    prediction: PredictionData
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    error: str = Field(..., example="Invalid ticker symbol")
    detail: str = Field(..., example="Ticker 'INVALID' is not a valid stock symbol")
