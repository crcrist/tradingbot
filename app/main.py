# app/main.py - NVIDIA-focused version
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import random
import logging
import os
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from app.models.schemas import HealthResponse, PredictionResponse, PredictionData, ErrorResponse
from app.services.data_service import DataService
from app.services.cache_service import CacheService
from app.services.prediction_service import PredictionService
from app.services.model_warmup_service import ModelWarmupService
from app.middleware.performance_middleware import PerformanceMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services
data_service = DataService()
cache_service = CacheService(
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
)
prediction_service = PredictionService(cache_service=cache_service)
model_warmup_service = ModelWarmupService(prediction_service)

# Global performance middleware instance
performance_middleware = None

# FOCUS ON NVIDIA ONLY
FOCUS_TICKER = "NVDA"
VALID_TICKERS = {"NVDA"}  # Only NVIDIA for now

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown - NVIDIA focused"""
    # Startup
    logger.info("Starting Stock Prediction API - NVIDIA Focus Mode")
    
    # Connect to Redis with connection pooling
    await cache_service.connect()
    
    # Set cache service reference in prediction service
    prediction_service.cache_service = cache_service
    
    # Connect performance middleware to cache service
    cache_service.set_performance_middleware(performance_middleware)
    
    # Start model warmup for NVIDIA only (much faster!)
    logger.info("Starting NVIDIA-only model warmup...")
    warmup_task = asyncio.create_task(
        model_warmup_service.warmup_single_ticker(FOCUS_TICKER)
    )
    
    logger.info("API ready - NVIDIA model warmup running in background")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NVIDIA-focused Stock Prediction API")
    
    # Cancel warmup task if still running
    if not warmup_task.done():
        warmup_task.cancel()
    
    await cache_service.close()

app = FastAPI(
    title="Stock Prediction API - NVIDIA Focus",
    description="High-performance NVIDIA stock prediction with ensemble ML models",
    version="5.1.0-NVIDIA",
    lifespan=lifespan
)

# Add performance middleware
performance_middleware = PerformanceMiddleware(app)
app.add_middleware(PerformanceMiddleware)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy")

@app.get("/predict", response_model=PredictionResponse)
@app.get("/predict/", response_model=PredictionResponse) 
async def predict_nvidia():
    """Get NVIDIA stock prediction (default endpoint)"""
    return await predict_stock("NVDA")

@app.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict_stock(ticker: str):
    """
    Get ensemble ML prediction for supported stocks.
    Currently focused on NVIDIA (NVDA) only.
    """
    ticker = ticker.upper()
    
    # Basic ticker validation - only NVIDIA for now
    if ticker not in VALID_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Currently only NVIDIA (NVDA) is supported in focus mode. "
                   f"You requested: {ticker}. Use /predict or /predict/NVDA"
        )
    
    # Try to get cached data first (optimized with memory cache)
    cache_key = cache_service.generate_cache_key("stock_data", ticker)
    cached_data = await cache_service.get(cache_key)
    
    current_price = None
    data_source = "fallback"
    
    if cached_data:
        logger.debug(f"Using cached data for {ticker}")
        current_price = cached_data['current_price']
        data_source = "cache"
    else:
        # Fetch real stock data
        logger.debug(f"Fetching real data for {ticker}")
        stock_data = await data_service.get_stock_data(ticker)
        
        if stock_data:
            current_price = stock_data['current_price']
            data_source = "real_time"
            # Cache the data for 5 minutes
            await cache_service.set(cache_key, stock_data)
            logger.debug(f"Successfully fetched and cached real data for {ticker}: ${current_price}")
        else:
            # Fallback to mock data if real data fails
            logger.warning(f"Failed to fetch real data for {ticker}, using fallback")
            current_price = _get_fallback_price(ticker)
    
    # Get ensemble ML prediction (Random Forest + LSTM)
    ml_prediction = await prediction_service.get_ml_prediction(ticker, current_price)
    
    if ml_prediction:
        prediction_data = PredictionData(
            price_target=ml_prediction['price_target'],
            confidence=ml_prediction['confidence'],
            direction=ml_prediction['direction']
        )
        
        models_info = f"models: {ml_prediction.get('models_used', 0)}"
        if ml_prediction.get('direction_agreement'):
            models_info += " (agreement)"
        
        logger.debug(f"Ensemble prediction for {ticker}: {ml_prediction['direction']} "
                    f"target ${ml_prediction['price_target']:.2f} "
                    f"({ml_prediction['predicted_change_pct']:+.2f}%) "
                    f"confidence: {ml_prediction['confidence']:.2f} "
                    f"(type: {ml_prediction['model_type']}, {models_info})")
    else:
        # Fallback to simple prediction if ensemble fails
        logger.warning(f"Ensemble prediction failed for {ticker}, using simple fallback")
        price_change_percent = random.uniform(-0.02, 0.02)  # ±2%
        price_target = current_price * (1 + price_change_percent)
        confidence = 0.5
        direction = "bullish" if price_change_percent > 0 else "bearish"
        
        prediction_data = PredictionData(
            price_target=round(price_target, 2),
            confidence=confidence,
            direction=direction
        )
    
    return PredictionResponse(
        ticker=ticker,
        current_price=current_price,
        prediction=prediction_data,
        timestamp=datetime.utcnow()
    )

@app.get("/model/status")
@app.get("/model/status/")
async def get_nvidia_model_status():
    """Get NVIDIA model status (default endpoint)"""
    return await get_model_status("NVDA")

@app.get("/model/status/{ticker}")
async def get_model_status(ticker: str):
    """Get ensemble model status for a ticker"""
    ticker = ticker.upper()
    
    if ticker not in VALID_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Currently only NVIDIA (NVDA) is supported. You requested: {ticker}"
        )
    
    status = prediction_service.get_model_status(ticker)
    return status

@app.get("/models/compare")
@app.get("/models/compare/")
async def compare_nvidia_models():
    """Compare NVIDIA model predictions (default endpoint)"""
    return await compare_models("NVDA")

@app.get("/models/compare/{ticker}")
async def compare_models(ticker: str):
    """Compare individual model predictions"""
    ticker = ticker.upper()
    
    if ticker not in VALID_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Currently only NVIDIA (NVDA) is supported. You requested: {ticker}"
        )
    
    try:
        # Get current price for comparison
        cache_key = cache_service.generate_cache_key("stock_data", ticker)
        cached_data = await cache_service.get(cache_key)
        
        if cached_data:
            current_price = cached_data['current_price']
        else:
            current_price = _get_fallback_price(ticker)
        
        # Get model comparison
        comparison = await prediction_service.get_model_comparison(ticker, current_price)
        
        if comparison is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not generate model comparison for {ticker}"
            )
        
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing models for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to compare models")

@app.get("/models/status")
async def get_all_models_status():
    """Get comprehensive status of NVIDIA ensemble models"""
    try:
        # Get ensemble service status
        ensemble_status = prediction_service.ensemble_service.get_models_status()
        ensemble_health = prediction_service.ensemble_service.get_ensemble_health()
        
        return {
            "focus_mode": "NVIDIA_ONLY",
            "supported_tickers": list(VALID_TICKERS),
            "ensemble_system": ensemble_status,
            "ensemble_health": ensemble_health,
            "performance": prediction_service.get_performance_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting all models status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve models status")

@app.post("/model/retrain")
@app.post("/model/retrain/")
async def retrain_nvidia_model(background_tasks: BackgroundTasks):
    """Trigger NVIDIA model retraining (default endpoint)"""
    return await retrain_model("NVDA", background_tasks)

@app.post("/model/retrain/{ticker}")
async def retrain_model(ticker: str, background_tasks: BackgroundTasks):
    """Trigger ensemble model retraining for a ticker"""
    ticker = ticker.upper()
    
    if ticker not in VALID_TICKERS:
        raise HTTPException(
            status_code=400,
            detail=f"Currently only NVIDIA (NVDA) is supported. You requested: {ticker}"
        )
    
    # Add retraining task to background
    background_tasks.add_task(prediction_service.retrain_model, ticker)
    
    return {
        "message": f"Ensemble model retraining started for {ticker}",
        "ticker": ticker,
        "models": ["random_forest", "lstm"],
        "status": "training_initiated",
        "focus_mode": "NVIDIA_ONLY"
    }

@app.get("/performance/metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    try:
        # Get middleware performance stats
        middleware_stats = performance_middleware.get_stats() if performance_middleware else {}
        
        # Get prediction service performance stats
        prediction_stats = prediction_service.get_performance_stats()
        
        # Get cache statistics
        cache_stats = cache_service.get_cache_stats()
        
        # Get warmup statistics
        warmup_stats = model_warmup_service.get_warmup_stats()
        
        # Get ensemble health
        ensemble_health = prediction_service.ensemble_service.get_ensemble_health()
        
        return {
            "focus_mode": "NVIDIA_ONLY",
            "timestamp": datetime.utcnow().isoformat(),
            "request_performance": middleware_stats.get('request_stats', {}),
            "memory_usage": middleware_stats.get('memory_stats', {}),
            "cache_performance": {
                **middleware_stats.get('cache_stats', {}),
                **cache_stats
            },
            "prediction_performance": prediction_stats,
            "ensemble_health": ensemble_health,
            "model_warmup": warmup_stats,
            "system_status": "healthy"
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")

@app.get("/performance/summary")
async def get_performance_summary():
    """Get simplified performance summary for NVIDIA ensemble system"""
    try:
        middleware_stats = performance_middleware.get_stats() if performance_middleware else {}
        prediction_stats = prediction_service.get_performance_stats()
        ensemble_health = prediction_service.ensemble_service.get_ensemble_health()
        
        # Calculate key metrics
        predict_endpoint_stats = middleware_stats.get('request_stats', {}).get('GET /predict/{ticker}', {})
        
        summary = {
            "focus_mode": "NVIDIA_ONLY",
            "api_performance": {
                "avg_response_time": predict_endpoint_stats.get('avg_time', 0),
                "p95_response_time": predict_endpoint_stats.get('p95_time', 0),
                "total_requests": predict_endpoint_stats.get('count', 0),
                "error_rate": predict_endpoint_stats.get('error_rate', 0)
            },
            "ensemble_performance": {
                "avg_prediction_time": prediction_stats.get('avg_time', 0),
                "predictions_under_200ms": prediction_stats.get('under_200ms', 0),
                "total_predictions": prediction_stats.get('total_predictions', 0),
                "ensemble_error_rate": ensemble_health.get('error_rate', 0),
                "models_available": ensemble_health.get('models_available', 0)
            },
            "cache_performance": middleware_stats.get('cache_stats', {}).get('hit_rate', 0),
            "memory_usage_mb": middleware_stats.get('memory_stats', {}).get('current_mb', 0),
            "status": "optimal" if predict_endpoint_stats.get('p95_time', 0) < 2.0 else "degraded"
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance summary")

@app.get("/warmup/status")
async def get_warmup_status():
    """Get model warmup status"""
    return model_warmup_service.get_warmup_stats()

@app.get("/focus-mode")
async def get_focus_mode_info():
    """Get information about current focus mode"""
    return {
        "mode": "NVIDIA_ONLY",
        "supported_tickers": list(VALID_TICKERS),
        "focus_ticker": FOCUS_TICKER,
        "description": "System is optimized for NVIDIA stock predictions only",
        "endpoints": {
            "prediction": ["/predict", "/predict/NVDA"],
            "model_status": ["/model/status", "/model/status/NVDA"],
            "model_comparison": ["/models/compare", "/models/compare/NVDA"],
            "retrain": ["/model/retrain", "/model/retrain/NVDA"]
        },
        "benefits": [
            "No API rate limiting issues",
            "Faster startup (only 1 stock to warm up)",
            "Lower memory usage",
            "Simplified testing and debugging"
        ]
    }

def _get_fallback_price(ticker: str) -> float:
    """Fallback price for NVIDIA"""
    if ticker == "NVDA":
        return 125.00  # Realistic NVIDIA price
    return 100.00

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for better error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": str(exc.detail)}
    )