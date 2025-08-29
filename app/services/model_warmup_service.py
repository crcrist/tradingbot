import asyncio
import logging
from typing import List, Dict
import time
import numpy as np
from pathlib import Path

from .prediction_service import PredictionService

logger = logging.getLogger(__name__)


class ModelWarmupService:
    """Service for warming up ML models on application startup"""
    
    def __init__(self, prediction_service: PredictionService):
        self.prediction_service = prediction_service
        self.supported_tickers = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", 
            "META", "NVDA", "NFLX", "ORCL", "CRM"
        ]
        self.warmup_stats = {
            'started_at': None,
            'completed_at': None,
            'duration_seconds': 0,
            'models_loaded': 0,
            'models_trained': 0,
            'errors': []
        }
    
    async def warmup_models(self, max_concurrent: int = 3) -> Dict:
        """
        Warm up models for all supported tickers
        
        Args:
            max_concurrent: Maximum number of concurrent model operations
            
        Returns:
            Dictionary with warmup statistics
        """
        logger.info("Starting model warmup process")
        start_time = time.time()
        self.warmup_stats['started_at'] = start_time
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create warmup tasks for all tickers
        tasks = []
        for ticker in self.supported_tickers:
            task = self._warmup_single_model(ticker, semaphore)
            tasks.append(task)
        
        # Execute warmup tasks concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for ticker, result in zip(self.supported_tickers, results):
                if isinstance(result, Exception):
                    error_msg = f"Warmup failed for {ticker}: {str(result)}"
                    logger.error(error_msg)
                    self.warmup_stats['errors'].append(error_msg)
                elif result:
                    if result.get('model_loaded'):
                        self.warmup_stats['models_loaded'] += 1
                    if result.get('model_trained'):
                        self.warmup_stats['models_trained'] += 1
        
        except Exception as e:
            error_msg = f"Warmup process error: {str(e)}"
            logger.error(error_msg)
            self.warmup_stats['errors'].append(error_msg)
        
        # Finalize stats
        end_time = time.time()
        self.warmup_stats['completed_at'] = end_time
        self.warmup_stats['duration_seconds'] = end_time - start_time
        
        logger.info(f"Model warmup completed in {self.warmup_stats['duration_seconds']:.2f}s")
        logger.info(f"Models loaded: {self.warmup_stats['models_loaded']}")
        logger.info(f"Models trained: {self.warmup_stats['models_trained']}")
        
        if self.warmup_stats['errors']:
            logger.warning(f"Warmup errors: {len(self.warmup_stats['errors'])}")
        
        return self.warmup_stats.copy()
    
    async def _warmup_single_model(self, ticker: str, semaphore: asyncio.Semaphore) -> Dict:
        """Warm up a single model with concurrency control"""
        async with semaphore:
            try:
                logger.debug(f"Starting warmup for {ticker}")
                start_time = time.time()
                
                # Check if model exists and load it - use the correct method name
                model_ready = await self.prediction_service._ensure_models_ready(ticker)
                
                warmup_time = time.time() - start_time
                
                # Check if any models are ready
                models_available = any(model_ready.values()) if isinstance(model_ready, dict) else False
                
                if models_available:
                    # Perform a dummy prediction to warm up the model
                    dummy_price = 100.0
                    await self._dummy_prediction(ticker, dummy_price)
                    
                    logger.info(f"Model warmed up for {ticker} in {warmup_time:.2f}s")
                    return {
                        'ticker': ticker,
                        'success': True,
                        'model_loaded': ticker in self.prediction_service.loaded_models or f"{ticker}_rf" in self.prediction_service.loaded_models,
                        'model_trained': models_available,
                        'warmup_time': warmup_time
                    }
                else:
                    logger.warning(f"Failed to warm up model for {ticker}")
                    return {
                        'ticker': ticker,
                        'success': False,
                        'model_loaded': False,
                        'model_trained': False,
                        'warmup_time': warmup_time
                    }
                    
            except Exception as e:
                logger.error(f"Error warming up model for {ticker}: {str(e)}")
                return {
                    'ticker': ticker,
                    'success': False,
                    'error': str(e),
                    'model_loaded': False,
                    'model_trained': False
                }
    
    async def _dummy_prediction(self, ticker: str, dummy_price: float) -> None:
        """Perform a dummy prediction to warm up model inference"""
        try:
            # This loads the model into memory and exercises the prediction pipeline
            await self.prediction_service.get_ml_prediction(ticker, dummy_price)
            logger.debug(f"Dummy prediction completed for {ticker}")
        except Exception as e:
            logger.debug(f"Dummy prediction failed for {ticker}: {str(e)}")
    
    def get_warmup_stats(self) -> Dict:
        """Get warmup statistics"""
        return self.warmup_stats.copy()