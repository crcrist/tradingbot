import logging
from typing import Optional, Dict
import asyncio
from datetime import datetime
import time
import numpy as np

from .historical_data_service import HistoricalDataService
from .feature_engineering_service import FeatureEngineeringService
from .ml_model_service import MLModelService

logger = logging.getLogger(__name__)


class PredictionService:
    """Optimized orchestration service for ML-based stock price predictions"""
    
    def __init__(self, cache_service=None):
        self.historical_service = HistoricalDataService()
        self.feature_service = FeatureEngineeringService()
        self.ml_service = MLModelService()
        self.cache_service = cache_service
        
        # Cache for models to avoid reloading
        self.loaded_models = set()
        
        # Performance tracking
        self.prediction_times = []
        
        logger.info("Optimized prediction service initialized")
    
    async def get_ml_prediction(self, ticker: str, current_price: float) -> Optional[Dict]:
        """
        Get ML-based prediction for a stock with performance optimizations
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            
        Returns:
            Dictionary with prediction results or None if failed
        """
        start_time = time.time()
        try:
            ticker = ticker.upper()
            
            # Ensure model is loaded/trained
            model_ready = await self._ensure_model_ready(ticker)
            if not model_ready:
                logger.warning(f"Model not ready for {ticker}, using fallback prediction")
                return self._fallback_prediction(current_price)
            
            # Get recent historical data for feature calculation
            historical_data = await self.historical_service.get_historical_data(ticker, days=60)
            if historical_data is None:
                logger.warning(f"No historical data for {ticker}, using fallback")
                return self._fallback_prediction(current_price)
            
            # Create features with optimizations
            features_df = self.feature_service.create_features(historical_data)
            if features_df is None:
                logger.warning(f"Could not create features for {ticker}, using fallback")
                return self._fallback_prediction(current_price)
            
            # Get latest features for prediction with caching optimization
            latest_features = self.feature_service.get_latest_features_optimized(features_df)
            if latest_features is None:
                logger.warning(f"Could not extract latest features for {ticker}, using fallback")
                return self._fallback_prediction(current_price)
            
            # Check prediction cache first
            cached_prediction = None
            if self.cache_service:
                features_hash = self.cache_service.generate_features_hash(latest_features)
                cached_prediction = await self.cache_service.get_prediction_cache(ticker, features_hash)
            
            if cached_prediction:
                logger.info(f"Using cached prediction for {ticker}")
                # Update timestamp for cached prediction
                cached_prediction['timestamp'] = datetime.utcnow().isoformat()
                prediction_time = time.time() - start_time
                self.prediction_times.append(prediction_time)
                return cached_prediction
            
            # Make ML prediction
            ml_result = self.ml_service.predict(latest_features)
            if ml_result is None:
                logger.warning(f"ML prediction failed for {ticker}, using fallback")
                return self._fallback_prediction(current_price)
            
            # Convert predicted change percentage to target price
            predicted_change = ml_result['predicted_change_pct']
            target_price = current_price * (1 + predicted_change)
            
            prediction_result = {
                'price_target': round(target_price, 2),
                'predicted_change_pct': round(predicted_change * 100, 2),  # Convert to percentage
                'direction': ml_result['direction'],
                'confidence': round(ml_result['confidence'], 2),
                'model_type': 'ml_random_forest',
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache the prediction for future requests with identical features
            if self.cache_service:
                await self.cache_service.set_prediction_cache(ticker, features_hash, prediction_result)
            
            # Track performance
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            logger.info(f"ML prediction for {ticker}: {ml_result['direction']} "
                       f"({predicted_change:+.3f}%) target ${target_price:.2f} "
                       f"(took {prediction_time:.3f}s)")
            
            return prediction_result
            
        except Exception as e:
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            logger.error(f"Error in ML prediction for {ticker} (took {prediction_time:.3f}s): {str(e)}")
            return self._fallback_prediction(current_price)
    
    async def _ensure_model_ready(self, ticker: str) -> bool:
        """Ensure model is loaded and trained for the ticker with optimizations"""
        try:
            # Check if model is already loaded
            if ticker in self.loaded_models and self.ml_service.is_model_ready():
                return True
            
            # Try to load existing model
            if self.ml_service.load_model(ticker):
                self.loaded_models.add(ticker)
                return True
            
            # No model exists, train new one
            logger.info(f"Training new model for {ticker}")
            return await self._train_new_model(ticker)
            
        except Exception as e:
            logger.error(f"Error ensuring model ready for {ticker}: {str(e)}")
            return False
    
    async def _train_new_model(self, ticker: str) -> bool:
        """Train a new model for the ticker with optimizations"""
        try:
            # Get historical data for training (1 year)
            historical_data = await self.historical_service.get_historical_data(ticker, days=252)
            if historical_data is None or len(historical_data) < 100:
                logger.error(f"Insufficient historical data for training {ticker}")
                return False
            
            # Create features with optimizations
            features_df = self.feature_service.create_features(historical_data)
            if features_df is None:
                logger.error(f"Could not create features for training {ticker}")
                return False
            
            # Prepare training data with optimizations
            X, y = self.feature_service.prepare_training_data_optimized(features_df)
            if len(X) == 0:
                logger.error(f"No training data prepared for {ticker}")
                return False
            
            # Set feature names
            self.ml_service.set_feature_names(self.feature_service.get_feature_names())
            
            # Train model
            success = self.ml_service.train_model(X, y, ticker)
            if success:
                self.loaded_models.add(ticker)
                logger.info(f"Successfully trained and loaded model for {ticker}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error training new model for {ticker}: {str(e)}")
            return False
    
    def _fallback_prediction(self, current_price: float) -> Dict:
        """Fallback prediction when ML model is not available"""
        import random
        
        # Simple fallback logic similar to Phase 2
        change_pct = random.uniform(-2.0, 2.0)  # ±2%
        target_price = current_price * (1 + change_pct / 100)
        direction = "bullish" if change_pct > 0 else "bearish"
        
        logger.info(f"Using fallback prediction: {direction} ({change_pct:+.2f}%)")
        
        return {
            'price_target': round(target_price, 2),
            'predicted_change_pct': round(change_pct, 2),
            'direction': direction,
            'confidence': 0.5,  # Low confidence for fallback
            'model_type': 'fallback',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def retrain_model(self, ticker: str) -> bool:
        """Force retrain model for a ticker"""
        try:
            ticker = ticker.upper()
            logger.info(f"Force retraining model for {ticker}")
            
            # Remove from loaded models to force retrain
            self.loaded_models.discard(ticker)
            
            # Clear feature engineering caches for fresh training
            self.feature_service.clear_cache()
            
            return await self._train_new_model(ticker)
            
        except Exception as e:
            logger.error(f"Error retraining model for {ticker}: {str(e)}")
            return False
    
    def get_model_status(self, ticker: str) -> Dict:
        """Get status information for a ticker's model"""
        ticker = ticker.upper()
        
        status = {
            'ticker': ticker,
            'model_loaded': ticker in self.loaded_models,
            'service_ready': self.ml_service.is_model_ready()
        }
        
        if status['model_loaded']:
            status.update(self.ml_service.get_model_info())
        
        return status
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for predictions"""
        if not self.prediction_times:
            return {
                'total_predictions': 0,
                'avg_time': 0,
                'min_time': 0,
                'max_time': 0,
                'p95_time': 0
            }
        
        times = self.prediction_times[-1000:]  # Keep last 1000 predictions
        times.sort()
        
        return {
            'total_predictions': len(self.prediction_times),
            'recent_predictions': len(times),
            'avg_time': sum(times) / len(times),
            'min_time': times[0],
            'max_time': times[-1],
            'p95_time': times[int(len(times) * 0.95)] if len(times) > 20 else times[-1],
            'under_100ms': sum(1 for t in times if t < 0.1) / len(times)
        }