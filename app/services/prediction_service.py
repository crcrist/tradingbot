import logging
from typing import Optional, Dict
import asyncio
from datetime import datetime
import time
import numpy as np

from .historical_data_service import HistoricalDataService
from .feature_engineering_service import FeatureEngineeringService
from .ensemble_service import EnsembleService

logger = logging.getLogger(__name__)


class PredictionService:
    """Enhanced prediction service with ensemble ML models (Phase 5)"""
    
    def __init__(self, cache_service=None):
        self.historical_service = HistoricalDataService()
        self.feature_service = FeatureEngineeringService()
        self.ensemble_service = EnsembleService()
        self.cache_service = cache_service
        
        # Cache for models to avoid reloading
        self.loaded_models = set()
        
        # Performance tracking
        self.prediction_times = []
        
        logger.info("Enhanced prediction service initialized (Phase 5)")
    
    async def get_ml_prediction(self, ticker: str, current_price: float) -> Optional[Dict]:
        """
        Get ensemble ML prediction combining Random Forest and LSTM models
        
        Args:
            ticker: Stock ticker symbol
            current_price: Current stock price
            
        Returns:
            Dictionary with ensemble prediction results
        """
        start_time = time.time()
        try:
            ticker = ticker.upper()
            
            # Ensure models are ready
            models_ready = await self._ensure_models_ready(ticker)
            if not any(models_ready.values()):
                logger.warning(f"No models ready for {ticker}, using fallback prediction")
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
            
            # Get latest features for prediction
            latest_features = self.feature_service.get_latest_features_optimized(features_df)
            if latest_features is None:
                logger.warning(f"Could not extract latest features for {ticker}, using fallback")
                return self._fallback_prediction(current_price)
            
            # Check prediction cache first
            cached_prediction = None
            if self.cache_service:
                features_hash = self.cache_service.generate_features_hash(latest_features)
                cached_prediction = await self.cache_service.get_prediction_cache(ticker, features_hash)
            
            if cached_prediction and cached_prediction.get('model_type') == 'ensemble':
                logger.info(f"Using cached ensemble prediction for {ticker}")
                cached_prediction['timestamp'] = datetime.utcnow().isoformat()
                prediction_time = time.time() - start_time
                self.prediction_times.append(prediction_time)
                return cached_prediction
            
            # Get ensemble prediction
            ensemble_result = self.ensemble_service.get_ensemble_prediction(ticker, latest_features)
            if ensemble_result is None:
                logger.warning(f"Ensemble prediction failed for {ticker}, using fallback")
                return self._fallback_prediction(current_price)
            
            # Convert predicted change percentage to target price
            predicted_change_pct = ensemble_result['predicted_change_pct'] / 100.0  # Convert from percentage
            target_price = current_price * (1 + predicted_change_pct)
            
            prediction_result = {
                'price_target': round(target_price, 2),
                'predicted_change_pct': ensemble_result['predicted_change_pct'],
                'direction': ensemble_result['direction'],
                'confidence': ensemble_result['confidence'],
                'model_type': ensemble_result['model_type'],
                'models_used': ensemble_result.get('models_used', 0),
                'direction_agreement': ensemble_result.get('direction_agreement', False),
                'model_weights': ensemble_result.get('model_weights', {}),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Cache the prediction
            if self.cache_service:
                await self.cache_service.set_prediction_cache(ticker, features_hash, prediction_result)
            
            # Track performance
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            logger.info(f"Ensemble prediction for {ticker}: {ensemble_result['direction']} "
                       f"({predicted_change_pct:+.3f}%) target ${target_price:.2f} "
                       f"(took {prediction_time:.3f}s, models: {ensemble_result.get('models_used', 0)})")
            
            return prediction_result
            
        except Exception as e:
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            logger.error(f"Error in ensemble prediction for {ticker} (took {prediction_time:.3f}s): {str(e)}")
            return self._fallback_prediction(current_price)
    
    async def _ensure_models_ready(self, ticker: str) -> Dict[str, bool]:
        """Ensure ensemble models are loaded and trained"""
        try:
            # Try to load existing models first
            load_results = self.ensemble_service.load_models(ticker)
            
            # If no models loaded successfully, try training
            if not any(load_results.values()):
                logger.info(f"No existing models found for {ticker}, training new ones")
                return await self._train_ensemble_models(ticker)
            
            # Add to loaded models cache
            if load_results.get('random_forest'):
                self.loaded_models.add(f"{ticker}_rf")
            if load_results.get('lstm'):
                self.loaded_models.add(f"{ticker}_lstm")
            
            return load_results
            
        except Exception as e:
            logger.error(f"Error ensuring models ready for {ticker}: {str(e)}")
            return {'random_forest': False, 'lstm': False}
    
    async def _train_ensemble_models(self, ticker: str) -> Dict[str, bool]:
        """Train ensemble models for the ticker"""
        try:
            # Get historical data for training (1 year)
            historical_data = await self.historical_service.get_historical_data(ticker, days=252)
            if historical_data is None or len(historical_data) < 100:
                logger.error(f"Insufficient historical data for training {ticker}")
                return {'random_forest': False, 'lstm': False}
            
            # Create features
            features_df = self.feature_service.create_features(historical_data)
            if features_df is None:
                logger.error(f"Could not create features for training {ticker}")
                return {'random_forest': False, 'lstm': False}
            
            # Prepare training data
            X, y = self.feature_service.prepare_training_data_optimized(features_df)
            if len(X) == 0:
                logger.error(f"No training data prepared for {ticker}")
                return {'random_forest': False, 'lstm': False}
            
            # Train ensemble models
            train_results = self.ensemble_service.train_models(X, y, ticker)
            
            # Update loaded models cache
            for model_type, success in train_results.items():
                if success:
                    self.loaded_models.add(f"{ticker}_{model_type[:2]}")  # rf or ls
            
            logger.info(f"Training results for {ticker}: {train_results}")
            return train_results
            
        except Exception as e:
            logger.error(f"Error training ensemble models for {ticker}: {str(e)}")
            return {'random_forest': False, 'lstm': False}
    
    def _fallback_prediction(self, current_price: float) -> Dict:
        """Fallback prediction when ensemble models are not available"""
        import random
        
        # Simple fallback logic
        change_pct = random.uniform(-2.0, 2.0)  # ±2%
        target_price = current_price * (1 + change_pct / 100)
        direction = "bullish" if change_pct > 0 else "bearish"
        
        logger.info(f"Using fallback prediction: {direction} ({change_pct:+.2f}%)")
        
        return {
            'price_target': round(target_price, 2),
            'predicted_change_pct': round(change_pct, 2),
            'direction': direction,
            'confidence': 0.5,
            'model_type': 'fallback',
            'models_used': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def retrain_model(self, ticker: str) -> bool:
        """Force retrain ensemble models for a ticker"""
        try:
            ticker = ticker.upper()
            logger.info(f"Force retraining ensemble models for {ticker}")
            
            # Remove from loaded models to force retrain
            self.loaded_models.discard(f"{ticker}_rf")
            self.loaded_models.discard(f"{ticker}_lstm")
            
            # Clear feature engineering caches
            self.feature_service.clear_cache()
            
            return await self._train_ensemble_models(ticker)
            
        except Exception as e:
            logger.error(f"Error retraining ensemble models for {ticker}: {str(e)}")
            return False
    
    def get_model_status(self, ticker: str) -> Dict:
        """Get status information for ensemble models"""
        ticker = ticker.upper()
        
        try:
            # Get ensemble status
            ensemble_status = self.ensemble_service.get_models_status()
            
            # Add ticker-specific information
            status = {
                'ticker': ticker,
                'ensemble_status': ensemble_status,
                'models_in_cache': {
                    'random_forest': f"{ticker}_rf" in self.loaded_models,
                    'lstm': f"{ticker}_lstm" in self.loaded_models
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting model status for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_model_comparison(self, ticker: str, current_price: float) -> Optional[Dict]:
        """Get individual model predictions for comparison"""
        try:
            ticker = ticker.upper()
            
            # Get historical data and features (same as prediction logic)
            historical_data = await self.historical_service.get_historical_data(ticker, days=60)
            if historical_data is None:
                return None
            
            features_df = self.feature_service.create_features(historical_data)
            if features_df is None:
                return None
            
            latest_features = self.feature_service.get_latest_features_optimized(features_df)
            if latest_features is None:
                return None
            
            # Get model comparison
            comparison = self.ensemble_service.get_model_comparison(ticker, latest_features)
            
            # Add price targets for each model
            if 'models' in comparison:
                for model_name, model_data in comparison['models'].items():
                    if model_data.get('available') and 'prediction_pct' in model_data:
                        change_pct = model_data['prediction_pct'] / 100.0
                        target_price = current_price * (1 + change_pct)
                        model_data['price_target'] = round(target_price, 2)
                        model_data['current_price'] = current_price
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error in model comparison for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for ensemble predictions"""
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
            'under_200ms': sum(1 for t in times if t < 0.2) / len(times)  # Phase 5 target
        }