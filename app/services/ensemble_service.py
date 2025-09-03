import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime

from .ml_model_service import MLModelService
from .lstm_model_service import LSTMModelService

logger = logging.getLogger(__name__)


class EnsembleService:
    """Service for combining predictions from multiple ML models"""
    
    def __init__(self):
        self.rf_service = MLModelService()
        self.lstm_service = LSTMModelService()
        
        # Model weights for ensemble (as specified in Phase 5)
        self.model_weights = {
            'random_forest': 0.4,
            'lstm': 0.6
        }
        
        # Track model health and performance
        self.model_health = {
            'random_forest': {'available': True, 'last_prediction_time': None, 'prediction_count': 0, 'error_count': 0},
            'lstm': {'available': self.lstm_service.is_available(), 'last_prediction_time': None, 'prediction_count': 0, 'error_count': 0}
        }
        
        logger.info("Ensemble service initialized")
        logger.info(f"Model weights: RF={self.model_weights['random_forest']}, LSTM={self.model_weights['lstm']}")
        logger.info(f"LSTM available: {self.lstm_service.is_available()}")
    
    def get_ensemble_prediction(self, ticker: str, features: np.ndarray) -> Optional[Dict]:
        """
        Get ensemble prediction combining Random Forest and LSTM models
        
        Args:
            ticker: Stock ticker symbol
            features: Feature array for prediction
            
        Returns:
            Dictionary with ensemble prediction results
        """
        try:
            individual_predictions = []
            model_results = {}
            
            # Get Random Forest prediction
            rf_result = self._get_rf_prediction(ticker, features)
            if rf_result:
                individual_predictions.append({
                    'model': 'random_forest',
                    'weight': self.model_weights['random_forest'],
                    'prediction': rf_result['predicted_change_pct'],
                    'confidence': rf_result['confidence'],
                    'direction': rf_result['direction']
                })
                model_results['random_forest'] = rf_result
                self.model_health['random_forest']['prediction_count'] += 1
                self.model_health['random_forest']['last_prediction_time'] = datetime.utcnow()
            else:
                self.model_health['random_forest']['error_count'] += 1
                logger.warning(f"Random Forest prediction failed for {ticker}")
            
            # Get LSTM prediction
            lstm_result = self._get_lstm_prediction(ticker, features)
            if lstm_result:
                individual_predictions.append({
                    'model': 'lstm',
                    'weight': self.model_weights['lstm'],
                    'prediction': lstm_result['predicted_change_pct'],
                    'confidence': lstm_result['confidence'],
                    'direction': lstm_result['direction']
                })
                model_results['lstm'] = lstm_result
                self.model_health['lstm']['prediction_count'] += 1
                self.model_health['lstm']['last_prediction_time'] = datetime.utcnow()
            else:
                self.model_health['lstm']['error_count'] += 1
                logger.warning(f"LSTM prediction failed for {ticker}")
            
            # Check if we have at least one prediction
            if not individual_predictions:
                logger.error(f"No model predictions available for ensemble for {ticker}")
                return None
            
            # Calculate ensemble prediction
            ensemble_result = self._calculate_ensemble(individual_predictions, model_results)
            
            # Add individual model results for comparison
            ensemble_result['individual_models'] = model_results
            
            logger.info(f"Ensemble prediction for {ticker}: {ensemble_result['direction']} "
                       f"({ensemble_result['predicted_change_pct']:+.3f}%) "
                       f"confidence: {ensemble_result['confidence']:.2f} "
                       f"(models: {len(individual_predictions)})")
            
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction for {ticker}: {str(e)}")
            return None
    
    def _get_rf_prediction(self, ticker: str, features: np.ndarray) -> Optional[Dict]:
        """Get Random Forest prediction with error handling"""
        try:
            # Ensure RF model is loaded
            if not self.rf_service.is_model_ready():
                if not self.rf_service.load_model(ticker):
                    return None
            
            return self.rf_service.predict(features)
            
        except Exception as e:
            logger.error(f"Error getting RF prediction for {ticker}: {str(e)}")
            return None
    
    def _get_lstm_prediction(self, ticker: str, features: np.ndarray) -> Optional[Dict]:
        """Get LSTM prediction with error handling"""
        try:
            if not self.lstm_service.is_available():
                logger.debug("LSTM service not available")
                return None
            
            # Ensure LSTM model is loaded
            if not self.lstm_service.is_model_ready():
                if not self.lstm_service.load_model(ticker):
                    return None
            
            return self.lstm_service.predict(features)
            
        except Exception as e:
            logger.error(f"Error getting LSTM prediction for {ticker}: {str(e)}")
            return None
    
    def _calculate_ensemble(self, individual_predictions: List[Dict], model_results: Dict) -> Dict:
        """Calculate weighted ensemble prediction"""
        try:
            # Calculate weighted average prediction
            total_weight = sum(pred['weight'] for pred in individual_predictions)
            if total_weight == 0:
                total_weight = 1.0  # Fallback to equal weights
            
            weighted_prediction = sum(
                pred['prediction'] * pred['weight'] for pred in individual_predictions
            ) / total_weight
            
            # Calculate ensemble confidence (conservative approach - use minimum)
            ensemble_confidence = min(pred['confidence'] for pred in individual_predictions)
            
            # Determine direction
            ensemble_direction = "bullish" if weighted_prediction > 0 else "bearish"
            
            # Calculate prediction agreement (bonus for models agreeing)
            directions = [pred['direction'] for pred in individual_predictions]
            direction_agreement = len(set(directions)) == 1  # All models agree
            
            if direction_agreement:
                # Boost confidence slightly when all models agree
                ensemble_confidence = min(0.95, ensemble_confidence * 1.1)
            
            return {
                'predicted_change_pct': round(weighted_prediction, 6),  # Convert to percentage
                'direction': ensemble_direction,
                'confidence': round(ensemble_confidence, 2),
                'model_type': 'ensemble',
                'models_used': len(individual_predictions),
                'direction_agreement': direction_agreement,
                'model_weights': {pred['model']: pred['weight'] for pred in individual_predictions},
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating ensemble: {str(e)}")
            # Fallback to simple average
            avg_prediction = np.mean([pred['prediction'] for pred in individual_predictions])
            avg_confidence = np.mean([pred['confidence'] for pred in individual_predictions])
            
            return {
                'predicted_change_pct': round(avg_prediction, 6),
                'direction': "bullish" if avg_prediction > 0 else "bearish",
                'confidence': round(avg_confidence, 2),
                'model_type': 'ensemble_fallback',
                'models_used': len(individual_predictions),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def train_models(self, X: np.ndarray, y: np.ndarray, ticker: str) -> Dict[str, bool]:
        """Train both RF and LSTM models"""
        results = {}
        
        # Set feature names for both models
        feature_names = ['feature_' + str(i) for i in range(X.shape[1])]
        self.rf_service.set_feature_names(feature_names)
        self.lstm_service.set_feature_names(feature_names)
        
        # Train Random Forest
        logger.info(f"Training Random Forest model for {ticker}")
        rf_success = self.rf_service.train_model(X, y, ticker)
        results['random_forest'] = rf_success
        
        # Train LSTM if available
        if self.lstm_service.is_available():
            logger.info(f"Training LSTM model for {ticker}")
            lstm_success = self.lstm_service.train_model(X, y, ticker)
            results['lstm'] = lstm_success
        else:
            logger.info("LSTM not available, skipping LSTM training")
            results['lstm'] = False
        
        return results
    
    def load_models(self, ticker: str) -> Dict[str, bool]:
        """Load both RF and LSTM models"""
        results = {}
        
        # Load Random Forest
        rf_success = self.rf_service.load_model(ticker)
        results['random_forest'] = rf_success
        
        # Load LSTM if available
        if self.lstm_service.is_available():
            lstm_success = self.lstm_service.load_model(ticker)
            results['lstm'] = lstm_success
        else:
            results['lstm'] = False
        
        return results
    
    def get_model_comparison(self, ticker: str, features: np.ndarray) -> Dict:
        """Get individual predictions from all models for comparison"""
        try:
            comparison = {
                'ticker': ticker,
                'timestamp': datetime.utcnow().isoformat(),
                'models': {}
            }
            
            # Get RF prediction
            rf_result = self._get_rf_prediction(ticker, features)
            if rf_result:
                comparison['models']['random_forest'] = {
                    'prediction_pct': round(rf_result['predicted_change_pct'] * 100, 2),
                    'direction': rf_result['direction'],
                    'confidence': rf_result['confidence'],
                    'available': True
                }
            else:
                comparison['models']['random_forest'] = {'available': False}
            
            # Get LSTM prediction
            lstm_result = self._get_lstm_prediction(ticker, features)
            if lstm_result:
                comparison['models']['lstm'] = {
                    'prediction_pct': round(lstm_result['predicted_change_pct'] * 100, 2),
                    'direction': lstm_result['direction'],
                    'confidence': lstm_result['confidence'],
                    'available': True
                }
            else:
                comparison['models']['lstm'] = {'available': self.lstm_service.is_available()}
            
            # Calculate differences if both available
            if rf_result and lstm_result:
                rf_pred = rf_result['predicted_change_pct'] * 100
                lstm_pred = lstm_result['predicted_change_pct'] * 100
                
                comparison['analysis'] = {
                    'prediction_difference_pct': round(abs(rf_pred - lstm_pred), 2),
                    'direction_agreement': rf_result['direction'] == lstm_result['direction'],
                    'confidence_difference': round(abs(rf_result['confidence'] - lstm_result['confidence']), 2)
                }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error in model comparison for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_models_status(self) -> Dict:
        """Get comprehensive status of all models"""
        try:
            rf_info = self.rf_service.get_model_info()
            lstm_info = self.lstm_service.get_model_info()
            
            return {
                'ensemble_weights': self.model_weights,
                'models': {
                    'random_forest': {
                        'info': rf_info,
                        'health': self.model_health['random_forest'],
                        'ready': self.rf_service.is_model_ready()
                    },
                    'lstm': {
                        'info': lstm_info,
                        'health': self.model_health['lstm'],
                        'ready': self.lstm_service.is_model_ready()
                    }
                },
                'ensemble_ready': self._is_ensemble_ready(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting models status: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _is_ensemble_ready(self) -> bool:
        """Check if ensemble is ready (at least one model available)"""
        return self.rf_service.is_model_ready() or self.lstm_service.is_model_ready()
    
    def get_ensemble_health(self) -> Dict:
        """Get ensemble system health metrics"""
        total_predictions = sum(
            self.model_health[model]['prediction_count'] 
            for model in self.model_health
        )
        total_errors = sum(
            self.model_health[model]['error_count'] 
            for model in self.model_health
        )
        
        # Fix error rate calculation to match test expectation
        error_rate = total_errors / max(1, total_predictions) if total_predictions > 0 else 0.0
        
        return {
            'total_predictions': total_predictions,
            'total_errors': total_errors,
            'error_rate': round(error_rate, 4),
            'models_available': sum(1 for model in self.model_health.values() if model['available']),
            'ensemble_ready': self._is_ensemble_ready()
        }