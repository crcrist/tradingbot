import joblib
import numpy as np
import pandas as pd
import logging
import os
from typing import Optional, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import time

logger = logging.getLogger(__name__)


class MLModelService:
    """Service for training and using Random Forest prediction models"""
    
    def __init__(self):
        self.models_dir = "models"
        self.model = None
        self.model_metrics = {}
        self.feature_names = []
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Random Forest parameters (simple but effective)
        self.rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1  # Use all CPU cores
        }
        
        logger.info("ML Model service initialized")
    
    def train_model(self, X: np.ndarray, y: np.ndarray, ticker: str) -> bool:
        """
        Train Random Forest model on historical data
        
        Args:
            X: Feature matrix
            y: Target vector (price change percentages)
            ticker: Stock ticker symbol
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            start_time = time.time()
            
            if len(X) < 30:
                logger.warning(f"Insufficient training data for {ticker}: {len(X)} samples")
                return False
            
            logger.info(f"Training Random Forest model for {ticker} with {len(X)} samples")
            
            # Split data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
            )
            
            # Train Random Forest model
            self.model = RandomForestRegressor(**self.rf_params)
            self.model.fit(X_train, y_train)
            
            # Validate model
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            
            self.model_metrics = {
                'ticker': ticker,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'training_time': time.time() - start_time
            }
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{ticker}_rf_model.joblib")
            joblib.dump({
                'model': self.model,
                'metrics': self.model_metrics,
                'feature_names': self.feature_names
            }, model_path)
            
            logger.info(f"Model training completed for {ticker}:")
            logger.info(f"  Training MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}")
            logger.info(f"  Training RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
            logger.info(f"  Training time: {self.model_metrics['training_time']:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
            return False
    
    def load_model(self, ticker: str) -> bool:
        """
        Load trained model from disk
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            model_path = os.path.join(self.models_dir, f"{ticker}_rf_model.joblib")
            
            if not os.path.exists(model_path):
                logger.warning(f"No saved model found for {ticker}")
                return False
            
            # Check model age (retrain if older than 7 days)
            model_age = time.time() - os.path.getmtime(model_path)
            if model_age > (7 * 24 * 3600):
                logger.info(f"Model for {ticker} is {model_age/86400:.1f} days old, should retrain")
            
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.model_metrics = model_data['metrics']
            self.feature_names = model_data.get('feature_names', [])
            
            logger.info(f"Loaded model for {ticker} (VAL MAE: {self.model_metrics.get('val_mae', 'N/A'):.4f})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {str(e)}")
            return False
    
    def predict(self, features: np.ndarray) -> Optional[Dict]:
        """
        Make prediction using trained model
        
        Args:
            features: Feature vector for prediction
            
        Returns:
            Dictionary with prediction results or None if failed
        """
        try:
            if self.model is None:
                logger.error("No model loaded for prediction")
                return None
            
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            predicted_change = self.model.predict(features)[0]
            
            # Calculate confidence based on feature importances and prediction consistency
            confidence = self._calculate_confidence(features, predicted_change)
            
            # Determine direction
            direction = "bullish" if predicted_change > 0 else "bearish"
            
            return {
                'predicted_change_pct': float(predicted_change),
                'direction': direction,
                'confidence': float(confidence),
                'model_type': 'random_forest'
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None
    
    def _calculate_confidence(self, features: np.ndarray, prediction: float) -> float:
        """Calculate prediction confidence based on model characteristics"""
        try:
            # Base confidence from validation metrics
            val_mae = self.model_metrics.get('val_mae', 0.02)
            base_confidence = max(0.1, min(0.9, 1.0 - (val_mae * 10)))
            
            # Adjust based on prediction magnitude
            pred_magnitude = abs(prediction)
            if pred_magnitude > 0.05:  # Large predicted changes are less reliable
                magnitude_penalty = min(0.3, pred_magnitude * 2)
                base_confidence *= (1 - magnitude_penalty)
            
            # Ensure confidence is in reasonable range
            confidence = max(0.1, min(0.9, base_confidence))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        if self.model is None:
            return {'status': 'no_model_loaded'}
        
        return {
            'status': 'model_loaded',
            'model_type': 'random_forest',
            'metrics': self.model_metrics,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0
        }
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return self.model is not None
    
    def set_feature_names(self, feature_names: list):
        """Set feature names for the model"""
        self.feature_names = feature_names