import numpy as np
import pandas as pd
import logging
import os
import time
from typing import Optional, Dict, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.get_logger().setLevel('ERROR')  # Reduce TensorFlow logging
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None

logger = logging.getLogger(__name__)


class LSTMModelService:
    """Service for training and using LSTM models for time series prediction"""
    
    def __init__(self):
        self.models_dir = "models"
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.model_metrics = {}
        self.feature_names = []
        self.lookback_window = 10  # Use 10 days of historical data
        
        # LSTM parameters
        self.lstm_params = {
            'units': 50,
            'dropout': 0.2,
            'recurrent_dropout': 0.2,
            'epochs': 50,
            'batch_size': 32,
            'patience': 10
        }
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - LSTM models disabled")
        else:
            logger.info("LSTM Model service initialized")
    
    def is_available(self) -> bool:
        """Check if LSTM models are available"""
        return TENSORFLOW_AVAILABLE
    
    def train_model(self, X: np.ndarray, y: np.ndarray, ticker: str) -> bool:
        """
        Train LSTM model on historical data with sequence preparation
        
        Args:
            X: Feature matrix
            y: Target vector (price change percentages)
            ticker: Stock ticker symbol
            
        Returns:
            True if training successful, False otherwise
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for LSTM training")
            return False
        
        try:
            start_time = time.time()
            
            if len(X) < 50:
                logger.warning(f"Insufficient data for LSTM training {ticker}: {len(X)} samples")
                return False
            
            logger.info(f"Training LSTM model for {ticker} with {len(X)} samples")
            
            # Prepare sequence data
            X_sequences, y_sequences = self._prepare_sequences(X, y)
            
            if len(X_sequences) < 30:
                logger.warning(f"Insufficient sequence data for {ticker}: {len(X_sequences)} sequences")
                return False
            
            # Split data for validation
            split_idx = int(len(X_sequences) * 0.8)
            X_train, X_val = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_val = y_sequences[:split_idx], y_sequences[split_idx:]
            
            # Scale the data
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            
            # Reshape for scaling
            X_train_scaled = self._scale_sequences(X_train, fit_scaler=True)
            X_val_scaled = self._scale_sequences(X_val, fit_scaler=False)
            
            y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            
            # Build LSTM model
            self.model = self._build_lstm_model(X_train_scaled.shape[1], X_train_scaled.shape[2])
            
            # Train model
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.lstm_params['patience'],
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
            
            history = self.model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                epochs=self.lstm_params['epochs'],
                batch_size=self.lstm_params['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Calculate metrics
            train_pred_scaled = self.model.predict(X_train_scaled, verbose=0)
            val_pred_scaled = self.model.predict(X_val_scaled, verbose=0)
            
            # Inverse transform predictions
            train_pred = self.scaler_y.inverse_transform(train_pred_scaled).flatten()
            val_pred = self.scaler_y.inverse_transform(val_pred_scaled).flatten()
            
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
                'training_time': time.time() - start_time,
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1]
            }
            
            # Save model and scalers
            model_path = os.path.join(self.models_dir, f"{ticker}_lstm_model.h5")
            scaler_path = os.path.join(self.models_dir, f"{ticker}_lstm_scalers.joblib")
            
            self.model.save(model_path)
            joblib.dump({
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'metrics': self.model_metrics,
                'feature_names': self.feature_names,
                'lookback_window': self.lookback_window
            }, scaler_path)
            
            logger.info(f"LSTM model training completed for {ticker}:")
            logger.info(f"  Training MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}")
            logger.info(f"  Training RMSE: {train_rmse:.4f}, Validation RMSE: {val_rmse:.4f}")
            logger.info(f"  Training time: {self.model_metrics['training_time']:.2f}s")
            logger.info(f"  Epochs: {self.model_metrics['epochs_trained']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model for {ticker}: {str(e)}")
            return False
    
    def load_model(self, ticker: str) -> bool:
        """Load trained LSTM model from disk"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available for LSTM loading")
            return False
        
        try:
            model_path = os.path.join(self.models_dir, f"{ticker}_lstm_model.h5")
            scaler_path = os.path.join(self.models_dir, f"{ticker}_lstm_scalers.joblib")
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.warning(f"No saved LSTM model found for {ticker}")
                return False
            
            # Check model age
            model_age = time.time() - os.path.getmtime(model_path)
            if model_age > (7 * 24 * 3600):
                logger.info(f"LSTM model for {ticker} is {model_age/86400:.1f} days old")
            
            # Load model and scalers
            self.model = keras.models.load_model(model_path)
            scaler_data = joblib.load(scaler_path)
            
            self.scaler_X = scaler_data['scaler_X']
            self.scaler_y = scaler_data['scaler_y']
            self.model_metrics = scaler_data['metrics']
            self.feature_names = scaler_data.get('feature_names', [])
            self.lookback_window = scaler_data.get('lookback_window', 10)
            
            logger.info(f"Loaded LSTM model for {ticker} (VAL MAE: {self.model_metrics.get('val_mae', 'N/A'):.4f})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading LSTM model for {ticker}: {str(e)}")
            return False
    
    def predict(self, features: np.ndarray) -> Optional[Dict]:
        """Make prediction using trained LSTM model"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            logger.error("No LSTM model available for prediction")
            return None
        
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # For LSTM, we need sequence data, but we only have current features
            # We'll create a sequence by repeating current features with slight variation
            sequence = self._create_sequence_from_current_features(features[0])
            
            # Scale the sequence
            sequence_scaled = self._scale_sequences(sequence.reshape(1, -1, sequence.shape[1]), fit_scaler=False)
            
            # Make prediction
            predicted_change_scaled = self.model.predict(sequence_scaled, verbose=0)
            predicted_change = self.scaler_y.inverse_transform(predicted_change_scaled).flatten()[0]
            
            # Calculate confidence based on model validation performance
            confidence = self._calculate_confidence(predicted_change)
            
            # Determine direction
            direction = "bullish" if predicted_change > 0 else "bearish"
            
            return {
                'predicted_change_pct': float(predicted_change),
                'direction': direction,
                'confidence': float(confidence),
                'model_type': 'lstm'
            }
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {str(e)}")
            return None
    
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequence data for LSTM training"""
        X_sequences = []
        y_sequences = []
        
        for i in range(self.lookback_window, len(X)):
            X_sequences.append(X[i-self.lookback_window:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _scale_sequences(self, sequences: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """Scale sequence data for LSTM"""
        original_shape = sequences.shape
        # Reshape to 2D for scaling
        sequences_2d = sequences.reshape(-1, sequences.shape[-1])
        
        if fit_scaler:
            sequences_scaled_2d = self.scaler_X.fit_transform(sequences_2d)
        else:
            sequences_scaled_2d = self.scaler_X.transform(sequences_2d)
        
        # Reshape back to 3D
        return sequences_scaled_2d.reshape(original_shape)
    
    def _build_lstm_model(self, timesteps: int, features: int) -> keras.Model:
        """Build LSTM model architecture"""
        model = keras.Sequential([
            layers.LSTM(
                self.lstm_params['units'],
                return_sequences=True,
                input_shape=(timesteps, features),
                dropout=self.lstm_params['dropout'],
                recurrent_dropout=self.lstm_params['recurrent_dropout']
            ),
            layers.LSTM(
                self.lstm_params['units'] // 2,
                dropout=self.lstm_params['dropout'],
                recurrent_dropout=self.lstm_params['recurrent_dropout']
            ),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_sequence_from_current_features(self, current_features: np.ndarray) -> np.ndarray:
        """Create a sequence from current features by adding slight historical variation"""
        sequence = []
        
        for i in range(self.lookback_window):
            # Add slight random variation to simulate historical data
            variation = np.random.normal(0, 0.01, size=current_features.shape)
            historical_features = current_features + variation
            sequence.append(historical_features)
        
        return np.array(sequence)
    
    def _calculate_confidence(self, prediction: float) -> float:
        """Calculate prediction confidence based on model performance"""
        try:
            val_mae = self.model_metrics.get('val_mae', 0.02)
            base_confidence = max(0.1, min(0.9, 1.0 - (val_mae * 10)))
            
            # Adjust based on prediction magnitude
            pred_magnitude = abs(prediction)
            if pred_magnitude > 0.05:  # Large predicted changes are less reliable
                magnitude_penalty = min(0.3, pred_magnitude * 2)
                base_confidence *= (1 - magnitude_penalty)
            
            return max(0.1, min(0.9, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating LSTM confidence: {str(e)}")
            return 0.5
    
    def get_model_info(self) -> Dict:
        """Get information about the current LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return {'status': 'tensorflow_not_available'}
        
        if self.model is None:
            return {'status': 'no_model_loaded'}
        
        return {
            'status': 'model_loaded',
            'model_type': 'lstm',
            'metrics': self.model_metrics,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'lookback_window': self.lookback_window,
            'tensorflow_version': tf.__version__ if tf else 'N/A'
        }
    
    def is_model_ready(self) -> bool:
        """Check if LSTM model is ready for predictions"""
        return TENSORFLOW_AVAILABLE and self.model is not None
    
    def set_feature_names(self, feature_names: list):
        """Set feature names for the model"""
        self.feature_names = feature_names