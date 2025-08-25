import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
import ta
from functools import lru_cache

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """Optimized service for creating ML features with vectorized operations"""
    
    def __init__(self):
        self.feature_columns = [
            'close_price',
            'volume',
            'price_change_pct',
            'ma_5',
            'ma_20',
            'ma_ratio',
            'rsi',
            'volume_ratio',
            'high_low_ratio'
        ]
        
        # Cache for expensive calculations
        self._indicator_cache = {}
    
    def create_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Create ML features from historical OHLCV data with optimized calculations
        
        Args:
            df: DataFrame with columns: date, open, high, low, close, volume
            
        Returns:
            DataFrame with features and target, or None if failed
        """
        try:
            if len(df) < 50:  # Need minimum data for indicators
                logger.warning(f"Insufficient data for feature creation: {len(df)} rows")
                return None
            
            # Sort by date to ensure correct order (vectorized)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Create features using vectorized operations
            features_df = self._calculate_features_vectorized(df)
            
            if features_df is None:
                return None
            
            # Create target variable (next day price change percentage)
            features_df = self._create_target_vectorized(features_df)
            
            # Remove rows with NaN values (due to indicator calculations)
            initial_len = len(features_df)
            features_df = features_df.dropna()
            
            logger.debug(f"Created features: {len(features_df)} rows from {initial_len} original rows")
            
            if len(features_df) < 30:  # Need minimum for training
                logger.warning(f"Insufficient feature data after cleaning: {len(features_df)} rows")
                return None
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return None
    
    def _calculate_features_vectorized(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate technical indicators using vectorized operations for speed"""
        try:
            # Pre-allocate DataFrame for better performance
            features = pd.DataFrame(index=df.index)
            
            # Extract series once for reuse (avoid repeated column access)
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            volumes = df['volume'].values
            
            # Basic price and volume features (vectorized)
            features['close_price'] = close_prices
            features['volume'] = volumes
            
            # Price change percentage (vectorized)
            features['price_change_pct'] = np.concatenate([[0], np.diff(close_prices) / close_prices[:-1]])
            
            # Moving averages (vectorized with pandas rolling)
            close_series = df['close']
            features['ma_5'] = close_series.rolling(window=5, min_periods=5).mean().values
            features['ma_20'] = close_series.rolling(window=20, min_periods=20).mean().values
            
            # Moving average ratio (vectorized)
            ma_5_values = features['ma_5'].values
            ma_20_values = features['ma_20'].values
            features['ma_ratio'] = np.divide(ma_5_values, ma_20_values, 
                                           out=np.ones_like(ma_5_values), 
                                           where=ma_20_values!=0)
            
            # RSI (optimized with caching)
            rsi_values = self._calculate_rsi_optimized(close_prices)
            features['rsi'] = rsi_values
            
            # Volume ratio (vectorized)
            volume_series = df['volume']
            volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
            features['volume_ratio'] = np.divide(volumes, volume_ma_20,
                                                out=np.ones_like(volumes, dtype=float),
                                                where=volume_ma_20!=0)
            
            # High-Low ratio (vectorized)
            features['high_low_ratio'] = (high_prices - low_prices) / close_prices
            
            # Add date for reference
            features['date'] = df['date'].values
            
            logger.debug(f"Calculated {len(features.columns)} features using vectorized operations")
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features: {str(e)}")
            return None
    
    def _calculate_rsi_optimized(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Optimized RSI calculation with caching"""
        cache_key = f"rsi_{len(prices)}_{hash(prices.tobytes())}"
        
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key]
        
        # Vectorized RSI calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate rolling averages
        avg_gains = pd.Series(gains).rolling(window=window, min_periods=window).mean().values
        avg_losses = pd.Series(losses).rolling(window=window, min_periods=window).mean().values
        
        # Calculate RS and RSI
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Prepend NaN for the first value (no delta available)
        rsi_full = np.concatenate([[np.nan], rsi])
        
        # Cache the result
        if len(self._indicator_cache) < 100:  # Limit cache size
            self._indicator_cache[cache_key] = rsi_full
        
        return rsi_full
    
    def _create_target_vectorized(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable using vectorized operations"""
        try:
            # Target: next day's price change percentage (vectorized)
            close_prices = features_df['close_price'].values
            next_day_changes = np.concatenate([close_prices[1:] / close_prices[:-1] - 1, [np.nan]])
            features_df['target'] = next_day_changes
            
            # Remove the last row (no target available)
            features_df = features_df[:-1].copy()
            
            logger.debug(f"Created target variable with {len(features_df)} samples")
            return features_df
            
        except Exception as e:
            logger.error(f"Error creating target: {str(e)}")
            return features_df
    
    def prepare_training_data_optimized(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for ML model training with optimizations
        
        Returns:
            Tuple of (X, y) where X is features array and y is target array
        """
        try:
            # Select feature columns (exclude date and target) - vectorized
            X = features_df[self.feature_columns].values
            y = features_df['target'].values
            
            # Optimized cleaning: use numpy operations
            # Replace infinite values with zeros
            X = np.where(np.isfinite(X), X, 0)
            y = np.where(np.isfinite(y), y, 0)
            
            # Cap extreme values to prevent model issues (vectorized)
            X = np.clip(X, -10, 10)
            y = np.clip(y, -0.2, 0.2)  # Cap at ±20% daily change
            
            logger.debug(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            return np.array([]), np.array([])
    
    @lru_cache(maxsize=32)
    def get_latest_features_cached(self, features_hash: str, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get the latest feature vector for prediction with caching
        
        Returns:
            Feature vector for the most recent data point
        """
        try:
            if len(features_df) == 0:
                return None
            
            # Get the last row's features (vectorized)
            latest_features = features_df[self.feature_columns].iloc[-1:].values
            
            # Handle infinite or NaN values (vectorized)
            latest_features = np.where(np.isfinite(latest_features), latest_features, 0)
            latest_features = np.clip(latest_features, -10, 10)
            
            logger.debug(f"Extracted latest features: shape {latest_features.shape}")
            return latest_features
            
        except Exception as e:
            logger.error(f"Error getting latest features: {str(e)}")
            return None
    
    def get_latest_features_optimized(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Optimized version without caching for dynamic data"""
        try:
            if len(features_df) == 0:
                return None
            
            # Get the last row's features (vectorized)
            latest_features = features_df[self.feature_columns].iloc[-1:].values
            
            # Handle infinite or NaN values (vectorized)
            latest_features = np.where(np.isfinite(latest_features), latest_features, 0)
            latest_features = np.clip(latest_features, -10, 10)
            
            return latest_features
            
        except Exception as e:
            logger.error(f"Error getting latest features: {str(e)}")
            return None
    
    def get_feature_names(self) -> list:
        """Get list of feature column names"""
        return self.feature_columns.copy()
    
    def clear_cache(self):
        """Clear internal caches"""
        self._indicator_cache.clear()
        self.get_latest_features_cached.cache_clear()
        logger.debug("Feature engineering caches cleared")