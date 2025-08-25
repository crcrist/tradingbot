import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.feature_engineering_service import FeatureEngineeringService
from app.services.ml_model_service import MLModelService
from app.services.prediction_service import PredictionService


class TestFeatureEngineeringService:
    """Test cases for FeatureEngineeringService"""
    
    @pytest.fixture
    def feature_service(self):
        return FeatureEngineeringService()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        prices = []
        price = 100.0
        for _ in range(100):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price *= (1 + change)
            prices.append(price)
        
        return pd.DataFrame({
            'date': dates,
            'open': [p * 0.995 for p in prices],
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        })
    
    def test_create_features_success(self, feature_service, sample_data):
        """Test successful feature creation"""
        features = feature_service.create_features(sample_data)
        
        assert features is not None
        assert len(features) > 0
        assert 'target' in features.columns
        
        # Check that all required features are present
        expected_features = feature_service.get_feature_names()
        for feature in expected_features:
            assert feature in features.columns
    
    def test_create_features_insufficient_data(self, feature_service):
        """Test feature creation with insufficient data"""
        small_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000000] * 10
        })
        
        features = feature_service.create_features(small_data)
        assert features is None
    
    def test_prepare_training_data(self, feature_service, sample_data):
        """Test training data preparation"""
        features = feature_service.create_features(sample_data)
        assert features is not None
        
        X, y = feature_service.prepare_training_data(features)
        
        assert len(X) > 0
        assert len(y) > 0
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(feature_service.get_feature_names())
    
    def test_get_latest_features(self, feature_service, sample_data):
        """Test latest features extraction"""
        features = feature_service.create_features(sample_data)
        assert features is not None
        
        latest = feature_service.get_latest_features(features)
        
        assert latest is not None
        assert len(latest.shape) == 2
        assert latest.shape[1] == len(feature_service.get_feature_names())


class TestMLModelService:
    """Test cases for MLModelService"""
    
    @pytest.fixture
    def ml_service(self):
        return MLModelService()
    
    @pytest.fixture
    def training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        X = np.random.randn(100, 9)  # 9 features
        y = np.random.randn(100) * 0.02  # Price changes ±2%
        return X, y
    
    def test_train_model_success(self, ml_service, training_data):
        """Test successful model training"""
        X, y = training_data
        ml_service.set_feature_names(['feature_' + str(i) for i in range(9)])
        
        success = ml_service.train_model(X, y, 'TEST')
        
        assert success is True
        assert ml_service.is_model_ready()
        assert ml_service.model is not None
        assert 'val_mae' in ml_service.model_metrics
    
    def test_train_model_insufficient_data(self, ml_service):
        """Test training with insufficient data"""
        X = np.random.randn(10, 9)  # Too few samples
        y = np.random.randn(10)
        
        success = ml_service.train_model(X, y, 'TEST')
        assert success is False
    
    def test_predict_success(self, ml_service, training_data):
        """Test successful prediction"""
        X, y = training_data
        ml_service.set_feature_names(['feature_' + str(i) for i in range(9)])
        
        # Train model first
        ml_service.train_model(X, y, 'TEST')
        
        # Make prediction
        test_features = np.random.randn(1, 9)
        result = ml_service.predict(test_features)
        
        assert result is not None
        assert 'predicted_change_pct' in result
        assert 'direction' in result
        assert 'confidence' in result
        assert result['direction'] in ['bullish', 'bearish']
        assert 0 <= result['confidence'] <= 1
    
    def test_predict_no_model(self, ml_service):
        """Test prediction without trained model"""
        test_features = np.random.randn(1, 9)
        result = ml_service.predict(test_features)
        
        assert result is None
    
    def test_model_info(self, ml_service, training_data):
        """Test model info retrieval"""
        # No model loaded
        info = ml_service.get_model_info()
        assert info['status'] == 'no_model_loaded'
        
        # Train model
        X, y = training_data
        ml_service.set_feature_names(['feature_' + str(i) for i in range(9)])
        ml_service.train_model(X, y, 'TEST')
        
        # Check info with model
        info = ml_service.get_model_info()
        assert info['status'] == 'model_loaded'
        assert info['model_type'] == 'random_forest'
        assert 'metrics' in info


class TestPredictionService:
    """Test cases for PredictionService"""
    
    @pytest.fixture
    def prediction_service(self):
        return PredictionService()
    
    @pytest.mark.asyncio
    async def test_fallback_prediction(self, prediction_service):
        """Test fallback prediction when ML fails"""
        result = prediction_service._fallback_prediction(100.0)
        
        assert result is not None
        assert 'price_target' in result
        assert 'predicted_change_pct' in result
        assert 'direction' in result
        assert 'confidence' in result
        assert result['model_type'] == 'fallback'
        assert result['direction'] in ['bullish', 'bearish']
    
    @pytest.mark.asyncio
    @patch('app.services.prediction_service.HistoricalDataService')
    @patch('app.services.prediction_service.MLModelService')
    async def test_get_ml_prediction_fallback(self, mock_ml_service, mock_hist_service, prediction_service):
        """Test ML prediction falling back when services fail"""
        # Mock services to fail
        mock_hist_service.return_value.get_historical_data.return_value = None
        
        result = await prediction_service.get_ml_prediction('AAPL', 150.0)
        
        assert result is not None
        assert result['model_type'] == 'fallback'
    
    def test_get_model_status(self, prediction_service):
        """Test model status retrieval"""
        status = prediction_service.get_model_status('AAPL')
        
        assert 'ticker' in status
        assert 'model_loaded' in status
        assert 'service_ready' in status
        assert status['ticker'] == 'AAPL'