import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.ensemble_service import EnsembleService
from app.services.lstm_model_service import LSTMModelService


class TestLSTMModelService:
    """Test cases for LSTM Model Service"""
    
    @pytest.fixture
    def lstm_service(self):
        return LSTMModelService()
    
    def test_lstm_availability(self, lstm_service):
        """Test LSTM availability check"""
        # This will depend on whether TensorFlow is installed
        availability = lstm_service.is_available()
        assert isinstance(availability, bool)
    
    def test_model_info_no_model(self, lstm_service):
        """Test model info when no model is loaded"""
        info = lstm_service.get_model_info()
        
        if lstm_service.is_available():
            assert info['status'] == 'no_model_loaded'
            assert info['model_type'] == 'lstm'
        else:
            assert info['status'] == 'tensorflow_not_available'
    
    @pytest.mark.skipif(not LSTMModelService().is_available(), reason="TensorFlow not available")
    def test_lstm_training_insufficient_data(self, lstm_service):
        """Test LSTM training with insufficient data"""
        # Very small dataset
        X = np.random.randn(10, 9)
        y = np.random.randn(10) * 0.02
        
        success = lstm_service.train_model(X, y, 'TEST')
        assert success is False
    
    def test_sequence_preparation(self, lstm_service):
        """Test sequence preparation for LSTM"""
        X = np.random.randn(50, 9)
        y = np.random.randn(50) * 0.02
        
        X_seq, y_seq = lstm_service._prepare_sequences(X, y)
        
        expected_length = len(X) - lstm_service.lookback_window
        assert len(X_seq) == expected_length
        assert len(y_seq) == expected_length
        assert X_seq.shape[1] == lstm_service.lookback_window
        assert X_seq.shape[2] == X.shape[1]
    
    def test_feature_sequence_creation(self, lstm_service):
        """Test creating sequence from current features"""
        current_features = np.random.randn(9)
        sequence = lstm_service._create_sequence_from_current_features(current_features)
        
        assert sequence.shape[0] == lstm_service.lookback_window
        assert sequence.shape[1] == len(current_features)


class TestEnsembleService:
    """Test cases for Ensemble Service"""
    
    @pytest.fixture
    def ensemble_service(self):
        return EnsembleService()
    
    def test_initialization(self, ensemble_service):
        """Test ensemble service initialization"""
        assert ensemble_service.model_weights['random_forest'] == 0.4
        assert ensemble_service.model_weights['lstm'] == 0.6
        assert 'random_forest' in ensemble_service.model_health
        assert 'lstm' in ensemble_service.model_health
    
    def test_model_weights_sum(self, ensemble_service):
        """Test that model weights sum to 1.0"""
        total_weight = sum(ensemble_service.model_weights.values())
        assert abs(total_weight - 1.0) < 0.001
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction_no_models(self, ensemble_service):
        """Test ensemble prediction when no models are available"""
        # Mock both models to fail
        with patch.object(ensemble_service, '_get_rf_prediction', return_value=None), \
             patch.object(ensemble_service, '_get_lstm_prediction', return_value=None):
            
            features = np.random.randn(1, 9)
            result = ensemble_service.get_ensemble_prediction('TEST', features)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction_rf_only(self, ensemble_service):
        """Test ensemble prediction with only Random Forest available"""
        # Mock RF to succeed, LSTM to fail
        rf_result = {
            'predicted_change_pct': 0.02,
            'confidence': 0.8,
            'direction': 'bullish'
        }
        
        with patch.object(ensemble_service, '_get_rf_prediction', return_value=rf_result), \
             patch.object(ensemble_service, '_get_lstm_prediction', return_value=None):
            
            features = np.random.randn(1, 9)
            result = ensemble_service.get_ensemble_prediction('TEST', features)
            
            assert result is not None
            assert result['models_used'] == 1
            assert result['direction'] == 'bullish'
            assert 'random_forest' in result['individual_models']
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction_both_models(self, ensemble_service):
        """Test ensemble prediction with both models available"""
        rf_result = {
            'predicted_change_pct': 0.01,
            'confidence': 0.7,
            'direction': 'bullish'
        }
        
        lstm_result = {
            'predicted_change_pct': 0.03,
            'confidence': 0.8,
            'direction': 'bullish'
        }
        
        with patch.object(ensemble_service, '_get_rf_prediction', return_value=rf_result), \
             patch.object(ensemble_service, '_get_lstm_prediction', return_value=lstm_result):
            
            features = np.random.randn(1, 9)
            result = ensemble_service.get_ensemble_prediction('TEST', features)
            
            assert result is not None
            assert result['models_used'] == 2
            assert result['direction'] == 'bullish'
            assert result['direction_agreement'] is True
            
            # Check weighted average: 0.01 * 0.4 + 0.03 * 0.6 = 0.022
            expected_weighted = (0.01 * 0.4 + 0.03 * 0.6) * 100
            assert abs(result['predicted_change_pct'] - expected_weighted) < 0.1
    
    @pytest.mark.asyncio
    async def test_ensemble_prediction_disagreement(self, ensemble_service):
        """Test ensemble prediction when models disagree on direction"""
        rf_result = {
            'predicted_change_pct': 0.01,
            'confidence': 0.7,
            'direction': 'bullish'
        }
        
        lstm_result = {
            'predicted_change_pct': -0.02,
            'confidence': 0.8,
            'direction': 'bearish'
        }
        
        with patch.object(ensemble_service, '_get_rf_prediction', return_value=rf_result), \
             patch.object(ensemble_service, '_get_lstm_prediction', return_value=lstm_result):
            
            features = np.random.randn(1, 9)
            result = ensemble_service.get_ensemble_prediction('TEST', features)
            
            assert result is not None
            assert result['models_used'] == 2
            assert result['direction_agreement'] is False
            
            # Weighted average: 0.01 * 0.4 + (-0.02) * 0.6 = -0.008 (bearish)
            assert result['direction'] == 'bearish'
    
    def test_calculate_ensemble_fallback(self, ensemble_service):
        """Test ensemble calculation fallback logic"""
        # Create predictions with zero weights to trigger fallback
        individual_predictions = [
            {'model': 'rf', 'weight': 0, 'prediction': 0.01, 'confidence': 0.7, 'direction': 'bullish'},
            {'model': 'lstm', 'weight': 0, 'prediction': 0.02, 'confidence': 0.8, 'direction': 'bullish'}
        ]
        
        result = ensemble_service._calculate_ensemble(individual_predictions, {})
        
        assert result is not None
        assert 'predicted_change_pct' in result
        assert 'confidence' in result
        assert 'direction' in result
    
    def test_model_comparison(self, ensemble_service):
        """Test model comparison functionality"""
        rf_result = {
            'predicted_change_pct': 0.015,
            'confidence': 0.75,
            'direction': 'bullish'
        }
        
        lstm_result = {
            'predicted_change_pct': 0.025,
            'confidence': 0.80,
            'direction': 'bullish'
        }
        
        with patch.object(ensemble_service, '_get_rf_prediction', return_value=rf_result), \
             patch.object(ensemble_service, '_get_lstm_prediction', return_value=lstm_result):
            
            features = np.random.randn(1, 9)
            comparison = ensemble_service.get_model_comparison('TEST', features)
            
            assert comparison is not None
            assert 'models' in comparison
            assert 'random_forest' in comparison['models']
            assert 'lstm' in comparison['models']
            assert 'analysis' in comparison
            assert comparison['analysis']['direction_agreement'] is True
    
    def test_models_status(self, ensemble_service):
        """Test getting models status"""
        with patch.object(ensemble_service.rf_service, 'get_model_info') as mock_rf_info, \
             patch.object(ensemble_service.lstm_service, 'get_model_info') as mock_lstm_info:
            
            mock_rf_info.return_value = {'status': 'model_loaded', 'model_type': 'random_forest'}
            mock_lstm_info.return_value = {'status': 'no_model_loaded', 'model_type': 'lstm'}
            
            status = ensemble_service.get_models_status()
            
            assert status is not None
            assert 'ensemble_weights' in status
            assert 'models' in status
            assert 'random_forest' in status['models']
            assert 'lstm' in status['models']
            assert 'ensemble_ready' in status
    
    def test_ensemble_health(self, ensemble_service):
        """Test ensemble health metrics"""
        # Simulate some predictions and errors
        ensemble_service.model_health['random_forest']['prediction_count'] = 10
        ensemble_service.model_health['random_forest']['error_count'] = 1
        ensemble_service.model_health['lstm']['prediction_count'] = 8
        ensemble_service.model_health['lstm']['error_count'] = 2
        
        health = ensemble_service.get_ensemble_health()
        
        assert health['total_predictions'] == 18
        assert health['total_errors'] == 3
        assert health['error_rate'] == 3 / (18 + 3)
        assert 'models_available' in health
        assert 'ensemble_ready' in health


class TestEnsembleIntegration:
    """Integration tests for ensemble system"""
    
    @pytest.mark.asyncio
    async def test_ensemble_training_integration(self):
        """Test ensemble training with mock data"""
        ensemble_service = EnsembleService()
        
        # Create mock training data
        X = np.random.randn(100, 9)
        y = np.random.randn(100) * 0.02
        
        with patch.object(ensemble_service.rf_service, 'train_model', return_value=True) as mock_rf_train, \
             patch.object(ensemble_service.lstm_service, 'train_model', return_value=True) as mock_lstm_train, \
             patch.object(ensemble_service.lstm_service, 'is_available', return_value=True):
            
            results = ensemble_service.train_models(X, y, 'TEST')
            
            assert results['random_forest'] is True
            assert results['lstm'] is True
            mock_rf_train.assert_called_once()
            mock_lstm_train.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensemble_loading_integration(self):
        """Test ensemble model loading"""
        ensemble_service = EnsembleService()
        
        with patch.object(ensemble_service.rf_service, 'load_model', return_value=True) as mock_rf_load, \
             patch.object(ensemble_service.lstm_service, 'load_model', return_value=True) as mock_lstm_load, \
             patch.object(ensemble_service.lstm_service, 'is_available', return_value=True):
            
            results = ensemble_service.load_models('TEST')
            
            assert results['random_forest'] is True
            assert results['lstm'] is True
            mock_rf_load.assert_called_once_with('TEST')
            mock_lstm_load.assert_called_once_with('TEST')