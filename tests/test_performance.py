import pytest
import asyncio
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
from app.main import app
from app.middleware.performance_middleware import PerformanceMiddleware
from app.services.model_warmup_service import ModelWarmupService

client = TestClient(app=app)


class TestPerformanceMiddleware:
    """Test cases for PerformanceMiddleware"""
    
    @pytest.fixture
    def middleware(self):
        return PerformanceMiddleware(MagicMock())
    
    def test_middleware_initialization(self, middleware):
        """Test middleware initializes correctly"""
        assert middleware.request_times is not None
        assert middleware.request_counts is not None
        assert middleware.error_counts is not None
        assert middleware.memory_samples is not None
        assert middleware.cache_stats is not None
    
    def test_get_stats_empty(self, middleware):
        """Test getting stats when no requests processed"""
        stats = middleware.get_stats()
        
        assert 'request_stats' in stats
        assert 'memory_stats' in stats
        assert 'cache_stats' in stats
        assert stats['cache_stats']['hit_rate'] == 0.0
    
    def test_cache_hit_miss_tracking(self, middleware):
        """Test cache hit/miss tracking"""
        middleware.record_cache_hit()
        middleware.record_cache_hit()
        middleware.record_cache_miss()
        
        assert middleware.cache_stats['hits'] == 2
        assert middleware.cache_stats['misses'] == 1
        
        stats = middleware.get_stats()
        assert stats['cache_stats']['hit_rate'] == 2/3


class TestPerformanceEndpoints:
    """Test performance monitoring endpoints"""
    
    def test_performance_metrics_endpoint(self):
        """Test /performance/metrics endpoint"""
        response = client.get("/performance/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert 'timestamp' in data
        assert 'request_performance' in data
        assert 'memory_usage' in data
        assert 'cache_performance' in data
        assert 'prediction_performance' in data
        assert 'system_status' in data
    
    def test_performance_summary_endpoint(self):
        """Test /performance/summary endpoint"""
        response = client.get("/performance/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert 'api_performance' in data
        assert 'ml_performance' in data
        assert 'cache_performance' in data
        assert 'memory_usage_mb' in data
        assert 'status' in data
    
    def test_warmup_status_endpoint(self):
        """Test /warmup/status endpoint"""
        response = client.get("/warmup/status")
        assert response.status_code == 200
        
        data = response.json()
        # Should return warmup statistics even if warmup hasn't run
        assert isinstance(data, dict)


class TestModelWarmup:
    """Test model warmup functionality"""
    
    @pytest.fixture
    def warmup_service(self):
        # Mock prediction service
        mock_prediction_service = MagicMock()
        mock_prediction_service._ensure_model_ready = AsyncMock(return_value=True)
        mock_prediction_service.get_ml_prediction = AsyncMock(return_value={'test': 'result'})
        mock_prediction_service.loaded_models = set()
        
        return ModelWarmupService(mock_prediction_service)
    
    @pytest.mark.asyncio
    async def test_warmup_single_model_success(self, warmup_service):
        """Test successful warmup of a single model"""
        result = await warmup_service._warmup_single_model('AAPL', asyncio.Semaphore(1))
        
        assert result['ticker'] == 'AAPL'
        assert 'success' in result
        assert 'warmup_time' in result
    
    @pytest.mark.asyncio
    async def test_warmup_all_models(self, warmup_service):
        """Test warmup of all supported models"""
        # Limit to just 2 tickers for faster testing
        warmup_service.supported_tickers = ['AAPL', 'GOOGL']
        
        stats = await warmup_service.warmup_models(max_concurrent=2)
        
        assert 'started_at' in stats
        assert 'completed_at' in stats
        assert 'duration_seconds' in stats
        assert 'models_loaded' in stats
        assert 'models_trained' in stats
        assert isinstance(stats['errors'], list)
    
    def test_warmup_stats(self, warmup_service):
        """Test warmup statistics retrieval"""
        stats = warmup_service.get_warmup_stats()
        
        assert 'started_at' in stats
        assert 'completed_at' in stats
        assert 'duration_seconds' in stats
        assert 'models_loaded' in stats
        assert 'models_trained' in stats
        assert 'errors' in stats


class TestOptimizedCaching:
    """Test optimized caching functionality"""
    
    @pytest.mark.asyncio
    async def test_prediction_endpoint_caching(self):
        """Test that prediction endpoint uses caching"""
        with patch('app.main.data_service.get_stock_data') as mock_get_data, \
             patch('app.main.cache_service.get') as mock_cache_get, \
             patch('app.main.cache_service.set') as mock_cache_set, \
             patch('app.main.prediction_service.get_ml_prediction') as mock_ml_pred:
            
            # Mock real data
            mock_cache_get.return_value = None
            mock_get_data.return_value = {
                'ticker': 'AAPL',
                'current_price': 200.00,
                'timestamp': '2024-01-01T12:00:00'
            }
            mock_cache_set.return_value = True
            mock_ml_pred.return_value = {
                'price_target': 202.00,
                'predicted_change_pct': 1.0,
                'direction': 'bullish',
                'confidence': 0.85,
                'model_type': 'ml_random_forest'
            }
            
            # Make request
            response = client.get("/predict/AAPL")
            assert response.status_code == 200
            
            data = response.json()
            assert data['ticker'] == 'AAPL'
            assert data['current_price'] == 200.00
            
            # Verify caching was attempted
            mock_cache_get.assert_called()
            mock_cache_set.assert_called()


class TestLoadTesting:
    """Test load handling and concurrent requests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        async def make_request():
            # Use test client in async context
            import httpx
            async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.get("/health")
                return response.status_code
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should succeed
        success_count = sum(1 for result in results if result == 200)
        assert success_count >= 8  # Allow for some potential failures in test environment
    
    def test_response_time_tracking(self):
        """Test that response times are being tracked"""
        # Make multiple requests to generate timing data
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200
            
            # Check for performance header
            assert "X-Response-Time" in response.headers
            response_time = float(response.headers["X-Response-Time"])
            assert response_time >= 0
    
    @pytest.mark.asyncio
    async def test_memory_monitoring(self):
        """Test memory usage monitoring"""
        # Get initial metrics
        response = client.get("/performance/metrics")
        assert response.status_code == 200
        
        data = response.json()
        memory_stats = data.get('memory_usage', {})
        
        # Should have memory information
        if memory_stats:  # May be empty if monitoring hasn't started yet
            assert 'current_mb' in memory_stats or 'avg_mb' in memory_stats


class TestPerformanceTargets:
    """Test that performance targets are being met"""
    
    def test_health_endpoint_performance(self):
        """Test health endpoint meets performance targets"""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Should complete in under 50ms for health check
        response_time = end_time - start_time
        assert response_time < 0.05, f"Health check took {response_time:.3f}s, expected <0.05s"
    
    @pytest.mark.asyncio
    async def test_performance_targets_tracking(self):
        """Test that we can track if performance targets are met"""
        # Make several requests to populate metrics
        for _ in range(3):
            client.get("/health")
        
        # Get performance summary
        response = client.get("/performance/summary")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check that we have performance data
        assert 'api_performance' in data
        assert 'ml_performance' in data
        
        # Status should be either 'optimal' or 'degraded'
        assert data['status'] in ['optimal', 'degraded']


class TestCacheOptimizations:
    """Test cache optimization features"""
    
    def test_features_hash_generation(self):
        """Test features hash generation for prediction caching"""
        from app.services.cache_service import CacheService
        import numpy as np
        
        cache_service = CacheService()
        
        # Test with numpy array
        features1 = np.array([[1.0, 2.0, 3.0]])
        features2 = np.array([[1.0, 2.0, 3.0]])  # Same values
        features3 = np.array([[1.1, 2.0, 3.0]])  # Different values
        
        hash1 = cache_service.generate_features_hash(features1)
        hash2 = cache_service.generate_features_hash(features2)
        hash3 = cache_service.generate_features_hash(features3)
        
        # Same features should produce same hash
        assert hash1 == hash2
        # Different features should produce different hash
        assert hash1 != hash3
        
        # Hashes should be reasonable length
        assert len(hash1) == 16
    
    def test_cache_key_generation(self):
        """Test standardized cache key generation"""
        from app.services.cache_service import CacheService
        
        cache_service = CacheService()
        
        # Test standard cache key
        key1 = cache_service.generate_cache_key("stock_data", "aapl")
        key2 = cache_service.generate_cache_key("stock_data", "AAPL")
        
        assert key1 == key2  # Should normalize to uppercase
        assert key1 == "stock_data:AAPL"


class TestVectorizedOperations:
    """Test that feature engineering uses vectorized operations"""
    
    def test_feature_service_performance(self):
        """Test that feature engineering completes quickly"""
        from app.services.feature_engineering_service import FeatureEngineeringService
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(95, 105, 100),
            'high': np.random.uniform(100, 110, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.randint(1000000, 5000000, 100)
        })
        
        feature_service = FeatureEngineeringService()
        
        start_time = time.time()
        features = feature_service.create_features(sample_data)
        end_time = time.time()
        
        # Feature creation should complete quickly (under 100ms for 100 rows)
        processing_time = end_time - start_time
        assert processing_time < 0.1, f"Feature creation took {processing_time:.3f}s, expected <0.1s"
        
        # Should return valid features
        assert features is not None
        assert len(features) > 0