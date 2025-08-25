import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.data_service import DataService
from app.services.cache_service import CacheService


class TestDataService:
    """Test cases for DataService"""
    
    @pytest.fixture
    def data_service(self):
        return DataService()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_get_stock_data_success(self, mock_get, data_service):
        """Test successful stock data fetching with Alpha Vantage"""
        # Mock Alpha Vantage API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "05. price": "150.0",
                "08. previous close": "149.0",
                "06. volume": "1000000",
                "03. high": "151.0",
                "04. low": "148.0",
                "02. open": "149.5",
                "10. change percent": "0.67%"
            }
        }
        
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Set API key for test
        data_service.api_key = "test_key"
        
        result = await data_service.get_stock_data('AAPL')
        
        assert result is not None
        assert result['ticker'] == 'AAPL'
        assert result['current_price'] == 150.0
        assert result['previous_close'] == 149.0
        assert result['data_source'] == 'alpha_vantage'
    
    @pytest.mark.asyncio
    async def test_get_stock_data_failure(self, data_service):
        """Test stock data fetching failure with no API key"""
        # Test with no API key
        data_service.api_key = None
        
        result = await data_service.get_stock_data('AAPL')
        
        assert result is None
    
    def test_validate_stock_data_valid(self, data_service):
        """Test stock data validation with valid data"""
        valid_data = {
            'ticker': 'AAPL',
            'current_price': 150.0,
            'timestamp': '2024-01-01T12:00:00'
        }
        
        assert data_service._validate_stock_data(valid_data) is True
    
    def test_validate_stock_data_invalid_price(self, data_service):
        """Test stock data validation with invalid price"""
        invalid_data = {
            'ticker': 'AAPL',
            'current_price': -10.0,  # Invalid negative price
            'timestamp': '2024-01-01T12:00:00'
        }
        
        assert data_service._validate_stock_data(invalid_data) is False
    
    def test_validate_stock_data_missing_fields(self, data_service):
        """Test stock data validation with missing fields"""
        incomplete_data = {
            'ticker': 'AAPL',
            # Missing current_price and timestamp
        }
        
        assert data_service._validate_stock_data(incomplete_data) is False


class TestCacheService:
    """Test cases for CacheService"""
    
    @pytest.fixture
    def cache_service(self):
        return CacheService("redis://localhost:6379")
    
    @pytest.mark.asyncio
    async def test_connect_success(self, cache_service):
        """Test successful Redis connection"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client
            mock_client.ping.return_value = True
            
            await cache_service.connect()
            
            assert cache_service.redis_client is not None
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self, cache_service):
        """Test Redis connection failure"""
        with patch('redis.asyncio.from_url') as mock_redis:
            mock_redis.side_effect = Exception("Connection failed")
            
            await cache_service.connect()
            
            assert cache_service.redis_client is None
    
    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_service):
        """Test cache get with hit"""
        mock_client = AsyncMock()
        cache_service.redis_client = mock_client
        mock_client.get.return_value = '{"ticker": "AAPL", "price": 150.0}'
        
        result = await cache_service.get("test_key")
        
        assert result is not None
        assert result["ticker"] == "AAPL"
        assert result["price"] == 150.0
    
    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_service):
        """Test cache get with miss"""
        mock_client = AsyncMock()
        cache_service.redis_client = mock_client
        mock_client.get.return_value = None
        
        result = await cache_service.get("test_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_success(self, cache_service):
        """Test cache set success"""
        mock_client = AsyncMock()
        cache_service.redis_client = mock_client
        mock_client.setex.return_value = True
        
        test_data = {"ticker": "AAPL", "price": 150.0}
        result = await cache_service.set("test_key", test_data, 300)
        
        assert result is True
        mock_client.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_set_no_client(self, cache_service):
        """Test cache set when Redis client unavailable"""
        cache_service.redis_client = None
        
        result = await cache_service.set("test_key", {"data": "test"})
        
        assert result is False
    
    def test_generate_cache_key(self, cache_service):
        """Test cache key generation"""
        key = cache_service.generate_cache_key("stock_data", "aapl")
        assert key == "stock_data:AAPL"
        
        key = cache_service.generate_cache_key("prediction", "MSFT")
        assert key == "prediction:MSFT"