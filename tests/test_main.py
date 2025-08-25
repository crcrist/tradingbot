import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app=app)


def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch('app.main.data_service.get_stock_data')
@patch('app.main.cache_service.get')
@patch('app.main.cache_service.set')
def test_predict_with_real_data(mock_cache_set, mock_cache_get, mock_get_stock_data):
    """Test prediction endpoint with mocked real data"""
    # Mock cache miss
    mock_cache_get.return_value = None
    
    # Mock real stock data
    mock_stock_data = {
        'ticker': 'AAPL',
        'current_price': 175.50,
        'previous_close': 174.00,
        'timestamp': '2024-01-01T12:00:00'
    }
    mock_get_stock_data.return_value = mock_stock_data
    mock_cache_set.return_value = True
    
    response = client.get("/predict/AAPL")
    assert response.status_code == 200
    
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["current_price"] == 175.50
    assert "prediction" in data
    assert "price_target" in data["prediction"]
    assert "confidence" in data["prediction"]
    assert "direction" in data["prediction"]
    assert data["prediction"]["direction"] in ["bullish", "bearish"]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0
    assert "timestamp" in data
    
    # Verify that data service was called
    mock_get_stock_data.assert_called_once_with('AAPL')


@patch('app.main.cache_service.get')
def test_predict_with_cached_data(mock_cache_get):
    """Test prediction endpoint with cached data"""
    # Mock cache hit
    cached_data = {
        'ticker': 'AAPL',
        'current_price': 180.25,
        'timestamp': '2024-01-01T11:55:00'
    }
    mock_cache_get.return_value = cached_data
    
    response = client.get("/predict/AAPL")
    assert response.status_code == 200
    
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["current_price"] == 180.25
    assert "prediction" in data
    
    # Verify cache was checked
    mock_cache_get.assert_called_once()


@patch('app.main.data_service.get_stock_data')
@patch('app.main.cache_service.get')
def test_predict_with_fallback_data(mock_cache_get, mock_get_stock_data):
    """Test prediction endpoint falls back when real data fails"""
    # Mock cache miss and data service failure
    mock_cache_get.return_value = None
    mock_get_stock_data.return_value = None
    
    response = client.get("/predict/AAPL")
    assert response.status_code == 200
    
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["current_price"] == 185.00  # Updated fallback price for Alpha Vantage version
    assert "prediction" in data


def test_predict_invalid_ticker():
    """Test prediction endpoint with invalid ticker"""
    response = client.get("/predict/INVALID")
    assert response.status_code == 400
    
    data = response.json()
    assert "error" in data
    assert "INVALID" in data["error"]


def test_predict_case_insensitive():
    """Test that ticker symbols are case insensitive"""
    with patch('app.main.data_service.get_stock_data') as mock_get_data, \
         patch('app.main.cache_service.get') as mock_cache_get, \
         patch('app.main.cache_service.set') as mock_cache_set:
        
        mock_cache_get.return_value = None
        mock_get_data.return_value = {
            'ticker': 'AAPL',
            'current_price': 175.50,
            'timestamp': '2024-01-01T12:00:00'
        }
        mock_cache_set.return_value = True
        
        response_upper = client.get("/predict/AAPL")
        response_lower = client.get("/predict/aapl")
        
        assert response_upper.status_code == 200
        assert response_lower.status_code == 200
        assert response_upper.json()["ticker"] == "AAPL"
        assert response_lower.json()["ticker"] == "AAPL"


@patch('app.main.data_service.get_stock_data')
@patch('app.main.cache_service.get')
@patch('app.main.cache_service.set')
def test_all_supported_tickers(mock_cache_set, mock_cache_get, mock_get_stock_data):
    """Test that all supported tickers work"""
    supported_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ORCL", "CRM"]
    
    mock_cache_get.return_value = None
    mock_cache_set.return_value = True
    
    for ticker in supported_tickers:
        mock_get_stock_data.return_value = {
            'ticker': ticker,
            'current_price': 100.00,
            'timestamp': '2024-01-01T12:00:00'
        }
        
        response = client.get(f"/predict/{ticker}")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == ticker


def test_prediction_values_reasonable():
    """Test that prediction values are within reasonable ranges"""
    with patch('app.main.data_service.get_stock_data') as mock_get_data, \
         patch('app.main.cache_service.get') as mock_cache_get, \
         patch('app.main.cache_service.set') as mock_cache_set:
        
        mock_cache_get.return_value = None
        mock_get_data.return_value = {
            'ticker': 'AAPL',
            'current_price': 100.00,
            'timestamp': '2024-01-01T12:00:00'
        }
        mock_cache_set.return_value = True
        
        response = client.get("/predict/AAPL")
        assert response.status_code == 200
        
        data = response.json()
        prediction = data["prediction"]
        
        # Check that price target is within reasonable range for ML predictions
        current_price = data["current_price"]
        price_target = prediction["price_target"]
        price_change = abs(price_target - current_price) / current_price
        
        # ML predictions can be more conservative than Phase 2 random predictions
        assert price_change <= 0.05, f"Price change {price_change:.2%} exceeds ±5% limit"
        
        # ML models can have higher confidence than Phase 2
        assert 0.5 <= prediction["confidence"] <= 1.0, f"Confidence {prediction['confidence']} outside valid range"