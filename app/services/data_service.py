import aiohttp
import asyncio
import logging
import os
from typing import Optional, Dict, Any
import time
import random
from datetime import datetime

logger = logging.getLogger(__name__)


class DataService:
    """Service for fetching real stock market data using Alpha Vantage API"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        self.max_retries = 2
        self.last_request_time = 0
        self.min_request_interval = 12.0  # Alpha Vantage free tier: 5 calls per minute = 12 seconds apart
        
        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY environment variable not set!")
        else:
            logger.info("Alpha Vantage API key configured successfully")
        
    async def get_stock_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch real stock data using Alpha Vantage API
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary with stock data or None if failed
        """
        ticker = ticker.upper()
        
        if not self.api_key:
            logger.error("No Alpha Vantage API key provided")
            return None
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting - Alpha Vantage is strict about this
                await self._rate_limit()
                
                # Fetch data
                stock_data = await self._fetch_alpha_vantage_data(ticker)
                
                if stock_data:
                    if self._validate_stock_data(stock_data):
                        logger.info(f"Successfully fetched Alpha Vantage data for {ticker} on attempt {attempt + 1}: ${stock_data['current_price']}")
                        return stock_data
                    else:
                        logger.warning(f"Invalid Alpha Vantage data for {ticker} on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Alpha Vantage error for {ticker} on attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    delay = 10 + random.uniform(2, 5)  # Longer delay for API limits
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
        
        logger.error(f"Failed to fetch Alpha Vantage data for {ticker} after {self.max_retries} attempts")
        return None
    
    async def _rate_limit(self):
        """Rate limiting for Alpha Vantage (5 calls per minute free tier)"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.info(f"Alpha Vantage rate limiting: waiting {sleep_time:.1f}s (5 calls/minute limit)")
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def _fetch_alpha_vantage_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Alpha Vantage API"""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker,
            "apikey": self.api_key
        }
        
        timeout = aiohttp.ClientTimeout(total=15)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug(f"Making Alpha Vantage request for {ticker}")
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Alpha Vantage response for {ticker}: {list(data.keys())}")
                        return self._parse_alpha_vantage_response(data, ticker)
                    else:
                        logger.error(f"Alpha Vantage HTTP {response.status} for {ticker}")
                        return None
        except Exception as e:
            logger.error(f"Alpha Vantage request failed for {ticker}: {str(e)}")
            return None
    
    def _parse_alpha_vantage_response(self, data: Dict, ticker: str) -> Optional[Dict[str, Any]]:
        """Parse Alpha Vantage API response"""
        try:
            # Check for API limit message
            if "Note" in data:
                logger.warning(f"Alpha Vantage API limit message: {data['Note']}")
                return None
            
            # Check for error message
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return None
            
            if "Global Quote" not in data:
                logger.error(f"No Global Quote data for {ticker}. Available keys: {list(data.keys())}")
                return None
            
            quote = data["Global Quote"]
            
            # Alpha Vantage field mappings
            current_price = float(quote.get("05. price", 0))
            previous_close = float(quote.get("08. previous close", current_price))
            change_percent = quote.get("10. change percent", "0%").replace("%", "")
            
            if current_price <= 0:
                logger.error(f"Invalid price from Alpha Vantage for {ticker}: {current_price}")
                return None
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'previous_close': previous_close,
                'change_percent': f"{change_percent}%",
                'volume': int(quote.get("06. volume", 0)),
                'high': float(quote.get("03. high", 0)),
                'low': float(quote.get("04. low", 0)),
                'open': float(quote.get("02. open", 0)),
                'timestamp': datetime.utcnow().isoformat(),
                'data_source': 'alpha_vantage'
            }
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing Alpha Vantage data for {ticker}: {str(e)}")
            logger.debug(f"Raw Alpha Vantage data: {data}")
            return None
    
    def _validate_stock_data(self, data: Dict[str, Any]) -> bool:
        """Validate that stock data is reasonable"""
        try:
            price = data.get('current_price')
            ticker = data.get('ticker')
            
            if not price or price <= 0:
                logger.warning(f"Invalid price for {ticker}: {price}")
                return False
            
            if not (0.01 <= price <= 50000):
                logger.warning(f"Price out of reasonable range for {ticker}: ${price}")
                return False
            
            required_fields = ['ticker', 'current_price', 'timestamp']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                logger.warning(f"Missing required fields for {ticker}: {missing_fields}")
                return False
            
            logger.debug(f"Alpha Vantage data validation passed for {ticker}: ${price}")
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return False