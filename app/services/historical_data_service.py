import aiohttp
import asyncio
import logging
import os
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import time
import json

logger = logging.getLogger(__name__)


class HistoricalDataService:
    """Service for fetching and managing historical stock data"""
    
    def __init__(self):
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        self.data_dir = "data/historical"
        self.cache_days = 1  # Cache historical data for 1 day
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        if not self.api_key:
            logger.warning("ALPHA_VANTAGE_API_KEY not set for historical data")
        else:
            logger.info("Historical data service initialized with Alpha Vantage")
    
    async def get_historical_data(self, ticker: str, days: int = 252) -> Optional[pd.DataFrame]:
        """
        Get historical data for a ticker (252 trading days = ~1 year)
        
        Args:
            ticker: Stock ticker symbol
            days: Number of trading days to fetch (default 252 = 1 year)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        ticker = ticker.upper()
        
        # Check cache first
        cached_data = self._load_cached_data(ticker)
        if cached_data is not None:
            logger.info(f"Using cached historical data for {ticker}")
            return cached_data.tail(days)  # Return last N days
        
        # Fetch fresh data
        logger.info(f"Fetching historical data for {ticker}")
        raw_data = await self._fetch_daily_data(ticker)
        
        if raw_data is not None:
            # Clean and validate data
            cleaned_data = self._clean_data(raw_data, ticker)
            if cleaned_data is not None:
                # Cache the data
                self._cache_data(cleaned_data, ticker)
                return cleaned_data.tail(days)
        
        logger.error(f"Failed to fetch historical data for {ticker}")
        return None
    
    async def _fetch_daily_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch daily data from Alpha Vantage"""
        if not self.api_key:
            logger.error("No Alpha Vantage API key for historical data")
            return None
        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",  # Get full historical data
            "apikey": self.api_key
        }
        
        timeout = aiohttp.ClientTimeout(total=30)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Add delay to respect rate limits
                await asyncio.sleep(2)
                
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_daily_response(data, ticker)
                    else:
                        logger.error(f"Alpha Vantage HTTP {response.status} for {ticker} historical data")
                        return None
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
            return None
    
    def _parse_daily_response(self, data: Dict, ticker: str) -> Optional[pd.DataFrame]:
        """Parse Alpha Vantage daily time series response"""
        try:
            # Check for API limit or error messages
            if "Note" in data:
                logger.warning(f"Alpha Vantage API limit: {data['Note']}")
                return None
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
                return None
            
            time_series_key = "Time Series (Daily)"
            if time_series_key not in data:
                logger.error(f"No time series data for {ticker}. Keys: {list(data.keys())}")
                return None
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append({
                    'date': pd.to_datetime(date_str),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            df = pd.DataFrame(df_data)
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Parsed {len(df)} days of historical data for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing historical data for {ticker}: {str(e)}")
            return None
    
    def _clean_data(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Clean and validate historical data"""
        try:
            original_len = len(df)
            
            # Remove rows with invalid prices
            df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
            
            # Remove rows where high < low (data errors)
            df = df[df['high'] >= df['low']]
            
            # Remove rows with extreme price changes (likely errors)
            df['price_change'] = df['close'].pct_change()
            df = df[abs(df['price_change']) < 0.5]  # Remove >50% daily changes
            
            # Remove rows with zero volume (likely holidays/errors)
            df = df[df['volume'] > 0]
            
            # Forward fill small gaps (up to 3 days)
            df = df.fillna(method='ffill', limit=3)
            
            # Drop any remaining NaN rows
            df = df.dropna()
            
            cleaned_len = len(df)
            removed = original_len - cleaned_len
            
            if removed > 0:
                logger.info(f"Cleaned {ticker} data: removed {removed} invalid rows")
            
            if len(df) < 50:  # Need minimum data for ML
                logger.warning(f"Insufficient clean data for {ticker}: {len(df)} rows")
                return None
            
            return df.drop('price_change', axis=1)  # Remove temporary column
            
        except Exception as e:
            logger.error(f"Error cleaning data for {ticker}: {str(e)}")
            return None
    
    def _load_cached_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load cached historical data if recent enough"""
        cache_file = os.path.join(self.data_dir, f"{ticker}_daily.csv")
        
        try:
            if os.path.exists(cache_file):
                # Check if cache is recent enough
                cache_age = time.time() - os.path.getmtime(cache_file)
                if cache_age < (self.cache_days * 24 * 3600):
                    df = pd.read_csv(cache_file)
                    df['date'] = pd.to_datetime(df['date'])
                    logger.debug(f"Loaded cached data for {ticker}: {len(df)} rows")
                    return df
                else:
                    logger.debug(f"Cache expired for {ticker}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading cached data for {ticker}: {str(e)}")
            return None
    
    def _cache_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """Cache historical data to CSV file"""
        cache_file = os.path.join(self.data_dir, f"{ticker}_daily.csv")
        
        try:
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached historical data for {ticker}: {len(df)} rows")
            return True
        except Exception as e:
            logger.error(f"Error caching data for {ticker}: {str(e)}")
            return False