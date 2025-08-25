import redis.asyncio as redis
import json
import logging
from typing import Optional, Any, Dict
import asyncio
import hashlib
import time

logger = logging.getLogger(__name__)


class CacheService:
    """Optimized Redis cache service with prediction caching and connection pooling"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.default_ttl = 300  # 5 minutes cache
        
        # In-memory cache for frequently accessed data
        self.memory_cache: Dict[str, Dict] = {}
        self.memory_cache_size = 100  # Max items in memory cache
        self.memory_cache_ttl = 60  # 1 minute TTL for memory cache
        
        # Performance tracking
        self.performance_middleware = None
        
    async def connect(self):
        """Initialize Redis connection with connection pooling"""
        try:
            # Create connection pool for better performance
            self.redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,  # Connection pool size
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test the connection
            await self.redis_client.ping()
            logger.info("Successfully connected to Redis with connection pooling")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None
    
    def set_performance_middleware(self, middleware):
        """Set reference to performance middleware for metrics"""
        self.performance_middleware = middleware
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with memory cache optimization"""
        # Check memory cache first (fastest)
        memory_result = self._get_from_memory_cache(key)
        if memory_result is not None:
            if self.performance_middleware:
                self.performance_middleware.record_cache_hit()
            logger.debug(f"Memory cache hit for key: {key}")
            return memory_result
        
        # Check Redis cache
        if not self.redis_client:
            logger.warning("Redis client not available, cache miss")
            if self.performance_middleware:
                self.performance_middleware.record_cache_miss()
            return None
            
        try:
            value = await self.redis_client.get(key)
            if value:
                parsed_value = json.loads(value)
                # Store in memory cache for faster future access
                self._set_memory_cache(key, parsed_value)
                
                if self.performance_middleware:
                    self.performance_middleware.record_cache_hit()
                logger.debug(f"Redis cache hit for key: {key}")
                return parsed_value
            else:
                if self.performance_middleware:
                    self.performance_middleware.record_cache_miss()
                logger.debug(f"Cache miss for key: {key}")
                return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            if self.performance_middleware:
                self.performance_middleware.record_cache_miss()
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with memory cache optimization"""
        if not self.redis_client:
            logger.warning("Redis client not available, skipping cache set")
            return False
            
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, default=str)
            
            await self.redis_client.setex(key, ttl, serialized_value)
            
            # Also store in memory cache
            self._set_memory_cache(key, value)
            
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    async def set_prediction_cache(self, ticker: str, features_hash: str, prediction: Dict) -> bool:
        """Cache ML predictions for identical inputs"""
        cache_key = f"prediction:{ticker}:{features_hash}"
        return await self.set(cache_key, prediction, ttl=1800)  # 30 minute TTL
    
    async def get_prediction_cache(self, ticker: str, features_hash: str) -> Optional[Dict]:
        """Get cached ML prediction for identical inputs"""
        cache_key = f"prediction:{ticker}:{features_hash}"
        return await self.get(cache_key)
    
    def _get_from_memory_cache(self, key: str) -> Optional[Any]:
        """Get from in-memory cache"""
        if key in self.memory_cache:
            cache_entry = self.memory_cache[key]
            if time.time() - cache_entry['timestamp'] < self.memory_cache_ttl:
                return cache_entry['value']
            else:
                # Expired, remove from memory cache
                del self.memory_cache[key]
        return None
    
    def _set_memory_cache(self, key: str, value: Any):
        """Set in memory cache with size limit"""
        # Remove oldest entries if cache is full
        if len(self.memory_cache) >= self.memory_cache_size:
            oldest_key = min(self.memory_cache.keys(), 
                           key=lambda k: self.memory_cache[k]['timestamp'])
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        if not self.redis_client:
            return False
            
        try:
            result = await self.redis_client.delete(key)
            if result:
                logger.debug(f"Deleted cache key: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    async def close(self):
        """Close Redis connection and pool"""
        if self.redis_client:
            await self.redis_client.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()
        logger.info("Redis connections closed")
    
    def generate_cache_key(self, prefix: str, ticker: str) -> str:
        """Generate standardized cache key"""
        return f"{prefix}:{ticker.upper()}"
    
    def generate_features_hash(self, features_array) -> str:
        """Generate hash of features array for prediction caching"""
        try:
            # Convert features to string and hash
            features_str = str(features_array.tolist()) if hasattr(features_array, 'tolist') else str(features_array)
            return hashlib.md5(features_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Error generating features hash: {str(e)}")
            return "unknown"
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'memory_cache_size': len(self.memory_cache),
            'memory_cache_max_size': self.memory_cache_size,
            'redis_connected': self.redis_client is not None
        }