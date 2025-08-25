import time
import logging
from typing import Callable, Dict, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque
import psutil
import asyncio

logger = logging.getLogger(__name__)


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking performance metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.memory_samples: deque = deque(maxlen=100)
        self.cache_stats = {
            'hits': 0,
            'misses': 0
        }
        
        # Start memory monitoring task
        asyncio.create_task(self._monitor_memory())
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        endpoint = f"{request.method} {request.url.path}"
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record timing
            duration = time.time() - start_time
            self.request_times[endpoint].append(duration)
            self.request_counts[endpoint] += 1
            
            # Log slow requests
            if duration > 1.0:  # Log requests over 1 second
                logger.warning(f"Slow request: {endpoint} took {duration:.3f}s")
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}"
            
            return response
            
        except Exception as e:
            # Record error
            duration = time.time() - start_time
            self.error_counts[endpoint] += 1
            self.request_counts[endpoint] += 1
            
            logger.error(f"Request error: {endpoint} - {str(e)} (took {duration:.3f}s)")
            raise
    
    async def _monitor_memory(self):
        """Monitor memory usage in background"""
        while True:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_samples.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'memory_percent': process.memory_percent()
                })
                
                # Log memory warnings
                if memory_mb > 500:  # Warn if over 500MB
                    logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                
                await asyncio.sleep(30)  # Sample every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {
            'request_stats': {},
            'memory_stats': {},
            'cache_stats': self.cache_stats.copy()
        }
        
        # Calculate request statistics
        for endpoint, times in self.request_times.items():
            if times:
                times_list = list(times)
                times_list.sort()
                
                count = len(times_list)
                stats['request_stats'][endpoint] = {
                    'count': self.request_counts[endpoint],
                    'errors': self.error_counts[endpoint],
                    'error_rate': self.error_counts[endpoint] / max(1, self.request_counts[endpoint]),
                    'avg_time': sum(times_list) / count,
                    'median_time': times_list[count // 2],
                    'p95_time': times_list[int(count * 0.95)] if count > 20 else times_list[-1],
                    'min_time': times_list[0],
                    'max_time': times_list[-1]
                }
        
        # Calculate memory statistics
        if self.memory_samples:
            recent_memory = [s['memory_mb'] for s in list(self.memory_samples)[-10:]]
            stats['memory_stats'] = {
                'current_mb': recent_memory[-1] if recent_memory else 0,
                'avg_mb': sum(recent_memory) / len(recent_memory),
                'max_mb': max(recent_memory),
                'samples': len(self.memory_samples)
            }
        
        # Calculate cache statistics
        total_cache_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_cache_requests > 0:
            stats['cache_stats']['hit_rate'] = self.cache_stats['hits'] / total_cache_requests
        else:
            stats['cache_stats']['hit_rate'] = 0.0
        
        return stats
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_stats['hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_stats['misses'] += 1