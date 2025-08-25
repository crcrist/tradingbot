#!/usr/bin/env python3
"""
Load testing script for Stock Prediction API Phase 4
Tests concurrent requests and performance under load
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict
import argparse


class LoadTester:
    """Load testing utility for the Stock Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict] = []
        self.tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str) -> Dict:
        """Make a single HTTP request and measure performance"""
        start_time = time.time()
        
        try:
            async with session.get(f"{self.base_url}{endpoint}") as response:
                end_time = time.time()
                
                response_time = end_time - start_time
                status_code = response.status
                
                # Try to read response body
                try:
                    body = await response.json()
                    response_size = len(json.dumps(body))
                except:
                    body = await response.text()
                    response_size = len(body)
                
                return {
                    'endpoint': endpoint,
                    'status_code': status_code,
                    'response_time': response_time,
                    'response_size': response_size,
                    'success': 200 <= status_code < 400,
                    'timestamp': start_time
                }
                
        except Exception as e:
            end_time = time.time()
            return {
                'endpoint': endpoint,
                'status_code': 0,
                'response_time': end_time - start_time,
                'response_size': 0,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            }
    
    async def run_concurrent_requests(self, endpoints: List[str], concurrent_requests: int) -> List[Dict]:
        """Run multiple concurrent requests"""
        print(f"Running {len(endpoints)} requests with {concurrent_requests} concurrent connections...")
        
        connector = aiohttp.TCPConnector(limit=concurrent_requests, limit_per_host=concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self.make_request(session, endpoint) for endpoint in endpoints]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and convert to regular results
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    print(f"Request failed with exception: {result}")
                else:
                    valid_results.append(result)
            
            return valid_results
    
    async def test_health_endpoint(self, num_requests: int = 100, concurrent: int = 10):
        """Test health endpoint performance"""
        print(f"\n=== Health Endpoint Load Test ===")
        print(f"Requests: {num_requests}, Concurrent: {concurrent}")
        
        endpoints = ["/health"] * num_requests
        results = await self.run_concurrent_requests(endpoints, concurrent)
        self.analyze_results(results, "Health Endpoint")
        
        return results
    
    async def test_prediction_endpoints(self, num_requests: int = 50, concurrent: int = 5):
        """Test prediction endpoints performance"""
        print(f"\n=== Prediction Endpoints Load Test ===")
        print(f"Requests: {num_requests}, Concurrent: {concurrent}")
        
        # Create requests for different tickers
        endpoints = []
        for i in range(num_requests):
            ticker = self.tickers[i % len(self.tickers)]
            endpoints.append(f"/predict/{ticker}")
        
        results = await self.run_concurrent_requests(endpoints, concurrent)
        self.analyze_results(results, "Prediction Endpoints")
        
        return results
    
    async def test_performance_endpoints(self, num_requests: int = 20, concurrent: int = 5):
        """Test performance monitoring endpoints"""
        print(f"\n=== Performance Endpoints Load Test ===")
        print(f"Requests: {num_requests}, Concurrent: {concurrent}")
        
        endpoints = ["/performance/summary", "/performance/metrics", "/warmup/status"] * (num_requests // 3)
        endpoints.extend(["/performance/summary"] * (num_requests % 3))
        
        results = await self.run_concurrent_requests(endpoints, concurrent)
        self.analyze_results(results, "Performance Endpoints")
        
        return results
    
    def analyze_results(self, results: List[Dict], test_name: str):
        """Analyze and print test results"""
        if not results:
            print(f"No results for {test_name}")
            return
        
        # Calculate statistics
        response_times = [r['response_time'] for r in results if r['success']]
        success_count = sum(1 for r in results if r['success'])
        total_requests = len(results)
        error_rate = (total_requests - success_count) / total_requests * 100
        
        if response_times:
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_time = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 20 else max_time
            p99_time = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 100 else max_time
            
            # Performance targets
            under_100ms = sum(1 for t in response_times if t < 0.1) / len(response_times) * 100
            under_1s = sum(1 for t in response_times if t < 1.0) / len(response_times) * 100
        else:
            avg_time = median_time = min_time = max_time = p95_time = p99_time = 0
            under_100ms = under_1s = 0
        
        # Print results
        print(f"\n{test_name} Results:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Successful: {success_count} ({(success_count/total_requests)*100:.1f}%)")
        print(f"  Error Rate: {error_rate:.1f}%")
        print(f"  Response Times:")
        print(f"    Average: {avg_time*1000:.1f}ms")
        print(f"    Median: {median_time*1000:.1f}ms")
        print(f"    95th percentile: {p95_time*1000:.1f}ms")
        print(f"    99th percentile: {p99_time*1000:.1f}ms")
        print(f"    Min: {min_time*1000:.1f}ms")
        print(f"    Max: {max_time*1000:.1f}ms")
        print(f"  Performance Targets:")
        print(f"    Under 100ms: {under_100ms:.1f}%")
        print(f"    Under 1s: {under_1s:.1f}%")
        
        # Highlight issues
        if error_rate > 1.0:
            print(f"  ⚠️  High error rate: {error_rate:.1f}%")
        if p95_time > 1.0:
            print(f"  ⚠️  Slow 95th percentile: {p95_time*1000:.1f}ms")
        if under_100ms < 90 and "Prediction" in test_name:
            print(f"  ⚠️  Low sub-100ms rate for predictions: {under_100ms:.1f}%")
        
        # Success indicators
        if error_rate < 1.0:
            print(f"  ✅ Low error rate")
        if p95_time < 0.1 or ("Health" in test_name and p95_time < 0.05):
            print(f"  ✅ Fast 95th percentile response time")
    
    async def run_comprehensive_test(self):
        """Run all load tests"""
        print("🚀 Starting Comprehensive Load Test for Stock Prediction API")
        print("=" * 60)
        
        start_time = time.time()
        
        # Test different aspects
        health_results = await self.test_health_endpoint(100, 20)
        prediction_results = await self.test_prediction_endpoints(30, 10)
        performance_results = await self.test_performance_endpoints(15, 5)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"Load Test Complete - Total Time: {total_time:.2f}s")
        
        # Overall summary
        all_results = health_results + prediction_results + performance_results
        total_requests = len(all_results)
        total_success = sum(1 for r in all_results if r['success'])
        overall_error_rate = (total_requests - total_success) / total_requests * 100
        
        print(f"\n📊 Overall Summary:")
        print(f"  Total Requests: {total_requests}")
        print(f"  Overall Success Rate: {(total_success/total_requests)*100:.1f}%")
        print(f"  Overall Error Rate: {overall_error_rate:.1f}%")
        print(f"  Requests per Second: {total_requests/total_time:.1f}")
        
        # Phase 4 validation criteria
        print(f"\n✅ Phase 4 Validation Criteria:")
        prediction_times = [r['response_time'] for r in prediction_results if r['success']]
        if prediction_times:
            p95_pred_time = sorted(prediction_times)[int(len(prediction_times) * 0.95)]
            print(f"  Inference time 95th percentile: {p95_pred_time*1000:.1f}ms (target: <100ms)")
            print(f"  Error rate: {overall_error_rate:.1f}% (target: <1%)")
            
            if p95_pred_time < 0.1:
                print(f"  ✅ Inference time target MET")
            else:
                print(f"  ❌ Inference time target MISSED")
            
            if overall_error_rate < 1.0:
                print(f"  ✅ Error rate target MET")
            else:
                print(f"  ❌ Error rate target MISSED")


async def main():
    parser = argparse.ArgumentParser(description='Load test the Stock Prediction API')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL of the API')
    parser.add_argument('--health-requests', type=int, default=100, help='Number of health check requests')
    parser.add_argument('--prediction-requests', type=int, default=30, help='Number of prediction requests')
    parser.add_argument('--concurrent', type=int, default=10, help='Number of concurrent connections')
    
    args = parser.parse_args()
    
    tester = LoadTester(args.url)
    
    try:
        await tester.run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n❌ Load test interrupted by user")
    except Exception as e:
        print(f"\n❌ Load test failed: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())