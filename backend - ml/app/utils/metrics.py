from functools import wraps
import time
from flask import request, g
from app.core.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    track_request_metrics
)

def track_api_call():
    """
    Decorator to track API calls with metrics
    
    Usage:
        @track_api_call()
        def my_endpoint():
            return "Hello"
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            try:
                response = f(*args, **kwargs)
                status = response.status_code
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.endpoint,
                    status=status
                ).inc()
                REQUEST_LATENCY.labels(
                    method=request.method,
                    endpoint=request.endpoint
                ).observe(time.time() - start_time)
                return response
            except Exception as e:
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.endpoint,
                    status=500
                ).inc()
                raise e
        return decorated_function
    return decorator

def track_error(error_type: str):
    """
    Track error occurrences
    
    Args:
        error_type: Type of error (e.g., 'validation', 'database', 'auth')
    """
    from app.core.metrics import ERROR_COUNT
    ERROR_COUNT.labels(error_type=error_type).inc()

def track_performance(operation: str):
    """
    Decorator to track performance metrics
    
    Args:
        operation: Name of the operation being tracked
        
    Usage:
        @track_performance('database_query')
        def my_function():
            return "Hello"
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                from app.core.metrics import OPERATION_DURATION
                OPERATION_DURATION.labels(operation=operation).observe(duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                from app.core.metrics import OPERATION_DURATION
                OPERATION_DURATION.labels(operation=operation).observe(duration)
                raise e
        return decorated_function
    return decorator

def track_cache_operation(operation: str, hit: bool):
    """
    Track cache operation results
    
    Args:
        operation: Type of cache operation
        hit: Whether it was a cache hit
    """
    from app.core.metrics import CACHE_HITS, CACHE_MISSES
    if hit:
        CACHE_HITS.labels(operation=operation).inc()
    else:
        CACHE_MISSES.labels(operation=operation).inc()

def track_database_operation(operation: str, duration: float):
    """
    Track database operation metrics
    
    Args:
        operation: Type of database operation
        duration: Operation duration in seconds
    """
    from app.core.metrics import DB_QUERY_COUNT, DB_QUERY_LATENCY
    DB_QUERY_COUNT.labels(operation=operation).inc()
    DB_QUERY_LATENCY.labels(operation=operation).observe(duration) 