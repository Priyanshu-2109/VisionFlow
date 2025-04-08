from prometheus_client import Counter, Histogram, Gauge
from functools import wraps
import time
from flask import request

def get_metrics():
    """Get all metrics"""
    return {
        'request_count': REQUEST_COUNT,
        'request_latency': REQUEST_LATENCY,
        'db_query_count': DB_QUERY_COUNT,
        'db_query_latency': DB_QUERY_LATENCY,
        'cache_hits': CACHE_HITS,
        'cache_misses': CACHE_MISSES,
        'memory_usage': MEMORY_USAGE,
        'cpu_usage': CPU_USAGE
    }

def setup_metrics(app=None):
    """Initialize metrics with Flask app"""
    if app is not None:
        # Register metrics endpoint
        from prometheus_client import make_wsgi_app
        app.wsgi_app = make_wsgi_app() / app.wsgi_app
        
        # Start system metrics update
        def update_metrics():
            update_system_metrics()
            app.after_request(update_metrics)
    
    return get_metrics()

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# Database metrics
DB_QUERY_COUNT = Counter(
    'db_queries_total',
    'Total number of database queries',
    ['operation']
)

DB_QUERY_LATENCY = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation']
)

# Cache metrics
CACHE_HITS = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

# System metrics
MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Current memory usage in bytes'
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'Current CPU usage percentage'
)

def track_request_metrics():
    """Decorator to track request metrics"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
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
        return wrapped
    return decorator

def track_db_metrics(operation):
    """Decorator to track database metrics"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                DB_QUERY_COUNT.labels(operation=operation).inc()
                DB_QUERY_LATENCY.labels(operation=operation).observe(time.time() - start_time)
                return result
            except Exception as e:
                DB_QUERY_COUNT.labels(operation=operation).inc()
                raise e
        return wrapped
    return decorator

def track_cache_metrics(cache_type):
    """Decorator to track cache metrics"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            try:
                result = f(*args, **kwargs)
                if result is not None:
                    CACHE_HITS.labels(cache_type=cache_type).inc()
                else:
                    CACHE_MISSES.labels(cache_type=cache_type).inc()
                return result
            except Exception as e:
                CACHE_MISSES.labels(cache_type=cache_type).inc()
                raise e
        return wrapped
    return decorator

def update_system_metrics():
    """Update system metrics"""
    import psutil
    process = psutil.Process()
    
    # Update memory usage
    MEMORY_USAGE.set(process.memory_info().rss)
    
    # Update CPU usage
    CPU_USAGE.set(process.cpu_percent()) 