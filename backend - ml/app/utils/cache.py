import json
from functools import wraps
from flask import current_app
import redis
from datetime import timedelta
from redis import Redis
from app.core.config import settings

class RedisCache:
    def __init__(self, app=None):
        self.redis_client = None
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the cache with the Flask app"""
        self.redis_client = redis.Redis(
            host=app.config['REDIS_HOST'],
            port=app.config['REDIS_PORT'],
            db=0,
            password=app.config['REDIS_PASSWORD'],
            decode_responses=True
        )
    
    def get(self, key):
        """Get value from cache"""
        if self.redis_client is None:
            self.init_app(current_app)
        value = self.redis_client.get(key)
        return json.loads(value) if value else None
    
    def set(self, key, value, expire=3600):
        """Set value in cache with expiration"""
        if self.redis_client is None:
            self.init_app(current_app)
        self.redis_client.setex(
            key,
            expire,
            json.dumps(value)
        )
    
    def delete(self, key):
        """Delete value from cache"""
        if self.redis_client is None:
            self.init_app(current_app)
        self.redis_client.delete(key)
    
    def clear(self):
        """Clear all cache"""
        if self.redis_client is None:
            self.init_app(current_app)
        self.redis_client.flushdb()

# Create cache instance
cache = RedisCache()

# Initialize Redis client
redis_client = Redis.from_url(
    settings.REDIS_URL,
    db=settings.REDIS_DB,
    password=settings.REDIS_PASSWORD,
    ssl=settings.REDIS_SSL,
    decode_responses=True
)

def cache_key_prefix(prefix: str):
    """Generate cache key with prefix"""
    return f"{settings.CACHE_KEY_PREFIX}:{prefix}"

def cache(ttl: int = None, prefix: str = None):
    """
    Cache decorator for functions.
    
    Args:
        ttl: Time to live in seconds
        prefix: Cache key prefix
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Generate cache key
            key = cache_key_prefix(prefix or f.__name__)
            if args:
                key += f":{':'.join(str(arg) for arg in args)}"
            if kwargs:
                key += f":{':'.join(f'{k}={v}' for k, v in sorted(kwargs.items()))}"
                
            # Try to get from cache
            cached = redis_client.get(key)
            if cached is not None:
                return cached
                
            # Get fresh value
            value = f(*args, **kwargs)
            
            # Cache value
            if ttl is not None:
                redis_client.setex(key, ttl, str(value))
            else:
                redis_client.set(key, str(value))
                
            return value
            
        return decorated_function
    return decorator

def invalidate_cache(prefix: str = None):
    """
    Invalidate cache for a prefix.
    
    Args:
        prefix: Cache key prefix to invalidate
    """
    pattern = cache_key_prefix(prefix or "*")
    keys = redis_client.keys(pattern)
    if keys:
        redis_client.delete(*keys)

def invalidate_cache_decorator(pattern):
    """
    Decorator to invalidate cache entries matching a pattern
    Usage: @invalidate_cache(pattern='prefix:*')
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            result = f(*args, **kwargs)
            # Invalidate cache entries matching pattern
            if cache.redis_client is None:
                cache.init_app(current_app)
            keys = cache.redis_client.keys(pattern)
            if keys:
                cache.redis_client.delete(*keys)
            return result
        return decorated_function
    return decorator

def cache_key_prefix_decorator(prefix):
    """
    Decorator to add a prefix to cache keys
    Usage: @cache_key_prefix('user:')
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Add prefix to function name
            f.__name__ = f"{prefix}{f.__name__}"
            return f(*args, **kwargs)
        return decorated_function
    return decorator 