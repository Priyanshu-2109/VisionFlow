import json
from typing import Any, Optional, Union
from redis import Redis, ConnectionPool
from functools import wraps
import logging
from datetime import datetime, timedelta

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create Redis connection pool
redis_pool = ConnectionPool(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True,
    max_connections=10,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True
)

# Create Redis client
redis_client = Redis(connection_pool=redis_pool)

def get_redis() -> Redis:
    """Get Redis client"""
    return redis_client

def check_redis_connection() -> bool:
    """Check Redis connection"""
    try:
        redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis connection check failed: {e}")
        return False

class RedisCache:
    """Redis cache wrapper"""
    
    def __init__(self, prefix: str = settings.CACHE_KEY_PREFIX):
        self.client = redis_client
        self.prefix = prefix
    
    def _make_key(self, key: str) -> str:
        """Make cache key with prefix"""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.client.get(self._make_key(key))
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        expire: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value in cache"""
        try:
            key = self._make_key(key)
            value = json.dumps(value)
            if expire:
                if isinstance(expire, timedelta):
                    expire = int(expire.total_seconds())
                return bool(self.client.setex(key, expire, value))
            return bool(self.client.set(key, value))
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return bool(self.client.delete(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def increment(self, key: str) -> Optional[int]:
        """Increment value in cache"""
        try:
            return self.client.incr(self._make_key(key))
        except Exception as e:
            logger.error(f"Redis increment error: {e}")
            return None
    
    def expire(self, key: str, seconds: Union[int, timedelta]) -> bool:
        """Set key expiration"""
        try:
            if isinstance(seconds, timedelta):
                seconds = int(seconds.total_seconds())
            return bool(self.client.expire(self._make_key(key), seconds))
        except Exception as e:
            logger.error(f"Redis expire error: {e}")
            return False

class RateLimiter:
    """Rate limiter using Redis"""
    
    def __init__(
        self,
        key_prefix: str = "rate_limit",
        max_requests: int = settings.RATE_LIMIT_REQUESTS,
        period: int = settings.RATE_LIMIT_PERIOD
    ):
        self.client = redis_client
        self.key_prefix = key_prefix
        self.max_requests = max_requests
        self.period = period
    
    def _make_key(self, key: str) -> str:
        """Make rate limit key with prefix"""
        return f"{self.key_prefix}:{key}"
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        redis_key = self._make_key(key)
        pipe = self.client.pipeline()
        
        try:
            # Get current count
            current = self.client.get(redis_key)
            if not current:
                # First request
                pipe.setex(redis_key, self.period, 1)
                pipe.execute()
                return True
            
            current = int(current)
            if current >= self.max_requests:
                # Rate limit exceeded
                return False
            
            # Increment counter
            pipe.incr(redis_key)
            pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Rate limit error: {e}")
            return True  # Allow request on error
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests"""
        try:
            current = self.client.get(self._make_key(key))
            if not current:
                return self.max_requests
            return max(0, self.max_requests - int(current))
        except Exception as e:
            logger.error(f"Get remaining error: {e}")
            return 0
    
    def reset(self, key: str) -> bool:
        """Reset rate limit"""
        try:
            return bool(self.client.delete(self._make_key(key)))
        except Exception as e:
            logger.error(f"Reset rate limit error: {e}")
            return False

def cache(
    expire: Optional[Union[int, timedelta]] = None,
    key_prefix: Optional[str] = None
):
    """Cache decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{key_prefix or func.__name__}:{args}:{kwargs}"
            cache = RedisCache()
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Get fresh value
            value = await func(*args, **kwargs)
            
            # Cache value
            cache.set(cache_key, value, expire)
            
            return value
        return wrapper
    return decorator

def rate_limit(
    key_func,
    max_requests: Optional[int] = None,
    period: Optional[int] = None
):
    """Rate limit decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get rate limit key
            key = key_func(*args, **kwargs)
            
            # Check rate limit
            limiter = RateLimiter(
                max_requests=max_requests or settings.RATE_LIMIT_REQUESTS,
                period=period or settings.RATE_LIMIT_PERIOD
            )
            
            if not limiter.is_allowed(key):
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator 