from functools import wraps
from flask import request, jsonify
from redis import Redis
from datetime import datetime, timedelta
from typing import Optional, Callable
from app.config import Config

class RateLimiter:
    """Rate limiter using Redis for storage"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        
    def is_rate_limited(self, key: str, limit: int, window: int) -> bool:
        """
        Check if a key has exceeded the rate limit.
        
        Args:
            key: Unique identifier for the rate limit
            limit: Maximum number of requests allowed
            window: Time window in seconds
            
        Returns:
            bool: True if rate limited, False otherwise
        """
        current = self.redis.get(key)
        if current is None:
            self.redis.setex(key, window, 1)
            return False
            
        current = int(current)
        if current >= limit:
            return True
            
        self.redis.incr(key)
        return False
        
    def get_remaining_requests(self, key: str, limit: int) -> int:
        """
        Get remaining requests for a key.
        
        Args:
            key: Unique identifier for the rate limit
            limit: Maximum number of requests allowed
            
        Returns:
            int: Number of remaining requests
        """
        current = self.redis.get(key)
        if current is None:
            return limit
        return max(0, limit - int(current))
        
    def get_reset_time(self, key: str) -> Optional[datetime]:
        """
        Get time when rate limit will reset.
        
        Args:
            key: Unique identifier for the rate limit
            
        Returns:
            datetime: Time when rate limit will reset
        """
        ttl = self.redis.ttl(key)
        if ttl > 0:
            return datetime.utcnow() + timedelta(seconds=ttl)
        return None

def rate_limit(limit: int, window: int, key_func: Optional[Callable] = None):
    """
    Decorator to rate limit endpoints.
    
    Args:
        limit: Maximum number of requests allowed
        window: Time window in seconds
        key_func: Optional function to generate rate limit key
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            redis_client = Redis.from_url(Config.REDIS_URL)
            limiter = RateLimiter(redis_client)
            
            if key_func:
                key = key_func()
            else:
                key = f"{request.remote_addr}:{request.endpoint}"
                
            if limiter.is_rate_limited(key, limit, window):
                reset_time = limiter.get_reset_time(key)
                return jsonify({
                    "message": "Rate limit exceeded",
                    "reset_time": reset_time.isoformat() if reset_time else None
                }), 429
                
            response = f(*args, **kwargs)
            
            # Add rate limit headers
            remaining = limiter.get_remaining_requests(key, limit)
            reset_time = limiter.get_reset_time(key)
            
            if isinstance(response, tuple):
                response_obj, status_code = response
            else:
                response_obj, status_code = response, 200
                
            headers = {
                'X-RateLimit-Limit': str(limit),
                'X-RateLimit-Remaining': str(remaining),
                'X-RateLimit-Reset': reset_time.isoformat() if reset_time else None
            }
            
            return response_obj, status_code, headers
            
        return decorated_function
    return decorator 