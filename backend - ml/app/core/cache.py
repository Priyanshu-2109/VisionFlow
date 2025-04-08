import json
from typing import Any, Optional, Union
import redis
from app.core.config import settings

class RedisCache:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=settings.REDIS_DB,
            decode_responses=True
        )
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception:
            return None
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """Set value in cache with optional expiration"""
        try:
            value = json.dumps(value)
            self.redis_client.set(key, value, ex=expire)
            return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            self.redis_client.delete(key)
            return True
        except Exception:
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.redis_client.exists(key))
        except Exception:
            return False
    
    def clear(self) -> bool:
        """Clear all keys from cache"""
        try:
            self.redis_client.flushdb()
            return True
        except Exception:
            return False

# Create global cache instance
cache = RedisCache()

def get_cache() -> RedisCache:
    """Get cache instance"""
    return cache

def setup_cache(app=None):
    """Initialize cache with Flask app"""
    global cache
    if app is not None:
        # You can add any app-specific cache setup here
        pass
    return cache 