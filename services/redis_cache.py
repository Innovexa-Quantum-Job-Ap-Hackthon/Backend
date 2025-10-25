# redis_cache.py
import redis
import json
from typing import Any, Optional

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.use_redis = True
            print("✅ Connected to Redis")
        except Exception as e:
            print(f"⚠️ Redis not available, using in-memory cache instead. Error: {e}")
            self.use_redis = False
            self.cache = {}

    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if self.use_redis:
            return self.redis_client.get(key)
        return self.cache.get(key)

    def set(self, key: str, value: Any, ex: int = None) -> bool:
        """Set value in cache with optional expiration"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)

        if self.use_redis:
            return self.redis_client.set(key, value, ex=ex)
        else:
            self.cache[key] = value
            return True

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if self.use_redis:
            return self.redis_client.delete(key) > 0
        else:
            return self.cache.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if self.use_redis:
            return self.redis_client.exists(key) > 0
        return key in self.cache
