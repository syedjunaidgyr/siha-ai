"""
Simple in-memory cache service
"""
import hashlib
import time
from typing import Optional, Dict, Any


class CacheService:
    """Simple in-memory cache with TTL"""
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def initialize(self):
        """Initialize cache service"""
        print("Cache service initialized")
    
    def generate_key(self, data: bytes) -> str:
        """Generate cache key from data"""
        return hashlib.md5(data).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            # Check if expired
            if time.time() < entry['expires_at']:
                return entry['value']
            else:
                # Remove expired entry
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL (seconds)"""
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }

