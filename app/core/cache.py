"""
In-memory TTL cache for reducing redundant API calls and client instantiation.

Used to cache:
- Model configurations (by config_id)
- Prompt versions (by prompt_id)
- Built LLM model services (by model_configuration_id)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional
import asyncio

logger = logging.getLogger(__name__)


class TTLCache:
    """
    Thread-safe in-memory cache with TTL (time-to-live).

    Entries expire after ttl_seconds. Uses asyncio.Lock for concurrent access.
    """

    def __init__(self, ttl_seconds: int, max_size: Optional[int] = None):
        """
        Args:
            ttl_seconds: Time-to-live for each entry in seconds
            max_size: Optional max entries (LRU eviction when exceeded). None = unbounded.
        """
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()
        self._access_order: list[str] = []  # For LRU when max_size is set

    async def get[T](self, key: str) -> Optional[T]:
        """
        Get value by key. Returns None if not found or expired.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.monotonic() > expires_at:
                del self._cache[key]
                if self._max_size and key in self._access_order:
                    self._access_order.remove(key)
                return None
            if self._max_size and key in self._access_order:
                self._access_order.remove(key)
                self._access_order.append(key)
            return value

    async def set[T](self, key: str, value: T) -> None:
        """
        Set value with TTL. Evicts oldest if at max_size.
        """
        async with self._lock:
            if self._max_size and len(self._cache) >= self._max_size and key not in self._cache:
                while len(self._cache) >= self._max_size and self._access_order:
                    oldest = self._access_order.pop(0)
                    if oldest in self._cache:
                        del self._cache[oldest]
            expires_at = time.monotonic() + self._ttl
            self._cache[key] = (value, expires_at)
            if self._max_size and key not in self._access_order:
                self._access_order.append(key)

    async def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            if self._max_size and key in self._access_order:
                self._access_order.remove(key)

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()


# Default TTLs (seconds)
DEFAULT_MODEL_CONFIG_TTL = 300  # 5 minutes
DEFAULT_PROMPT_VERSION_TTL = 300  # 5 minutes
DEFAULT_MODEL_SERVICE_TTL = 600  # 10 minutes - built clients are heavier to create
DEFAULT_MODEL_SERVICE_MAX_SIZE = 50  # Limit memory for many distinct configs
