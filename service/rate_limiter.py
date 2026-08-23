"""
service/rate_limiter.py
------------------------
Two interchangeable fixed-window rate limiters for service/api.py:

  RateLimiter       — in-memory. Correct for exactly one process. Zero
                      extra dependencies, zero network hop, default.

  RedisRateLimiter   — shared-store. Correct across multiple workers/replicas
                      counting against the same key, because the counter
                      lives in Redis instead of each process's own memory.
                      Opt-in via REDIS_URL — see get_rate_limiter().

service/api.py calls get_rate_limiter() rather than constructing either
class directly, so the choice is one env var, not a code change.
"""

import os
import time
from collections import defaultdict, deque

from rag.logging_config import get_logger

log = get_logger(__name__)


class RateLimiter:
    """Fixed-window limiter: at most `max_requests` per `window_seconds`, per key."""

    def __init__(self, max_requests: int, window_seconds: float):
        if max_requests <= 0:
            raise ValueError("max_requests must be positive.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        """
        Returns True and records the request if `key` is under its limit,
        False (without recording) if it would exceed the limit.
        """
        now = time.monotonic()
        timestamps = self._requests[key]

        while timestamps and now - timestamps[0] >= self._window_seconds:
            timestamps.popleft()

        if len(timestamps) >= self._max_requests:
            return False

        timestamps.append(now)
        return True

    def retry_after_seconds(self, key: str) -> float:
        """How long until `key`'s oldest recorded request ages out of the window."""
        timestamps = self._requests.get(key)
        if not timestamps:
            return 0.0
        elapsed = time.monotonic() - timestamps[0]
        return max(0.0, round(self._window_seconds - elapsed, 2))


class RedisRateLimiter:
    """
    Fixed-window limiter backed by Redis, so the count is shared across every
    process/replica hitting the same Redis instance — the actual fix for the
    "multiple workers each count independently" problem RateLimiter's
    docstring used to warn about.

    Uses one INCR + conditional EXPIRE per key per window (not a sliding
    log like RateLimiter's deque) — cheaper at scale, at the cost of allowing
    a short burst at window boundaries. That trade-off is the right one once
    there's more than one process to coordinate.
    """

    def __init__(self, redis_client, max_requests: int, window_seconds: int):
        if max_requests <= 0:
            raise ValueError("max_requests must be positive.")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive.")
        self._redis = redis_client
        self._max_requests = max_requests
        self._window_seconds = int(window_seconds)

    def allow(self, key: str) -> bool:
        redis_key = f"ratelimit:{key}"
        count = self._redis.incr(redis_key)
        if count == 1:
            self._redis.expire(redis_key, self._window_seconds)
        return count <= self._max_requests

    def retry_after_seconds(self, key: str) -> float:
        ttl = self._redis.ttl(f"ratelimit:{key}")
        return float(ttl) if ttl and ttl > 0 else 0.0


def get_rate_limiter(max_requests: int, window_seconds: int):
    """
    Returns a RedisRateLimiter if REDIS_URL is set (shared across
    processes/replicas), otherwise a RateLimiter (correct for one process
    only — see its docstring). This is the one place that decides which
    limiter service/api.py gets; callers don't branch on REDIS_URL themselves.
    """
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    try:
        import redis
    except ImportError:
        log.warning(
            "REDIS_URL is set but the redis package isn't installed — falling back to the in-memory limiter (uncomment redis in requirements.txt)."
        )
        return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    client = redis.Redis.from_url(redis_url)
    try:
        client.ping()
    except redis.exceptions.RedisError as exc:
        log.warning(
            "REDIS_URL is set but Redis is unreachable (%s) — falling back to the in-memory limiter.", exc
        )
        return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    log.info("Using RedisRateLimiter (%s) — rate limit is shared across all processes/replicas.", redis_url)
    return RedisRateLimiter(client, max_requests=max_requests, window_seconds=window_seconds)
