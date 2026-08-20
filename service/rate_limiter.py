"""
service/rate_limiter.py
------------------------
A minimal in-memory fixed-window rate limiter for service/api.py.

Deliberately hand-rolled instead of pulling in slowapi/limits: this is a
single-process service (see docker-compose.yml — one `api` container, no
horizontal scaling configured), so an in-memory counter is exactly as
correct as a distributed one would be, at zero extra dependencies. If this
service is ever scaled to multiple workers/replicas, this stops being
correct (each process would count independently, so the effective limit
multiplies by the worker count) — the fix at that point is a shared store
(Redis, most likely), not a bigger in-process data structure.
"""

import time
from collections import defaultdict, deque


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
