"""
rate_limiter.py
===============
Centralised, thread-safe (asyncio-native) rate limiter that enforces
Zerodha's two hard limits simultaneously:

    • 10 REST requests per second (any endpoint)
    • 400 orders per minute   (place_order / modify_order / cancel_order)

Design: Token Bucket algorithm backed by asyncio.Lock and asyncio.Semaphore.

Usage:
    limiter = RateLimiter()

    # Before any Kite REST call:
    async with limiter.request_slot():
        response = kite.some_api_call()

    # Before any order placement call:
    async with limiter.order_slot():
        kite.place_order(...)
"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Generic async token-bucket rate limiter.

    Args:
        rate:     Maximum number of tokens consumed per `period` seconds.
        period:   Rolling window in seconds.
    """

    def __init__(self, rate: int, period: float) -> None:
        self._rate = rate
        self._period = period
        self._tokens = float(rate)          # Start with a full bucket
        self._last_refill: float = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time (called under lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        # How many tokens have been earned in the elapsed window?
        new_tokens = elapsed * (self._rate / self._period)
        self._tokens = min(float(self._rate), self._tokens + new_tokens)
        self._last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        """
        Block until `tokens` tokens are available, then consume them.
        Respects FIFO ordering via the asyncio Lock.
        """
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                # Calculate sleep needed to accumulate missing tokens
                deficit = tokens - self._tokens
                sleep_for = deficit * (self._period / self._rate)
                logger.debug(
                    "Rate limit hit — sleeping %.3f s to acquire %d token(s).",
                    sleep_for, tokens
                )
                # Release lock while sleeping so other coroutines can check
                self._lock.release()
                try:
                    await asyncio.sleep(sleep_for)
                finally:
                    await self._lock.acquire()


class RateLimiter:
    """
    Centralised rate-limiter enforcing both Zerodha API constraints.

    Attributes:
        _request_bucket:  10 req/s  — governs all REST calls.
        _order_bucket:   400 ord/min — governs order mutation calls only.

    Both buckets are token-bucket instances; callers must acquire a slot
    before making the corresponding API call.
    """

    def __init__(
        self,
        max_requests_per_second: int = 10,
        max_orders_per_minute: int = 400,
    ) -> None:
        self._request_bucket = TokenBucket(
            rate=max_requests_per_second, period=1.0
        )
        self._order_bucket = TokenBucket(
            rate=max_orders_per_minute, period=60.0
        )

        # Telemetry counters (monotonic; never reset)
        self._total_requests: int = 0
        self._total_orders: int = 0
        self._total_request_waits: int = 0
        self._total_order_waits: int = 0

        logger.info(
            "RateLimiter initialised: %d req/s | %d orders/min",
            max_requests_per_second, max_orders_per_minute
        )

    @asynccontextmanager
    async def request_slot(self) -> AsyncGenerator[None, None]:
        """
        Async context manager: acquire one request token before the block
        and release (no cleanup needed) afterwards.

        Example:
            async with limiter.request_slot():
                await kite_async.instruments()
        """
        t0 = time.monotonic()
        await self._request_bucket.acquire()
        elapsed = time.monotonic() - t0
        if elapsed > 0.05:
            self._total_request_waits += 1
            logger.debug("Request token waited %.3f s (total waits: %d)",
                         elapsed, self._total_request_waits)
        self._total_requests += 1
        try:
            yield
        finally:
            pass  # Token bucket is already consumed; nothing to release.

    @asynccontextmanager
    async def order_slot(self) -> AsyncGenerator[None, None]:
        """
        Async context manager: acquire both a request token (order calls are
        still REST requests) and an order token before the block.

        Example:
            async with limiter.order_slot():
                kite.place_order(...)
        """
        t0 = time.monotonic()
        # Acquire both tokens — request first (finer-grained limit), then order
        await self._request_bucket.acquire()
        await self._order_bucket.acquire()
        elapsed = time.monotonic() - t0
        if elapsed > 0.05:
            self._total_order_waits += 1
            logger.debug("Order token waited %.3f s (total waits: %d)",
                         elapsed, self._total_order_waits)
        self._total_orders += 1
        try:
            yield
        finally:
            pass

    @property
    def stats(self) -> dict:
        """Return a snapshot of rate-limiter telemetry."""
        return {
            "total_requests": self._total_requests,
            "total_orders": self._total_orders,
            "request_throttle_events": self._total_request_waits,
            "order_throttle_events": self._total_order_waits,
        }
