"""Task-6 unit tests for OrderExecutor.place_passive_limit.

Covers four decision branches:
  1. Happy path: order rests, fills COMPLETE within TTL → passive fill returned.
  2. TTL timeout: order never fills → cancel + aggressive fallback for full qty.
  3. Invalid book (bid >= ask): skip passive, go straight to aggressive.
  4. Partial fill + TTL: cancel, aggressive fallback for remaining qty,
     returned fill_price is a weighted blend of passive + aggressive legs.

All tests stub the kite client and rate limiter — no network or env vars
beyond the minimal config bootstrap.
"""
from __future__ import annotations

import asyncio
import os
import sys
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

for _k, _v in {
    "KITE_API_KEY": "stub",
    "KITE_API_SECRET": "stub",
    "KITE_ACCESS_TOKEN": "stub",
    "OPENWEATHER_API_KEY": "stub",
    "GEMINI_API_KEY": "stub",
    "TELEGRAM_BOT_TOKEN": "stub",
    "TELEGRAM_CHAT_ID": "0",
}.items():
    os.environ.setdefault(_k, _v)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from execution import OrderExecutor             # noqa: E402
from strategy import TradeDirection              # noqa: E402
from config import settings                      # noqa: E402


class _StubLimiter:
    """Minimal stand-in for RateLimiter — both slots are no-op async CMs."""

    @asynccontextmanager
    async def order_slot(self):
        yield

    @asynccontextmanager
    async def request_slot(self):
        yield


def _fresh_executor(kite: MagicMock) -> OrderExecutor:
    # TelegramNotifier needs a coroutine-able .send(); mock the whole object.
    tg = MagicMock()
    tg.send = AsyncMock()
    ex = OrderExecutor(kite=kite, rate_limiter=_StubLimiter(), telegram=tg)
    return ex


def _quote_response(bid: float, ask: float, ltp: float) -> Dict[str, Any]:
    key = f"{settings.kite.EXCHANGE}:ACME"
    return {
        key: {
            "last_price": ltp,
            "depth": {
                "buy":  [{"price": bid}] if bid else [],
                "sell": [{"price": ask}] if ask else [],
            },
        }
    }


class PassiveLimitTests(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Force short TTL so timeout tests finish quickly.
        from config import settings as _s
        self._orig_ttl  = _s.kite.PASSIVE_LIMIT_TTL_S
        self._orig_poll = _s.kite.PASSIVE_POLL_INTERVAL_S
        object.__setattr__(_s.kite, "PASSIVE_LIMIT_TTL_S", 0.3)
        object.__setattr__(_s.kite, "PASSIVE_POLL_INTERVAL_S", 0.05)

    def tearDown(self):
        from config import settings as _s
        object.__setattr__(_s.kite, "PASSIVE_LIMIT_TTL_S", self._orig_ttl)
        object.__setattr__(_s.kite, "PASSIVE_POLL_INTERVAL_S", self._orig_poll)

    async def test_happy_path_order_fills_before_ttl(self):
        kite = MagicMock()
        kite.TRANSACTION_TYPE_BUY  = "BUY"
        kite.TRANSACTION_TYPE_SELL = "SELL"
        kite.quote.return_value = _quote_response(bid=100.0, ask=100.20, ltp=100.10)
        kite.place_order.return_value = "ORD-1"
        # First poll → COMPLETE at ₹100.05
        kite.order_history.return_value = [
            {"status": "COMPLETE", "average_price": 100.05, "filled_quantity": 10}
        ]

        ex = _fresh_executor(kite)
        oid, fill, err = await ex.place_passive_limit(
            symbol="ACME", direction=TradeDirection.BUY, quantity=10,
            signal_price=100.10,
        )
        self.assertEqual(oid, "ORD-1")
        self.assertAlmostEqual(fill, 100.05, places=2)
        self.assertIsNone(err)

        # Verify BUY rested BELOW mid (100.10) at mid - spread/4 = 100.05
        place_kwargs = kite.place_order.call_args.kwargs
        self.assertEqual(place_kwargs["transaction_type"], "BUY")
        self.assertLessEqual(place_kwargs["price"], 100.10)
        self.assertGreaterEqual(place_kwargs["price"], 100.0)

    async def test_ttl_timeout_falls_back_to_aggressive(self):
        kite = MagicMock()
        kite.TRANSACTION_TYPE_BUY  = "BUY"
        kite.TRANSACTION_TYPE_SELL = "SELL"
        kite.quote.return_value = _quote_response(bid=500.0, ask=500.40, ltp=500.20)
        # place_order called twice: passive leg then aggressive fallback
        kite.place_order.side_effect = ["PASSIVE-1", "AGG-1"]
        # order_history always "OPEN" until timeout
        kite.order_history.return_value = [
            {"status": "OPEN", "average_price": 0.0, "filled_quantity": 0}
        ]
        # cancel_order succeeds
        kite.cancel_order.return_value = "cancelled"
        # Aggressive _get_ltp needs ltp()
        kite.ltp.return_value = {f"{settings.kite.EXCHANGE}:ACME": {"last_price": 500.30}}

        ex = _fresh_executor(kite)
        oid, fill, err = await ex.place_passive_limit(
            symbol="ACME", direction=TradeDirection.BUY, quantity=5,
            signal_price=500.20,
        )

        self.assertIsNone(err)
        self.assertEqual(oid, "AGG-1")  # aggressive leg's id is returned
        self.assertEqual(kite.place_order.call_count, 2)
        kite.cancel_order.assert_called_once()
        # Aggressive leg used signal price × (1 + 0.002) = 501.20 rounded to 1dp
        agg_kwargs = kite.place_order.call_args_list[1].kwargs
        self.assertAlmostEqual(agg_kwargs["price"], 501.2, places=2)

    async def test_invalid_book_skips_passive(self):
        kite = MagicMock()
        kite.TRANSACTION_TYPE_BUY  = "BUY"
        kite.TRANSACTION_TYPE_SELL = "SELL"
        # Crossed book → ask <= bid → invalid
        kite.quote.return_value = _quote_response(bid=0.0, ask=0.0, ltp=250.0)
        kite.place_order.return_value = "AGG-ONLY"
        kite.ltp.return_value = {f"{settings.kite.EXCHANGE}:ACME": {"last_price": 250.1}}

        ex = _fresh_executor(kite)
        oid, fill, err = await ex.place_passive_limit(
            symbol="ACME", direction=TradeDirection.BUY, quantity=3,
            signal_price=250.0,
        )
        self.assertEqual(oid, "AGG-ONLY")
        self.assertIsNone(err)
        # Only ONE place_order call — straight to aggressive
        self.assertEqual(kite.place_order.call_count, 1)

    async def test_partial_fill_blends_with_aggressive_fallback(self):
        kite = MagicMock()
        kite.TRANSACTION_TYPE_BUY  = "BUY"
        kite.TRANSACTION_TYPE_SELL = "SELL"
        kite.quote.return_value = _quote_response(bid=1000.0, ask=1000.80, ltp=1000.40)
        kite.place_order.side_effect = ["PASSIVE-2", "AGG-2"]
        # Partial fill stuck at 4/10
        kite.order_history.return_value = [
            {"status": "OPEN", "average_price": 1000.20, "filled_quantity": 4}
        ]
        kite.cancel_order.return_value = "cancelled"
        kite.ltp.return_value = {f"{settings.kite.EXCHANGE}:ACME": {"last_price": 1001.0}}

        ex = _fresh_executor(kite)
        oid, fill, err = await ex.place_passive_limit(
            symbol="ACME", direction=TradeDirection.BUY, quantity=10,
            signal_price=1000.40,
        )
        self.assertIsNone(err)
        self.assertEqual(oid, "AGG-2")
        self.assertEqual(kite.place_order.call_count, 2)

        # Aggressive leg only re-ordered the remaining 6 shares
        agg_kwargs = kite.place_order.call_args_list[1].kwargs
        self.assertEqual(agg_kwargs["quantity"], 6)

        # Blended fill = (1000.20*4 + agg_fill*6) / 10
        # agg_fill comes from _get_ltp mock → 1001.0
        expected_blend = (1000.20 * 4 + 1001.0 * 6) / 10
        self.assertAlmostEqual(fill, expected_blend, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
