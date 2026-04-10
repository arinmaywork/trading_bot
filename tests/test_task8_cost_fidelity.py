"""Task-8 unit tests — full Zerodha cost fidelity (R-16).

Verifies that backtest.cost_model.leg_cost matches a Zerodha contract note
line-by-line, and that every consumer (logbook.py, portfolio_risk.py,
strategy.py) routes through the same canonical helper — so live, backtest
and P&L reports can never drift.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

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

from backtest.cost_model import (  # noqa: E402
    CostBreakdown,
    leg_cost,
    round_trip_cost,
)


class ZerodhaContractNoteTests(unittest.TestCase):
    """
    Hand-verified contract-note line items for a 10-share MIS round trip
    on RELIANCE @ ₹2500 → ₹2550. See backtest.cost_model._parity_check for
    the full arithmetic.
    """

    def test_mis_buy_leg_components(self):
        leg = leg_cost(2500.0, 10, "BUY", "MIS")
        # Brokerage = min(20, 0.03% × 25000) = 7.5
        self.assertAlmostEqual(leg.brokerage, 7.5, places=4)
        # No STT on MIS BUY
        self.assertEqual(leg.stt, 0.0)
        # Exchange = 0.00297% × 25000
        self.assertAlmostEqual(leg.exchange, 0.7425, places=4)
        # SEBI = 0.0001% × 25000
        self.assertAlmostEqual(leg.sebi, 0.025, places=6)
        # Stamp duty = 0.003% × 25000
        self.assertAlmostEqual(leg.stamp, 0.75, places=4)
        # GST = 18% × (brokerage + exchange + SEBI)
        self.assertAlmostEqual(leg.gst, 0.18 * (7.5 + 0.7425 + 0.025), places=4)
        self.assertAlmostEqual(
            leg.total,
            7.5 + 0.7425 + 0.025 + 0.75 + 0.18 * (7.5 + 0.7425 + 0.025),
            places=4,
        )

    def test_mis_sell_leg_components(self):
        leg = leg_cost(2550.0, 10, "SELL", "MIS")
        self.assertAlmostEqual(leg.brokerage, 7.65, places=4)
        # STT on sell = 0.025% × 25500
        self.assertAlmostEqual(leg.stt, 6.375, places=4)
        self.assertAlmostEqual(leg.exchange, 0.75735, places=4)
        self.assertAlmostEqual(leg.sebi, 0.0255, places=6)
        # No stamp duty on SELL
        self.assertEqual(leg.stamp, 0.0)
        # GST excludes STT and stamp
        self.assertAlmostEqual(leg.gst, 0.18 * (7.65 + 0.75735 + 0.0255), places=4)

    def test_round_trip_total_matches_hand_computation(self):
        rt = round_trip_cost(2500.0, 2550.0, 10, "MIS")
        # From _parity_check: 10.5056 + 16.3257 ≈ 26.8313
        self.assertAlmostEqual(rt, 26.8313, places=3)

    def test_brokerage_flat_cap_kicks_in_for_large_orders(self):
        # At ₹10_000 × 10 shares = ₹100_000 order val → 0.03% = ₹30 > ₹20 cap
        leg = leg_cost(10_000.0, 10, "BUY", "MIS")
        self.assertEqual(leg.brokerage, 20.0)


class CncDeliveryTests(unittest.TestCase):
    """Delivery trades have BOTH-side STT (0.1%) and higher stamp duty (0.015%)."""

    def test_cnc_buy_leg(self):
        leg = leg_cost(1000.0, 100, "BUY", "CNC")
        order_val = 100_000.0
        # Brokerage = 0 for delivery... but our model uses the same tier.
        # Note: Zerodha delivery is actually ₹0, but SentiStack intentionally
        # uses the same intraday schedule (conservative). We verify the
        # configured math, not the real-world delivery exemption.
        self.assertEqual(leg.stt, 0.001 * order_val)         # ₹100 STT on CNC buy
        # Stamp duty 0.015% on CNC BUY
        self.assertAlmostEqual(leg.stamp, 0.00015 * order_val, places=4)

    def test_cnc_sell_leg(self):
        leg = leg_cost(1000.0, 100, "SELL", "CNC")
        self.assertEqual(leg.stt, 0.001 * 100_000.0)          # both-side STT
        self.assertEqual(leg.stamp, 0.0)                      # no stamp on SELL


class ConsumerParityTests(unittest.TestCase):
    """Every consumer must return the same number as backtest.cost_model."""

    def test_portfolio_risk_uses_canonical_model(self):
        from portfolio_risk import _trade_cost as pr_cost
        row = {
            "fill_price": "2500",
            "qty": "10",
            "direction": "BUY",
            "product_type": "MIS",
        }
        expected = leg_cost(2500.0, 10, "BUY", "MIS").total
        self.assertAlmostEqual(pr_cost(row), expected, places=4)

    def test_logbook_sources_canonical_model(self):
        """Source-level check — logbook.py must import from backtest.cost_model."""
        logbook_src = (ROOT / "logbook.py").read_text(encoding="utf-8")
        self.assertIn("from backtest.cost_model import leg_cost", logbook_src)

    def test_strategy_sources_canonical_model(self):
        """Source-level check — strategy.py must import round_trip_cost."""
        strategy_src = (ROOT / "strategy.py").read_text(encoding="utf-8")
        self.assertIn("from backtest.cost_model import round_trip_cost", strategy_src)


class RoundTripHurdleTests(unittest.TestCase):

    def test_strategy_cost_filter_uses_canonical(self):
        # strategy.py imports round_trip_cost from backtest.cost_model directly;
        # a black-box check: round_trip_cost should be monotonic in qty
        c1 = round_trip_cost(2500, 2500, 1, "MIS")
        c2 = round_trip_cost(2500, 2500, 10, "MIS")
        c3 = round_trip_cost(2500, 2500, 100, "MIS")
        self.assertLess(c1, c2)
        self.assertLess(c2, c3)

    def test_small_qty_has_positive_total(self):
        rt = round_trip_cost(100.0, 100.0, 1, "MIS")
        self.assertGreater(rt, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
