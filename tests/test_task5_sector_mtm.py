"""Task-5 unit tests: sector exposure cap + intraday MTM drawdown stop.

Runs without requiring any broker / telegram env vars — stubs the bare
minimum for `config.Settings` to import cleanly.
"""
from __future__ import annotations

import os
import sys
import unittest
from datetime import timedelta
from pathlib import Path

# ── Stub env before importing settings ───────────────────────────────────
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

import portfolio_risk as pr                       # noqa: E402
from portfolio_risk import (                      # noqa: E402
    PortfolioRiskMonitor,
    compute_sector_exposure,
    compute_unrealised_pnl,
    load_sector_map,
    sector_for,
)


class SectorExposureTests(unittest.TestCase):

    def test_sector_map_loads(self):
        m = load_sector_map(force_reload=True)
        self.assertGreater(len(m), 50, "sector map should have many rows")
        self.assertEqual(sector_for("TCS"), "IT")
        self.assertEqual(sector_for("HDFCBANK"), "Banking")
        self.assertEqual(sector_for("NOT_A_REAL_SYMBOL_XYZ"), "Unknown")

    def test_compute_sector_exposure_groups_by_sector(self):
        positions = {"TCS": 10, "INFY": 5, "HDFCBANK": 20, "ICICIBANK": -10}
        ltp       = {"TCS": 4000, "INFY": 1500, "HDFCBANK": 1600, "ICICIBANK": 1100}
        exp = compute_sector_exposure(positions, ltp)
        # IT       = 10*4000 + 5*1500 = 47500
        # Banking  = 20*1600 + 10*1100 (abs qty) = 43000
        self.assertAlmostEqual(exp["IT"],      47500.0, places=2)
        self.assertAlmostEqual(exp["Banking"], 43000.0, places=2)

    def test_would_breach_sector_cap(self):
        mon = PortfolioRiskMonitor(capital=100_000)   # cap = 30% = ₹30k
        positions = {"TCS": 5}                         # ₹20k in IT
        ltp       = {"TCS": 4000, "INFY": 1500}
        # +5 INFY × 1500 = 7500 → IT total 27500 < 30000 → allowed
        blocked, _ = mon.would_breach_sector_cap("INFY", 5, 1500, positions, ltp)
        self.assertFalse(blocked)
        # +10 INFY × 1500 = 15000 → IT total 35000 > 30000 → blocked
        blocked, reason = mon.would_breach_sector_cap("INFY", 10, 1500, positions, ltp)
        self.assertTrue(blocked)
        self.assertIn("IT", reason)


class MtmStopTests(unittest.TestCase):

    def test_mtm_stop_latches_and_releases(self):
        mon = PortfolioRiskMonitor(capital=100_000)
        now = pr._ist_now()

        # Well within budget → stop inactive
        mtm, limit, active, until = mon.evaluate_mtm_stop(
            realised_day_pnl=-100, unrealised_pnl=-200, now_ist=now,
        )
        self.assertFalse(active)
        self.assertEqual(limit, 1500.0)  # 1.5% of 100k

        # Bust the threshold — should latch
        mtm, limit, active, until = mon.evaluate_mtm_stop(
            realised_day_pnl=-800, unrealised_pnl=-900, now_ist=now,
        )
        self.assertTrue(active)
        self.assertIsNotNone(until)

        # Mid-day rebound should NOT release the latch
        mtm, limit, active2, _ = mon.evaluate_mtm_stop(
            realised_day_pnl=500, unrealised_pnl=200, now_ist=now + timedelta(minutes=30),
        )
        self.assertTrue(active2, "latch must survive intraday rebound")

        # Past the auto-clear time (next session) → released
        post = until + timedelta(minutes=1)
        mtm, limit, active3, until3 = mon.evaluate_mtm_stop(
            realised_day_pnl=500, unrealised_pnl=200, now_ist=post,
        )
        self.assertFalse(active3)
        self.assertIsNone(until3)

    def test_compute_unrealised_pnl(self):
        pos = {"TCS": 10, "INFY": -5}          # short 5 INFY
        entry = {"TCS": 4000.0, "INFY": 1600.0}
        ltp   = {"TCS": 4050.0, "INFY": 1550.0}
        u = compute_unrealised_pnl(pos, entry, ltp)
        # TCS:  (4050-4000)*10 = +500
        # INFY: (1550-1600)*-5 = +250  (short profits on drop)
        self.assertAlmostEqual(u, 750.0, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
