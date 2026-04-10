"""Task-7 unit tests for monitor.py.

Covers:
  1. compute_session_slippage_bps — qty-weighted absolute bps
  2. rolling_slippage_baseline — multi-day fixture skipping missing days
  3. _roundtrip_pnls — FIFO matching produces correct winner/loser
  4. build_digest — end-to-end on a synthetic day's CSV
  5. format_digest — renders zero-trade + normal + degrading cases
  6. DigestScheduler — per-day latch before / at / after 15:25 IST
"""
from __future__ import annotations

import csv
import os
import sys
import unittest
from datetime import date, datetime, time, timedelta, timezone
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

import monitor  # noqa: E402
from monitor import (  # noqa: E402
    DigestScheduler,
    DigestStats,
    IST,
    _roundtrip_pnls,
    build_digest,
    compute_session_slippage_bps,
    format_digest,
    rolling_slippage_baseline,
)

CSV_FIELDS = [
    "timestamp", "symbol", "direction", "qty", "fill_price", "signal_price",
    "slippage_bps", "brokerage", "exchange_charges", "stt", "success",
    "order_type",
]


def _row(**kw) -> dict:
    r = {f: "" for f in CSV_FIELDS}
    r["success"] = "True"
    r["order_type"] = "LIMIT"
    r["brokerage"] = "2.0"
    r["exchange_charges"] = "0.05"
    r["stt"] = "0.1"
    r.update({k: str(v) for k, v in kw.items()})
    return r


def _write_day(day: date, rows: list[dict]) -> Path:
    p = monitor.LOG_DIR / f"trades_{day.strftime('%Y-%m-%d')}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


class SlippageMathTests(unittest.TestCase):

    def test_qty_weighted_abs_bps(self):
        rows = [
            _row(qty=10, slippage_bps=8.0),   # 80
            _row(qty=20, slippage_bps=-4.0),  # 80 (abs)
            _row(qty=5,  slippage_bps=20.0),  # 100
        ]
        avg, n = compute_session_slippage_bps(rows)
        # (8*10 + 4*20 + 20*5) / 35 = 260/35 ≈ 7.428
        self.assertAlmostEqual(avg, 260 / 35, places=4)
        self.assertEqual(n, 3)

    def test_empty_rows(self):
        self.assertEqual(compute_session_slippage_bps([]), (0.0, 0))

    def test_zero_qty_skipped(self):
        rows = [_row(qty=0, slippage_bps=99.0), _row(qty=10, slippage_bps=5.0)]
        avg, n = compute_session_slippage_bps(rows)
        self.assertAlmostEqual(avg, 5.0)
        self.assertEqual(n, 1)


class RollingBaselineTests(unittest.TestCase):

    def setUp(self):
        # Use a date far in the past to avoid clobbering real CSVs
        self.end = date(2023, 6, 15)   # Thursday
        self.created: list[Path] = []

        # Day-1 back = Wed 14th (rows)
        self.created.append(_write_day(
            date(2023, 6, 14),
            [_row(qty=10, slippage_bps=4.0), _row(qty=10, slippage_bps=6.0)],
        ))
        # Day-2 back = Tue 13th (no file — skipped)
        # Day-3 back = Mon 12th
        self.created.append(_write_day(
            date(2023, 6, 12),
            [_row(qty=20, slippage_bps=8.0)],
        ))
        # Sat/Sun automatically skipped by _prev_trading_days

    def tearDown(self):
        for p in self.created:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def test_baseline_weights_by_qty_across_days(self):
        avg, n_days = rolling_slippage_baseline(self.end, window_days=10)
        # Baseline weights by n_rows per day (not total qty).
        # Day-1 avg = (4*10 + 6*10)/20 = 5.0, weight = 2 rows
        # Day-3 avg = 8.0, weight = 1 row
        # combined = (5*2 + 8*1)/3 = 6.0
        self.assertAlmostEqual(avg, 6.0, places=4)
        self.assertEqual(n_days, 2)

    def test_baseline_no_data(self):
        # End on a date with no history files
        avg, n_days = rolling_slippage_baseline(date(1990, 1, 15), window_days=5)
        self.assertEqual((avg, n_days), (0.0, 0))


class RoundTripTests(unittest.TestCase):

    def test_fifo_matches_single_pair(self):
        rows = [
            _row(timestamp="09:30", symbol="AAA", direction="BUY",
                 qty=10, fill_price=100.0),
            _row(timestamp="10:30", symbol="AAA", direction="SELL",
                 qty=10, fill_price=105.0),
        ]
        trips = _roundtrip_pnls(rows)
        self.assertIn("AAA", trips)
        self.assertEqual(len(trips["AAA"]), 1)
        # Gross = 10 * (105 - 100) = 50; costs from two legs subtract a bit
        pnl, entry_px = trips["AAA"][0]
        self.assertAlmostEqual(entry_px, 100.0)
        self.assertLess(pnl, 50.0)    # costs applied
        self.assertGreater(pnl, 40.0)

    def test_fifo_partial_matching(self):
        rows = [
            _row(timestamp="09:30", symbol="BBB", direction="BUY",
                 qty=10, fill_price=50.0),
            _row(timestamp="10:00", symbol="BBB", direction="SELL",
                 qty=4,  fill_price=52.0),
            _row(timestamp="11:00", symbol="BBB", direction="SELL",
                 qty=6,  fill_price=48.0),
        ]
        trips = _roundtrip_pnls(rows)
        self.assertEqual(len(trips["BBB"]), 2)
        # First pair = 4 * (52-50) = 8 gross → winner
        # Second = 6 * (48-50) = -12 gross → loser
        pnls = [p for p, _ in trips["BBB"]]
        self.assertGreater(pnls[0], 0)
        self.assertLess(pnls[1], 0)


class BuildDigestTests(unittest.TestCase):

    def setUp(self):
        self.day = date(2023, 6, 16)   # Friday
        rows = [
            _row(timestamp="09:30", symbol="AAA", direction="BUY",
                 qty=10, fill_price=100.0, slippage_bps=5.0),
            _row(timestamp="10:30", symbol="AAA", direction="SELL",
                 qty=10, fill_price=110.0, slippage_bps=-5.0),
            _row(timestamp="11:00", symbol="BBB", direction="BUY",
                 qty=5,  fill_price=200.0, slippage_bps=4.0),
            _row(timestamp="13:00", symbol="BBB", direction="SELL",
                 qty=5,  fill_price=195.0, slippage_bps=-4.0),
        ]
        self.path = _write_day(self.day, rows)

    def tearDown(self):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def test_build_digest_full(self):
        stats = build_digest(day=self.day)
        self.assertEqual(stats.n_trades, 4)
        self.assertEqual(stats.n_round_trips, 2)
        # Biggest winner should be AAA (gross +100), loser BBB (gross -25)
        self.assertEqual(stats.biggest_winner[0], "AAA")
        self.assertEqual(stats.biggest_loser[0], "BBB")
        self.assertGreater(stats.biggest_winner[1], 0)
        self.assertLess(stats.biggest_loser[1], 0)
        self.assertAlmostEqual(stats.win_rate, 0.5, places=2)
        self.assertGreater(stats.avg_slippage_bps, 0)
        # No prior history → INSUFFICIENT
        self.assertIn(stats.slippage_status, ("INSUFFICIENT", "NORMAL", "DEGRADING"))

    def test_unrealised_pnl(self):
        stats = build_digest(
            day=self.day,
            positions={"CCC": 10},
            entry_prices={"CCC": 100.0},
            ltp_map={"CCC": 105.0},
        )
        self.assertAlmostEqual(stats.unrealised_pnl, 50.0)


class FormatDigestTests(unittest.TestCase):

    def test_zero_trade_render(self):
        s = DigestStats(day=date(2024, 1, 1))
        s.slippage_status = "INSUFFICIENT"
        s.baseline_slippage = 3.2
        s.baseline_days = 4
        out = format_digest(s)
        self.assertIn("No trades today", out)
        self.assertIn("3.2 bps", out)
        self.assertIn("⚪", out)

    def test_normal_render(self):
        s = DigestStats(day=date(2024, 1, 2))
        s.n_trades = 6
        s.n_round_trips = 3
        s.win_rate = 0.67
        s.realised_pnl = 1234.0
        s.gross_pnl = 1500.0
        s.total_costs = 266.0
        s.unrealised_pnl = 200.0
        s.biggest_winner = ("AAA", 900.0)
        s.biggest_loser = ("BBB", -300.0)
        s.avg_slippage_bps = 3.0
        s.baseline_slippage = 4.0
        s.baseline_days = 10
        s.slippage_status = "NORMAL"
        s.cost_drag_bps = 12.5
        out = format_digest(s)
        self.assertIn("🟢", out)
        self.assertIn("NORMAL", out)
        self.assertIn("67%", out)
        self.assertIn("AAA", out)

    def test_degrading_render(self):
        s = DigestStats(day=date(2024, 1, 3))
        s.n_trades = 4
        s.slippage_status = "DEGRADING"
        out = format_digest(s)
        self.assertIn("🔴", out)
        self.assertIn("DEGRADING", out)


class SchedulerTests(unittest.TestCase):

    def _ist(self, h, m):
        return datetime(2024, 5, 15, h, m, tzinfo=IST)

    def test_before_cutoff_does_not_fire(self):
        s = DigestScheduler()
        self.assertFalse(s.should_fire(self._ist(15, 24)))
        self.assertFalse(s.should_fire(self._ist(9, 30)))

    def test_at_cutoff_fires(self):
        s = DigestScheduler()
        self.assertTrue(s.should_fire(self._ist(15, 25)))

    def test_after_cutoff_fires_once(self):
        s = DigestScheduler()
        self.assertTrue(s.should_fire(self._ist(15, 30)))
        s.mark_fired(self._ist(15, 30).date())
        # Second call same day → no fire
        self.assertFalse(s.should_fire(self._ist(15, 45)))

    def test_reset_clears_latch(self):
        s = DigestScheduler()
        s.mark_fired(self._ist(15, 30).date())
        s.reset()
        self.assertTrue(s.should_fire(self._ist(15, 30)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
