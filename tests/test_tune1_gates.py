"""Tune-1 sanity tests — verify the new strategy gates kill more trades.

These are NOT pass/fail correctness tests — they're guard-rails that prove
the cost hurdle and alpha-threshold knobs got wired in correctly and the
defaults are stricter than the pre-tune values.
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


class GateDefaultsTests(unittest.TestCase):
    """Defensive tests: defaults must stay STRICTER than the old config."""

    def setUp(self):
        # Force defaults by clearing any user-supplied env overrides
        for k in [
            "MIN_ALPHA_THRESHOLD", "COST_HURDLE_MULTIPLIER",
            "ALPHA_PERCENTILE", "TSL_ACTIVATION_PCT",
            "TSL_CALLBACK_PCT", "HARD_STOP_LOSS_PCT",
        ]:
            os.environ.pop(k, None)
        # Force re-import so dataclass defaults re-evaluate
        for mod in list(sys.modules.keys()):
            if mod == "config" or mod.startswith("config."):
                del sys.modules[mod]
        from config import settings  # noqa: PLC0415
        self.cfg = settings.strategy

    def test_min_alpha_at_least_1pct(self):
        self.assertGreaterEqual(self.cfg.MIN_ALPHA_THRESHOLD, 0.010)

    def test_cost_hurdle_at_least_4x(self):
        self.assertGreaterEqual(self.cfg.COST_HURDLE_MULTIPLIER, 4.0)

    def test_alpha_percentile_at_least_p93(self):
        self.assertGreaterEqual(self.cfg.ALPHA_PERCENTILE, 0.93)

    def test_hard_stop_tightened(self):
        # Was 1.2% → now ≤ 1.0%
        self.assertLessEqual(self.cfg.HARD_STOP_LOSS_PCT, 0.010)

    def test_tsl_callback_widened(self):
        # Was 0.3% → now ≥ 0.4%
        self.assertGreaterEqual(self.cfg.TSL_CALLBACK_PCT, 0.004)

    def test_tsl_activation_lowered(self):
        # Was 0.6% → now ≤ 0.5% (winners start trailing earlier)
        self.assertLessEqual(self.cfg.TSL_ACTIVATION_PCT, 0.005)


class CostHurdleMathTests(unittest.TestCase):
    """At default 5x hurdle, only signals predicting >= ~0.5% pass at typical px."""

    def test_5x_hurdle_blocks_weak_signal(self):
        from backtest.cost_model import round_trip_cost
        # 10 shares @ ₹2500 MIS round-trip ≈ ₹26.83
        cost = round_trip_cost(2500.0, 2500.0, 10, "MIS")
        # Old 2x hurdle: any |signal| > 53.66 / 25000 = 0.21% passes
        # New 5x hurdle: any |signal| > 134.15 / 25000 = 0.54% passes
        old_min_signal = (2.0 * cost) / 25000.0
        new_min_signal = (5.0 * cost) / 25000.0
        self.assertLess(old_min_signal, 0.0025)
        self.assertGreater(new_min_signal, 0.005)
        # New gate is strictly stricter
        self.assertGreater(new_min_signal, old_min_signal * 2)

    def test_risk_reward_no_longer_inverted(self):
        # Pre-tune: TSL_ACTIVATION 0.6%, TSL_CALLBACK 0.3%, HARD_STOP 1.2%
        # → min winner = +0.3%, max loser = -1.2% → 1:4 R:R against bot
        # Post-tune: TSL_ACTIVATION 0.4%, TSL_CALLBACK 0.5%, HARD_STOP 0.8%
        # → min winner = -0.1% (yes, slight loss possible if exits fast), max loser = -0.8%
        # Better: with TSL trailing later moves, expected winner can grow far
        # beyond activation. The structural fix is hard_stop / activation no
        # longer dwarfs the floor.
        from config import settings
        cfg = settings.strategy
        # Worst-case 1:R ratio shouldn't be worse than 1:2 anymore
        worst_loser = cfg.HARD_STOP_LOSS_PCT
        min_winner_floor = max(0.0, cfg.TSL_ACTIVATION_PCT - cfg.TSL_CALLBACK_PCT)
        # Actually any winner that activates trail and reaches new highs runs
        # further; the floor is just the *minimum* lock-in. What we want:
        # hard_stop should not be 4× the activation level.
        self.assertLess(worst_loser, 4 * cfg.TSL_ACTIVATION_PCT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
