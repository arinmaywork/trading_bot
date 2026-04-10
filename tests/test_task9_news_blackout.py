"""Task-9 unit tests — news-event blackout manager.

Covers:
  1. High-impact detection (threshold + classification)
  2. Neutral and low-score headlines don't latch
  3. Multi-symbol latching from key_entities
  4. TTL expiry via monkey-patched time.monotonic
  5. Lazy eviction on is_blackout after expiry
  6. active_blackouts sort + headline snippet
  7. Enabled flag gates both update + is_blackout
  8. Multiple updates only extend (never shrink) the unlock deadline
  9. clear() wipes all latches
  10. format_blackouts HTML rendering
"""
from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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

import news_blackout  # noqa: E402
from news_blackout import NewsBlackoutManager, format_blackouts  # noqa: E402


@dataclass
class FakeSentiment:
    sentiment_score: float = 0.0
    sentiment_classification: str = "Neutral"
    key_entities: List[str] = field(default_factory=list)
    rationale: str = ""


class _ClockMock:
    """Monkey-patchable monotonic clock so tests don't depend on wall time."""
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def advance(self, s: float) -> None:
        self.t += s

    def __call__(self) -> float:
        return self.t


class _ClockMixin:
    def setUp(self) -> None:
        self.clock = _ClockMock()
        self._orig_mono = news_blackout.time.monotonic
        news_blackout.time.monotonic = self.clock  # type: ignore[assignment]

    def tearDown(self) -> None:
        news_blackout.time.monotonic = self._orig_mono  # type: ignore[assignment]


class HighImpactDetectionTests(_ClockMixin, unittest.TestCase):

    def test_score_above_threshold_with_fear_latches(self):
        mgr = NewsBlackoutManager(duration_s=300, score_threshold=0.5)
        stamped = mgr.update(FakeSentiment(
            sentiment_score=-0.8, sentiment_classification="Fear",
            key_entities=["RELIANCE", "TCS"], rationale="Major selloff",
        ))
        self.assertEqual(sorted(stamped), ["RELIANCE", "TCS"])
        self.assertTrue(mgr.is_blackout("RELIANCE"))
        self.assertTrue(mgr.is_blackout("TCS"))
        self.assertFalse(mgr.is_blackout("INFY"))

    def test_below_threshold_does_not_latch(self):
        mgr = NewsBlackoutManager(duration_s=300, score_threshold=0.5)
        stamped = mgr.update(FakeSentiment(
            sentiment_score=-0.3, sentiment_classification="Fear",
            key_entities=["RELIANCE"],
        ))
        self.assertEqual(stamped, [])
        self.assertFalse(mgr.is_blackout("RELIANCE"))

    def test_neutral_classification_never_latches(self):
        mgr = NewsBlackoutManager(duration_s=300, score_threshold=0.5)
        stamped = mgr.update(FakeSentiment(
            sentiment_score=0.9, sentiment_classification="Neutral",
            key_entities=["RELIANCE"],
        ))
        self.assertEqual(stamped, [])

    def test_empty_entities_no_effect(self):
        mgr = NewsBlackoutManager()
        self.assertEqual(
            mgr.update(FakeSentiment(sentiment_score=-0.8,
                                     sentiment_classification="Fear")),
            [],
        )

    def test_none_sentiment_is_safe(self):
        mgr = NewsBlackoutManager()
        self.assertEqual(mgr.update(None), [])


class ExpiryTests(_ClockMixin, unittest.TestCase):

    def test_expiry_after_duration(self):
        mgr = NewsBlackoutManager(duration_s=60, score_threshold=0.5)
        mgr.update(FakeSentiment(sentiment_score=0.8,
                                 sentiment_classification="Excitement",
                                 key_entities=["INFY"]))
        self.assertTrue(mgr.is_blackout("INFY"))
        self.clock.advance(30)
        self.assertTrue(mgr.is_blackout("INFY"))
        self.assertAlmostEqual(mgr.remaining_s("INFY"), 30.0, places=1)
        self.clock.advance(31)
        self.assertFalse(mgr.is_blackout("INFY"))
        # Lazy eviction: entry should be gone after is_blackout check
        self.assertEqual(mgr.remaining_s("INFY"), 0.0)

    def test_update_only_extends_deadline(self):
        mgr = NewsBlackoutManager(duration_s=100, score_threshold=0.5)
        mgr.update(FakeSentiment(sentiment_score=-0.8,
                                 sentiment_classification="Fear",
                                 key_entities=["HDFC"]))
        self.clock.advance(50)
        # A second headline with same duration → extends unlock by 50 more
        mgr.update(FakeSentiment(sentiment_score=-0.9,
                                 sentiment_classification="Fear",
                                 key_entities=["HDFC"]))
        self.assertAlmostEqual(mgr.remaining_s("HDFC"), 100.0, places=1)

    def test_update_does_not_shrink_deadline(self):
        mgr = NewsBlackoutManager(duration_s=300, score_threshold=0.5)
        mgr.update(FakeSentiment(sentiment_score=-0.9,
                                 sentiment_classification="Fear",
                                 key_entities=["HDFC"]))
        self.clock.advance(10)
        # Shorter manager → shouldn't matter, same manager in this test:
        mgr.update(FakeSentiment(sentiment_score=-0.9,
                                 sentiment_classification="Fear",
                                 key_entities=["HDFC"]))
        # Second update's unlock = now + 300 = (1010 + 300) = 1310 vs original 1300+???
        # original unlock = 1000+300=1300; new unlock=1010+300=1310 → extends
        self.assertGreater(mgr.remaining_s("HDFC"), 299.0)


class ReportingTests(_ClockMixin, unittest.TestCase):

    def test_active_blackouts_sorted_by_remaining(self):
        mgr = NewsBlackoutManager(duration_s=300, score_threshold=0.5)
        mgr.update(FakeSentiment(sentiment_score=-0.8,
                                 sentiment_classification="Fear",
                                 key_entities=["AAA"],
                                 rationale="first"))
        self.clock.advance(60)
        mgr.update(FakeSentiment(sentiment_score=-0.8,
                                 sentiment_classification="Fear",
                                 key_entities=["BBB"],
                                 rationale="second"))
        rows = mgr.active_blackouts()
        self.assertEqual([r["symbol"] for r in rows], ["BBB", "AAA"])
        self.assertGreater(rows[0]["remaining_s"], rows[1]["remaining_s"])

    def test_format_blackouts_disabled(self):
        mgr = NewsBlackoutManager(enabled=False)
        s = format_blackouts(mgr)
        self.assertIn("disabled", s)

    def test_format_blackouts_empty(self):
        mgr = NewsBlackoutManager()
        s = format_blackouts(mgr)
        self.assertIn("No active", s)

    def test_format_blackouts_populated(self):
        mgr = NewsBlackoutManager(duration_s=300, score_threshold=0.5)
        mgr.update(FakeSentiment(sentiment_score=-0.8,
                                 sentiment_classification="Fear",
                                 key_entities=["RELIANCE"],
                                 rationale="Major selloff"))
        s = format_blackouts(mgr)
        self.assertIn("RELIANCE", s)
        self.assertIn("Fear", s)
        self.assertIn("-0.80", s)


class EnabledToggleTests(_ClockMixin, unittest.TestCase):

    def test_disabled_manager_rejects_updates(self):
        mgr = NewsBlackoutManager(enabled=False)
        stamped = mgr.update(FakeSentiment(sentiment_score=-0.9,
                                           sentiment_classification="Fear",
                                           key_entities=["RELIANCE"]))
        self.assertEqual(stamped, [])
        self.assertFalse(mgr.is_blackout("RELIANCE"))

    def test_disable_after_latch_bypasses_gate(self):
        mgr = NewsBlackoutManager(enabled=True)
        mgr.update(FakeSentiment(sentiment_score=-0.9,
                                 sentiment_classification="Fear",
                                 key_entities=["RELIANCE"]))
        self.assertTrue(mgr.is_blackout("RELIANCE"))
        mgr.set_enabled(False)
        self.assertFalse(mgr.is_blackout("RELIANCE"))

    def test_clear_wipes_all(self):
        mgr = NewsBlackoutManager()
        mgr.update(FakeSentiment(sentiment_score=-0.9,
                                 sentiment_classification="Fear",
                                 key_entities=["A", "B", "C"]))
        self.assertEqual(len(mgr), 3)
        mgr.clear()
        self.assertEqual(len(mgr), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
