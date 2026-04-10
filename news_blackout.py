"""
news_blackout.py
================

Task-9 — News-event blackout.

When a high-impact sentiment headline lands, the bot should NOT enter new
positions on the affected symbols for a short cooldown window. This
prevents:

  • getting whipsawed by the initial reaction bar to a surprise earnings
    miss or regulatory announcement,
  • stacking into already-crowded momentum after the news is priced in,
  • chasing AI-generated false positives from a single inflammatory
    headline.

Design:
  • ``NewsBlackoutManager`` is a stateless in-memory registry.
  • Call ``update(sentiment)`` from the sentiment loop on every new
    ``SentimentResult``. If the headline is "high impact" (big absolute
    score or classification = Fear/Excitement with score above
    ``NEWS_BLACKOUT_SCORE_THRESHOLD``), we stamp an unlock timestamp on
    every symbol listed in ``sentiment.key_entities``.
  • Call ``is_blackout(symbol)`` from the strategy gate before sizing.
  • ``active_blackouts()`` returns a dict for Telegram /blackouts and
    /status rendering.

The manager is deliberately process-local: a restart clears all
blackouts, which is the correct default because on startup we also
re-enter from a clean slate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("news_blackout")

# Hard upper bound on the number of symbols we track, in case a buggy LLM
# decides to return thousands of key_entities.
_MAX_TRACKED_SYMBOLS = 500


@dataclass
class _Entry:
    symbol:       str
    unlock_mono:  float   # monotonic deadline
    headline:     str     # short snippet for /blackouts
    score:        float
    classification: str


class NewsBlackoutManager:
    """
    Per-symbol blackout latches driven by ``SentimentResult`` updates.

    All timestamps are from ``time.monotonic()`` so the manager is
    immune to wall-clock jumps (DST, NTP steps). For user-facing /status
    output, we also stamp the remaining seconds at read time.
    """

    def __init__(
        self,
        duration_s:       float = 300.0,
        score_threshold:  float = 0.50,
        enabled:          bool  = True,
    ) -> None:
        self._duration_s       = float(duration_s)
        self._score_threshold  = float(score_threshold)
        self._enabled          = bool(enabled)
        self._entries: Dict[str, _Entry] = {}

    # ── config toggles ────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)

    @property
    def duration_s(self) -> float:
        return self._duration_s

    # ── core updater ──────────────────────────────────────────────────
    def update(self, sentiment: Any) -> List[str]:
        """
        Inspect a ``SentimentResult`` and latch blackouts on any
        newly-affected symbols. Returns the list of symbols that were
        freshly latched in this call (so the caller can log them).
        """
        if not self._enabled:
            return []
        if sentiment is None:
            return []
        try:
            score = float(getattr(sentiment, "sentiment_score", 0.0) or 0.0)
            cls   = str(getattr(sentiment, "sentiment_classification", "Neutral") or "Neutral")
            entities: Iterable[str] = getattr(sentiment, "key_entities", []) or []
            rationale = str(getattr(sentiment, "rationale", "") or "")
        except Exception:
            return []

        if not self._is_high_impact(score, cls):
            return []

        if not entities:
            return []

        snippet = rationale[:80].replace("\n", " ").strip()
        now_mono = time.monotonic()
        unlock   = now_mono + self._duration_s
        stamped: List[str] = []

        for raw in entities:
            sym = str(raw or "").upper().strip()
            if not sym or len(sym) > 32:
                continue
            if len(self._entries) >= _MAX_TRACKED_SYMBOLS and sym not in self._entries:
                continue
            prev = self._entries.get(sym)
            # Only extend/replace if the new unlock is LATER than the existing one.
            if prev is None or unlock > prev.unlock_mono:
                self._entries[sym] = _Entry(
                    symbol=sym,
                    unlock_mono=unlock,
                    headline=snippet,
                    score=score,
                    classification=cls,
                )
                stamped.append(sym)
        if stamped:
            logger.info(
                "NewsBlackout: latched %d symbol(s) for %.0fs — score=%+.2f (%s): %s",
                len(stamped), self._duration_s, score, cls, ",".join(stamped),
            )
        return stamped

    def _is_high_impact(self, score: float, cls: str) -> bool:
        """A headline is high-impact iff the absolute score clears the
        threshold AND the classification is non-neutral. Both conditions
        must hold so the blackout doesn't fire on mid-range scores that
        the model couldn't actually classify."""
        if abs(score) < self._score_threshold:
            return False
        if cls.lower() == "neutral":
            return False
        return True

    # ── strategy gate ─────────────────────────────────────────────────
    def is_blackout(self, symbol: str) -> bool:
        """Return True iff ``symbol`` is currently within a blackout window."""
        if not self._enabled:
            return False
        sym = (symbol or "").upper().strip()
        entry = self._entries.get(sym)
        if entry is None:
            return False
        if time.monotonic() >= entry.unlock_mono:
            # Lazy eviction — clear the stale entry on read
            self._entries.pop(sym, None)
            return False
        return True

    def remaining_s(self, symbol: str) -> float:
        """Seconds until ``symbol`` is cleared (0 if not blacked out)."""
        entry = self._entries.get((symbol or "").upper().strip())
        if entry is None:
            return 0.0
        rem = entry.unlock_mono - time.monotonic()
        return max(0.0, rem)

    # ── reporting ─────────────────────────────────────────────────────
    def active_blackouts(self) -> List[Dict[str, Any]]:
        """
        Return a sorted list of active blackouts for rendering in
        /blackouts and /status. Auto-evicts expired entries.
        """
        now_mono = time.monotonic()
        out: List[Dict[str, Any]] = []
        expired: List[str] = []
        for sym, entry in self._entries.items():
            rem = entry.unlock_mono - now_mono
            if rem <= 0:
                expired.append(sym)
                continue
            out.append({
                "symbol":   sym,
                "remaining_s": round(rem, 1),
                "headline": entry.headline,
                "score":    entry.score,
                "classification": entry.classification,
            })
        for sym in expired:
            self._entries.pop(sym, None)
        out.sort(key=lambda r: r["remaining_s"], reverse=True)
        return out

    def clear(self) -> None:
        """Wipe every latch — used by /blackouts clear and tests."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def format_blackouts(mgr: NewsBlackoutManager) -> str:
    """HTML-safe summary for Telegram /blackouts."""
    if not mgr.enabled:
        return "📴 News blackout is <b>disabled</b>."
    rows = mgr.active_blackouts()
    if not rows:
        return (
            "🟢 <b>No active news blackouts</b>\n"
            f"Window: {mgr.duration_s:.0f}s  |  enabled ✅"
        )
    lines = [
        f"🚫 <b>Active news blackouts ({len(rows)})</b>",
        "─" * 30,
    ]
    for r in rows[:15]:
        lines.append(
            f"<code>{r['symbol']:<14}</code>  "
            f"{int(r['remaining_s']):>3}s  "
            f"({r['classification']} {r['score']:+.2f})"
        )
    if len(rows) > 15:
        lines.append(f"…and {len(rows) - 15} more")
    lines.append("")
    lines.append(f"Window: {mgr.duration_s:.0f}s  |  /blackouts clear → wipe latches")
    return "\n".join(lines)
