"""
logbook.py  — V2
================
Trading Logbook: persistent, structured record of all bot activity.

Writes two files to the `logs/` directory (auto-created):

  1. logs/trades_YYYYMMDD.csv
     One row per executed order (paper or live).
     Columns: timestamp, symbol, direction, qty, fill_price, expected_price,
              slippage_bps, alpha, ml_signal, ml_confidence, sentiment_class,
              sentiment_score, gri, gri_level, geo_alpha_mult, geo_kelly_mult,
              busseti_f, mlofi, aflow_ratio, vol, vwap, order_id, mode, success

  2. logs/signals_YYYYMMDD.csv
     One row per signal evaluation (actionable or not).
     Columns: timestamp, symbol, direction, qty, alpha, ml_signal, ml_confidence,
              is_fallback, is_actionable, is_decayed, sentiment_class,
              sentiment_score, risk_context, gri, gri_level, mlofi, aflow_ratio,
              ofi, vol, vwap, busseti_f, rationale

  3. logs/session_YYYYMMDD_HHMMSS.log
     Human-readable daily session summary updated every 30 minutes:
     - P&L estimate (paper or live)
     - Win rate, avg slippage
     - Top performing symbols
     - GRI / sentiment evolution

All files are UTF-8 CSV with headers on first write.
Log rotation happens automatically at midnight IST.
"""

import asyncio
import csv
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# IST offset
_IST = timezone(timedelta(hours=5, minutes=30))

# Log directory
LOG_DIR = Path(__file__).parent / "logs"


def _ist_now() -> datetime:
    return datetime.now(_IST)


def _today_str() -> str:
    return _ist_now().strftime("%Y%m%d")


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Trade Log Row
# ---------------------------------------------------------------------------

@dataclass
class TradeLogRow:
    timestamp:        str
    symbol:           str
    direction:        str
    qty:              int
    fill_price:       float
    expected_price:   float
    slippage_bps:     float
    alpha:            float
    ml_signal:        float
    ml_confidence:    float
    ml_is_fallback:   bool
    sentiment_class:  str
    sentiment_score:  float
    gri:              float
    gri_level:        str
    geo_alpha_mult:   float
    geo_kelly_mult:   float
    busseti_f:        float
    mlofi:            float
    aflow_ratio:      float
    vol:              float
    vwap:             float
    order_id:         str
    mode:             str    # "PAPER" or "LIVE"
    success:          bool
    error:            str
    # ── Full reasoning chain ──────────────────────────────────────────
    sentiment_rationale: str   # Gemini explanation for sentiment score
    risk_context:        str   # RiskManagerAgent context (if invoked)
    gpr_normalised:      float # GPR geopolitical pressure [0,1]
    signal_why:          str   # Human-readable decision summary


TRADE_HEADERS = [
    "timestamp", "symbol", "direction", "qty",
    "fill_price", "expected_price", "slippage_bps",
    "alpha", "ml_signal", "ml_confidence", "ml_is_fallback",
    "sentiment_class", "sentiment_score",
    "gri", "gri_level", "geo_alpha_mult", "geo_kelly_mult",
    "busseti_f", "mlofi", "aflow_ratio", "vol", "vwap",
    "order_id", "mode", "success", "error",
    "sentiment_rationale", "risk_context", "gpr_normalised", "signal_why",
]


# ---------------------------------------------------------------------------
# Signal Log Row
# ---------------------------------------------------------------------------

@dataclass
class SignalLogRow:
    timestamp:       str
    symbol:          str
    direction:       str
    qty:             int
    alpha:           float
    ml_signal:       float
    ml_confidence:   float
    ml_is_fallback:  bool
    is_actionable:   bool
    is_decayed:      bool
    sentiment_class: str
    sentiment_score: float
    risk_context:    str
    gri:             float
    gri_level:       str
    mlofi:           float
    aflow_ratio:     float
    ofi:             float
    vol:             float
    vwap:            float
    busseti_f:       float
    rationale:       str


SIGNAL_HEADERS = [
    "timestamp", "symbol", "direction", "qty",
    "alpha", "ml_signal", "ml_confidence", "ml_is_fallback",
    "is_actionable", "is_decayed",
    "sentiment_class", "sentiment_score", "risk_context",
    "gri", "gri_level", "mlofi", "aflow_ratio", "ofi",
    "vol", "vwap", "busseti_f", "rationale",
]


# ---------------------------------------------------------------------------
# Logbook
# ---------------------------------------------------------------------------

class Logbook:
    """
    Async-safe trading logbook.
    Uses an asyncio.Lock to serialise file writes (no thread contention).
    """

    def __init__(self) -> None:
        _ensure_log_dir()
        self._lock = asyncio.Lock()
        self._trades:  List[TradeLogRow]  = []
        self._signals: List[SignalLogRow] = []
        self._session_start = _ist_now()
        logger.info("Logbook initialised → %s", LOG_DIR.resolve())

    # ── File paths (rotate at midnight IST) ───────────────────────────────

    def _trade_path(self) -> Path:
        return LOG_DIR / f"trades_{_today_str()}.csv"

    def _signal_path(self) -> Path:
        return LOG_DIR / f"signals_{_today_str()}.csv"

    def _session_path(self) -> Path:
        tag = self._session_start.strftime("%Y%m%d_%H%M%S")
        return LOG_DIR / f"session_{tag}.log"

    # ── CSV writer helpers ────────────────────────────────────────────────

    def _append_csv(self, path: Path, headers: List[str], row: Any) -> None:
        """Append one dataclass row to a CSV file, writing headers if new."""
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            row_dict = {k: getattr(row, k, "") for k in headers}
            writer.writerow(row_dict)

    # ── Public API ────────────────────────────────────────────────────────

    async def log_signal(
        self,
        signal: Any,               # SignalState
        sentiment_score: float,
        sentiment_class: str,
        risk_context:    str = "",
    ) -> None:
        """Record every signal evaluation — actionable or not."""
        row = SignalLogRow(
            timestamp      = _ist_now().isoformat(),
            symbol         = signal.symbol,
            direction      = signal.direction.value,
            qty            = signal.quantity,
            alpha          = round(signal.alpha, 8),
            ml_signal      = round(getattr(signal, "ml_signal", signal.alpha), 6),
            ml_confidence  = round(getattr(signal, "ml_confidence", 0.0), 4),
            ml_is_fallback = getattr(signal, "ml_is_fallback", True),
            is_actionable  = signal.is_actionable,
            is_decayed     = signal.is_decayed,
            sentiment_class= sentiment_class,
            sentiment_score= round(sentiment_score, 4),
            risk_context   = risk_context[:200],
            gri            = round(signal.geo_risk, 4),
            gri_level      = signal.geo_level,
            mlofi          = round(getattr(signal, "mlofi", 0.0), 6),
            aflow_ratio    = round(getattr(signal, "aflow_ratio", 0.0), 4),
            ofi            = round(signal.ofi, 6),
            vol            = round(signal.vol_regime, 4),
            vwap           = round(signal.vwap, 2),
            busseti_f      = round(getattr(signal, "busseti_f", 0.0), 6),
            rationale      = signal.rationale[:300],
        )
        async with self._lock:
            self._signals.append(row)
            self._append_csv(self._signal_path(), SIGNAL_HEADERS, row)

    async def log_trade(
        self,
        report: Any,       # ExecutionReport
        signal: Any,       # SignalState
        mode:   str,       # "PAPER" or "LIVE"
    ) -> None:
        """Record every executed trade (paper or live)."""
        # Use first successful slice fill price, or expected if none
        fill = report.avg_fill_price or signal.current_price
        order_ids = ",".join(
            s.order_id for s in report.slices if s.order_id
        )

        # Build human-readable decision summary
        ml_src = "V1-static" if getattr(signal, "ml_is_fallback", True) else "XGBoost-ensemble"
        signal_why = (
            f"Direction: {signal.direction.value} | "
            f"Alpha: {signal.alpha:+.6f} | "
            f"Sentiment: {signal.sentiment_class} ({signal.sentiment_score:+.3f}) | "
            f"GRI: {signal.geo_level} ({signal.geo_risk:.3f}) | "
            f"ML-src: {ml_src} | "
            f"Kelly-f: {getattr(signal, 'busseti_f', 0):.4f} | "
            f"MLOFI: {getattr(signal, 'mlofi', 0):+.4f} | "
            f"AFlow: {getattr(signal, 'aflow_ratio', 0):.3f} | "
            f"Rationale: {getattr(signal, 'rationale', '')[:150]}"
        )

        row = TradeLogRow(
            timestamp      = _ist_now().isoformat(),
            symbol         = signal.symbol,
            direction      = signal.direction.value,
            qty            = report.total_quantity,
            fill_price     = round(fill, 2),
            expected_price = round(signal.current_price, 2),
            slippage_bps   = round(report.slippage_bps, 2),
            alpha          = round(signal.alpha, 8),
            ml_signal      = round(getattr(signal, "ml_signal", signal.alpha), 6),
            ml_confidence  = round(getattr(signal, "ml_confidence", 0.0), 4),
            ml_is_fallback = getattr(signal, "ml_is_fallback", True),
            sentiment_class= signal.sentiment_class,
            sentiment_score= round(signal.sentiment_score, 4),
            gri            = round(signal.geo_risk, 4),
            gri_level      = signal.geo_level,
            geo_alpha_mult = round(signal.geo_alpha_multiplier, 4),
            geo_kelly_mult = round(signal.geo_kelly_multiplier, 4),
            busseti_f      = round(getattr(signal, "busseti_f", 0.0), 6),
            mlofi          = round(getattr(signal, "mlofi", 0.0), 6),
            aflow_ratio    = round(getattr(signal, "aflow_ratio", 0.0), 4),
            vol            = round(signal.vol_regime, 4),
            vwap           = round(signal.vwap, 2),
            order_id       = order_ids or "N/A",
            mode           = mode,
            success        = report.success,
            error          = (report.error or "")[:200],
            sentiment_rationale = getattr(signal, "rationale", "")[:400],
            risk_context        = getattr(signal, "risk_context", "")[:300],
            gpr_normalised      = round(getattr(signal, "gpr_normalised", 0.0), 4),
            signal_why          = signal_why,
        )
        async with self._lock:
            self._trades.append(row)
            self._append_csv(self._trade_path(), TRADE_HEADERS, row)

        logger.info(
            "📒 Logbook | %s %s %s qty=%d @ ₹%.2f | slip=%.1f bps | "
            "α=%.5f GRI=%.3f (%s) | %s",
            mode, row.direction, row.symbol, row.qty, row.fill_price,
            row.slippage_bps, row.alpha, row.gri, row.gri_level,
            "✅" if row.success else "❌",
        )

    # ── Session Summary ───────────────────────────────────────────────────

    async def write_session_summary(self) -> None:
        """Write a human-readable session summary to the log file."""
        async with self._lock:
            trades  = list(self._trades)
            signals = list(self._signals)

        now = _ist_now()
        elapsed = now - self._session_start
        hours   = int(elapsed.total_seconds() // 3600)
        minutes = int((elapsed.total_seconds() % 3600) // 60)

        live_trades  = [t for t in trades if t.success]
        paper_trades = [t for t in live_trades if t.mode == "PAPER"]
        real_trades  = [t for t in live_trades if t.mode == "LIVE"]

        # P&L estimate: sum of (fill_price - expected_price) * qty * direction_sign
        def pnl_sign(t: TradeLogRow) -> float:
            sign = 1 if t.direction == "BUY" else -1
            return (t.fill_price - t.expected_price) * t.qty * sign

        estimated_pnl = sum(pnl_sign(t) for t in live_trades)
        avg_slippage  = (
            sum(t.slippage_bps for t in live_trades) / len(live_trades)
            if live_trades else 0.0
        )

        # Win rate: signals where |alpha| > threshold and direction was correct
        actionable     = [s for s in signals if s.is_actionable]
        decayed        = [s for s in signals if s.is_decayed]
        fallback_count = sum(1 for s in signals if s.ml_is_fallback)

        # Symbol frequency
        sym_counts: Dict[str, int] = {}
        for t in live_trades:
            sym_counts[t.symbol] = sym_counts.get(t.symbol, 0) + 1
        top_symbols = sorted(sym_counts.items(), key=lambda x: -x[1])[:5]

        # Sentiment distribution
        sent_counts: Dict[str, int] = {}
        for s in signals:
            sent_counts[s.sentiment_class] = sent_counts.get(s.sentiment_class, 0) + 1

        # GRI evolution
        gri_vals = [s.gri for s in signals if s.gri > 0]
        gri_min  = min(gri_vals) if gri_vals else 0
        gri_max  = max(gri_vals) if gri_vals else 0
        gri_avg  = sum(gri_vals) / len(gri_vals) if gri_vals else 0

        lines = [
            "=" * 65,
            f"  SENTISTACK V2 — SESSION SUMMARY",
            f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S IST')}",
            f"  Session duration: {hours}h {minutes}m",
            "=" * 65,
            "",
            "── EXECUTION SUMMARY ──────────────────────────────────────",
            f"  Total trades executed : {len(live_trades)}",
            f"  Paper trades          : {len(paper_trades)}",
            f"  Live trades           : {len(real_trades)}",
            f"  Failed orders         : {len(trades) - len(live_trades)}",
            f"  Estimated P&L         : ₹{estimated_pnl:+,.2f}",
            f"  Avg slippage          : {avg_slippage:.1f} bps",
            "",
            "── SIGNAL SUMMARY ─────────────────────────────────────────",
            f"  Total signals evaluated : {len(signals)}",
            f"  Actionable signals      : {len(actionable)}",
            f"  Decayed (risk-off)      : {len(decayed)}",
            f"  ML fallback (V1 alpha)  : {fallback_count}",
            f"  ML ensemble active      : {len(signals) - fallback_count}",
            "",
            "── TOP SYMBOLS ────────────────────────────────────────────",
        ]
        for sym, cnt in top_symbols:
            lines.append(f"  {sym:<15} {cnt} trades")

        lines += [
            "",
            "── SENTIMENT DISTRIBUTION ─────────────────────────────────",
        ]
        for cls, cnt in sorted(sent_counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / max(len(signals), 1)
            lines.append(f"  {cls:<15} {cnt:>4} signals  ({pct:.0f}%)")

        lines += [
            "",
            "── GEOPOLITICAL RISK (GRI) ────────────────────────────────",
            f"  Min  : {gri_min:.3f}",
            f"  Max  : {gri_max:.3f}",
            f"  Avg  : {gri_avg:.3f}",
            f"  Level: {signals[-1].gri_level if signals else 'N/A'}",
            "",
            "── RECENT TRADES (last 10) ────────────────────────────────",
        ]

        for t in trades[-10:]:
            status = "✓" if t.success else "✗"
            lines.append(
                f"  {status} {t.timestamp[11:19]}  "
                f"{t.mode:<6} {t.direction:<5} {t.symbol:<15} "
                f"qty={t.qty:<6} @ ₹{t.fill_price:>10.2f}  "
                f"slip={t.slippage_bps:+.1f}bps  α={t.alpha:+.5f}"
            )

        lines += ["", "=" * 65, ""]

        summary_text = "\n".join(lines)

        async with self._lock:
            with open(self._session_path(), "w", encoding="utf-8") as f:
                f.write(summary_text)

        logger.info("📒 Session summary written → %s", self._session_path().name)

    async def run_summary_loop(self, interval_seconds: int = 1800) -> None:
        """Write session summary every 30 minutes."""
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self.write_session_summary()
            except asyncio.CancelledError:
                # Write final summary on shutdown
                await self.write_session_summary()
                raise
            except Exception as exc:
                logger.error("Logbook summary error: %s", exc, exc_info=True)

    # ── Quick stats for Telegram / console ───────────────────────────────

    def today_stats(self) -> Dict[str, Any]:
        trades = [t for t in self._trades if t.success]
        return {
            "total_trades":    len(trades),
            "paper_trades":    sum(1 for t in trades if t.mode == "PAPER"),
            "live_trades":     sum(1 for t in trades if t.mode == "LIVE"),
            "avg_slippage_bps": round(
                sum(t.slippage_bps for t in trades) / max(len(trades), 1), 2
            ),
            "total_signals":   len(self._signals),
            "actionable":      sum(1 for s in self._signals if s.is_actionable),
        }
