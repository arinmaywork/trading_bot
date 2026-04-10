"""
monitor.py — Task 7
===================
Slippage monitor + end-of-day Telegram digest.

Reads the daily trade CSVs written by ``logbook.py`` (``logs/trades_YYYY-MM-DD.csv``)
and produces two things:

1.  **DigestStats** — a plain-data struct summarising the session:
        - trade count, win rate, realised P&L, cost drag
        - unrealised P&L from open positions (needs a live ltp_map)
        - biggest winner / biggest loser by per-trade net P&L
        - average slippage in basis points (entry + exit)
        - the 10-day rolling-baseline slippage, and whether today is
          "DEGRADING" (today_bps − baseline_bps ≥ SLIPPAGE_ALERT_DELTA_BPS).

2.  ``format_digest`` — an HTML-safe Telegram string.

The monitor is stateless: each call re-reads the CSVs from disk, so it is
safe to fire from the strategy loop once per trading day.

Firing policy: strategy_loop (see main.py) calls ``maybe_fire_digest``
every cycle after 15:25 IST; the first call after 15:25 IST sends the
digest and latches a per-day flag so we do not spam.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

IST      = timezone(timedelta(hours=5, minutes=30))
LOG_DIR  = Path(__file__).parent / "logs"

# Rolling window size for the slippage baseline.
SLIPPAGE_BASELINE_DAYS      = 10
SLIPPAGE_ALERT_DELTA_BPS    = 5.0   # today - baseline ≥ 5 bps → "DEGRADING"
SLIPPAGE_MIN_TRADES_FOR_SIG = 5     # need this many trades to trust the signal


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _trade_csv_path(day: date) -> Path:
    return LOG_DIR / f"trades_{day.strftime('%Y-%m-%d')}.csv"


def _load_rows(day: date) -> List[Dict[str, str]]:
    p = _trade_csv_path(day)
    if not p.exists():
        return []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if (r.get("success") or "True") == "True"]
    except Exception as exc:
        logger.warning("monitor: could not read %s (%s)", p.name, exc)
        return []


def _prev_trading_days(end: date, n: int) -> List[date]:
    """Return up to ``n`` dates strictly before ``end`` skipping Sat/Sun."""
    out: List[date] = []
    d = end
    while len(out) < n:
        d = d - timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Slippage computation
# ---------------------------------------------------------------------------
def compute_session_slippage_bps(rows: Iterable[Dict[str, str]]) -> Tuple[float, int]:
    """
    Qty-weighted average |slippage_bps| across all rows for one day.

    Using absolute value so BUY +12 bps and SELL −12 bps both count as
    execution friction rather than cancelling out. Returns
    ``(avg_abs_bps, n_rows_considered)``.
    """
    total_abs_bps_qty = 0.0
    total_qty         = 0
    n = 0
    for r in rows:
        qty    = _safe_int(r.get("qty"))
        bps    = _safe_float(r.get("slippage_bps"))
        if qty <= 0:
            continue
        total_abs_bps_qty += abs(bps) * qty
        total_qty         += qty
        n += 1
    if total_qty == 0:
        return 0.0, 0
    return total_abs_bps_qty / total_qty, n


def rolling_slippage_baseline(
    end_day: date,
    window_days: int = SLIPPAGE_BASELINE_DAYS,
) -> Tuple[float, int]:
    """
    Average qty-weighted |slippage_bps| over the last ``window_days``
    trading days STRICTLY before ``end_day``. Days with no trades are
    skipped (they don't pull the mean). Returns ``(avg_bps, n_days)``.
    """
    bps_samples: List[Tuple[float, int]] = []
    for d in _prev_trading_days(end_day, window_days):
        rows = _load_rows(d)
        if not rows:
            continue
        avg, n_rows = compute_session_slippage_bps(rows)
        if n_rows >= 1:
            bps_samples.append((avg, n_rows))
    if not bps_samples:
        return 0.0, 0
    total_w = sum(n for _, n in bps_samples)
    if total_w == 0:
        return 0.0, 0
    weighted = sum(bps * n for bps, n in bps_samples) / total_w
    return weighted, len(bps_samples)


# ---------------------------------------------------------------------------
# FIFO P&L per closed round-trip (shared cost model from portfolio_risk)
# ---------------------------------------------------------------------------
def _roundtrip_pnls(rows: List[Dict[str, str]]) -> Dict[str, List[Tuple[float, float]]]:
    """
    FIFO-match BUY/SELL legs and return, per symbol, an ordered list of
    ``(net_pnl, entry_price)`` tuples. Uses the same cost model as
    logbook via the shared helper.
    """
    try:
        from portfolio_risk import _trade_cost
    except Exception:                      # pragma: no cover
        def _trade_cost(_r):                # type: ignore[misc]
            return 0.0

    by_symbol: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_symbol.setdefault(r.get("symbol", ""), []).append(r)

    out: Dict[str, List[Tuple[float, float]]] = {}
    for sym, sym_rows in by_symbol.items():
        sym_rows.sort(key=lambda x: x.get("timestamp", ""))
        buys  = [r for r in sym_rows if (r.get("direction") or "").upper() == "BUY"]
        sells = [r for r in sym_rows if (r.get("direction") or "").upper() == "SELL"]

        bi = si = 0
        b_rem = _safe_int(buys[0].get("qty"))  if buys  else 0
        s_rem = _safe_int(sells[0].get("qty")) if sells else 0
        trips: List[Tuple[float, float]] = []
        while bi < len(buys) and si < len(sells):
            matched = min(b_rem, s_rem)
            bpx = _safe_float(buys[bi].get("fill_price"))
            spx = _safe_float(sells[si].get("fill_price"))
            gross = matched * (spx - bpx)
            b_share = _trade_cost(buys[bi])  * (matched / max(_safe_int(buys[bi].get("qty")),  1))
            s_share = _trade_cost(sells[si]) * (matched / max(_safe_int(sells[si].get("qty")), 1))
            trips.append((gross - b_share - s_share, bpx))
            b_rem -= matched
            s_rem -= matched
            if b_rem == 0:
                bi += 1
                b_rem = _safe_int(buys[bi].get("qty")) if bi < len(buys) else 0
            if s_rem == 0:
                si += 1
                s_rem = _safe_int(sells[si].get("qty")) if si < len(sells) else 0
        if trips:
            out[sym] = trips
    return out


# ---------------------------------------------------------------------------
# DigestStats
# ---------------------------------------------------------------------------
@dataclass
class DigestStats:
    day: date
    n_trades:            int   = 0                # CSV rows (legs)
    n_round_trips:       int   = 0                # closed BUY/SELL pairs
    win_rate:            float = 0.0              # 0..1 over round-trips
    realised_pnl:        float = 0.0              # net, after costs
    gross_pnl:           float = 0.0
    total_costs:         float = 0.0
    unrealised_pnl:      float = 0.0
    biggest_winner:      Tuple[str, float] = ("", 0.0)
    biggest_loser:       Tuple[str, float] = ("", 0.0)
    avg_slippage_bps:    float = 0.0
    baseline_slippage:   float = 0.0
    baseline_days:       int   = 0
    slippage_status:     str   = "NORMAL"         # NORMAL | DEGRADING | INSUFFICIENT
    cost_drag_bps:       float = 0.0              # total_costs / turnover × 10_000


def build_digest(
    day:           Optional[date]                  = None,
    positions:     Optional[Mapping[str, int]]     = None,
    entry_prices:  Optional[Mapping[str, float]]   = None,
    ltp_map:       Optional[Mapping[str, float]]   = None,
) -> DigestStats:
    day = day or datetime.now(IST).date()
    rows = _load_rows(day)
    stats = DigestStats(day=day)
    stats.n_trades = len(rows)

    if not rows:
        # Still compute unrealised P&L if positions are supplied
        stats.unrealised_pnl = _compute_unrealised(positions, entry_prices, ltp_map)
        baseline, n_days = rolling_slippage_baseline(day)
        stats.baseline_slippage = round(baseline, 2)
        stats.baseline_days     = n_days
        stats.slippage_status   = "INSUFFICIENT"
        return stats

    # ── Round-trip P&L ──────────────────────────────────────────────────
    trips = _roundtrip_pnls(rows)
    flat: List[Tuple[str, float]] = []   # (symbol, net_pnl)
    for sym, pnls in trips.items():
        for pnl, _ in pnls:
            flat.append((sym, pnl))
    stats.n_round_trips = len(flat)
    if flat:
        stats.realised_pnl    = round(sum(p for _, p in flat), 2)
        stats.win_rate        = round(sum(1 for _, p in flat if p > 0) / len(flat), 4)
        winner = max(flat, key=lambda kv: kv[1])
        loser  = min(flat, key=lambda kv: kv[1])
        stats.biggest_winner = (winner[0], round(winner[1], 2))
        stats.biggest_loser  = (loser[0],  round(loser[1],  2))

    # ── Cost drag ───────────────────────────────────────────────────────
    try:
        from portfolio_risk import _trade_cost, _fifo_net_pnl
    except Exception:                                  # pragma: no cover
        _trade_cost = None
        _fifo_net_pnl = None
    if _trade_cost is not None:
        total_cost = sum(_trade_cost(r) for r in rows)
        turnover   = sum(_safe_float(r.get("fill_price")) * _safe_int(r.get("qty")) for r in rows)
        stats.total_costs   = round(total_cost, 2)
        stats.cost_drag_bps = round((total_cost / turnover) * 10_000, 2) if turnover > 0 else 0.0
        if _fifo_net_pnl is not None:
            stats.gross_pnl = round(_fifo_net_pnl(rows) + total_cost, 2)

    # ── Slippage ────────────────────────────────────────────────────────
    today_bps, n_rows = compute_session_slippage_bps(rows)
    stats.avg_slippage_bps = round(today_bps, 2)
    baseline, n_days = rolling_slippage_baseline(day)
    stats.baseline_slippage = round(baseline, 2)
    stats.baseline_days     = n_days
    if n_rows < SLIPPAGE_MIN_TRADES_FOR_SIG or n_days == 0:
        stats.slippage_status = "INSUFFICIENT"
    elif today_bps - baseline >= SLIPPAGE_ALERT_DELTA_BPS:
        stats.slippage_status = "DEGRADING"
    else:
        stats.slippage_status = "NORMAL"

    # ── Unrealised P&L ──────────────────────────────────────────────────
    stats.unrealised_pnl = _compute_unrealised(positions, entry_prices, ltp_map)

    return stats


def _compute_unrealised(
    positions:    Optional[Mapping[str, int]],
    entry_prices: Optional[Mapping[str, float]],
    ltp_map:      Optional[Mapping[str, float]],
) -> float:
    if not positions:
        return 0.0
    entry_prices = entry_prices or {}
    ltp_map      = ltp_map      or {}
    total = 0.0
    for sym, qty in positions.items():
        if not qty:
            continue
        ltp   = _safe_float(ltp_map.get(sym, 0.0))
        entry = _safe_float(entry_prices.get(sym, 0.0))
        if ltp <= 0 or entry <= 0:
            continue
        total += (ltp - entry) * int(qty)
    return round(total, 2)


# ---------------------------------------------------------------------------
# Telegram formatter
# ---------------------------------------------------------------------------
def format_digest(stats: DigestStats) -> str:
    flag = {
        "NORMAL":       "🟢",
        "DEGRADING":    "🔴",
        "INSUFFICIENT": "⚪",
    }.get(stats.slippage_status, "⚪")

    win_loss_line = ""
    if stats.biggest_winner[0] or stats.biggest_loser[0]:
        win_loss_line = (
            f"🏆 Biggest winner: {stats.biggest_winner[0] or '—'}  "
            f"₹{stats.biggest_winner[1]:+,.0f}\n"
            f"💀 Biggest loser:  {stats.biggest_loser[0] or '—'}  "
            f"₹{stats.biggest_loser[1]:+,.0f}\n"
        )

    if stats.n_trades == 0:
        body = (
            "No trades today.\n\n"
            f"{flag} 10-day baseline slippage: {stats.baseline_slippage:.1f} bps "
            f"(over {stats.baseline_days} days)\n"
        )
    else:
        body = (
            f"Trades:      {stats.n_trades} legs  ({stats.n_round_trips} round-trips)\n"
            f"Win rate:    {stats.win_rate:.0%}\n"
            f"Realised:    ₹{stats.realised_pnl:+,.0f}   "
            f"(gross ₹{stats.gross_pnl:+,.0f} − costs ₹{stats.total_costs:,.0f})\n"
            f"Unrealised:  ₹{stats.unrealised_pnl:+,.0f}\n"
            f"{win_loss_line}\n"
            f"<b>Execution</b>\n"
            f"  Avg slippage: {stats.avg_slippage_bps:.1f} bps  {flag} {stats.slippage_status}\n"
            f"  10d baseline: {stats.baseline_slippage:.1f} bps "
            f"({stats.baseline_days} day{'s' if stats.baseline_days != 1 else ''})\n"
            f"  Cost drag:    {stats.cost_drag_bps:.1f} bps of turnover\n"
        )

    return (
        f"📈 <b>Daily Digest — {stats.day.strftime('%a %d %b %Y')}</b>\n"
        f"{'─' * 32}\n"
        f"{body}"
    )


# ---------------------------------------------------------------------------
# Scheduler — called every strategy-loop cycle
# ---------------------------------------------------------------------------
# Fire at 15:25 IST (5 minutes before auto square-off) so the digest
# reflects the full session including exits, but beats the end-of-day
# noise. Latches per-day to avoid double-send.

DIGEST_HOUR_IST   = 15
DIGEST_MINUTE_IST = 25


class DigestScheduler:
    """Stateful per-session latch so the digest fires exactly once per day."""

    def __init__(self) -> None:
        self._fired_on: Optional[date] = None

    def should_fire(self, now_ist: Optional[datetime] = None) -> bool:
        now = now_ist or datetime.now(IST)
        if self._fired_on == now.date():
            return False
        at_or_past = (
            now.hour > DIGEST_HOUR_IST
            or (now.hour == DIGEST_HOUR_IST and now.minute >= DIGEST_MINUTE_IST)
        )
        return bool(at_or_past)

    def mark_fired(self, on_day: Optional[date] = None) -> None:
        self._fired_on = on_day or datetime.now(IST).date()

    def reset(self) -> None:
        self._fired_on = None
