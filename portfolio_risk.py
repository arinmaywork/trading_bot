"""
portfolio_risk.py  — R-13
=========================
Portfolio-level loss limits and consecutive-loss guard.

Design reference: SENTISTACK_RETIREMENT_PLAN §3.1.

This is the single most important risk protection in the whole system. Its
job is to save the account from the mistakes that have not been found yet.

Windows and actions:
    DAILY:    -2% of TOTAL_CAPITAL  → HALT new entries until next 09:15 IST
    WEEKLY:   -5% of TOTAL_CAPITAL  → HALT new entries until next Monday 09:15
    MONTHLY:  -8% of TOTAL_CAPITAL  → FULL HALT + PAPER_MONITOR mode, manual
                                       /resume required
    CONSECUTIVE LOSSES: N in a row per symbol → blacklist symbol for session

Halt actions are expressed by setting BotState fields:
    state.no_new_buys   = True
    state.no_new_sells  = True
    state.risk_halt_scope   ∈ {"NONE", "DAY", "WEEK", "MONTH"}
    state.risk_halt_reason  = human-readable
    state.risk_halt_until   = datetime when halt auto-lifts (None for MONTH)
    state.risk_blacklist    = set of symbols to skip
    state.mode              = TradingMode.PAPER_MONITOR  (only for MONTH halt)

strategy_loop in main.py checks these flags before placing orders and before
executing `StrategyEngine.evaluate()` entries for blacklisted symbols.

P&L is computed by FIFO-matching BUY/SELL legs from the daily trade CSVs
written by logbook.py. The cost model matches `Logbook.get_pnl_report`.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

from config import settings

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
LOG_DIR = Path(__file__).parent / "logs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ist_now() -> datetime:
    return datetime.now(IST)


def _trade_csv_path(day: date) -> Path:
    return LOG_DIR / f"trades_{day.strftime('%Y-%m-%d')}.csv"


def _load_trades_for_day(day: date) -> List[Dict[str, str]]:
    """Read one day's trades CSV. Returns [] if missing or malformed."""
    p = _trade_csv_path(day)
    if not p.exists():
        return []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return [row for row in csv.DictReader(f)]
    except Exception as exc:
        logger.warning("portfolio_risk: could not read %s (%s)", p.name, exc)
        return []


def _trade_cost(row: Dict[str, str]) -> float:
    """Per-trade cost model — delegates to canonical backtest.cost_model."""
    try:
        price = float(row.get("fill_price", 0))
        qty   = int(float(row.get("qty", 0)))
    except (ValueError, TypeError):
        return 0.0
    if price <= 0 or qty <= 0:
        return 0.0
    direction = (row.get("direction") or "BUY").upper()
    product   = (row.get("product_type") or "MIS").upper()
    if direction not in ("BUY", "SELL"):
        direction = "BUY"
    if product not in ("MIS", "CNC"):
        product = "MIS"
    try:
        from backtest.cost_model import leg_cost
        return float(leg_cost(price, qty, direction, product).total)  # type: ignore[arg-type]
    except Exception:
        # Fallback: brokerage + exchange only (should never hit in practice)
        cfg = settings.strategy
        order_val = price * qty
        brokerage = min(cfg.BROKERAGE_PER_ORDER, cfg.BROKERAGE_PCT * order_val)
        exchange  = cfg.EXCHANGE_CHARGE_RATE * order_val
        return brokerage + exchange


def _fifo_net_pnl(rows: List[Dict[str, str]]) -> float:
    """
    FIFO-match BUY/SELL legs across symbols and return net P&L (gross − cost).
    Unmatched legs are ignored (treated as open positions).
    """
    if not rows:
        return 0.0

    by_symbol: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        if (r.get("success") or "True") != "True":
            continue
        by_symbol.setdefault(r.get("symbol", ""), []).append(r)

    gross = 0.0
    total_cost = 0.0

    for sym, sym_rows in by_symbol.items():
        # Sort by timestamp so FIFO is meaningful across the day
        sym_rows.sort(key=lambda x: x.get("timestamp", ""))
        buys  = [(int(float(r["qty"])), float(r["fill_price"])) for r in sym_rows
                 if (r.get("direction") or "").upper() == "BUY"]
        sells = [(int(float(r["qty"])), float(r["fill_price"])) for r in sym_rows
                 if (r.get("direction") or "").upper() == "SELL"]

        bi = si = 0
        b_rem = buys[0][0]  if buys  else 0
        s_rem = sells[0][0] if sells else 0
        while bi < len(buys) and si < len(sells):
            matched = min(b_rem, s_rem)
            gross += matched * (sells[si][1] - buys[bi][1])
            b_rem -= matched
            s_rem -= matched
            if b_rem == 0:
                bi += 1
                b_rem = buys[bi][0] if bi < len(buys) else 0
            if s_rem == 0:
                si += 1
                s_rem = sells[si][0] if si < len(sells) else 0

        total_cost += sum(_trade_cost(r) for r in sym_rows)

    return round(gross - total_cost, 2)


def _per_symbol_roundtrip_pnl(rows: List[Dict[str, str]]) -> Dict[str, List[float]]:
    """
    Return ordered list of round-trip net P&Ls per symbol (for
    consecutive-loss detection). Each entry is (sell_price - buy_price)*qty
    minus the cost of both legs.
    """
    cfg = settings.strategy
    out: Dict[str, List[float]] = {}
    by_symbol: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        if (r.get("success") or "True") != "True":
            continue
        by_symbol.setdefault(r.get("symbol", ""), []).append(r)

    for sym, sym_rows in by_symbol.items():
        sym_rows.sort(key=lambda x: x.get("timestamp", ""))
        buys:  List[Tuple[int, float, Dict[str, str]]] = []
        sells: List[Tuple[int, float, Dict[str, str]]] = []
        for r in sym_rows:
            try:
                q = int(float(r.get("qty", 0)))
                p = float(r.get("fill_price", 0))
            except (ValueError, TypeError):
                continue
            if (r.get("direction") or "").upper() == "BUY":
                buys.append((q, p, r))
            else:
                sells.append((q, p, r))

        rt_list: List[float] = []
        bi = si = 0
        b_rem = buys[0][0]  if buys  else 0
        s_rem = sells[0][0] if sells else 0
        while bi < len(buys) and si < len(sells):
            matched   = min(b_rem, s_rem)
            pnl_gross = matched * (sells[si][1] - buys[bi][1])
            # Attribute a proportional share of leg cost to this match.
            b_share = _trade_cost(buys[bi][2])  * (matched / max(buys[bi][0],  1))
            s_share = _trade_cost(sells[si][2]) * (matched / max(sells[si][0], 1))
            rt_list.append(pnl_gross - b_share - s_share)
            b_rem -= matched
            s_rem -= matched
            if b_rem == 0:
                bi += 1
                b_rem = buys[bi][0] if bi < len(buys) else 0
            if s_rem == 0:
                si += 1
                s_rem = sells[si][0] if si < len(sells) else 0

        if rt_list:
            out[sym] = rt_list
    return out


def _current_week_start(now_ist: datetime) -> date:
    """Monday of the current IST week."""
    d = now_ist.date()
    return d - timedelta(days=d.weekday())


def _current_month_start(now_ist: datetime) -> date:
    return now_ist.date().replace(day=1)


def _next_session_start(now_ist: datetime) -> datetime:
    """Next trading-day 09:15 IST (skips weekends; does not honour holidays)."""
    candidate = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    if now_ist >= candidate:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:   # Sat/Sun
        candidate += timedelta(days=1)
    return candidate


def _next_monday_915(now_ist: datetime) -> datetime:
    monday = now_ist.date() + timedelta(days=(7 - now_ist.weekday()))
    return datetime.combine(monday, time(9, 15), IST)


# ---------------------------------------------------------------------------
# Task-5: sector map loader + exposure / MTM helpers
# ---------------------------------------------------------------------------

_SECTOR_MAP_CACHE: Optional[Dict[str, str]] = None
_SECTOR_MAP_MTIME: float = 0.0
_UNKNOWN_SECTOR = "Unknown"


def _sector_map_path() -> Path:
    """Resolve the sector map CSV relative to the project root."""
    raw = settings.strategy.SECTOR_MAP_CSV
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    return p


def load_sector_map(force_reload: bool = False) -> Dict[str, str]:
    """
    Read ``data/nse_sector_map.csv`` → ``{symbol: sector}``.

    Cached on module load and auto-refreshed whenever the file's mtime
    advances, so a running bot picks up sector edits without restart.
    Returns an empty dict on read failure (caller falls back to the
    ``Unknown`` bucket for every symbol).
    """
    global _SECTOR_MAP_CACHE, _SECTOR_MAP_MTIME
    path = _sector_map_path()
    if not path.exists():
        if _SECTOR_MAP_CACHE is None:
            logger.warning("sector map not found at %s — using 'Unknown' bucket", path)
            _SECTOR_MAP_CACHE = {}
        return _SECTOR_MAP_CACHE

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0

    if (
        not force_reload
        and _SECTOR_MAP_CACHE is not None
        and mtime == _SECTOR_MAP_MTIME
    ):
        return _SECTOR_MAP_CACHE

    out: Dict[str, str] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = (row.get("symbol") or "").strip().upper()
                sec = (row.get("sector") or "").strip()
                if sym and sec:
                    out[sym] = sec
    except Exception as exc:
        logger.warning("sector map read failed (%s): %s", path.name, exc)
        return _SECTOR_MAP_CACHE or {}

    _SECTOR_MAP_CACHE = out
    _SECTOR_MAP_MTIME = mtime
    logger.info("Loaded %d symbols from sector map %s", len(out), path.name)
    return out


def sector_for(symbol: str) -> str:
    """Lookup helper. Unknown symbols bucket into ``'Unknown'``."""
    return load_sector_map().get(symbol.upper(), _UNKNOWN_SECTOR)


def compute_sector_exposure(
    positions: Mapping[str, int],
    ltp_map:   Mapping[str, float],
) -> Dict[str, float]:
    """
    Convert ``{symbol: net_qty}`` + ``{symbol: ltp}`` into
    ``{sector: notional_₹}`` where notional = abs(qty) × ltp.

    Shorts count toward the sector's gross exposure (``abs(qty)``) because
    a correlated-sector crash hurts longs and shorts asymmetrically — a
    single sector concentration is a problem either way.
    """
    out: Dict[str, float] = {}
    for sym, qty in positions.items():
        if not qty:
            continue
        ltp = float(ltp_map.get(sym, 0.0))
        if ltp <= 0:
            continue
        sec   = sector_for(sym)
        notl  = abs(int(qty)) * ltp
        out[sec] = out.get(sec, 0.0) + notl
    return out


def compute_unrealised_pnl(
    positions:      Mapping[str, int],
    entry_prices:   Mapping[str, float],
    ltp_map:        Mapping[str, float],
) -> float:
    """Sum of (ltp - entry) × qty across open positions (longs positive)."""
    total = 0.0
    for sym, qty in positions.items():
        if not qty:
            continue
        ltp   = float(ltp_map.get(sym, 0.0))
        entry = float(entry_prices.get(sym, 0.0))
        if ltp <= 0 or entry <= 0:
            continue
        total += (ltp - entry) * int(qty)
    return total


# ---------------------------------------------------------------------------
# Risk budget result
# ---------------------------------------------------------------------------

@dataclass
class RiskBudget:
    day_pnl:         float
    week_pnl:        float
    month_pnl:       float
    day_limit_inr:   float
    week_limit_inr:  float
    month_limit_inr: float
    blacklisted:     Set[str]
    halt_scope:      str       # "NONE" | "DAY" | "WEEK" | "MONTH" | "MTM"
    halt_reason:     str
    halt_until:      Optional[datetime]
    # Task-5 additions — do NOT participate in the escalation scope rank;
    # they are orthogonal guards that the strategy loop consults in parallel.
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    sector_cap_inr:  float            = 0.0
    sectors_blocked: Set[str]         = field(default_factory=set)
    mtm_pnl:         float            = 0.0
    mtm_limit_inr:   float            = 0.0
    mtm_stop_active: bool             = False

    @property
    def day_used_pct(self) -> float:
        return 0.0 if self.day_limit_inr <= 0 else min(
            1.0, max(0.0, -self.day_pnl / self.day_limit_inr)
        )

    @property
    def week_used_pct(self) -> float:
        return 0.0 if self.week_limit_inr <= 0 else min(
            1.0, max(0.0, -self.week_pnl / self.week_limit_inr)
        )

    @property
    def month_used_pct(self) -> float:
        return 0.0 if self.month_limit_inr <= 0 else min(
            1.0, max(0.0, -self.month_pnl / self.month_limit_inr)
        )


# ---------------------------------------------------------------------------
# PortfolioRiskMonitor
# ---------------------------------------------------------------------------

class PortfolioRiskMonitor:
    """
    Evaluates the three loss-window budgets + consecutive-loss guard.
    Holds no mutable state of its own — writes results into BotState so
    the strategy loop can read them on every cycle.
    """

    def __init__(self, capital: float) -> None:
        self._capital = float(capital)
        self._last_check_ts: float = 0.0
        # Task-5: intraday MTM stop latches to True the first time the stop
        # fires each session. It resets at the next 09:15 IST open, the same
        # way a DAY-window halt auto-lifts.
        self._mtm_stop_latched_until: Optional[datetime] = None

    def update_capital(self, capital: float) -> None:
        if capital > 0:
            self._capital = float(capital)

    # ── Task-5: sector exposure + MTM drawdown stop ──────────────────────

    def evaluate_sector_exposure(
        self,
        positions: Mapping[str, int],
        ltp_map:   Mapping[str, float],
    ) -> Tuple[Dict[str, float], float, Set[str]]:
        """
        Returns (sector_exposure_₹, sector_cap_₹, blocked_sectors).

        ``blocked_sectors`` is the set of sectors that are currently AT or
        OVER the cap — new BUY entries in those sectors should be rejected
        by the strategy loop's pre-trade check.
        """
        cfg = settings.strategy
        exposure = compute_sector_exposure(positions, ltp_map)
        cap_inr  = cfg.MAX_SECTOR_EXPOSURE_PCT * self._capital
        if not cfg.SECTOR_CAP_ENABLED or cap_inr <= 0:
            return exposure, cap_inr, set()
        blocked = {s for s, notl in exposure.items() if notl >= cap_inr}
        return exposure, cap_inr, blocked

    def would_breach_sector_cap(
        self,
        symbol:         str,
        additional_qty: int,
        price:          float,
        positions:      Mapping[str, int],
        ltp_map:        Mapping[str, float],
    ) -> Tuple[bool, str]:
        """
        Pre-trade check: would placing an ``additional_qty`` order for
        ``symbol`` at ``price`` push its sector above the cap?

        Returns ``(blocked, reason)`` — the strategy loop calls this BEFORE
        the order is submitted. If the sector cap is disabled or capital is
        not yet known, this is a no-op that always returns ``(False, "")``.
        """
        cfg = settings.strategy
        if not cfg.SECTOR_CAP_ENABLED:
            return False, ""
        cap_inr = cfg.MAX_SECTOR_EXPOSURE_PCT * self._capital
        if cap_inr <= 0 or price <= 0 or additional_qty <= 0:
            return False, ""
        sector    = sector_for(symbol)
        existing  = compute_sector_exposure(positions, ltp_map)
        current   = existing.get(sector, 0.0)
        prospective = current + abs(int(additional_qty)) * float(price)
        if prospective > cap_inr:
            reason = (
                f"Sector cap breach: {sector} would reach "
                f"₹{prospective:,.0f} > ₹{cap_inr:,.0f} "
                f"({cfg.MAX_SECTOR_EXPOSURE_PCT:.0%} of ₹{self._capital:,.0f})"
            )
            return True, reason
        return False, ""

    def evaluate_mtm_stop(
        self,
        realised_day_pnl: float,
        unrealised_pnl:   float,
        now_ist:          Optional[datetime] = None,
    ) -> Tuple[float, float, bool, Optional[datetime]]:
        """
        Returns ``(mtm_pnl, mtm_limit_inr, stop_active, auto_clear_at)``.

        ``mtm_pnl`` is realised + unrealised P&L for the session. Once it
        crosses below the ``-INTRADAY_MTM_STOP_PCT × capital`` threshold the
        stop latches ON and stays ON until the next session 09:15 IST, so a
        brief mid-day rebound past the threshold does not re-enable entries.
        """
        cfg = settings.strategy
        now_ist = now_ist or _ist_now()
        mtm_pnl  = float(realised_day_pnl) + float(unrealised_pnl)
        limit    = cfg.INTRADAY_MTM_STOP_PCT * self._capital

        if not cfg.INTRADAY_MTM_STOP_ENABLED or limit <= 0:
            # Disabled → report numbers for telemetry but never latch.
            return mtm_pnl, limit, False, None

        # Auto-clear at session rollover
        if (
            self._mtm_stop_latched_until is not None
            and now_ist >= self._mtm_stop_latched_until
        ):
            self._mtm_stop_latched_until = None

        # Latch when we first cross below the threshold
        if mtm_pnl <= -limit and self._mtm_stop_latched_until is None:
            self._mtm_stop_latched_until = _next_session_start(now_ist)
            logger.warning(
                "Task-5 MTM stop LATCHED: mtm=₹%+.0f ≤ -₹%.0f (%.1f%% of ₹%.0f). "
                "Auto-clears at %s",
                mtm_pnl, limit, cfg.INTRADAY_MTM_STOP_PCT * 100, self._capital,
                self._mtm_stop_latched_until.strftime("%a %H:%M IST"),
            )

        stop_active = self._mtm_stop_latched_until is not None
        return mtm_pnl, limit, stop_active, self._mtm_stop_latched_until

    # ── Core budget computation ──────────────────────────────────────────

    def compute_budget(
        self,
        now_ist:      Optional[datetime]       = None,
        positions:    Optional[Mapping[str, int]]   = None,
        ltp_map:      Optional[Mapping[str, float]] = None,
        entry_prices: Optional[Mapping[str, float]] = None,
    ) -> RiskBudget:
        now_ist = now_ist or _ist_now()
        cfg = settings.strategy
        positions    = positions    or {}
        ltp_map      = ltp_map      or {}
        entry_prices = entry_prices or {}

        # Load trades for the three windows
        today_rows = _load_trades_for_day(now_ist.date())

        week_rows: List[Dict[str, str]] = []
        d = _current_week_start(now_ist)
        while d <= now_ist.date():
            week_rows.extend(_load_trades_for_day(d))
            d += timedelta(days=1)

        month_rows: List[Dict[str, str]] = []
        d = _current_month_start(now_ist)
        while d <= now_ist.date():
            month_rows.extend(_load_trades_for_day(d))
            d += timedelta(days=1)

        day_pnl   = _fifo_net_pnl(today_rows)
        week_pnl  = _fifo_net_pnl(week_rows)
        month_pnl = _fifo_net_pnl(month_rows)

        day_limit_inr   = cfg.DAILY_LOSS_LIMIT_PCT   * self._capital
        week_limit_inr  = cfg.WEEKLY_LOSS_LIMIT_PCT  * self._capital
        month_limit_inr = cfg.MONTHLY_LOSS_LIMIT_PCT * self._capital

        # Consecutive-loss blacklist from today's trades only
        blacklist: Set[str] = set()
        rt_pnls = _per_symbol_roundtrip_pnl(today_rows)
        for sym, pnls in rt_pnls.items():
            # Count trailing losses
            trail = 0
            for p in reversed(pnls):
                if p < 0:
                    trail += 1
                else:
                    break
            if trail >= cfg.MAX_CONSECUTIVE_LOSSES:
                blacklist.add(sym)

        # Halt scope — strictest wins
        halt_scope = "NONE"
        halt_reason = ""
        halt_until: Optional[datetime] = None

        if month_pnl <= -month_limit_inr:
            halt_scope  = "MONTH"
            halt_reason = (
                f"Month P&L ₹{month_pnl:+,.0f} ≤ limit -₹{month_limit_inr:,.0f} "
                f"({cfg.MONTHLY_LOSS_LIMIT_PCT:.0%} of ₹{self._capital:,.0f}). "
                "Full halt. Manual /resume required."
            )
            halt_until = None       # manual resume only
        elif week_pnl <= -week_limit_inr:
            halt_scope  = "WEEK"
            halt_reason = (
                f"Week P&L ₹{week_pnl:+,.0f} ≤ limit -₹{week_limit_inr:,.0f} "
                f"({cfg.WEEKLY_LOSS_LIMIT_PCT:.0%}). "
                "Halted until Monday 09:15 IST."
            )
            halt_until = _next_monday_915(now_ist)
        elif day_pnl <= -day_limit_inr:
            halt_scope  = "DAY"
            halt_reason = (
                f"Day P&L ₹{day_pnl:+,.0f} ≤ limit -₹{day_limit_inr:,.0f} "
                f"({cfg.DAILY_LOSS_LIMIT_PCT:.0%}). "
                "Halted until next session 09:15 IST."
            )
            halt_until = _next_session_start(now_ist)

        # Task-5: sector exposure snapshot (based on live open positions)
        sector_exposure, sector_cap_inr, sectors_blocked = self.evaluate_sector_exposure(
            positions=positions, ltp_map=ltp_map,
        )

        # Task-5: intraday MTM drawdown stop.
        # realised_day_pnl is today's FIFO-matched net P&L from closed trades
        # (what `day_pnl` already measures). Unrealised = open MTM from the
        # current positions dict.
        unrealised_pnl = compute_unrealised_pnl(
            positions=positions, entry_prices=entry_prices, ltp_map=ltp_map,
        )
        mtm_pnl, mtm_limit_inr, mtm_stop_active, mtm_until = self.evaluate_mtm_stop(
            realised_day_pnl = day_pnl,
            unrealised_pnl   = unrealised_pnl,
            now_ist          = now_ist,
        )

        # If MTM stop fires AND no stricter window halt is already active,
        # promote the halt scope so `apply_to_state` will flip the entry
        # guards. MTM slots between NONE and DAY in the escalation ladder:
        # it blocks new entries but auto-clears at the next session, exactly
        # like a DAY halt.
        if mtm_stop_active and halt_scope == "NONE":
            halt_scope  = "MTM"
            halt_reason = (
                f"Intraday MTM stop: realised₹{day_pnl:+,.0f} + "
                f"unrealised₹{unrealised_pnl:+,.0f} = ₹{mtm_pnl:+,.0f} "
                f"≤ -₹{mtm_limit_inr:,.0f} "
                f"({cfg.INTRADAY_MTM_STOP_PCT:.1%} of ₹{self._capital:,.0f}). "
                "Halted until next session 09:15 IST."
            )
            halt_until = mtm_until

        return RiskBudget(
            day_pnl         = day_pnl,
            week_pnl        = week_pnl,
            month_pnl       = month_pnl,
            day_limit_inr   = day_limit_inr,
            week_limit_inr  = week_limit_inr,
            month_limit_inr = month_limit_inr,
            blacklisted     = blacklist,
            halt_scope      = halt_scope,
            halt_reason     = halt_reason,
            halt_until      = halt_until,
            sector_exposure = sector_exposure,
            sector_cap_inr  = sector_cap_inr,
            sectors_blocked = sectors_blocked,
            mtm_pnl         = mtm_pnl,
            mtm_limit_inr   = mtm_limit_inr,
            mtm_stop_active = mtm_stop_active,
        )

    # ── State sync ────────────────────────────────────────────────────────

    def apply_to_state(self, state, budget: RiskBudget) -> Tuple[bool, str]:
        """
        Write risk budget into BotState and set the trade guards.
        Returns (halt_state_changed, change_reason) so main.py can fire a
        Telegram alert only on transitions.
        """
        cfg = settings.strategy
        prev_scope = state.risk_halt_scope
        prev_black = set(state.risk_blacklist)

        # Always cache the raw numbers for /risk
        state.risk_day_pnl     = budget.day_pnl
        state.risk_week_pnl    = budget.week_pnl
        state.risk_month_pnl   = budget.month_pnl
        state.risk_day_limit   = budget.day_limit_inr
        state.risk_week_limit  = budget.week_limit_inr
        state.risk_month_limit = budget.month_limit_inr
        state.risk_blacklist   = budget.blacklisted
        state.risk_last_check  = _ist_now().strftime("%H:%M:%S")

        # Task-5: always mirror the sector + MTM telemetry into state so the
        # /risk command and strategy pre-trade checks can see them even when
        # auto-halt is off.
        try:
            state.risk_sector_exposure = dict(budget.sector_exposure)
            state.risk_sector_cap_inr  = budget.sector_cap_inr
            state.risk_sector_blocked  = set(budget.sectors_blocked)
            state.risk_mtm_pnl         = budget.mtm_pnl
            state.risk_mtm_limit       = budget.mtm_limit_inr
            state.risk_mtm_stop_active = budget.mtm_stop_active
        except AttributeError:
            # Older BotState without Task-5 fields — ignore silently.
            pass

        if not cfg.RISK_ENABLE_AUTOHALT:
            # Dry-run mode: record numbers but do not set halt flags
            state.risk_halt_scope  = "NONE"
            state.risk_halt_reason = "AUTOHALT disabled (dry run)"
            state.risk_halt_until  = None
            return (False, "")

        changed = False
        reason  = ""

        # Auto-lift expired halts (MTM/day/week only — MONTH is manual)
        now = _ist_now()
        if (
            state.risk_halt_until is not None
            and now >= state.risk_halt_until
            and state.risk_halt_scope in ("MTM", "DAY", "WEEK")
            and budget.halt_scope == "NONE"
        ):
            state.risk_halt_scope  = "NONE"
            state.risk_halt_reason = ""
            state.risk_halt_until  = None
            state.no_new_buys      = False
            state.no_new_sells     = False
            changed = True
            reason  = f"Halt auto-lifted ({prev_scope} window expired)"

        # New halt (escalation only — never downgrade until expiry).
        # MTM sits between NONE and DAY: it blocks new entries and auto-clears
        # at the next session open, but any DAY/WEEK/MONTH halt supersedes it.
        scope_rank = {"NONE": 0, "MTM": 1, "DAY": 2, "WEEK": 3, "MONTH": 4}
        if scope_rank[budget.halt_scope] > scope_rank[state.risk_halt_scope]:
            state.risk_halt_scope  = budget.halt_scope
            state.risk_halt_reason = budget.halt_reason
            state.risk_halt_until  = budget.halt_until
            state.no_new_buys      = True
            state.no_new_sells     = True
            if budget.halt_scope == "MONTH":
                # Full halt + switch to paper-monitor mode
                try:
                    from telegram_controller import TradingMode
                    state.mode   = TradingMode.PAPER_MONITOR
                    state.paused = True
                except Exception:
                    pass
            changed = True
            reason  = budget.halt_reason

        # Blacklist delta (log new additions as separate alerts)
        new_black = budget.blacklisted - prev_black
        if new_black:
            changed = True
            reason  = (reason + " | " if reason else "") + (
                f"Blacklisted (consecutive losses ≥ "
                f"{cfg.MAX_CONSECUTIVE_LOSSES}): {', '.join(sorted(new_black))}"
            )

        return changed, reason

    # ── Convenience: one-shot check returning both the budget & change ──

    def check(
        self,
        state,
        positions:    Optional[Mapping[str, int]]   = None,
        ltp_map:      Optional[Mapping[str, float]] = None,
        entry_prices: Optional[Mapping[str, float]] = None,
    ) -> Tuple[RiskBudget, bool, str]:
        budget = self.compute_budget(
            positions    = positions,
            ltp_map      = ltp_map,
            entry_prices = entry_prices,
        )
        changed, reason = self.apply_to_state(state, budget)
        return budget, changed, reason


# ---------------------------------------------------------------------------
# Formatter used by the /risk Telegram command
# ---------------------------------------------------------------------------

def format_risk_report(state) -> str:
    def _bar(pct: float, width: int = 20) -> str:
        filled = int(round(pct * width))
        return "█" * filled + "·" * (width - filled)

    def _pct(used: float, limit: float) -> float:
        return 0.0 if limit <= 0 else min(1.0, max(0.0, used / limit))

    d_pct = _pct(-state.risk_day_pnl,   state.risk_day_limit)
    w_pct = _pct(-state.risk_week_pnl,  state.risk_week_limit)
    m_pct = _pct(-state.risk_month_pnl, state.risk_month_limit)

    if state.risk_halt_scope == "NONE":
        halt_line = "✅ <b>No halt active</b>"
    else:
        until = (
            state.risk_halt_until.strftime("%a %d %b %H:%M IST")
            if state.risk_halt_until else "MANUAL /resume required"
        )
        halt_line = (
            f"🛑 <b>HALT: {state.risk_halt_scope}</b>\n"
            f"   Reason: {state.risk_halt_reason}\n"
            f"   Resume: {until}"
        )

    blacklist_line = (
        f"🚷 Blacklisted today: {', '.join(sorted(state.risk_blacklist))}"
        if state.risk_blacklist else "🚷 Blacklisted today: none"
    )

    # ── Task-5: Sector exposure panel ────────────────────────────────────
    sector_exposure = getattr(state, "risk_sector_exposure", {}) or {}
    sector_cap      = float(getattr(state, "risk_sector_cap_inr", 0.0) or 0.0)
    sectors_blocked = getattr(state, "risk_sector_blocked", set()) or set()
    if sector_exposure and sector_cap > 0:
        rows = sorted(sector_exposure.items(), key=lambda kv: -kv[1])
        lines = []
        for sec, notl in rows[:8]:   # show top 8 sectors only
            pct = min(1.0, max(0.0, notl / sector_cap))
            marker = "⛔" if sec in sectors_blocked else "  "
            lines.append(
                f"{marker} {sec:<16} ₹{notl:>10,.0f}  "
                f"<code>{_bar(pct, width=12)}</code> {pct:>4.0%}"
            )
        sector_panel = (
            f"\n<b>Sector exposure</b> (cap ₹{sector_cap:,.0f} each)\n"
            + "\n".join(lines)
        )
    else:
        sector_panel = "\n<b>Sector exposure</b>: (none open)"

    # ── Task-5: Intraday MTM panel ───────────────────────────────────────
    mtm_pnl        = float(getattr(state, "risk_mtm_pnl",   0.0) or 0.0)
    mtm_limit      = float(getattr(state, "risk_mtm_limit", 0.0) or 0.0)
    mtm_active     = bool(getattr(state, "risk_mtm_stop_active", False))
    if mtm_limit > 0:
        mtm_pct = _pct(-mtm_pnl, mtm_limit)
        mtm_flag = "🛑 LATCHED" if mtm_active else "✅ OK"
        mtm_panel = (
            f"\n<b>Intraday MTM</b>  ₹{mtm_pnl:+,.0f} / -₹{mtm_limit:,.0f}  {mtm_flag}\n"
            f"  <code>{_bar(mtm_pct)}</code> {mtm_pct:>5.0%}"
        )
    else:
        mtm_panel = ""

    return (
        "📊 <b>Portfolio Risk Budget</b>\n"
        f"{'─' * 32}\n"
        f"Last check: {state.risk_last_check or 'pending'}\n\n"
        f"<b>Day</b>    ₹{state.risk_day_pnl:+,.0f} / "
        f"-₹{state.risk_day_limit:,.0f}\n"
        f"  <code>{_bar(d_pct)}</code> {d_pct:>5.0%}\n"
        f"<b>Week</b>   ₹{state.risk_week_pnl:+,.0f} / "
        f"-₹{state.risk_week_limit:,.0f}\n"
        f"  <code>{_bar(w_pct)}</code> {w_pct:>5.0%}\n"
        f"<b>Month</b>  ₹{state.risk_month_pnl:+,.0f} / "
        f"-₹{state.risk_month_limit:,.0f}\n"
        f"  <code>{_bar(m_pct)}</code> {m_pct:>5.0%}"
        f"{mtm_panel}\n"
        f"{sector_panel}\n\n"
        f"{halt_line}\n"
        f"{blacklist_line}"
    )
