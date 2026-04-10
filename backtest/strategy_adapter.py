"""
backtest/strategy_adapter.py
============================

Bridges the live SentiStack risk engine (`strategy.RiskManager`) into the
offline backtester so historical validation uses the SAME sizing
mathematics as production — parametric Kelly with the Busseti VaR cap,
GRI gating, vol-spike decay, the R-10 cost filter, and the Task-1
bootstrap-mode envelope.

What this adapter does NOT bring along
--------------------------------------
The live StrategyEngine depends on a live Redis tick stream, ML feature
store, Gemini LLM calls, GRI feeds, and the news pipeline — replaying
all of that against history is the kind of gold-plating that would
blow Task 3's session budget for zero marginal validation value.

Instead the adapter substitutes the smallest possible stand-ins:

  • `ml_signal`  — a lightweight technical z-score built from the rolling
                   log-return mean, scaled to match the live ensemble's
                   output range (fwd_log_return × SIGNAL_SCALE). It is
                   deliberately NOT a "good" predictor — Task 4 will
                   replace it with a walk-forward calibrated threshold.
                   The point here is to feed the real RiskManager
                   something shaped like its production input.
  • `vol`        — annualised realised vol from the same rolling window.
  • `gri`        — `GeopoliticalRiskIndex.neutral()` (the same fallback
                   the live code uses when the GRI fetch fails).
  • `redis`      — an in-memory fake whose `lrange()` returns an empty
                   list; the live `get_return_distribution()` catches
                   the exception path and returns [] anyway, so the
                   RiskManager falls through to parametric-only
                   sizing exactly as it does when Redis is cold.

Portfolio-level guards
----------------------
A lightweight inline mirror of `PortfolioRiskMonitor` (`_BudgetTracker`)
applies the SAME D/W/M loss-limit percentages from
`settings.strategy.DAILY/WEEKLY/MONTHLY_LOSS_LIMIT_PCT` against the
backtest's own cumulative P&L. It also replicates the consecutive-loss
blacklist. The full monitor reads trades from disk CSVs via
`_load_trades_for_day`; rather than writing pretend CSVs into the live
logbook directory, the adapter tracks trades in memory and applies the
same rules. Parity with the live monitor is enforced by reading the
same config constants.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from geopolitical import GeopoliticalRiskIndex
from strategy import RiskManager

from .engine import Signal, Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fake Redis — RiskManager only calls `lrange` via get_return_distribution.
# Returning [] drives it down the parametric-only branch, matching the
# behaviour of a freshly-started bot before the feature store warms up.
# ---------------------------------------------------------------------------
class _FakeRedis:
    async def lrange(self, key: str, start: int, end: int) -> List[str]:
        return []


# ---------------------------------------------------------------------------
# Inline budget tracker — parity with portfolio_risk.PortfolioRiskMonitor
# for the backtest-relevant subset (loss limits + consecutive-loss blacklist).
# Uses the SAME config constants so any tightening in production is
# automatically reflected in backtests.
# ---------------------------------------------------------------------------
@dataclass
class _BudgetTracker:
    capital: float

    def __post_init__(self) -> None:
        self._day_pnl:   Dict[date, float]   = defaultdict(float)
        self._week_pnl:  Dict[date, float]   = defaultdict(float)  # keyed by Monday
        self._month_pnl: Dict[Tuple[int, int], float] = defaultdict(float)
        self._halt_scope: str = "NONE"
        self._halt_until_date: Optional[date] = None
        self._consecutive: Dict[str, int] = defaultdict(int)
        self._blacklist:   set[str]       = set()

    @staticmethod
    def _week_key(d: date) -> date:
        return d - pd.Timedelta(days=d.weekday()).to_pytimedelta()

    def is_halted(self, now: datetime) -> bool:
        if self._halt_scope == "NONE":
            return False
        if self._halt_scope == "MONTH":
            return True        # manual resume only in live; stays halted for rest of backtest
        if self._halt_until_date is None:
            return False
        return now.date() < self._halt_until_date

    def is_blacklisted(self, symbol: str) -> bool:
        return symbol in self._blacklist

    def record_trade(self, symbol: str, net_pnl: float, ts: datetime) -> None:
        d  = ts.date()
        wk = self._week_key(d)
        mo = (d.year, d.month)

        self._day_pnl[d]   += net_pnl
        self._week_pnl[wk] += net_pnl
        self._month_pnl[mo] += net_pnl

        # Consecutive-loss guard: tracked per symbol per calendar day only
        key = f"{symbol}:{d.isoformat()}"
        if net_pnl < 0:
            self._consecutive[key] += 1
            if self._consecutive[key] >= settings.strategy.MAX_CONSECUTIVE_LOSSES:
                self._blacklist.add(symbol)
        else:
            self._consecutive[key] = 0

        # Re-evaluate halt scope
        cfg = settings.strategy
        day_lim   = cfg.DAILY_LOSS_LIMIT_PCT   * self.capital
        week_lim  = cfg.WEEKLY_LOSS_LIMIT_PCT  * self.capital
        month_lim = cfg.MONTHLY_LOSS_LIMIT_PCT * self.capital

        if self._month_pnl[mo] <= -month_lim and self._halt_scope != "MONTH":
            self._halt_scope = "MONTH"
            self._halt_until_date = None
            logger.warning(
                "BACKTEST MONTH HALT: pnl=₹%.0f <= -₹%.0f",
                self._month_pnl[mo], month_lim,
            )
        elif self._week_pnl[wk] <= -week_lim and self._halt_scope not in ("MONTH", "WEEK"):
            self._halt_scope = "WEEK"
            # Next Monday
            days_ahead = 7 - d.weekday()
            self._halt_until_date = d + pd.Timedelta(days=days_ahead).to_pytimedelta()
        elif self._day_pnl[d] <= -day_lim and self._halt_scope == "NONE":
            self._halt_scope = "DAY"
            self._halt_until_date = d + pd.Timedelta(days=1).to_pytimedelta()

    def maybe_expire(self, now: datetime) -> None:
        """Lift expired DAY/WEEK halts at the new session start."""
        if self._halt_scope in ("DAY", "WEEK") and self._halt_until_date is not None:
            if now.date() >= self._halt_until_date:
                self._halt_scope = "NONE"
                self._halt_until_date = None


# ---------------------------------------------------------------------------
# Signal generator — scaled to match the live ensemble's output range.
# Task 4 will replace the threshold with a self-calibrating percentile.
# ---------------------------------------------------------------------------
class _TechnicalSignal:
    """
    Produces an `ml_signal` in roughly the same magnitude as the live
    ensemble. The live output is `fwd_log_return × SIGNAL_SCALE` where
    SIGNAL_SCALE = 20, so typical live values sit in the ±0.01-0.10 band.

    Here we use the rolling mean log-return of the last `short_win` bars
    scaled by SIGNAL_SCALE. This gives the RiskManager a signal in its
    native units without pretending to be an alpha source — the
    direction is whatever mean-reversion or momentum happens to be in
    the bars, and the magnitude is in-range for the alpha threshold.
    """

    def __init__(self, short_win: int = 20, long_win: int = 120) -> None:
        self.short_win = short_win
        self.long_win  = long_win
        self._closes: Deque[float] = deque(maxlen=long_win + 2)
        self._signal_scale = float(getattr(settings.ml, "SIGNAL_SCALE", 20.0))

    def reset(self) -> None:
        self._closes.clear()

    def push(self, close: float) -> None:
        self._closes.append(float(close))

    def ready(self) -> bool:
        return len(self._closes) >= self.long_win

    def signal_and_vol(self) -> Tuple[float, float]:
        closes = np.asarray(self._closes, dtype=float)
        if len(closes) < self.short_win + 2:
            return 0.0, 0.0
        log_rets = np.diff(np.log(closes))
        if len(log_rets) < self.short_win:
            return 0.0, 0.0
        recent  = log_rets[-self.short_win:]
        mu_1bar = float(np.mean(recent))
        ml_signal = mu_1bar * self._signal_scale

        # Annualised realised vol from the longer window
        baseline = log_rets[-min(len(log_rets), self.long_win):]
        if len(baseline) > 1:
            sigma_bar = float(np.std(baseline, ddof=1))
        else:
            sigma_bar = 0.0
        # Convert per-bar σ to annualised σ (NSE: 252 × 375 minute bars)
        vol_ann = sigma_bar * math.sqrt(252 * 375) if sigma_bar > 0 else 0.0
        return ml_signal, vol_ann


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------
class BacktestStrategyAdapter:
    """
    `Strategy` protocol implementation (see backtest.engine.Strategy) that
    drives the real `RiskManager` against historical bars.

    Positions are held LONG only (scaffold — shorts are trivially added by
    accepting negative signals and flipping the leg directions in the
    engine). Exit is triggered when the signal crosses back through zero
    OR when the cumulative unrealised P&L on the open position exceeds
    the hard stop loss config field `HARD_STOP_LOSS_PCT`.
    """

    def __init__(
        self,
        capital:     float,
        product:     str = "MIS",
        short_win:   int = 20,
        long_win:    int = 120,
        enable_budgets: bool = True,
    ) -> None:
        self.capital = float(capital)
        self.product = product
        self.enable_budgets = enable_budgets
        self._risk_manager = RiskManager(capital=capital)
        self._signal      = _TechnicalSignal(short_win=short_win, long_win=long_win)
        self._gri          = GeopoliticalRiskIndex.neutral()
        self._fake_redis   = _FakeRedis()
        self._budget       = _BudgetTracker(capital=capital)
        self._open_qty:    int   = 0
        self._open_entry:  float = 0.0
        self._loop         = asyncio.new_event_loop()

    # Strategy protocol --------------------------------------------------

    def reset(self) -> None:
        self._signal.reset()
        self._risk_manager = RiskManager(capital=self.capital)
        self._budget = _BudgetTracker(capital=self.capital)
        self._open_qty   = 0
        self._open_entry = 0.0

    def on_bar(self, ts: pd.Timestamp, symbol: str, bar: pd.Series) -> Optional[Signal]:
        close = float(bar["close"])
        self._signal.push(close)

        # Convert pandas Timestamp → python datetime for the budget tracker
        now_dt = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
        self._budget.maybe_expire(now_dt)

        # Existing position: check hard stop first
        if self._open_qty > 0:
            loss_pct = (close - self._open_entry) / self._open_entry
            if loss_pct <= -settings.strategy.HARD_STOP_LOSS_PCT:
                qty = self._open_qty
                self._open_qty   = 0
                self._open_entry = 0.0
                return Signal(ts, symbol, "SELL", qty=qty, reason="hard_stop")

        if not self._signal.ready():
            return None

        # Halted by D/W/M budget or blacklisted — stand down
        if self.enable_budgets and (
            self._budget.is_halted(now_dt) or self._budget.is_blacklisted(symbol)
        ):
            return None

        ml_signal, vol_ann = self._signal.signal_and_vol()

        # Position-management path: exit on sign flip
        if self._open_qty > 0 and ml_signal < 0:
            qty = self._open_qty
            self._open_qty   = 0
            self._open_entry = 0.0
            return Signal(ts, symbol, "SELL", qty=qty, reason="signal_flip")

        # Entry path: only on positive signal with no position
        if self._open_qty > 0 or ml_signal <= 0:
            return None
        if abs(ml_signal) < settings.strategy.MIN_ALPHA_THRESHOLD:
            return None

        # Call the REAL RiskManager for sizing. Uses asyncio so we bounce
        # the coroutine through our persistent loop.
        try:
            qty, f_final, f_busseti, decayed, reason = self._loop.run_until_complete(
                self._risk_manager.compute_quantity(
                    symbol        = symbol,
                    ml_signal     = float(ml_signal),
                    current_price = close,
                    vol           = float(vol_ann),
                    gri           = self._gri,
                    redis_client  = self._fake_redis,  # type: ignore[arg-type]
                )
            )
        except Exception as exc:
            logger.debug("RiskManager sizing failed at %s %s: %s", ts, symbol, exc)
            return None

        if qty <= 0 or decayed:
            return None

        self._open_qty   = qty
        self._open_entry = close
        return Signal(ts, symbol, "BUY", qty=qty, reason=f"kelly_entry|{reason[:40]}")

    # Engine callback hook (invoked by a small engine extension below) --

    def on_trade_closed(self, symbol: str, net_pnl: float, ts: datetime) -> None:
        if self.enable_budgets:
            self._budget.record_trade(symbol, net_pnl, ts)

    def close(self) -> None:
        try:
            self._loop.close()
        except Exception:
            pass
