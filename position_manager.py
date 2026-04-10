"""
position_manager.py
===================
Live position and Risk-Management (TSL/TP) layer.

Tracks each open trade: entry price, current quantity, peak price reached,
and evaluates Trailing Stop Loss (TSL) and Hard Stop Loss (SL) conditions.

Bug 2.4 fix — vol-scaled stops:
  The fixed percent stops in config (HARD_STOP_LOSS_PCT, TSL_ACTIVATION_PCT,
  TSL_CALLBACK_PCT) are too tight for quiet regimes and too loose for
  volatile ones. Each PositionState now carries the annualised realised vol
  at entry; check_exit() scales all three thresholds by σ_bar (1-minute
  realised stdev) with multipliers k_hsl = 3.0, k_act = 2.5, k_cbk = 1.5.
  A floor at the config value preserves the previous behaviour when vol is
  extremely low (so we are never *looser* than the fixed default).

  A hard time-stop (TIME_STOP_MINUTES) also flattens dead trades that have
  neither hit their TSL activation nor their hard stop within N minutes.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import settings
from strategy import TradeDirection

logger = logging.getLogger(__name__)

# Vol-scaling multipliers (expressed in multiples of σ_bar = 1-min realised stdev)
HSL_SIGMA_MULT = 3.0    # hard stop at 3σ
ACT_SIGMA_MULT = 2.5    # TSL activates at +2.5σ profit
CBK_SIGMA_MULT = 1.5    # TSL trails 1.5σ below peak
BARS_PER_YEAR  = 252 * 375
TIME_STOP_MINUTES = 25  # kill dead trades after 25 minutes of no progress


@dataclass
class PositionState:
    """Live state of an open trade."""
    symbol:           str
    direction:        TradeDirection
    entry_price:      float
    quantity:         int
    peak_price:       float      # highest price (BUY) or lowest price (SELL) since entry
    entry_time:       float      # monotonic timestamp
    sigma_ann:        float = 0.20  # annualised realised vol at entry (fallback 20%)
    is_tsl_active:    bool = False
    stop_loss:        float = 0.0
    take_profit:      float = 0.0
    last_update:      float = field(default_factory=time.monotonic)

    @property
    def sigma_bar(self) -> float:
        """Per-bar (1-minute) realised stdev implied by annualised vol."""
        return max(self.sigma_ann, 1e-4) / math.sqrt(BARS_PER_YEAR)

    def update_peak(self, current_price: float) -> bool:
        """Update peak price and return True if a new peak was reached."""
        if self.direction == TradeDirection.BUY:
            if current_price > self.peak_price:
                self.peak_price = current_price
                return True
        elif self.direction == TradeDirection.SELL:
            if current_price < self.peak_price or self.peak_price == 0:
                self.peak_price = current_price
                return True
        return False

    def _effective_thresholds(self) -> Tuple[float, float, float]:
        """
        Compute the effective (hsl_pct, tsl_activation_pct, tsl_callback_pct)
        as the max of (a) the fixed config percent and (b) a vol-scaled
        multiple of per-bar realised stdev. This makes stops adapt to the
        volatility regime while never being *looser* than the configured
        fixed floor.
        """
        cfg = settings.strategy
        sig_b = self.sigma_bar
        hsl_vol = HSL_SIGMA_MULT * sig_b
        act_vol = ACT_SIGMA_MULT * sig_b
        cbk_vol = CBK_SIGMA_MULT * sig_b
        return (
            max(cfg.HARD_STOP_LOSS_PCT, hsl_vol),
            max(cfg.TSL_ACTIVATION_PCT, act_vol),
            max(cfg.TSL_CALLBACK_PCT,   cbk_vol),
        )

    def check_exit(self, current_price: float) -> Tuple[bool, str]:
        """
        Evaluate SL, TSL and time-stop conditions.
        Returns (should_exit, reason).
        """
        hsl_pct, act_pct, cbk_pct = self._effective_thresholds()

        # ── 1. Hard Stop Loss (vol-scaled) ─────────────────────────────
        if self.direction == TradeDirection.BUY:
            sl_price = self.entry_price * (1.0 - hsl_pct)
            if current_price <= sl_price:
                return True, (
                    f"Hard SL (vol-scaled {hsl_pct:.2%}): "
                    f"{current_price:.2f} <= {sl_price:.2f} "
                    f"[σ_ann={self.sigma_ann:.1%}]"
                )
        else:
            sl_price = self.entry_price * (1.0 + hsl_pct)
            if current_price >= sl_price:
                return True, (
                    f"Hard SL (vol-scaled {hsl_pct:.2%}): "
                    f"{current_price:.2f} >= {sl_price:.2f} "
                    f"[σ_ann={self.sigma_ann:.1%}]"
                )

        # ── 2. Trailing Stop Loss (vol-scaled) ─────────────────────────
        if not self.is_tsl_active:
            if self.direction == TradeDirection.BUY:
                profit_pct = (current_price - self.entry_price) / self.entry_price
            else:
                profit_pct = (self.entry_price - current_price) / self.entry_price

            if profit_pct >= act_pct:
                self.is_tsl_active = True
                logger.info(
                    "TSL Activated %s: profit %.2f%% ≥ vol-scaled threshold %.2f%% "
                    "(σ_ann=%.1f%%)",
                    self.symbol, profit_pct * 100, act_pct * 100, self.sigma_ann * 100,
                )

        if self.is_tsl_active:
            if self.direction == TradeDirection.BUY:
                tsl_price = self.peak_price * (1.0 - cbk_pct)
                if current_price <= tsl_price:
                    return True, (
                        f"TSL (vol-scaled {cbk_pct:.2%}): "
                        f"{current_price:.2f} <= {tsl_price:.2f} "
                        f"[peak {self.peak_price:.2f}]"
                    )
            else:
                tsl_price = self.peak_price * (1.0 + cbk_pct)
                if current_price >= tsl_price:
                    return True, (
                        f"TSL (vol-scaled {cbk_pct:.2%}): "
                        f"{current_price:.2f} >= {tsl_price:.2f} "
                        f"[peak {self.peak_price:.2f}]"
                    )

        # ── 3. Time-stop: kill dead trades that never activated TSL ────
        if not self.is_tsl_active:
            age_min = (time.monotonic() - self.entry_time) / 60.0
            if age_min >= TIME_STOP_MINUTES:
                return True, (
                    f"Time-stop: position alive {age_min:.1f}m with no TSL "
                    f"activation (threshold {TIME_STOP_MINUTES}m)"
                )

        return False, ""

class PositionManager:
    """Tracks all live positions and evaluates exit criteria."""
    
    def __init__(self) -> None:
        self._positions: Dict[str, PositionState] = {}

    def on_trade_executed(
        self,
        symbol: str,
        direction: TradeDirection,
        price: float,
        quantity: int,
        sigma_ann: float = 0.20,
    ) -> None:
        """
        Called after a new trade is filled.

        Args:
            sigma_ann: Annualised realised vol at entry (from StrategyEngine).
                       Used to vol-scale stop-loss and trailing-stop thresholds.
                       Defaults to 20% if the caller does not provide it.
        """
        if quantity == 0:
            if symbol in self._positions:
                del self._positions[symbol]
                logger.info("PositionManager: %s position cleared.", symbol)
            return

        # If existing position, we could average out, but SentiStack
        # position guard currently prevents multiple entries.
        # We replace/update the state.
        self._positions[symbol] = PositionState(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            quantity=quantity,
            peak_price=price,
            entry_time=time.monotonic(),
            sigma_ann=max(float(sigma_ann), 0.05),
        )
        logger.info(
            "PositionManager: Tracking %s %s %d @ %.2f (σ_ann=%.1f%%)",
            direction.value, symbol, quantity, price, sigma_ann * 100,
        )

    def update(self, ltp_map: Dict[str, float]) -> List[Tuple[str, int, TradeDirection, str]]:
        """
        Update peak prices and check for exits.
        Returns list of (symbol, quantity, exit_direction, reason) for trades that should close.
        """
        exits = []
        for symbol, pos in list(self._positions.items()):
            price = ltp_map.get(symbol)
            if not price or price <= 0:
                continue
            
            pos.update_peak(price)
            pos.last_update = time.monotonic()
            
            should_exit, reason = pos.check_exit(price)
            if should_exit:
                exit_dir = TradeDirection.SELL if pos.direction == TradeDirection.BUY else TradeDirection.BUY
                exits.append((symbol, pos.quantity, exit_dir, reason))
                # Remove from tracking immediately to avoid double-exit
                del self._positions[symbol]
                
        return exits

    def get_position(self, symbol: str) -> Optional[PositionState]:
        return self._positions.get(symbol)

    @property
    def active_symbols(self) -> List[str]:
        return list(self._positions.keys())

    # ── Task-5 helpers: snapshot open positions for portfolio_risk ──────
    def snapshot_positions(self) -> Dict[str, int]:
        """``{symbol: signed_qty}`` where BUY is positive, SELL is negative."""
        out: Dict[str, int] = {}
        for sym, pos in self._positions.items():
            sign = 1 if pos.direction == TradeDirection.BUY else -1
            out[sym] = sign * int(pos.quantity)
        return out

    def snapshot_entry_prices(self) -> Dict[str, float]:
        """``{symbol: entry_price}`` for computing unrealised MTM."""
        return {sym: float(pos.entry_price) for sym, pos in self._positions.items()}
