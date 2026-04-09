"""
position_manager.py
===================
Live position and Risk-Management (TSL/TP) layer.

Tracks each open trade: entry price, current quantity, peak price reached,
and evaluates Trailing Stop Loss (TSL) and Hard Stop Loss (SL) conditions.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from config import settings
from strategy import TradeDirection

logger = logging.getLogger(__name__)

@dataclass
class PositionState:
    """Live state of an open trade."""
    symbol:           str
    direction:        TradeDirection
    entry_price:      float
    quantity:         int
    peak_price:       float      # highest price (BUY) or lowest price (SELL) since entry
    entry_time:       float      # monotonic timestamp
    is_tsl_active:    bool = False
    stop_loss:        float = 0.0
    take_profit:      float = 0.0
    last_update:      float = field(default_factory=time.monotonic)

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

    def check_exit(self, current_price: float) -> Tuple[bool, str]:
        """
        Evaluate SL and TSL conditions.
        Returns (should_exit, reason).
        """
        cfg = settings.strategy
        
        # ── 1. Hard Stop Loss ──────────────────────────────────────────
        if self.direction == TradeDirection.BUY:
            sl_price = self.entry_price * (1.0 - cfg.HARD_STOP_LOSS_PCT)
            if current_price <= sl_price:
                return True, f"Hard SL: {current_price:.2f} <= {sl_price:.2f} (-{cfg.HARD_STOP_LOSS_PCT:.1%})"
        else:
            sl_price = self.entry_price * (1.0 + cfg.HARD_STOP_LOSS_PCT)
            if current_price >= sl_price:
                return True, f"Hard SL: {current_price:.2f} >= {sl_price:.2f} (-{cfg.HARD_STOP_LOSS_PCT:.1%})"

        # ── 2. Trailing Stop Loss (TSL) ────────────────────────────────
        # Activation check
        if not self.is_tsl_active:
            if self.direction == TradeDirection.BUY:
                profit_pct = (current_price - self.entry_price) / self.entry_price
            else:
                profit_pct = (self.entry_price - current_price) / self.entry_price
                
            if profit_pct >= cfg.TSL_ACTIVATION_PCT:
                self.is_tsl_active = True
                logger.info("TSL Activated for %s: Profit %.2f%% reached threshold %.2f%%", 
                            self.symbol, profit_pct*100, cfg.TSL_ACTIVATION_PCT*100)

        # Execution check
        if self.is_tsl_active:
            if self.direction == TradeDirection.BUY:
                # Exit if price drops more than TSL_CALLBACK from peak
                tsl_price = self.peak_price * (1.0 - cfg.TSL_CALLBACK_PCT)
                if current_price <= tsl_price:
                    return True, f"TSL: {current_price:.2f} <= {tsl_price:.2f} (Peak: {self.peak_price:.2f})"
            else:
                # Exit if price rises more than TSL_CALLBACK from peak (lowest price)
                tsl_price = self.peak_price * (1.0 + cfg.TSL_CALLBACK_PCT)
                if current_price >= tsl_price:
                    return True, f"TSL: {current_price:.2f} >= {tsl_price:.2f} (Peak: {self.peak_price:.2f})"

        return False, ""

class PositionManager:
    """Tracks all live positions and evaluates exit criteria."""
    
    def __init__(self) -> None:
        self._positions: Dict[str, PositionState] = {}

    def on_trade_executed(self, symbol: str, direction: TradeDirection, price: float, quantity: int) -> None:
        """Called after a new trade is filled."""
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
            entry_time=time.monotonic()
        )
        logger.info("PositionManager: Tracking %s %s %d @ %.2f", 
                    direction.value, symbol, quantity, price)

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
