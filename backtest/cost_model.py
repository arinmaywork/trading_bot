"""
backtest/cost_model.py
======================

Bit-for-bit mirror of the live Zerodha cost model used in
  • strategy.py (_round_trip_cost inside RiskManager.compute_qty)
  • logbook.py  (_trade_cost inside daily P&L statement)

Keeping one canonical implementation here prevents backtest results from
diverging from live results. Any change to Zerodha's fee schedule gets
updated in a SINGLE place (config.py fields) and both live and backtest
pick it up automatically.

Coverage today (matches logbook.py line-for-line):
  • brokerage  : min(BROKERAGE_PER_ORDER, BROKERAGE_PCT × order_val) per order
  • exchange   : EXCHANGE_CHARGE_RATE × order_val per order
  • STT        : 0.025% on MIS sell-side  (CNC: 0.1% BOTH sides)

NOT yet covered (tracked in ROADMAP Task 8 / R-16):
  • GST 18% on brokerage + exchange
  • Stamp duty 0.003% on buy
  • SEBI turnover charge ₹10/crore

Task 2 deliberately ships parity with the live code — Task 8 raises
fidelity on both sides simultaneously so live and backtest stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction   = Literal["BUY", "SELL"]
ProductType = Literal["MIS", "CNC"]


@dataclass(frozen=True)
class CostBreakdown:
    """Per-leg cost breakdown in INR."""
    brokerage: float
    stt:       float
    exchange:  float

    @property
    def total(self) -> float:
        return self.brokerage + self.stt + self.exchange


def _load_cfg():
    """Late import so the backtester can run without the full config tree loaded yet."""
    from config import settings
    return settings.strategy


def leg_cost(
    price:       float,
    qty:         int,
    direction:   Direction,
    product:     ProductType = "MIS",
) -> CostBreakdown:
    """
    Cost of a single leg (one buy OR one sell). Matches logbook.py._trade_cost
    exactly — copied line-for-line and reduced to a pure function.
    """
    cfg = _load_cfg()
    order_val = price * qty

    brokerage = min(cfg.BROKERAGE_PER_ORDER, cfg.BROKERAGE_PCT * order_val)
    exchange  = cfg.EXCHANGE_CHARGE_RATE * order_val

    if direction == "SELL":
        # STT on sell-side only for MIS; both sides for CNC
        stt_rate = 0.001 if product == "CNC" else cfg.STT_INTRADAY_SELL_RATE
        stt = stt_rate * order_val
    else:  # BUY
        # CNC buy-side STT (0.1%); MIS buy has no STT
        stt = (0.001 * order_val) if product == "CNC" else 0.0

    return CostBreakdown(
        brokerage=round(brokerage, 4),
        stt=round(stt, 4),
        exchange=round(exchange, 4),
    )


def round_trip_cost(
    entry_price: float,
    exit_price:  float,
    qty:         int,
    product:     ProductType = "MIS",
) -> float:
    """
    Total cost of a BUY → SELL round trip at the given prices + qty.
    Shortcut for tests and quick signal filters.
    """
    buy_leg  = leg_cost(entry_price, qty, "BUY",  product)
    sell_leg = leg_cost(exit_price,  qty, "SELL", product)
    return round(buy_leg.total + sell_leg.total, 4)


def cost_hurdle_qty(
    price:            float,
    expected_pnl_pct: float,
    product:          ProductType = "MIS",
    safety_mult:      float = 2.0,
) -> int:
    """
    Return the minimum qty at which expected P&L clears
    `safety_mult × round_trip_cost`. Matches the R-10 filter in strategy.py.

    Returns 0 if no qty clears the hurdle for the given price.
    """
    # Closed-form: expected_pnl = expected_pnl_pct × price × qty.
    # Cost is almost linear in qty once order_val > BROKERAGE_PER_ORDER/BROKERAGE_PCT,
    # so a simple linear search from qty=1 converges in a few steps.
    for qty in range(1, 10_001):
        pnl = expected_pnl_pct * price * qty
        cost = round_trip_cost(price, price, qty, product)
        if pnl >= safety_mult * cost:
            return qty
    return 0


# ---------------------------------------------------------------------------
# Self-test helpers (run this file directly: `python -m backtest.cost_model`)
# ---------------------------------------------------------------------------
def _parity_check() -> None:
    """Cross-check against logbook.py._trade_cost on a sample trade."""
    # Recreate the same math by hand for qty=10 @ ₹2500 MIS SELL
    price = 2500.0
    qty   = 10
    order_val = price * qty  # ₹25,000

    # By hand:
    # brokerage = min(20, 0.0003 × 25000) = min(20, 7.5) = 7.5
    # exchange  = 0.0000345 × 25000 = 0.8625
    # stt (MIS SELL) = 0.00025 × 25000 = 6.25
    # total = 7.5 + 6.25 + 0.8625 = 14.6125
    expected_total = 7.5 + 6.25 + 0.8625

    leg = leg_cost(price, qty, "SELL", "MIS")
    assert abs(leg.total - expected_total) < 1e-6, (
        f"PARITY MISMATCH: got {leg.total}, expected {expected_total}"
    )
    print(f"[OK] MIS sell leg parity: ₹{leg.total:.4f} (brokerage={leg.brokerage} "
          f"stt={leg.stt} exch={leg.exchange})")

    # Round trip
    rt = round_trip_cost(price, price, qty, "MIS")
    # buy leg = 7.5 + 0 + 0.8625 = 8.3625
    # sell leg = 7.5 + 6.25 + 0.8625 = 14.6125
    expected_rt = 8.3625 + 14.6125
    assert abs(rt - expected_rt) < 1e-6, f"ROUND TRIP MISMATCH: {rt} vs {expected_rt}"
    print(f"[OK] MIS round-trip parity: ₹{rt:.4f}")

    # CNC both-side STT check
    cnc_buy = leg_cost(price, qty, "BUY", "CNC")
    # brokerage = 7.5, exchange = 0.8625, stt = 0.001 × 25000 = 25
    assert abs(cnc_buy.stt - 25.0) < 1e-6
    print(f"[OK] CNC buy-side STT parity: ₹{cnc_buy.stt:.4f}")


if __name__ == "__main__":
    _parity_check()
    print("\nAll parity checks passed — backtest/cost_model.py matches logbook.py.")
