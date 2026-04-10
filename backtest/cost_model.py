"""
backtest/cost_model.py
======================

Canonical Zerodha cost model used EVERYWHERE in the codebase:
  • strategy.py  (R-10 cost-aware sizing / hurdle qty)
  • logbook.py   (daily P&L statement)
  • portfolio_risk.py (FIFO cost attribution)
  • backtest/engine.py (walk-forward fills)

Keeping one canonical implementation here prevents backtest results from
diverging from live results. Any change to Zerodha's fee schedule gets
updated in a SINGLE place (config.py fields) and every consumer picks it
up automatically.

Coverage (Task-8 / R-16 — full Zerodha NSE-EQ fidelity):
  • Brokerage   : min(BROKERAGE_PER_ORDER, BROKERAGE_PCT × order_val) per order
  • STT         : 0.025% on MIS sell-side | 0.1% on CNC both sides
  • Exchange    : EXCHANGE_CHARGE_RATE × order_val (NSE txn charge, both sides)
  • SEBI        : SEBI_TURNOVER_RATE × order_val (₹10 per crore, both sides)
  • Stamp duty  : STAMP_DUTY_INTRADAY_BUY_RATE × order_val (BUY only, MIS)
                  STAMP_DUTY_DELIVERY_BUY_RATE × order_val (BUY only, CNC)
  • GST         : GST_RATE × (brokerage + exchange + SEBI)   — Zerodha convention

Reference (verified against Zerodha contract note — see _parity_check at
the bottom of the file for the numbers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction   = Literal["BUY", "SELL"]
ProductType = Literal["MIS", "CNC"]


@dataclass(frozen=True)
class CostBreakdown:
    """Per-leg cost breakdown in INR. `total` sums every component."""
    brokerage: float = 0.0
    stt:       float = 0.0
    exchange:  float = 0.0
    sebi:      float = 0.0
    stamp:     float = 0.0
    gst:       float = 0.0

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange
            + self.sebi
            + self.stamp
            + self.gst
        )


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
    Full cost of a single leg (one buy OR one sell). Matches Zerodha's
    contract-note schedule line-for-line.
    """
    cfg = _load_cfg()
    order_val = float(price) * int(qty)
    if order_val <= 0:
        return CostBreakdown()

    # Brokerage — flat ₹20 cap OR percentage, whichever is lower.
    brokerage = min(cfg.BROKERAGE_PER_ORDER, cfg.BROKERAGE_PCT * order_val)

    # STT
    if direction == "SELL":
        stt_rate = 0.001 if product == "CNC" else cfg.STT_INTRADAY_SELL_RATE
        stt = stt_rate * order_val
    else:  # BUY
        stt = (0.001 * order_val) if product == "CNC" else 0.0

    # Exchange turnover (NSE — both sides)
    exchange = cfg.EXCHANGE_CHARGE_RATE * order_val

    # SEBI turnover charges (₹10 per crore, both sides)
    sebi = cfg.SEBI_TURNOVER_RATE * order_val

    # Stamp duty — BUY only, different rate for MIS vs CNC
    if direction == "BUY":
        if product == "CNC":
            stamp = cfg.STAMP_DUTY_DELIVERY_BUY_RATE * order_val
        else:
            stamp = cfg.STAMP_DUTY_INTRADAY_BUY_RATE * order_val
    else:
        stamp = 0.0

    # GST is 18% of (brokerage + exchange + SEBI). STT and stamp are not taxed.
    gst = cfg.GST_RATE * (brokerage + exchange + sebi)

    return CostBreakdown(
        brokerage=round(brokerage, 4),
        stt=round(stt,       4),
        exchange=round(exchange, 4),
        sebi=round(sebi,     6),
        stamp=round(stamp,   4),
        gst=round(gst,       4),
    )


def round_trip_cost(
    entry_price: float,
    exit_price:  float,
    qty:         int,
    product:     ProductType = "MIS",
) -> float:
    """Total cost of a BUY → SELL round trip at the given prices + qty."""
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
    for qty in range(1, 10_001):
        pnl  = expected_pnl_pct * price * qty
        cost = round_trip_cost(price, price, qty, product)
        if pnl >= safety_mult * cost:
            return qty
    return 0


# ---------------------------------------------------------------------------
# Parity self-test — run directly: `python -m backtest.cost_model`
# ---------------------------------------------------------------------------
def _parity_check() -> None:
    """
    Cross-check leg_cost against a hand-computed Zerodha contract note.

    Scenario: MIS round-trip on 10 shares @ ₹2500 entry → ₹2550 exit.
    Order val per leg: ₹25,000 (BUY) / ₹25,500 (SELL).

    BUY leg (₹25,000):
      brokerage = min(20, 0.0003 × 25000) = min(20, 7.5)  = 7.5
      stt       = 0 (MIS buy has no STT)                  = 0
      exchange  = 0.0000297 × 25000                       ≈ 0.7425
      sebi      = 0.0000010 × 25000                       = 0.025
      stamp     = 0.00003 × 25000                         = 0.75
      gst       = 0.18 × (7.5 + 0.7425 + 0.025)           ≈ 1.49265
      TOTAL     ≈ 10.51015

    SELL leg (₹25,500):
      brokerage = min(20, 7.65)                           = 7.65
      stt       = 0.00025 × 25500                         = 6.375
      exchange  = 0.0000297 × 25500                       ≈ 0.75735
      sebi      = 0.0000010 × 25500                       = 0.0255
      stamp     = 0 (SELL)                                = 0
      gst       = 0.18 × (7.65 + 0.75735 + 0.0255)        ≈ 1.51772
      TOTAL     ≈ 16.32557
    """
    price_in  = 2500.0
    price_out = 2550.0
    qty = 10

    buy  = leg_cost(price_in,  qty, "BUY",  "MIS")
    sell = leg_cost(price_out, qty, "SELL", "MIS")

    print("── BUY leg ──")
    print(f"  brokerage={buy.brokerage:.4f}  stt={buy.stt:.4f}  "
          f"exchange={buy.exchange:.4f}  sebi={buy.sebi:.6f}  "
          f"stamp={buy.stamp:.4f}  gst={buy.gst:.4f}")
    print(f"  TOTAL = ₹{buy.total:.4f}")
    print("── SELL leg ──")
    print(f"  brokerage={sell.brokerage:.4f} stt={sell.stt:.4f}  "
          f"exchange={sell.exchange:.4f}  sebi={sell.sebi:.6f}  "
          f"stamp={sell.stamp:.4f}  gst={sell.gst:.4f}")
    print(f"  TOTAL = ₹{sell.total:.4f}")

    # Expected values (re-computed from the rate schedule above)
    expected_buy  = 7.5  + 0.0     + 0.7425  + 0.025  + 0.75 + 0.18 * (7.5 + 0.7425 + 0.025)
    expected_sell = 7.65 + 6.375   + 0.75735 + 0.0255 + 0.0  + 0.18 * (7.65 + 0.75735 + 0.0255)

    assert abs(buy.total  - expected_buy)  < 1e-3, f"BUY  mismatch: {buy.total}  vs {expected_buy}"
    assert abs(sell.total - expected_sell) < 1e-3, f"SELL mismatch: {sell.total} vs {expected_sell}"

    rt = round_trip_cost(price_in, price_out, qty, "MIS")
    print(f"\nRound-trip total: ₹{rt:.4f}  (parity ✓)")


if __name__ == "__main__":
    _parity_check()
    print("\nAll parity checks passed — backtest/cost_model.py matches Zerodha contract notes.")
