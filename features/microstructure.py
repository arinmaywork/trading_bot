"""
Microstructure features for short-term (1-15 min) prediction.

Tier-1 features derived from order book dynamics, trade flow, and price action.
All functions are pure (no side effects except logging) and can be called from
data_ingestion.py and ml_signal.py without codebase dependencies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureState:
    """
    Per-symbol rolling state for microstructure feature computation.

    Tracks EMA of order book imbalance, running tick count, spread statistics,
    recent price action, and signed trade flow over a 5-minute window.
    """
    symbol: str
    obi_ema: float = 0.0
    tick_count_30s: int = 0
    tick_count_avg: float = 10.0
    spread_mean: float = 5.0
    spread_var: float = 4.0
    last_30s_reset: float = 0.0
    signed_trades_5min: list = field(default_factory=list)
    recent_closes_1min: list = field(default_factory=list)


def order_book_imbalance_momentum(obi_current: float, obi_ema: float) -> float:
    """
    Compute the acceleration of order book imbalance (MLOFI momentum).

    The momentum signal captures how quickly the order book imbalance is
    changing relative to its recent exponential moving average.

    Args:
        obi_current: Current MLOFI (Modified Limit Order Book Imbalance).
        obi_ema: EMA of MLOFI from prior observation(s).

    Returns:
        Acceleration of order book imbalance (obi_current - obi_ema).
    """
    return obi_current - obi_ema


def update_obi_ema(obi_current: float, obi_ema_prev: float, span: int = 20) -> float:
    """
    Update the exponential moving average of order book imbalance.

    Uses standard EMA formula with smoothing factor alpha = 2 / (span + 1).

    Args:
        obi_current: Current MLOFI value.
        obi_ema_prev: Previous EMA value.
        span: EMA span (default 20 observations). Lower span = more responsive.

    Returns:
        Updated EMA value.
    """
    alpha = 2.0 / (span + 1)
    return alpha * obi_current + (1.0 - alpha) * obi_ema_prev


def trade_arrival_rate(tick_count_30s: int, avg_tick_count_30s: float) -> float:
    """
    Compute normalised trade arrival rate over a 30-second window.

    Metric indicates whether current trading activity is elevated relative to
    the running baseline. A value of 1.0 indicates normal activity, >2.0
    indicates unusual bursts.

    Args:
        tick_count_30s: Number of trades (ticks) in the last 30 seconds.
        avg_tick_count_30s: Running average baseline tick count.

    Returns:
        Ratio: tick_count_30s / max(avg_tick_count_30s, 1.0).
    """
    return tick_count_30s / max(avg_tick_count_30s, 1.0)


def update_tick_count_avg(
    tick_count_30s: int, avg_prev: float, alpha: float = 0.05
) -> float:
    """
    Update the EMA of the 30-second tick count baseline.

    Tracks the moving baseline of trading activity to detect anomalies
    in trade arrival rate.

    Args:
        tick_count_30s: Current 30-second tick count.
        avg_prev: Previous EMA of tick count.
        alpha: Smoothing factor (default 0.05 for slower adaptation).

    Returns:
        Updated EMA of tick count.
    """
    return alpha * tick_count_30s + (1.0 - alpha) * avg_prev


def spread_z_score(
    best_bid: float,
    best_ask: float,
    mid_price: float,
    spread_mean: float,
    spread_std: float,
) -> float:
    """
    Compute the z-score of the spread in basis points.

    Normalises the bid-ask spread by its running mean and standard deviation.
    A high z-score indicates unusually wide spreads (liquidity stress).

    Args:
        best_bid: Best bid price.
        best_ask: Best ask price.
        mid_price: Mid price (e.g., (bid + ask) / 2).
        spread_mean: Running mean of spread in basis points.
        spread_std: Running standard deviation of spread in basis points.

    Returns:
        Z-score of spread, clamped to [-5, 5].
    """
    spread_bps = (best_ask - best_bid) / max(mid_price, 0.01) * 10000.0
    z_score = (spread_bps - spread_mean) / max(spread_std, 0.01)
    return max(-5.0, min(5.0, z_score))


def update_spread_stats(
    spread_bps: float, mean_prev: float, var_prev: float, alpha: float = 0.01
) -> tuple[float, float]:
    """
    Update running mean and variance of spread using Welford's online algorithm.

    Efficiently tracks the first two moments of the spread distribution without
    storing the full history.

    Args:
        spread_bps: Current spread in basis points.
        mean_prev: Previous running mean.
        var_prev: Previous running variance.
        alpha: Smoothing factor (default 0.01 for slow adaptation).

    Returns:
        Tuple of (new_mean, new_variance).
    """
    new_mean = alpha * spread_bps + (1.0 - alpha) * mean_prev
    delta = spread_bps - mean_prev
    new_var = (1.0 - alpha) * (var_prev + alpha * delta * delta)
    return new_mean, new_var


def volume_weighted_price_pressure(signed_trades: list[tuple[float, float]]) -> float:
    """
    Compute the volume-weighted price pressure over a 5-minute window.

    Integrates the directional impact of trades: buy-classified trades (positive
    signed quantity) should correlate with price increases, and vice versa.

    Args:
        signed_trades: List of (signed_qty, price_change) tuples.
                      signed_qty is +qty for buy-classified, -qty for sell-classified.
                      price_change is the price move (absolute, in price units) during
                      the trade or immediately after.

    Returns:
        Aggregated volume-weighted price pressure, clamped to [-1, 1].
        Positive value indicates buy pressure correlating with price increases.
    """
    if not signed_trades:
        return 0.0

    total_impact = 0.0
    total_volume = 0.0

    for signed_qty, price_change in signed_trades:
        total_impact += signed_qty * price_change
        total_volume += abs(signed_qty)

    if total_volume < 1e-9:
        return 0.0

    pressure = total_impact / total_volume
    return max(-1.0, min(1.0, pressure))


def price_momentum(prices: list[float], horizons: list[int] | None = None) -> list[float]:
    """
    Compute log-return momentum at multiple horizons.

    Captures price momentum over short time horizons (1, 5, 15 minutes).
    Uses log returns for mathematical consistency.

    Args:
        prices: List of 1-minute close prices, most recent price last.
        horizons: List of lookback horizons in minutes (default [1, 5, 15]).

    Returns:
        List of log-returns for each horizon. Value is 0.0 if insufficient history.
    """
    if horizons is None:
        horizons = [1, 5, 15]

    momenta = []
    for h in horizons:
        if len(prices) > h and prices[-1] > 0 and prices[-1 - h] > 0:
            log_ret = math.log(prices[-1] / prices[-1 - h])
            momenta.append(log_ret)
        else:
            momenta.append(0.0)

    return momenta


def compute_tier1_features(
    state: MicrostructureState,
    mlofi: float,
    aflow_ratio: float,
    best_bid: float,
    best_ask: float,
) -> dict:
    """
    Compute all Tier-1 microstructure features for a symbol.

    This is the main entry point for feature engineering. It:
    1. Updates the state in-place (EMA, statistics, etc.)
    2. Computes all derived features using the updated state
    3. Returns a dictionary of features for downstream ML models

    Args:
        state: MicrostructureState object for the symbol (modified in-place).
        mlofi: Current MLOFI (Modified Limit Order Book Imbalance).
        aflow_ratio: Ask flow to bid flow ratio (or similar aggressiveness metric).
        best_bid: Current best bid price.
        best_ask: Current best ask price.

    Returns:
        Dictionary with keys:
            - obi_momentum: Acceleration of order book imbalance
            - trade_arrival_rate: Normalised tick count
            - spread_z: Z-score of bid-ask spread
            - vwpp: Volume-weighted price pressure
            - ret_1min: Log return over 1 minute
            - ret_5min: Log return over 5 minutes
            - ret_15min: Log return over 15 minutes
            - mlofi: Current MLOFI (pass-through)
            - aflow_ratio: Current aflow ratio (pass-through)
    """
    mid_price = (best_bid + best_ask) / 2.0

    obi_momentum = order_book_imbalance_momentum(mlofi, state.obi_ema)
    state.obi_ema = update_obi_ema(mlofi, state.obi_ema, span=20)

    tar = trade_arrival_rate(state.tick_count_30s, state.tick_count_avg)
    state.tick_count_avg = update_tick_count_avg(
        state.tick_count_30s, state.tick_count_avg, alpha=0.05
    )

    spread_bps = (best_ask - best_bid) / max(mid_price, 0.01) * 10000.0
    spread_z = spread_z_score(best_bid, best_ask, mid_price, state.spread_mean,
                               math.sqrt(state.spread_var))
    state.spread_mean, state.spread_var = update_spread_stats(
        spread_bps, state.spread_mean, state.spread_var, alpha=0.01
    )

    vwpp = volume_weighted_price_pressure(state.signed_trades_5min)

    momenta = price_momentum(state.recent_closes_1min, horizons=[1, 5, 15])
    ret_1min, ret_5min, ret_15min = momenta[0], momenta[1], momenta[2]

    logger.debug(
        f"{state.symbol}: obi_mom={obi_momentum:.4f}, tar={tar:.2f}, "
        f"spread_z={spread_z:.2f}, vwpp={vwpp:.3f}, ret_1m={ret_1min:.4f}"
    )

    return {
        "obi_momentum": obi_momentum,
        "trade_arrival_rate": tar,
        "spread_z": spread_z,
        "vwpp": vwpp,
        "ret_1min": ret_1min,
        "ret_5min": ret_5min,
        "ret_15min": ret_15min,
        "mlofi": mlofi,
        "aflow_ratio": aflow_ratio,
    }
