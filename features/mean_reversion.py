"""
Mean-reversion features for Tier-2 (30-min to intraday) prediction.

Standalone module providing pure functions for computing mean-reversion signals,
opening range breakout positions, VWAP deviations, and sector-relative strength.
All functions are stateless except for the MeanReversionState dataclass which
tracks per-symbol state across multiple updates.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


def vwap_z_score(price: float, session_vwap: float, price_vwap_std: float) -> float:
    """
    Compute Z-score of price relative to session VWAP.

    Args:
        price: Current price
        session_vwap: Session volume-weighted average price
        price_vwap_std: Standard deviation of (price - vwap)

    Returns:
        Z-score clamped to [-5, 5]
    """
    if price_vwap_std < 0.001:
        return 0.0
    z_score = (price - session_vwap) / max(price_vwap_std, 0.001)
    return max(-5.0, min(5.0, z_score))


def update_vwap_deviation_stats(
    price: float,
    vwap: float,
    mean_prev: float,
    var_prev: float,
    alpha: float = 0.005,
) -> Tuple[float, float]:
    """
    Online Welford update for (price - vwap) mean and variance.

    Uses exponential moving average with decay factor alpha for adaptive
    tracking of deviation statistics.

    Args:
        price: Current price
        vwap: Current session VWAP
        mean_prev: Previous mean of (price - vwap)
        var_prev: Previous variance of (price - vwap)
        alpha: Decay factor for exponential moving average (default 0.005)

    Returns:
        Tuple of (new_mean, new_variance)
    """
    deviation = price - vwap
    # Exponential moving average update
    new_mean = (1.0 - alpha) * mean_prev + alpha * deviation
    # Update variance using Welford-like approach
    diff = deviation - new_mean
    new_var = (1.0 - alpha) * var_prev + alpha * diff * diff
    return new_mean, new_var


def sector_relative_strength(stock_ret: float, sector_avg_ret: float) -> float:
    """
    Compute sector-relative strength: stock return minus sector average return.

    Args:
        stock_ret: Stock return over the period
        sector_avg_ret: Sector average return over the period

    Returns:
        Relative strength clamped to [-0.05, 0.05]
    """
    rel_strength = stock_ret - sector_avg_ret
    return max(-0.05, min(0.05, rel_strength))


def opening_range_breakout_position(
    price: float, orb_high: float, orb_low: float
) -> float:
    """
    Compute normalized position within opening range.

    Indicates how far price has moved from the opening range low toward the high.
    Values > 1 indicate breakout above the range; values < 0 indicate breakdown.

    Args:
        price: Current price
        orb_high: Opening range high (typically first 15 minutes)
        orb_low: Opening range low (typically first 15 minutes)

    Returns:
        Normalized position clamped to [0, 1] within range, unbounded outside
    """
    range_width = orb_high - orb_low
    if range_width < 0.01:
        return 0.5
    position = (price - orb_low) / max(range_width, 0.01)
    return max(0.0, min(1.0, position))


def rsi_mean_reversion_signal(rsi: float) -> float:
    """
    Map RSI to a mean-reversion trading signal.

    Encodes the mean-reversion hypothesis: extreme RSI values (very high or very low)
    suggest the instrument has moved too far and should revert.

    Args:
        rsi: Relative Strength Index (0-100)

    Returns:
        Signal in range [-1.0, 1.0]:
        - +1.0: Strong buy signal (RSI < 25, oversold)
        - 0.0: Neutral (RSI 40-60)
        - -1.0: Strong sell signal (RSI > 75, overbought)
    """
    if rsi < 25:
        return 1.0  # Strong buy
    elif rsi < 40:
        # Linear interpolation from 1.0 at RSI=25 to 0.0 at RSI=40
        return 1.0 - (rsi - 25) / 15.0
    elif rsi < 60:
        return 0.0  # Neutral zone
    elif rsi < 75:
        # Linear interpolation from 0.0 at RSI=60 to -1.0 at RSI=75
        return -(rsi - 60) / 15.0
    else:
        return -1.0  # Strong sell


@dataclass
class MeanReversionState:
    """
    Per-symbol state for mean-reversion feature computation.

    Tracks opening range breakout formation, VWAP deviation statistics,
    and sector return data across multiple updates.

    Attributes:
        symbol: Ticker symbol
        orb_high: Opening range high (set after first 15 minutes)
        orb_low: Opening range low (set after first 15 minutes)
        orb_set: True if opening range has been established
        orb_candle_count: Number of candles processed for ORB formation
        vwap_dev_mean: Running mean of (price - vwap) deviations
        vwap_dev_var: Running variance of (price - vwap) deviations
        sector_returns_15min: Dict mapping sector symbol to average return
    """

    symbol: str
    orb_high: float = 0.0
    orb_low: float = float("inf")
    orb_set: bool = False
    orb_candle_count: int = 0
    vwap_dev_mean: float = 0.0
    vwap_dev_var: float = 1.0
    sector_returns_15min: Dict[str, float] = field(default_factory=dict)


def compute_tier2_features(
    state: MeanReversionState,
    price: float,
    vwap: float,
    rsi: float,
    stock_ret_15min: float,
    sector_avg_ret_15min: float,
    time_features: Dict,
) -> Dict:
    """
    Compute complete Tier-2 feature set for mean-reversion prediction.

    Combines VWAP deviation, RSI mean-reversion signal, sector-relative strength,
    opening range breakout position, and time-of-day features into a unified
    feature vector.

    Args:
        state: MeanReversionState tracking per-symbol statistics
        price: Current price
        vwap: Session VWAP
        rsi: RSI indicator (0-100)
        stock_ret_15min: 15-minute stock return
        sector_avg_ret_15min: 15-minute sector average return
        time_features: Dict from seasonality.time_of_day_features()

    Returns:
        Dict with keys:
            - vwap_z: VWAP Z-score
            - rsi_mr_signal: Mean-reversion signal from RSI
            - sector_rs: Sector-relative strength
            - orb_position: Position within opening range
            - time_sin: Cyclical time component (sine)
            - time_cos: Cyclical time component (cosine)
            - ret_15min: 15-minute return
            - vol_normalised: Normalized volatility (VWAP std)
            - aflow_ratio_15min: Sector relative strength
    """
    features = {
        "vwap_z": vwap_z_score(price, vwap, (state.vwap_dev_var ** 0.5)),
        "rsi_mr_signal": rsi_mean_reversion_signal(rsi),
        "sector_rs": sector_relative_strength(stock_ret_15min, sector_avg_ret_15min),
        "orb_position": opening_range_breakout_position(
            price, state.orb_high, state.orb_low
        ) if state.orb_set else 0.5,
        "time_sin": time_features.get("time_sin", 0.0),
        "time_cos": time_features.get("time_cos", 0.0),
        "ret_15min": stock_ret_15min,
        "vol_normalised": state.vwap_dev_var ** 0.5,
        "aflow_ratio_15min": sector_relative_strength(
            stock_ret_15min, sector_avg_ret_15min
        ),
    }
    return features
