from __future__ import annotations
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Enum for market regime classification."""
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeConfig:
    """Configuration parameters for regime detection."""
    adx_trending_threshold: float = 25.0
    adx_mean_revert_threshold: float = 20.0
    vix_volatile_threshold: float = 20.0
    nifty_ret_trending_threshold: float = 0.003  # 0.3%
    bollinger_bw_mean_revert_threshold: float = 0.02  # 2%
    vol_annualised_volatile_threshold: float = 0.25
    update_interval_s: float = 900.0  # re-evaluate every 15 min


@dataclass
class RegimeState:
    """Current state of market regime and supporting metrics."""
    current_regime: MarketRegime = MarketRegime.UNKNOWN
    confidence: float = 0.0
    last_update: float = 0.0  # monotonic time
    adx_value: float = 0.0
    vix_value: float = 0.0
    nifty_ret_60min: float = 0.0
    bollinger_bw: float = 0.0
    realised_vol_1h: float = 0.0


def compute_adx(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14
) -> float:
    """
    Compute Average Directional Index (ADX) from OHLC data.

    Args:
        highs: List of high prices
        lows: List of low prices
        closes: List of closing prices
        period: ADX period (default 14)

    Returns:
        ADX value in range [0, 100]

    Requires at least period + 1 bars for calculation.
    """
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        logger.warning(
            f"Not enough bars for ADX calculation. Required: {period + 1}, "
            f"Got: {len(highs)}"
        )
        return 0.0

    # Calculate True Range (TR)
    tr_values = []
    for i in range(1, len(closes)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        tr = max(high_low, high_close, low_close)
        tr_values.append(tr)

    # Calculate Directional Movements
    plus_dm = []
    minus_dm = []
    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        plus_val = up_move if up_move > 0 and up_move > down_move else 0.0
        minus_val = down_move if down_move > 0 and down_move > up_move else 0.0

        plus_dm.append(plus_val)
        minus_dm.append(minus_val)

    # Smooth TR and DM over period
    atr = sum(tr_values[:period]) / period
    plus_dm_sum = sum(plus_dm[:period]) / period
    minus_dm_sum = sum(minus_dm[:period]) / period

    for i in range(period, len(tr_values)):
        atr = (atr * (period - 1) + tr_values[i]) / period
        plus_dm_sum = (plus_dm_sum * (period - 1) + plus_dm[i]) / period
        minus_dm_sum = (minus_dm_sum * (period - 1) + minus_dm[i]) / period

    # Calculate Directional Indicators
    if atr == 0:
        return 0.0

    plus_di = 100.0 * plus_dm_sum / atr
    minus_di = 100.0 * minus_dm_sum / atr

    # Calculate ADX
    dx_values = []
    for _ in range(period):
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0.0
        else:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
        dx_values.append(dx)

    adx = sum(dx_values) / period

    logger.debug(f"Computed ADX: {adx:.2f}")
    return min(adx, 100.0)


def compute_bollinger_bandwidth(
    closes: List[float],
    period: int = 20,
    num_std: float = 2.0
) -> float:
    """
    Compute Bollinger Band bandwidth.

    Bandwidth = (upper - lower) / middle = 2 * num_std * std / mean

    Args:
        closes: List of closing prices
        period: SMA period (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Bandwidth as decimal (e.g., 0.02 = 2%)
    """
    if len(closes) < period:
        logger.warning(
            f"Not enough closes for BB calculation. Required: {period}, "
            f"Got: {len(closes)}"
        )
        return 0.0

    # Calculate SMA
    recent_closes = closes[-period:]
    sma = sum(recent_closes) / period

    if sma == 0:
        return 0.0

    # Calculate standard deviation
    variance = sum((c - sma) ** 2 for c in recent_closes) / period
    std_dev = math.sqrt(variance)

    # Calculate bandwidth
    bandwidth = (2.0 * num_std * std_dev) / sma

    logger.debug(f"Computed Bollinger Bandwidth: {bandwidth:.4f}")
    return bandwidth


class RegimeDetector:
    """
    Market regime detector using ADX, VIX, returns, and volatility.

    Classifies market into TRENDING, MEAN_REVERTING, VOLATILE, or UNKNOWN.
    """

    def __init__(self, config: RegimeConfig = None):
        """
        Initialize regime detector.

        Args:
            config: RegimeConfig instance. Uses defaults if None.
        """
        self._config = config or RegimeConfig()
        self._state = RegimeState()
        logger.info(
            f"RegimeDetector initialized with ADX thresholds: "
            f"trending={self._config.adx_trending_threshold}, "
            f"mean_revert={self._config.adx_mean_revert_threshold}"
        )

    def update(
        self,
        nifty_highs: List[float],
        nifty_lows: List[float],
        nifty_closes: List[float],
        india_vix: float,
        nifty_ret_60min: float,
        realised_vol_1h: float
    ) -> RegimeState:
        """
        Re-evaluate market regime using latest OHLC and indicators.

        Only updates state if update_interval has elapsed since last update.

        Args:
            nifty_highs: List of Nifty 50 high prices
            nifty_lows: List of Nifty 50 low prices
            nifty_closes: List of Nifty 50 closing prices
            india_vix: Current India VIX level
            nifty_ret_60min: 60-minute Nifty 50 return (as decimal)
            realised_vol_1h: 1-hour realized volatility (annualized)

        Returns:
            Updated RegimeState
        """
        current_time = time.monotonic()

        # Check if update interval has elapsed
        if current_time - self._state.last_update < self._config.update_interval_s:
            logger.debug(
                f"Update interval not elapsed. "
                f"Next update in {self._config.update_interval_s - (current_time - self._state.last_update):.1f}s"
            )
            return self._state

        # Compute metrics
        adx = compute_adx(nifty_highs, nifty_lows, nifty_closes)
        bollinger_bw = compute_bollinger_bandwidth(nifty_closes)

        # Classify regime
        regime, confidence = self.classify(
            adx=adx,
            vix=india_vix,
            nifty_ret=nifty_ret_60min,
            bollinger_bw=bollinger_bw,
            vol_1h=realised_vol_1h
        )

        # Update state
        self._state.current_regime = regime
        self._state.confidence = confidence
        self._state.last_update = current_time
        self._state.adx_value = adx
        self._state.vix_value = india_vix
        self._state.nifty_ret_60min = nifty_ret_60min
        self._state.bollinger_bw = bollinger_bw
        self._state.realised_vol_1h = realised_vol_1h

        logger.info(
            f"Regime updated: {regime.value} (confidence={confidence:.2f}, "
            f"ADX={adx:.1f}, VIX={india_vix:.1f}, vol_1h={realised_vol_1h:.3f})"
        )

        return self._state

    def classify(
        self,
        adx: float,
        vix: float,
        nifty_ret: float,
        bollinger_bw: float,
        vol_1h: float
    ) -> tuple[MarketRegime, float]:
        """
        Pure classification logic without state mutation.

        Priority:
        1. VOLATILE: if VIX > threshold OR vol_1h > threshold
        2. TRENDING: if ADX > threshold AND abs(nifty_ret) > threshold
        3. MEAN_REVERTING: if ADX < threshold AND bollinger_bw < threshold
        4. UNKNOWN: default fallback

        Args:
            adx: ADX value [0, 100]
            vix: India VIX level
            nifty_ret: 60-minute return as decimal
            bollinger_bw: Bollinger bandwidth as decimal
            vol_1h: 1-hour realized volatility (annualized)

        Returns:
            Tuple of (MarketRegime, confidence) where confidence in [0, 1]
        """
        # Check VOLATILE first (highest priority)
        if vix > self._config.vix_volatile_threshold or \
           vol_1h > self._config.vol_annualised_volatile_threshold:
            confidence = min(
                (vix / 25.0 + vol_1h / 0.3) / 2.0,  # normalize to ~1.0
                1.0
            )
            logger.debug(f"Classified as VOLATILE (VIX={vix:.1f}, vol_1h={vol_1h:.3f})")
            return MarketRegime.VOLATILE, confidence

        # Check TRENDING
        abs_ret = abs(nifty_ret)
        if adx > self._config.adx_trending_threshold and \
           abs_ret > self._config.nifty_ret_trending_threshold:
            # Confidence based on how strongly trending
            adx_norm = min(adx / 50.0, 1.0)  # normalize ADX
            ret_norm = min(abs_ret / 0.01, 1.0)  # normalize return
            confidence = (adx_norm + ret_norm) / 2.0
            logger.debug(
                f"Classified as TRENDING (ADX={adx:.1f}, ret={nifty_ret:.4f})"
            )
            return MarketRegime.TRENDING, confidence

        # Check MEAN_REVERTING
        if adx < self._config.adx_mean_revert_threshold and \
           bollinger_bw < self._config.bollinger_bw_mean_revert_threshold:
            # Confidence based on how tight the bands are
            bw_norm = 1.0 - min(bollinger_bw / 0.05, 1.0)  # inverted
            adx_norm = 1.0 - min(adx / 30.0, 1.0)  # inverted (lower is better)
            confidence = (bw_norm + adx_norm) / 2.0
            logger.debug(
                f"Classified as MEAN_REVERTING (ADX={adx:.1f}, BW={bollinger_bw:.4f})"
            )
            return MarketRegime.MEAN_REVERTING, confidence

        # Default: UNKNOWN
        logger.debug(
            f"Classified as UNKNOWN (ADX={adx:.1f}, VIX={vix:.1f}, "
            f"BW={bollinger_bw:.4f}, vol_1h={vol_1h:.3f})"
        )
        return MarketRegime.UNKNOWN, 0.0

    @property
    def state(self) -> RegimeState:
        """Get current regime state."""
        return self._state

    @property
    def regime(self) -> MarketRegime:
        """Get current market regime."""
        return self._state.current_regime
