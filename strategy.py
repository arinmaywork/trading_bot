"""
strategy.py  — V2
=================
Strategy & Risk Engine: ML Ensemble Signal + Busseti Risk-Constrained Kelly.

Enhancement 4 integration:
  EnsembleSignalEngine.predict() replaces the static compute_alpha() formula.
  The ML signal ∈ [−1, 1] drives both direction and Kelly sizing.

Enhancement 5 — Busseti et al. (2016) Risk-Constrained Kelly:
  Reference: Busseti, Ryu, Boyd — "Risk-Constrained Kelly Gambling" (2016).
  Core idea: maximise E[log W] subject to the constraint that the probability
  of wealth falling below a target floor W_target is at most ε.

  Algorithm (bisection on position fraction f):
    1. Collect the empirical distribution of per-period returns r_i
       from the rolling FeatureStore (labelled forward log-returns).
    2. For a given fraction f, compute the empirical probability:
         P_ruin(f) = fraction of {r_i} where (1 + f × r_i) < W_floor
    3. Binary search on f ∈ [0, f_max]:
         If P_ruin(f) > ε_target → f is too large → upper = f
         If P_ruin(f) ≤ ε_target → f is safe → lower = f
    4. f* = lower bound at convergence (conservative)
    5. Apply GRI scaling and hard decay checks on f*

  Parameters:
    ε_target  = 0.05  (max 5% probability of ruin per period)
    W_floor   = 0.95  (wealth must not fall below 95% of current value)
    bisection_iters = 20 (always converges to tolerance < 0.001)

V2 SignalState additions:
    ml_signal        — raw ML ensemble signal ∈ [−1, 1]
    ml_confidence    — model agreement score
    ml_model_version — retraining timestamp
    busseti_f        — risk-constrained Kelly fraction from bisection
    mlofi            — V2 Multi-Level OFI
    aflow_ratio      — aggressive flow ratio
"""

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis.asyncio as aioredis

from alternative_data import SentimentResult
from config import settings
from data_ingestion import (
    get_latest_mlofi, get_latest_ofi, get_aggressive_flow, CandleAggregator
)
from geopolitical import GeopoliticalRiskIndex
from ml_signal import EnsembleSignalEngine, FeatureVector, SignalOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TradeDirection(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    FLAT = "FLAT"


# ---------------------------------------------------------------------------
# V2 SignalState — expanded to carry all V2 metadata
# ---------------------------------------------------------------------------

@dataclass
class SignalState:
    """Complete V2 signal bundle passed to the execution module."""
    symbol:               str
    timestamp:            datetime
    current_price:        float
    vwap:                 float
    # V1 microstructure
    ofi:                  float
    # V2 microstructure
    mlofi:                float
    aflow_ratio:          float
    # Sentiment
    sentiment_score:      float
    sentiment_class:      str
    # ML ensemble
    ml_signal:            float     # Primary signal from EnsembleSignalEngine
    ml_confidence:        float
    ml_model_version:     str
    ml_is_fallback:       bool
    # Legacy alpha fields (preserved for telemetry compatibility)
    alpha_raw:            float
    alpha:                float
    # Risk
    direction:            TradeDirection
    quantity:             int
    position_fraction:    float      # Busseti-constrained Kelly fraction
    busseti_f:            float      # Raw bisection output before GRI scaling
    vol_regime:           float      # Annualised realised vol
    # Geo
    geo_risk:             float
    geo_level:            str
    geo_alpha_multiplier: float
    geo_kelly_multiplier: float
    is_decayed:           bool
    rationale:            str

    @property
    def is_actionable(self) -> bool:
        return (
            self.direction != TradeDirection.FLAT
            and self.quantity > 0
            and not self.is_decayed
            and abs(self.ml_signal) >= settings.strategy.MIN_ALPHA_THRESHOLD
        )


# ---------------------------------------------------------------------------
# VWAP and realised vol  (unchanged from V1)
# ---------------------------------------------------------------------------

async def compute_rolling_vwap(
    redis_client: aioredis.Redis,
    symbol: str,
    lookback_minutes: int = 30,
) -> Optional[float]:
    now = datetime.now(timezone.utc)
    total_pv = total_vol = 0.0
    for i in range(lookback_minutes):
        ts  = now - timedelta(minutes=i)
        key = CandleAggregator._bucket_key(symbol, ts)
        raw = await redis_client.hgetall(key)
        if not raw:
            continue
        try:
            h = float(raw.get(b"high",   0))
            l = float(raw.get(b"low",    0))
            c = float(raw.get(b"close",  0))
            v = float(raw.get(b"volume", 0))
            if v > 0:
                total_pv  += ((h + l + c) / 3.0) * v
                total_vol += v
        except (ValueError, TypeError):
            continue
    return (total_pv / total_vol) if total_vol > 0 else None


async def compute_realised_vol(
    redis_client: aioredis.Redis,
    symbol: str,
    lookback_minutes: int = 20,
) -> float:
    now    = datetime.now(timezone.utc)
    closes: List[float] = []
    for i in range(lookback_minutes + 1):
        ts  = now - timedelta(minutes=i)
        key = CandleAggregator._bucket_key(symbol, ts)
        raw = await redis_client.hgetall(key)
        if raw and raw.get(b"close"):
            try:
                closes.append(float(raw[b"close"]))
            except (ValueError, TypeError):
                pass
    if len(closes) < 3:
        return 0.20   # default 20% vol when insufficient data
    log_returns = [
        math.log(closes[i] / closes[i + 1])
        for i in range(len(closes) - 1)
        if closes[i + 1] > 0
    ]
    if not log_returns:
        return 0.20
    mean_r   = sum(log_returns) / len(log_returns)
    # B-05 FIX: use Bessel correction (n-1) for unbiased sample variance
    variance = sum((r - mean_r) ** 2 for r in log_returns) / max(len(log_returns) - 1, 1)
    return math.sqrt(variance) * math.sqrt(252 * 375)


async def get_close_series(
    redis_client: aioredis.Redis,
    symbol: str,
    lookback: int = 20,
) -> List[float]:
    """Retrieve recent closing prices for RSI computation."""
    now    = datetime.now(timezone.utc)
    closes: List[float] = []
    for i in range(lookback + 1):
        ts  = now - timedelta(minutes=i)
        key = CandleAggregator._bucket_key(symbol, ts)
        raw = await redis_client.hgetall(key)
        if raw and raw.get(b"close"):
            try:
                closes.append(float(raw[b"close"]))
            except (ValueError, TypeError):
                pass
    return closes


# ---------------------------------------------------------------------------
# Enhancement 5: Busseti Risk-Constrained Kelly (2016)
# ---------------------------------------------------------------------------

def busseti_kelly_bisection(
    returns: List[float],
    epsilon: float = 0.05,
    w_floor: float = 0.95,
    f_max:   float = 1.0,
    n_iters: int   = 20,
) -> float:
    """
    Bisection algorithm implementing the Busseti, Ryu & Boyd (2016)
    risk-constrained Kelly criterion.

    Finds the maximum fraction f* such that:
        P(W_{t+1} / W_t < w_floor) ≤ epsilon

    Equivalently, finds f* = argmax E[log(1 + f·r)]
    subject to: empirical CVaR constraint ≤ log(w_floor).

    Args:
        returns:  Empirical distribution of per-period returns r_i
                  (forward log-returns from FeatureStore labelled observations)
        epsilon:  Maximum allowed ruin probability (default 5%)
        w_floor:  Minimum acceptable wealth ratio (default 0.95 = max 5% drawdown)
        f_max:    Upper bound for bisection (typically half-Kelly or 1.0)
        n_iters:  Bisection iterations (20 → precision < 0.001 on [0, 1])

    Returns:
        f* — risk-constrained optimal fraction ∈ [0, f_max]
        Returns 0.0 if all returns are non-positive (no safe fraction exists).

    Mathematical note:
        The constraint P(1 + f·r < w_floor) ≤ ε is equivalent to requiring
        that the ε-quantile of (1 + f·r) is ≥ w_floor.
        This is solved via bisection: monotone decreasing P_ruin(f) in f.
    """
    if not returns or len(returns) < 5:
        return 0.0

    returns_arr = np.array(returns, dtype=np.float64)

    def p_ruin(f: float) -> float:
        """Empirical P(1 + f·r < w_floor)."""
        terminal_wealth = 1.0 + f * returns_arr
        return float(np.mean(terminal_wealth < w_floor))

    # Guard: if even f=0 has issues, return 0
    if p_ruin(0.0) > epsilon:
        return 0.0

    # If even full f_max is safe, return f_max (no constraint binding)
    if p_ruin(f_max) <= epsilon:
        return f_max

    # Bisection: find largest f where P_ruin(f) ≤ epsilon
    lower, upper = 0.0, f_max
    for _ in range(n_iters):
        mid = (lower + upper) / 2.0
        if p_ruin(mid) > epsilon:
            upper = mid   # too aggressive — decrease
        else:
            lower = mid   # safe — can increase

    return round(lower, 6)


async def get_return_distribution(
    redis_client: aioredis.Redis,
    symbol: str,
) -> List[float]:
    """
    Retrieve empirical return distribution from FeatureStore for Busseti Kelly.
    Falls back to a synthetic normal distribution if insufficient data.
    """
    key = f"ml:features:{symbol}"
    try:
        raw_list = await redis_client.lrange(key, 0, -1)
        returns: List[float] = []
        for r in raw_list:
            try:
                d = json.loads(r)
                ret = d.get("fwd_log_return")
                if ret is not None:
                    returns.append(float(ret))
            except Exception:
                pass
        return returns
    except Exception:
        return []


# ---------------------------------------------------------------------------
# RiskManager — V2 with Busseti Kelly
# ---------------------------------------------------------------------------

class RiskManager:
    """
    V2 RiskManager: Busseti (2016) risk-constrained Kelly with GRI scaling.

    Pipeline:
      1. Fetch empirical return distribution from Redis FeatureStore
      2. Run bisection → f_busseti (maximum safe fraction given ruin constraint)
      3. Apply GRI.kelly_multiplier scaling → f_geo
      4. Apply GRI position cap → f_final
      5. Hard decay checks (vol spike, GRI threshold)

    Parameters:
      ε = 0.05  → max 5% probability of a ruin event per period
      W_floor = 0.95  → ruin = wealth falling below 95% of current
    """

    BUSSETI_EPSILON = 0.05
    BUSSETI_W_FLOOR = 0.95
    BISECTION_ITERS = 20

    def __init__(self, capital: float) -> None:
        self._capital        = capital
        self._cfg            = settings.strategy
        self._baseline_vols: Dict[str, float] = {}

    def _update_baseline_vol(self, symbol: str, vol: float) -> None:
        prev = self._baseline_vols.get(symbol, vol)
        self._baseline_vols[symbol] = 0.9 * prev + 0.1 * vol

    def _should_decay(
        self,
        symbol: str,
        vol: float,
        gri: GeopoliticalRiskIndex,
    ) -> Tuple[bool, str]:
        baseline = self._baseline_vols.get(symbol, vol)
        if baseline > 0 and vol > self._cfg.VOL_SPIKE_THRESHOLD * baseline:
            return True, (
                f"Vol spike: σ={vol:.3f} > "
                f"{self._cfg.VOL_SPIKE_THRESHOLD}× baseline={baseline:.3f}"
            )
        if gri.composite > self._cfg.GEOPOLITICAL_RISK_THRESHOLD:
            return True, (
                f"GRI={gri.composite:.3f} ({gri.level}) "
                f"exceeds threshold {self._cfg.GEOPOLITICAL_RISK_THRESHOLD}"
            )
        return False, ""

    async def compute_quantity(
        self,
        symbol:        str,
        ml_signal:     float,
        current_price: float,
        vol:           float,
        gri:           GeopoliticalRiskIndex,
        redis_client:  aioredis.Redis,
    ) -> Tuple[int, float, float, bool, str]:
        """
        Full Busseti + GRI position sizing.

        Returns:
            (quantity, f_final, f_busseti, is_decayed, reason)
        """
        if current_price <= 0:
            return 0, 0.0, 0.0, False, "Invalid price"

        self._update_baseline_vol(symbol, vol)

        # Hard decay check
        decayed, reason = self._should_decay(symbol, vol, gri)
        if decayed:
            logger.warning("RiskManager DECAY %s: %s", symbol, reason)
            return 0, 0.0, 0.0, True, reason

        if abs(ml_signal) < self._cfg.MIN_ALPHA_THRESHOLD:
            return 0, 0.0, 0.0, False, f"Signal below threshold: {ml_signal:.6f}"

        # Fetch return distribution from FeatureStore
        returns = await get_return_distribution(redis_client, symbol)

        if len(returns) >= 20:
            # Busseti (2016) bisection
            # Scale ml_signal as directional weight on returns
            # (positive signal → use raw returns; negative → flip sign)
            directional_returns = [r * math.copysign(1.0, ml_signal) for r in returns]
            f_busseti = busseti_kelly_bisection(
                returns   = directional_returns,
                epsilon   = self.BUSSETI_EPSILON,
                w_floor   = self.BUSSETI_W_FLOOR,
                f_max     = self._cfg.MAX_POSITION_FRACTION * 2,
                n_iters   = self.BISECTION_ITERS,
            )
            method = f"Busseti(ε={self.BUSSETI_EPSILON}, Wf={self.BUSSETI_W_FLOOR}, n={len(returns)})"
        else:
            # Not enough data — fall back to half-Kelly
            vol_safe   = max(vol, 0.05)
            full_kelly = abs(ml_signal) / (vol_safe ** 2)
            f_busseti  = min(self._cfg.KELLY_FRACTION * full_kelly,
                             self._cfg.MAX_POSITION_FRACTION)
            method = "HalfKelly-fallback (insufficient return data)"

        # GRI graduated scaling (Layer 2)
        f_geo  = f_busseti * gri.kelly_multiplier

        # GRI position cap tightening (Layer 3)
        geo_cap = gri.max_position_fraction_cap(self._cfg.MAX_POSITION_FRACTION)
        f_final = min(f_geo, geo_cap)

        if f_final <= 0:
            return 0, f_final, f_busseti, False, "Zero position after GRI scaling"

        raw_qty  = int((f_final * self._capital) / current_price)
        cap_qty  = int((geo_cap * self._capital) / current_price)
        final    = min(raw_qty, cap_qty, 10_000)   # hard absolute cap

        # ── R-10: Transaction cost filter ────────────────────────────────────
        # Zerodha charges ₹20/order flat (or 0.03% if lower) + 0.025% STT on
        # the sell side. For a round trip that's up to ₹40 + STT.
        # Only place the trade if the expected P&L exceeds total round-trip cost.
        # If the Kelly qty is too small, try bumping to MIN_TRADE_VALUE first.
        if final > 0:
            cfg_costs = settings.strategy

            def _round_trip_cost(qty: int) -> float:
                order_val   = current_price * qty
                brok_buy    = min(cfg_costs.BROKERAGE_PER_ORDER, cfg_costs.BROKERAGE_PCT * order_val)
                brok_sell   = min(cfg_costs.BROKERAGE_PER_ORDER, cfg_costs.BROKERAGE_PCT * order_val)
                stt_sell    = cfg_costs.STT_INTRADAY_SELL_RATE * order_val
                exch        = cfg_costs.EXCHANGE_CHARGE_RATE * order_val * 2  # both sides
                return brok_buy + brok_sell + stt_sell + exch

            def _expected_pnl(qty: int) -> float:
                return abs(ml_signal) * current_price * qty

            cost   = _round_trip_cost(final)
            exp_pl = _expected_pnl(final)

            if exp_pl < cost:
                # Kelly qty is too small — try bumping up to MIN_TRADE_VALUE
                min_qty = math.ceil(cfg_costs.MIN_TRADE_VALUE / current_price)
                min_qty = max(min_qty, final)
                max_allowed = int(cfg_costs.MAX_POSITION_FRACTION * self._capital / current_price)
                if min_qty <= max_allowed and _expected_pnl(min_qty) > _round_trip_cost(min_qty):
                    logger.info(
                        "CostFilter %s: bumping qty %d→%d to clear brokerage hurdle "
                        "(exp_pnl=₹%.2f cost=₹%.2f)",
                        symbol, final, min_qty, _expected_pnl(min_qty), _round_trip_cost(min_qty),
                    )
                    final = min_qty
                else:
                    logger.info(
                        "CostFilter %s: skipping — exp_pnl=₹%.2f < cost=₹%.2f "
                        "(brokerage+STT hurdle not cleared even at min_trade_value=₹%.0f)",
                        symbol, exp_pl, cost, cfg_costs.MIN_TRADE_VALUE,
                    )
                    return 0, f_final, f_busseti, False, (
                        f"Trade cost ₹{cost:.2f} > expected P&L ₹{exp_pl:.2f} — "
                        f"increase capital or wait for stronger signal"
                    )
        # ── End cost filter ──────────────────────────────────────────────────

        reason = (
            f"{method} | f_busseti={f_busseti:.4f} "
            f"→ f_geo={f_geo:.4f} (GRI_mult={gri.kelly_multiplier:.2f}) "
            f"→ f_final={f_final:.4f} | qty_raw={raw_qty} cap={cap_qty} "
            f"| GRI={gri.composite:.3f} ({gri.level})"
        )
        logger.info("RiskManager %s: qty=%d | %s", symbol, final, reason)
        return final, f_final, f_busseti, False, reason


# ---------------------------------------------------------------------------
# GRI Alpha multiplier (V2 — kept for telemetry; ML signal handles dampening)
# ---------------------------------------------------------------------------

def geo_alpha_multiplier(gri_composite: float) -> float:
    """Piecewise linear GRI dampener for display purposes."""
    if gri_composite <= 0.25:
        return 1.0
    if gri_composite >= 0.65:
        return 0.0
    return round(1.0 - (gri_composite - 0.25) / (0.65 - 0.25), 4)


# ---------------------------------------------------------------------------
# StrategyEngine — V2
# ---------------------------------------------------------------------------

class StrategyEngine:
    """
    V2 orchestrator: combines MLOFI, aggressive flow, ML ensemble signal,
    Busseti Kelly sizing, and GRI dampening.
    """

    def __init__(
        self,
        redis_client:    aioredis.Redis,
        risk_manager:    RiskManager,
        ml_engine:       EnsembleSignalEngine,
    ) -> None:
        self._redis = redis_client
        self._risk  = risk_manager
        self._ml    = ml_engine

    async def evaluate(
        self,
        symbol:        str,
        current_price: float,
        sentiment:     SentimentResult,
        gri:           GeopoliticalRiskIndex,
        gpr_normalised: float = 0.10,
        vol_regime:    str    = "MODERATE",
    ) -> SignalState:
        """
        Full V2 signal pipeline:
          0. Label the previous feature vector with current price (training data)
          1. VWAP + OFI + MLOFI + Aggressive Flow from Redis
          2. Realised vol + close series for RSI
          3. Build FeatureVector
          4. EnsembleSignalEngine.predict() → ML signal
          5. Busseti Kelly sizing with GRI constraints
          6. Trade direction determination
        """

        # ── 0. Labeling (for training) ──────────────────────────────────
        await self._ml.label_and_store(symbol, current_price)

        # ── 1. Microstructure ──────────────────────────────────────────
        vwap = await compute_rolling_vwap(
            self._redis, symbol, settings.strategy.VWAP_LOOKBACK
        )
        if vwap is None or vwap <= 0:
            vwap = current_price

        ofi   = await get_latest_ofi(self._redis, symbol)
        mlofi = await get_latest_mlofi(self._redis, symbol)
        aflow = await get_aggressive_flow(self._redis, symbol)

        # ── 2. Volatility + close series ──────────────────────────────
        vol    = await compute_realised_vol(self._redis, symbol, lookback_minutes=20)
        closes = await get_close_series(self._redis, symbol, lookback=20)

        # ── 3. Feature vector construction ────────────────────────────
        fv = self._ml.build_feature_vector(
            symbol          = symbol,
            mlofi           = mlofi,
            ofi             = ofi,
            aflow_ratio     = aflow.get("ratio", 0.0),
            aflow_delta     = aflow.get("delta", 0.0),
            price           = current_price,
            vwap            = vwap,
            closes          = closes,
            sentiment_score = sentiment.sentiment_score,
            vol             = vol,
            gri_composite   = gri.composite,
            gpr_normalised  = gpr_normalised,
            vol_regime      = vol_regime,
        )

        # ── 4. ML ensemble prediction ─────────────────────────────────
        signal_out: SignalOutput = self._ml.predict(symbol, fv)

        # Apply GRI alpha dampening to ML signal
        geo_mult  = geo_alpha_multiplier(gri.composite)
        ml_signal = signal_out.signal * geo_mult

        # For telemetry compatibility — treat ML signal as alpha
        alpha_raw = signal_out.signal
        alpha_adj = ml_signal

        # ── 5. Busseti Kelly sizing ────────────────────────────────────
        qty, f_final, f_busseti, is_decayed, size_reason = (
            await self._risk.compute_quantity(
                symbol        = symbol,
                ml_signal     = ml_signal,
                current_price = current_price,
                vol           = vol,
                gri           = gri,
                redis_client  = self._redis,
            )
        )

        # ── 6. Direction ───────────────────────────────────────────────
        if is_decayed or abs(ml_signal) < settings.strategy.MIN_ALPHA_THRESHOLD:
            direction = TradeDirection.FLAT
            qty       = 0
        elif ml_signal > 0:
            direction = TradeDirection.BUY
        else:
            direction = TradeDirection.SELL

        rationale = (
            f"ML: signal={signal_out.signal:+.4f} geo_adj={ml_signal:+.4f} "
            f"({'fallback' if signal_out.is_fallback else f'v{signal_out.model_version}'}) "
            f"conf={signal_out.confidence:.2f} dir={direction.value} "
            f"| MLOFI={mlofi:+.3f} OFI={ofi:+.3f} AFR={aflow.get('ratio',0):+.3f} "
            f"| VWAP={vwap:.2f} σ={vol:.2%} "
            f"| S={sentiment.sentiment_score:+.3f} ({sentiment.sentiment_classification}) "
            f"| GRI={gri.composite:.3f} ({gri.level}) geo_mult={geo_mult:.2f} "
            f"| {size_reason}"
        )

        return SignalState(
            symbol               = symbol,
            timestamp            = datetime.now(timezone.utc),
            current_price        = current_price,
            vwap                 = vwap,
            ofi                  = ofi,
            mlofi                = mlofi,
            aflow_ratio          = aflow.get("ratio", 0.0),
            sentiment_score      = sentiment.sentiment_score,
            sentiment_class      = sentiment.sentiment_classification,
            ml_signal            = ml_signal,
            ml_confidence        = signal_out.confidence,
            ml_model_version     = signal_out.model_version,
            ml_is_fallback       = signal_out.is_fallback,
            alpha_raw            = alpha_raw,
            alpha                = alpha_adj,
            direction            = direction,
            quantity             = qty,
            position_fraction    = f_final,
            busseti_f            = f_busseti,
            vol_regime           = vol,
            geo_risk             = gri.composite,
            geo_level            = gri.level,
            geo_alpha_multiplier = geo_mult,
            geo_kelly_multiplier = gri.kelly_multiplier,
            is_decayed           = is_decayed,
            rationale            = rationale,
        )
