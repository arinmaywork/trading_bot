"""
ml_signal.py  — V2
==================
Enhancement 4: Machine Learning Ensemble Signal Generator.

Architecture: Two-stage ensemble meta-learner
  Stage 1 — Base learners trained independently:
    • XGBoostSignal:  gradient-boosted trees on microstructure features
    • RidgeSignal:    L2-regularised linear regression (interpretable baseline)
  Stage 2 — Meta-learner:
    • RidgeRegression on [xgb_pred, ridge_pred] → final signal ∈ [−1, 1]

Feature Vector (11 dimensions per observation):
  [0]  mlofi              — Multi-Level OFI (primary microstructure signal)
  [1]  ofi                — Level-0 OFI (V1 compat; correlated with mlofi)
  [2]  aflow_ratio        — Aggressive Flow Ratio ∈ [−1, 1]
  [3]  aflow_delta_norm   — Normalised aggressive delta (buy−sell) / capital
  [4]  vwap_dev           — (price − VWAP) / VWAP ∈ ~[−0.05, 0.05]
  [5]  sentiment_score    — Gemini/Agent score ∈ [−1, 1]
  [6]  vol_normalised     — Annualised vol / 1.0 (already dimensionless)
  [7]  gri_composite      — GRI ∈ [0, 1] (from geopolitical module)
  [8]  gpr_normalised     — Caldara-Iacoviello GPR ∈ [0, 1]
  [9]  rsi_normalised     — RSI / 100 ∈ [0, 1]
  [10] vol_regime_encoded — 0=LOW, 0.33=MOD, 0.67=HIGH, 1.0=EXTREME

Target: rolling 1-minute forward log-return (sign gives direction,
        magnitude gives Kelly input — replaces static alpha formula).

Training:
  • Rolling window of 200 in-sample observations stored in Redis
  • Retrain every 30 minutes (async, non-blocking)
  • Minimum 50 observations required before model replaces static alpha
  • StandardScaler applied per retraining cycle (fitted on training data)

Prediction:
  • Returns SignalOutput: {signal, confidence, direction, feature_importances}
  • If model not ready → falls back to V1 static alpha (gradual cutover)
"""

import asyncio
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Safe import of ML libs — model degrades gracefully if not installed
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not installed. Run: pip install xgboost")
    _XGB_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Run: pip install scikit-learn")
    _SKLEARN_AVAILABLE = False

_ML_READY = _XGB_AVAILABLE and _SKLEARN_AVAILABLE


# ---------------------------------------------------------------------------
# Feature and Signal structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """Single observation for training or inference."""
    mlofi:              float
    ofi:                float
    aflow_ratio:        float
    aflow_delta_norm:   float
    vwap_dev:           float
    sentiment_score:    float
    vol_normalised:     float
    gri_composite:      float
    gpr_normalised:     float
    rsi_normalised:     float
    vol_regime_encoded: float
    # Metadata for labeling (not features)
    reference_price:    float = 0.0
    reference_ts:       float = field(default_factory=time.monotonic)  # wall-clock for delay
    # Target (None during inference)
    fwd_log_return:     Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    FEATURE_NAMES = [
        "mlofi", "ofi", "aflow_ratio", "aflow_delta_norm",
        "vwap_dev", "sentiment_score", "vol_normalised",
        "gri_composite", "gpr_normalised", "rsi_normalised",
        "vol_regime_encoded",
    ]

    def to_array(self) -> List[float]:
        return [
            self.mlofi, self.ofi, self.aflow_ratio, self.aflow_delta_norm,
            self.vwap_dev, self.sentiment_score, self.vol_normalised,
            self.gri_composite, self.gpr_normalised, self.rsi_normalised,
            self.vol_regime_encoded,
        ]

    def to_dict(self) -> Dict[str, Any]:
        d = {n: v for n, v in zip(self.FEATURE_NAMES, self.to_array())}
        if self.fwd_log_return is not None:
            d["fwd_log_return"] = self.fwd_log_return
        d["reference_price"] = self.reference_price
        d["ts"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureVector":
        return cls(
            mlofi              = float(d.get("mlofi", 0)),
            ofi                = float(d.get("ofi", 0)),
            aflow_ratio        = float(d.get("aflow_ratio", 0)),
            aflow_delta_norm   = float(d.get("aflow_delta_norm", 0)),
            vwap_dev           = float(d.get("vwap_dev", 0)),
            sentiment_score    = float(d.get("sentiment_score", 0)),
            vol_normalised     = float(d.get("vol_normalised", 0)),
            gri_composite      = float(d.get("gri_composite", 0)),
            gpr_normalised     = float(d.get("gpr_normalised", 0)),
            rsi_normalised     = float(d.get("rsi_normalised", 0)),
            vol_regime_encoded = float(d.get("vol_regime_encoded", 0)),
            reference_price    = float(d.get("reference_price", 0)),
            fwd_log_return     = d.get("fwd_log_return"),
        )


@dataclass
class Tier1FeatureVector:
    """Tier-1 microstructure features for short-term (1-15 min) prediction."""
    mlofi: float
    obi_momentum: float        # OBI acceleration
    aflow_ratio: float
    trade_arrival_rate: float  # tick rate vs baseline
    spread_z: float            # bid-ask spread z-score
    vwpp: float                # volume-weighted price pressure
    ret_1min: float            # 1-minute log return
    ret_5min: float            # 5-minute log return
    reference_price: float = 0.0
    reference_ts: float = field(default_factory=time.monotonic)
    fwd_log_return: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    FEATURE_NAMES = [
        "mlofi", "obi_momentum", "aflow_ratio", "trade_arrival_rate",
        "spread_z", "vwpp", "ret_1min", "ret_5min",
    ]

    def to_array(self) -> List[float]:
        return [self.mlofi, self.obi_momentum, self.aflow_ratio,
                self.trade_arrival_rate, self.spread_z, self.vwpp,
                self.ret_1min, self.ret_5min]

    def to_dict(self) -> Dict[str, Any]:
        d = {n: v for n, v in zip(self.FEATURE_NAMES, self.to_array())}
        if self.fwd_log_return is not None:
            d["fwd_log_return"] = self.fwd_log_return
        d["reference_price"] = self.reference_price
        d["ts"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tier1FeatureVector":
        return cls(**{k: float(d.get(k, 0)) for k in cls.FEATURE_NAMES},
                   reference_price=float(d.get("reference_price", 0)),
                   fwd_log_return=d.get("fwd_log_return"))


@dataclass
class Tier2FeatureVector:
    """Tier-2 mean-reversion features for medium-term (30-60 min) prediction."""
    vwap_z: float              # standardised VWAP deviation
    rsi_mr_signal: float       # RSI mean-reversion signal
    sector_rs: float           # sector relative strength
    orb_position: float        # opening range breakout position
    time_sin: float            # cyclical time-of-day (sin)
    time_cos: float            # cyclical time-of-day (cos)
    ret_15min: float           # 15-minute log return
    vol_normalised: float      # annualised volatility
    aflow_ratio_15min: float   # 15-min avg aggressive flow
    reference_price: float = 0.0
    reference_ts: float = field(default_factory=time.monotonic)
    fwd_log_return: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    FEATURE_NAMES = [
        "vwap_z", "rsi_mr_signal", "sector_rs", "orb_position",
        "time_sin", "time_cos", "ret_15min", "vol_normalised",
        "aflow_ratio_15min",
    ]

    def to_array(self) -> List[float]:
        return [self.vwap_z, self.rsi_mr_signal, self.sector_rs,
                self.orb_position, self.time_sin, self.time_cos,
                self.ret_15min, self.vol_normalised, self.aflow_ratio_15min]

    def to_dict(self) -> Dict[str, Any]:
        d = {n: v for n, v in zip(self.FEATURE_NAMES, self.to_array())}
        if self.fwd_log_return is not None:
            d["fwd_log_return"] = self.fwd_log_return
        d["reference_price"] = self.reference_price
        d["ts"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tier2FeatureVector":
        return cls(**{k: float(d.get(k, 0)) for k in cls.FEATURE_NAMES},
                   reference_price=float(d.get("reference_price", 0)),
                   fwd_log_return=d.get("fwd_log_return"))


@dataclass
class SignalOutput:
    """ML ensemble output."""
    symbol:              str
    signal:              float     # ∈ [−1, 1] — primary signal strength + direction
    confidence:          float     # ∈ [0, 1]  — model confidence
    xgb_pred:            float     # XGBoost base learner output
    ridge_pred:          float     # Ridge base learner output
    meta_pred:           float     # Meta-learner output (= signal if ensemble ready)
    feature_importances: Dict[str, float]
    model_version:       str       # timestamp of last retraining
    is_fallback:         bool      # True = using V1 static alpha (model not ready)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def direction(self) -> str:
        if self.signal > 0.001:
            return "BUY"
        elif self.signal < -0.001:
            return "SELL"
        return "FLAT"


# ---------------------------------------------------------------------------
# Vol regime encoder
# ---------------------------------------------------------------------------

_VOL_REGIME_MAP = {"LOW": 0.0, "MODERATE": 0.33, "HIGH": 0.67, "EXTREME": 1.0}


def encode_vol_regime(vol_regime: str) -> float:
    return _VOL_REGIME_MAP.get(vol_regime.upper(), 0.33)


def compute_rsi(closes: List[float], period: int = 14) -> float:
    """
    Wilder's RSI from a list of closing prices.
    B-07 FIX: uses Wilder's EMA smoothing (not simple average) for gains/losses.
    Simple average gives values 3-8 points off vs. industry-standard RSI.
    """
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    if not gains:
        return 50.0
    # Wilder's initial SMA seed over first `period` bars
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    # Wilder's EMA smoothing: avg = (prev_avg * (period-1) + current) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 2)


# ---------------------------------------------------------------------------
# Feature Store (Redis-backed rolling window)
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    Maintains a rolling window of labelled feature vectors per symbol in Redis.
    Window size: MAX_WINDOW observations (configurable).
    """

    # Bug 2.2 fix: 200 observations is guaranteed overfitting for XGBoost
    # (80 estimators × depth 4 ≈ 400+ internal splits on 11 features).
    # Raising to 5000 gives roughly 12-13 trading days of 1-minute bars per
    # symbol, which is the minimum for stable out-of-sample behaviour.
    # MIN_TRAIN is also raised so we don't fit on 50 samples.
    MAX_WINDOW   = 5000
    MIN_TRAIN    = 500
    REDIS_TTL    = 7 * 86400   # 7 days — aligns with the larger window

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis = redis_client

    def _key(self, symbol: str) -> str:
        return f"ml:features:{symbol}"

    async def append(self, symbol: str, fv: FeatureVector) -> int:
        """Append a labelled observation. Returns current window size."""
        key     = self._key(symbol)
        payload = json.dumps(fv.to_dict())
        # Redis-fix: pipeline 4 commands into one round-trip instead of 4
        # separate connections. With 50+ symbols calling append() each cycle,
        # this cuts connection pressure by 75%.
        pipe = self._redis.pipeline()
        try:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -self.MAX_WINDOW, -1)
            pipe.expire(key, self.REDIS_TTL)
            pipe.llen(key)
            results = await pipe.execute()
            return int(results[-1])  # llen result
        finally:
            await pipe.reset()

    async def get_all(self, symbol: str) -> List[FeatureVector]:
        """Retrieve all stored observations for a symbol."""
        key  = self._key(symbol)
        raw  = await self._redis.lrange(key, 0, -1)
        out: List[FeatureVector] = []
        for r in raw:
            try:
                out.append(FeatureVector.from_dict(json.loads(r)))
            except Exception:
                pass
        return out

    async def window_size(self, symbol: str) -> int:
        return await self._redis.llen(self._key(symbol))

    async def ready(self, symbol: str) -> bool:
        return await self.window_size(symbol) >= self.MIN_TRAIN


class TieredFeatureStore:
    """
    Feature store for Tier-1 and Tier-2 feature vectors.
    Separate Redis keys from the original FeatureStore to avoid conflicts.
    """
    MAX_WINDOW = 5000
    MIN_TRAIN = 500
    REDIS_TTL = 7 * 86400

    def __init__(self, redis_client: aioredis.Redis, tier: str) -> None:
        self._redis = redis_client
        self._tier = tier  # "t1" or "t2"

    def _key(self, symbol: str) -> str:
        return f"ml:{self._tier}:features:{symbol}"

    async def append(self, symbol: str, fv_dict: Dict[str, Any]) -> int:
        key = self._key(symbol)
        payload = json.dumps(fv_dict)
        pipe = self._redis.pipeline()
        try:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -self.MAX_WINDOW, -1)
            pipe.expire(key, self.REDIS_TTL)
            pipe.llen(key)
            results = await pipe.execute()
            return int(results[-1])
        finally:
            await pipe.reset()

    async def get_all(self, symbol: str) -> List[Dict[str, Any]]:
        key = self._key(symbol)
        raw = await self._redis.lrange(key, 0, -1)
        out = []
        for r in raw:
            try:
                out.append(json.loads(r))
            except Exception:
                pass
        return out

    async def window_size(self, symbol: str) -> int:
        return await self._redis.llen(self._key(symbol))

    async def ready(self, symbol: str) -> bool:
        return await self.window_size(symbol) >= self.MIN_TRAIN


# ---------------------------------------------------------------------------
# Base Learners
# ---------------------------------------------------------------------------

class XGBoostLearner:
    """
    Gradient-boosted tree model for microstructure + sentiment features.
    Suited for non-linear interactions (e.g., MLOFI × sentiment, GRI × vol).
    """

    def __init__(self) -> None:
        self._model = None
        self._scaler = StandardScaler() if _SKLEARN_AVAILABLE else None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not _XGB_AVAILABLE or not _SKLEARN_AVAILABLE:
            return
        X_scaled = self._scaler.fit_transform(X)
        self._model = xgb.XGBRegressor(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self._model.fit(X_scaled, y)

    def predict(self, x: np.ndarray) -> float:
        if self._model is None or not _XGB_AVAILABLE:
            return 0.0
        x_scaled = self._scaler.transform(x.reshape(1, -1))
        return float(self._model.predict(x_scaled)[0])

    def feature_importances(self, names: List[str]) -> Dict[str, float]:
        if self._model is None or not _XGB_AVAILABLE:
            return {}
        imps = self._model.feature_importances_
        total = sum(imps) or 1.0
        return {n: round(float(v) / total, 4) for n, v in zip(names, imps)}

    @property
    def is_fitted(self) -> bool:
        return self._model is not None


class RidgeLearner:
    """
    L2-regularised linear regression — interpretable baseline.
    Captures linear relationships; good for VWAP deviation and sentiment.
    """

    def __init__(self) -> None:
        self._pipeline = (
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
            if _SKLEARN_AVAILABLE else None
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._pipeline is None:
            return
        self._pipeline.fit(X, y)

    def predict(self, x: np.ndarray) -> float:
        if self._pipeline is None or not hasattr(self._pipeline, "predict"):
            return 0.0
        try:
            return float(self._pipeline.predict(x.reshape(1, -1))[0])
        except Exception:
            return 0.0

    def feature_importances(self, names: List[str]) -> Dict[str, float]:
        if self._pipeline is None:
            return {}
        try:
            coefs = self._pipeline.named_steps["ridge"].coef_
            total = sum(abs(c) for c in coefs) or 1.0
            return {n: round(abs(float(c)) / total, 4) for n, c in zip(names, coefs)}
        except Exception:
            return {}

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None and hasattr(
            self._pipeline.named_steps.get("ridge", object()), "coef_"
        )


# ---------------------------------------------------------------------------
# Meta-Learner (stacking)
# ---------------------------------------------------------------------------

class MetaLearner:
    """
    Ridge regression meta-learner trained on [xgb_pred, ridge_pred] → target.
    Learns optimal blending weights between the two base learners.
    """

    def __init__(self) -> None:
        self._model = Ridge(alpha=0.5) if _SKLEARN_AVAILABLE else None
        self._fitted = False

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> None:
        if self._model is None:
            return
        self._model.fit(X_meta, y)
        self._fitted = True

    def predict(self, xgb_pred: float, ridge_pred: float) -> float:
        if not self._fitted or self._model is None:
            return (xgb_pred + ridge_pred) / 2.0   # simple average fallback
        try:
            return float(self._model.predict([[xgb_pred, ridge_pred]])[0])
        except Exception:
            return (xgb_pred + ridge_pred) / 2.0

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# EnsembleSignalEngine — orchestrates training, prediction, retraining
# ---------------------------------------------------------------------------

class EnsembleSignalEngine:
    """
    V2 ML signal engine.  Replaces the static compute_alpha() formula.

    Lifecycle:
      1. On each tick cycle: call .build_feature_vector() to create FV
      2. Periodically (after getting next-period return): call .label_and_store()
      3. Every 30 min: retrain() — fits XGB, Ridge, MetaLearner on rolling window
      4. On each signal request: call .predict() — returns SignalOutput

    Thread safety: retrain() is dispatched to a thread executor (CPU-bound).
    """

    RETRAIN_INTERVAL_S = 1800   # 30 minutes
    # How long to wait before labeling a feature vector with its forward return.
    # The docstring specifies "1-minute forward log-return": without this delay,
    # cycles running at ~1 s each would label with 1-second returns (≈ 0),
    # causing the model to always predict ≈ 0 and generate only FLAT signals.
    LABEL_DELAY_S = 60          # label FV with the price 60 s later

    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._store      = FeatureStore(redis_client)
        self._xgb:  Dict[str, XGBoostLearner]  = {}
        self._ridge: Dict[str, RidgeLearner]    = {}
        self._meta:  Dict[str, MetaLearner]     = {}
        self._last_retrain: Dict[str, float]    = {}
        self._model_version: Dict[str, str]     = {}
        # Per-symbol queue of unlabelled FVs waiting for their forward return.
        # Each entry: (FeatureVector, monotonic timestamp when it was enqueued).
        self._pending_queues: Dict[str, Deque[FeatureVector]] = {}

    def build_feature_vector(
        self,
        symbol:          str,
        mlofi:           float,
        ofi:             float,
        aflow_ratio:     float,
        aflow_delta:     float,
        price:           float,
        vwap:            float,
        closes:          List[float],
        sentiment_score: float,
        vol:             float,
        gri_composite:   float,
        gpr_normalised:  float,
        vol_regime:      str,
    ) -> FeatureVector:
        """Build a FeatureVector from current market state."""
        aflow_delta_norm = aflow_delta / max(abs(aflow_delta) + 1.0, 1.0)
        vwap_dev         = (price - vwap) / vwap if vwap > 0 else 0.0
        rsi              = compute_rsi(closes) / 100.0
        vol_enc          = encode_vol_regime(vol_regime)

        fv = FeatureVector(
            mlofi              = mlofi,
            ofi                = ofi,
            aflow_ratio        = aflow_ratio,
            aflow_delta_norm   = aflow_delta_norm,
            vwap_dev           = vwap_dev,
            sentiment_score    = sentiment_score,
            vol_normalised     = vol,
            gri_composite      = gri_composite,
            gpr_normalised     = gpr_normalised,
            rsi_normalised     = rsi,
            vol_regime_encoded = vol_enc,
            reference_price    = price,
        )
        # Push to per-symbol pending queue (reference_ts already set by field default)
        if symbol not in self._pending_queues:
            self._pending_queues[symbol] = deque()
        self._pending_queues[symbol].append(fv)
        return fv

    async def label_and_store(self, symbol: str, current_price: float) -> None:
        """
        Labels pending FeatureVectors with their realized 1-minute forward
        log-return and stores them for training.

        A FV is only labelled once it is at least LABEL_DELAY_S (60 s) old,
        ensuring the target is a genuine 1-minute return rather than a
        sub-second return that collapses the training signal to ≈ 0.

        Multiple FVs may mature on a single call (burst catch-up). The queue
        is bounded implicitly because cycles run faster than 60 s, so the
        queue depth stays at roughly LABEL_DELAY_S / cycle_interval ≈ 60.
        """
        queue = self._pending_queues.get(symbol)
        if not queue:
            return

        now = time.monotonic()
        labelled_count = 0
        while queue:
            fv = queue[0]
            age = now - fv.reference_ts
            if age < self.LABEL_DELAY_S:
                break   # oldest item not yet mature; nothing else will be either

            # Compute 1-minute forward log-return
            if fv.reference_price > 0 and current_price > 0:
                fv.fwd_log_return = round(math.log(current_price / fv.reference_price), 8)
                await self._store.append(symbol, fv)
                labelled_count += 1

            queue.popleft()

        if labelled_count > 1:
            logger.debug("label_and_store %s: labelled %d mature FVs in one pass", symbol, labelled_count)

    async def maybe_retrain(self, symbol: str) -> bool:
        """
        Trigger retraining if 30 minutes have elapsed since last retrain
        and sufficient observations are available.
        Returns True if retraining was triggered.
        """
        now = time.monotonic()
        elapsed = now - self._last_retrain.get(symbol, 0.0)
        if elapsed < self.RETRAIN_INTERVAL_S:
            return False
        if not await self._store.ready(symbol):
            return False

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._retrain_sync, symbol,
                                   await self._store.get_all(symbol))
        self._last_retrain[symbol] = now
        return True

    def _retrain_sync(self, symbol: str, fvs: List[FeatureVector]) -> None:
        """CPU-bound retraining — runs in thread executor."""
        if not _ML_READY:
            return

        labelled = [fv for fv in fvs if fv.fwd_log_return is not None]
        if len(labelled) < FeatureStore.MIN_TRAIN:
            return

        X = np.array([fv.to_array() for fv in labelled], dtype=np.float32)
        y = np.array([fv.fwd_log_return for fv in labelled], dtype=np.float32)

        # Clip extreme targets (outlier protection)
        y = np.clip(y, -0.05, 0.05)

        # B-09 FIX: true holdout stacking to eliminate meta-learner data leakage.
        # Split: train base models on first 80%, use their OOF preds on last 20% to
        # train the meta-learner, then re-fit base models on the full dataset.
        split = max(int(len(labelled) * 0.80), FeatureStore.MIN_TRAIN)
        split = min(split, len(labelled) - 10)  # ensure ≥10 holdout rows

        # Step 1: Fit base learners on the TRAINING portion only
        xgb_train   = XGBoostLearner()
        ridge_train  = RidgeLearner()
        xgb_train.fit(X[:split], y[:split])
        ridge_train.fit(X[:split], y[:split])

        # Step 2: Generate out-of-fold (holdout) predictions on the remaining 20%
        holdout_xgb   = np.array([xgb_train.predict(X[i])   for i in range(split, len(labelled))])
        holdout_ridge = np.array([ridge_train.predict(X[i]) for i in range(split, len(labelled))])
        X_meta_train  = np.column_stack([holdout_xgb, holdout_ridge])
        y_meta        = y[split:]

        # Step 3: Fit meta-learner on the true out-of-fold predictions
        meta_model = MetaLearner()
        if len(X_meta_train) >= 10:
            meta_model.fit(X_meta_train, y_meta)

        # Step 4: Re-fit base learners on the FULL dataset for inference
        xgb_model   = XGBoostLearner()
        ridge_model  = RidgeLearner()
        xgb_model.fit(X, y)
        ridge_model.fit(X, y)

        self._xgb[symbol]           = xgb_model
        self._ridge[symbol]         = ridge_model
        self._meta[symbol]          = meta_model
        self._model_version[symbol] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")

        logger.info(
            "EnsembleSignalEngine retrained %s: %d obs | XGB=%s Ridge=%s Meta=%s",
            symbol, len(labelled),
            xgb_model.is_fitted, ridge_model.is_fitted, meta_model.is_fitted,
        )

    def predict(self, symbol: str, fv: FeatureVector) -> SignalOutput:
        """
        Generate ensemble signal from a FeatureVector.
        Falls back to a VWAP+sentiment static signal if model not fitted.
        """
        xgb_model   = self._xgb.get(symbol)
        ridge_model  = self._ridge.get(symbol)
        meta_model   = self._meta.get(symbol)

        x = np.array(fv.to_array(), dtype=np.float32)

        is_fallback = not (
            xgb_model and xgb_model.is_fitted and
            ridge_model and ridge_model.is_fitted
        )

        if is_fallback:
            # V1 static alpha fallback — sentiment-primary, VWAP/OFI as modifiers
            import math as _math

            # --- Sentiment component (primary driver) ---
            sent_score = fv.sentiment_score           # [-1, 1]
            sent_dir   = _math.copysign(1.0, sent_score) if sent_score != 0 else 0.0
            sent_mag   = _math.log1p(abs(sent_score))  # 0..0.69 for score [-1,1]

            # --- VWAP dislocation modifier (secondary) ---
            # vwap_dev = (price - vwap) / vwap; usually small
            # Use as a confirming signal, not a multiplier (avoids zeroing out)
            vwap_confirm = 0.0
            if abs(fv.vwap_dev) > 0.001:              # only if meaningful
                # Same direction as sentiment = confirm; opposite = attenuate
                vwap_confirm = _math.copysign(
                    min(abs(fv.vwap_dev) * 5.0, 0.3),  # cap at ±0.3
                    fv.vwap_dev,
                )

            # --- OFI/MLOFI modifier ---
            ofi_boost  = 0.0
            if abs(fv.mlofi) >= 0.10:
                ofi_boost = _math.copysign(
                    min(abs(fv.mlofi) * 0.5, 0.25),
                    fv.mlofi,
                )

            # --- GRI modifier: fear-regime boosts short sentiment ---
            gri_mod = 0.0
            if fv.gri_composite > 0.40:
                gri_mod = -(fv.gri_composite - 0.40) * 0.5  # negative = bearish

            # --- Combine ---
            signal = (sent_dir * sent_mag            # primary: 0..0.69
                      + vwap_confirm                 # confirming: ±0.3
                      + ofi_boost                   # order flow: ±0.25
                      + gri_mod)                    # geo: typically -0.1..0
            signal = max(-1.0, min(1.0, signal))

            return SignalOutput(
                symbol=symbol, signal=round(signal, 6), confidence=0.35,
                xgb_pred=0.0, ridge_pred=signal, meta_pred=signal,
                feature_importances={},
                model_version="fallback-v1-static",
                is_fallback=True,
            )

        xgb_pred   = xgb_model.predict(x)
        ridge_pred  = ridge_model.predict(x)
        meta_pred   = meta_model.predict(xgb_pred, ridge_pred) if meta_model else (xgb_pred + ridge_pred) / 2.0

        # Scale signal to [−1, 1] (returns are in ~[−0.05, 0.05])
        signal = max(-1.0, min(1.0, meta_pred * 20.0))

        # Confidence: agreement between base learners (correlation proxy)
        agreement = 1.0 - min(abs(xgb_pred - ridge_pred) / (abs(xgb_pred) + abs(ridge_pred) + 1e-9), 1.0)
        confidence = round(agreement * abs(signal), 4)

        importances = xgb_model.feature_importances(FeatureVector.FEATURE_NAMES)

        return SignalOutput(
            symbol              = symbol,
            signal              = round(signal, 6),
            confidence          = confidence,
            xgb_pred            = round(xgb_pred,  8),
            ridge_pred          = round(ridge_pred, 8),
            meta_pred           = round(meta_pred,  8),
            feature_importances = importances,
            model_version       = self._model_version.get(symbol, "untrained"),
            is_fallback         = False,
        )

    async def run_retrain_loop(self, symbols: List[str]) -> None:
        """
        Background task: checks each symbol every 5 minutes and retrains if due.
        """
        logger.info("EnsembleSignalEngine retrain loop started (%d symbols).", len(symbols))
        while True:
            try:
                for sym in symbols:
                    retrained = await self.maybe_retrain(sym)
                    if retrained:
                        logger.info("Retrain completed: %s (model_v=%s)",
                                    sym, self._model_version.get(sym))
                    await asyncio.sleep(0.1)   # Yield between symbols
            except asyncio.CancelledError:
                logger.info("EnsembleSignalEngine retrain loop cancelled.")
                raise
            except Exception as exc:
                logger.error("Retrain loop error: %s", exc, exc_info=True)
            await asyncio.sleep(300)   # Check every 5 min


# ---------------------------------------------------------------------------
# TierSignalEngine — Multi-tier signal generation
# ---------------------------------------------------------------------------

class TierSignalEngine:
    """
    Multi-tier signal engine that wraps the original EnsembleSignalEngine
    and adds Tier-1 (microstructure) and Tier-2 (mean-reversion) signal paths.

    This is an ADDITIVE layer — the original engine continues to work unchanged.
    The TierSignalEngine delegates to the original for backward-compat and
    provides separate predict methods for each tier.
    """

    # Tier-1: 5-min forward return (vs original 1-min)
    TIER1_LABEL_DELAY_S = 300
    # Tier-2: 30-min forward return
    TIER2_LABEL_DELAY_S = 1800
    RETRAIN_INTERVAL_S = 1800

    def __init__(self, redis_client: aioredis.Redis,
                 original_engine: EnsembleSignalEngine) -> None:
        self._original = original_engine
        self._redis = redis_client

        # Tier-1 stores and models
        self._t1_store = TieredFeatureStore(redis_client, "t1")
        self._t1_xgb: Dict[str, XGBoostLearner] = {}
        self._t1_ridge: Dict[str, RidgeLearner] = {}
        self._t1_meta: Dict[str, MetaLearner] = {}
        self._t1_last_retrain: Dict[str, float] = {}
        self._t1_pending: Dict[str, Deque[Tier1FeatureVector]] = {}

        # Tier-2 stores and models
        self._t2_store = TieredFeatureStore(redis_client, "t2")
        self._t2_xgb: Dict[str, XGBoostLearner] = {}
        self._t2_ridge: Dict[str, RidgeLearner] = {}
        self._t2_meta: Dict[str, MetaLearner] = {}
        self._t2_last_retrain: Dict[str, float] = {}
        self._t2_pending: Dict[str, Deque[Tier2FeatureVector]] = {}

    # -- Tier-1 methods --

    def enqueue_tier1(self, symbol: str, fv: Tier1FeatureVector) -> None:
        """Enqueue a Tier-1 FV for delayed labeling."""
        if symbol not in self._t1_pending:
            self._t1_pending[symbol] = deque()
        self._t1_pending[symbol].append(fv)

    async def label_tier1(self, symbol: str, current_price: float) -> None:
        """Label mature Tier-1 FVs (5-min delay) and store."""
        queue = self._t1_pending.get(symbol)
        if not queue:
            return
        now = time.monotonic()
        while queue:
            fv = queue[0]
            if now - fv.reference_ts < self.TIER1_LABEL_DELAY_S:
                break
            if fv.reference_price > 0 and current_price > 0:
                fv.fwd_log_return = round(math.log(current_price / fv.reference_price), 8)
                await self._t1_store.append(symbol, fv.to_dict())
            queue.popleft()

    def predict_tier1(self, symbol: str, fv: Tier1FeatureVector) -> SignalOutput:
        """Generate Tier-1 signal from microstructure features."""
        xgb_m = self._t1_xgb.get(symbol)
        ridge_m = self._t1_ridge.get(symbol)
        meta_m = self._t1_meta.get(symbol)

        if not (xgb_m and xgb_m.is_fitted and ridge_m and ridge_m.is_fitted):
            # Fallback: use MLOFI + aggressive flow as simple directional signal
            raw = 0.6 * fv.mlofi + 0.4 * fv.aflow_ratio
            signal = max(-1.0, min(1.0, raw))
            return SignalOutput(
                symbol=symbol, signal=round(signal, 6), confidence=0.25,
                xgb_pred=0.0, ridge_pred=signal, meta_pred=signal,
                feature_importances={}, model_version="t1-fallback",
                is_fallback=True,
            )

        x = np.array(fv.to_array(), dtype=np.float32)
        xgb_pred = xgb_m.predict(x)
        ridge_pred = ridge_m.predict(x)
        meta_pred = meta_m.predict(xgb_pred, ridge_pred) if meta_m else (xgb_pred + ridge_pred) / 2.0

        signal = max(-1.0, min(1.0, meta_pred * 20.0))
        agreement = 1.0 - min(abs(xgb_pred - ridge_pred) / (abs(xgb_pred) + abs(ridge_pred) + 1e-9), 1.0)
        confidence = round(agreement * abs(signal), 4)
        importances = xgb_m.feature_importances(Tier1FeatureVector.FEATURE_NAMES)

        return SignalOutput(
            symbol=symbol, signal=round(signal, 6), confidence=confidence,
            xgb_pred=round(xgb_pred, 8), ridge_pred=round(ridge_pred, 8),
            meta_pred=round(meta_pred, 8), feature_importances=importances,
            model_version=f"t1-{self._t1_last_retrain.get(symbol, 0):.0f}",
            is_fallback=False,
        )

    # -- Tier-2 methods (same pattern) --

    def enqueue_tier2(self, symbol: str, fv: Tier2FeatureVector) -> None:
        if symbol not in self._t2_pending:
            self._t2_pending[symbol] = deque()
        self._t2_pending[symbol].append(fv)

    async def label_tier2(self, symbol: str, current_price: float) -> None:
        queue = self._t2_pending.get(symbol)
        if not queue:
            return
        now = time.monotonic()
        while queue:
            fv = queue[0]
            if now - fv.reference_ts < self.TIER2_LABEL_DELAY_S:
                break
            if fv.reference_price > 0 and current_price > 0:
                fv.fwd_log_return = round(math.log(current_price / fv.reference_price), 8)
                await self._t2_store.append(symbol, fv.to_dict())
            queue.popleft()

    def predict_tier2(self, symbol: str, fv: Tier2FeatureVector) -> SignalOutput:
        xgb_m = self._t2_xgb.get(symbol)
        ridge_m = self._t2_ridge.get(symbol)
        meta_m = self._t2_meta.get(symbol)

        if not (xgb_m and xgb_m.is_fitted and ridge_m and ridge_m.is_fitted):
            # Fallback: VWAP z-score mean-reversion + RSI signal
            raw = -0.5 * fv.vwap_z / 5.0 + 0.5 * fv.rsi_mr_signal
            signal = max(-1.0, min(1.0, raw))
            return SignalOutput(
                symbol=symbol, signal=round(signal, 6), confidence=0.20,
                xgb_pred=0.0, ridge_pred=signal, meta_pred=signal,
                feature_importances={}, model_version="t2-fallback",
                is_fallback=True,
            )

        x = np.array(fv.to_array(), dtype=np.float32)
        xgb_pred = xgb_m.predict(x)
        ridge_pred = ridge_m.predict(x)
        meta_pred = meta_m.predict(xgb_pred, ridge_pred) if meta_m else (xgb_pred + ridge_pred) / 2.0

        signal = max(-1.0, min(1.0, meta_pred * 20.0))
        agreement = 1.0 - min(abs(xgb_pred - ridge_pred) / (abs(xgb_pred) + abs(ridge_pred) + 1e-9), 1.0)
        confidence = round(agreement * abs(signal), 4)
        importances = xgb_m.feature_importances(Tier2FeatureVector.FEATURE_NAMES)

        return SignalOutput(
            symbol=symbol, signal=round(signal, 6), confidence=confidence,
            xgb_pred=round(xgb_pred, 8), ridge_pred=round(ridge_pred, 8),
            meta_pred=round(meta_pred, 8), feature_importances=importances,
            model_version=f"t2-{self._t2_last_retrain.get(symbol, 0):.0f}",
            is_fallback=False,
        )

    # -- Retraining --

    async def maybe_retrain_tier1(self, symbol: str) -> bool:
        now = time.monotonic()
        if now - self._t1_last_retrain.get(symbol, 0.0) < self.RETRAIN_INTERVAL_S:
            return False
        if not await self._t1_store.ready(symbol):
            return False

        raw_dicts = await self._t1_store.get_all(symbol)
        fvs = [Tier1FeatureVector.from_dict(d) for d in raw_dicts]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._retrain_tier_sync, symbol, fvs,
            Tier1FeatureVector.FEATURE_NAMES,
            self._t1_xgb, self._t1_ridge, self._t1_meta,
        )
        self._t1_last_retrain[symbol] = now
        logger.info("Tier-1 retrained %s", symbol)
        return True

    async def maybe_retrain_tier2(self, symbol: str) -> bool:
        now = time.monotonic()
        if now - self._t2_last_retrain.get(symbol, 0.0) < self.RETRAIN_INTERVAL_S:
            return False
        if not await self._t2_store.ready(symbol):
            return False

        raw_dicts = await self._t2_store.get_all(symbol)
        fvs = [Tier2FeatureVector.from_dict(d) for d in raw_dicts]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._retrain_tier_sync, symbol, fvs,
            Tier2FeatureVector.FEATURE_NAMES,
            self._t2_xgb, self._t2_ridge, self._t2_meta,
        )
        self._t2_last_retrain[symbol] = now
        logger.info("Tier-2 retrained %s", symbol)
        return True

    @staticmethod
    def _retrain_tier_sync(
        symbol: str,
        fvs: list,
        feature_names: List[str],
        xgb_dict: Dict[str, XGBoostLearner],
        ridge_dict: Dict[str, RidgeLearner],
        meta_dict: Dict[str, MetaLearner],
    ) -> None:
        """CPU-bound retraining for any tier — runs in thread executor."""
        if not _ML_READY:
            return

        labelled = [fv for fv in fvs if fv.fwd_log_return is not None]
        if len(labelled) < FeatureStore.MIN_TRAIN:
            return

        X = np.array([fv.to_array() for fv in labelled], dtype=np.float32)
        y = np.array([fv.fwd_log_return for fv in labelled], dtype=np.float32)
        y = np.clip(y, -0.05, 0.05)

        split = max(int(len(labelled) * 0.80), FeatureStore.MIN_TRAIN)
        split = min(split, len(labelled) - 10)

        # Holdout stacking (same as original)
        xgb_train = XGBoostLearner()
        ridge_train = RidgeLearner()
        xgb_train.fit(X[:split], y[:split])
        ridge_train.fit(X[:split], y[:split])

        holdout_xgb = np.array([xgb_train.predict(X[i]) for i in range(split, len(labelled))])
        holdout_ridge = np.array([ridge_train.predict(X[i]) for i in range(split, len(labelled))])
        X_meta_train = np.column_stack([holdout_xgb, holdout_ridge])
        y_meta = y[split:]

        meta_model = MetaLearner()
        if len(X_meta_train) >= 10:
            meta_model.fit(X_meta_train, y_meta)

        xgb_model = XGBoostLearner()
        ridge_model = RidgeLearner()
        xgb_model.fit(X, y)
        ridge_model.fit(X, y)

        xgb_dict[symbol] = xgb_model
        ridge_dict[symbol] = ridge_model
        meta_dict[symbol] = meta_model

    # -- Convenience: delegate to original engine --

    @property
    def original(self) -> EnsembleSignalEngine:
        return self._original

    async def run_retrain_loop(self, symbols: List[str]) -> None:
        """Background task: retrain all tiers for all symbols."""
        logger.info("TierSignalEngine retrain loop started (%d symbols).", len(symbols))
        while True:
            try:
                for sym in symbols:
                    await self.maybe_retrain_tier1(sym)
                    await self.maybe_retrain_tier2(sym)
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.info("TierSignalEngine retrain loop cancelled.")
                raise
            except Exception as exc:
                logger.error("TierSignalEngine retrain loop error: %s", exc, exc_info=True)
            await asyncio.sleep(300)
