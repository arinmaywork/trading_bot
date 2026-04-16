"""
universe.py
===========
Sophisticated Universe Selection Layer

Automatically screens all ~2,000 NSE equities down to an optimally sized,
diversified, liquid, and alpha-rich active trading universe.

Pipeline Stages:
  Stage 1 — Static Daily Filter   (08:45 IST, pre-market, once per day)
    • Fetch complete NSE instrument list via KiteConnect
    • Hard filters: price range, instrument type (EQ only), exchange
    • Fetch 30-day historical data via yfinance for each candidate
    • Apply: min ADV, min turnover, vol range

  Stage 2 — Multi-Factor Composite Scoring
    • Liquidity Score   (35%) — ADV rank, turnover, bid-ask spread proxy
    • Momentum Score    (25%) — 20-day price return, RSI positioning
    • Volatility Score  (20%) — distance from target vol band (15-40%)
    • Sector Signal     (20%) — macro/weather tailwind per sector

  Stage 3 — Sector Diversification
    • Group by NSE sector classification
    • Cap at MAX_STOCKS_PER_SECTOR per sector
    • Boost stocks in major indices (NIFTY50/200/500 membership)

  Stage 4 — Intraday Dynamic Rescoring (every 30 minutes)
    • Volume surge detection: current vol vs. 5-day rolling avg
    • OFI momentum: trending OFI means the stock is "activating"
    • Add surging stocks to active universe; remove stale ones
    • Re-subscribe KiteTicker WebSocket when universe changes

  Stage 5 — Frequency Optimisation
    • Compute optimal strategy poll_interval based on active universe size
    • Target: process each symbol once per its own liquidity-adjusted cadence

Output:
    • Active universe stored in Redis Sorted Set (score = composite rank)
    • Metadata stored in Redis Hash per symbol
    • universe_changed event triggers WebSocket re-subscription
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

import redis.asyncio as aioredis
import yfinance as yf

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSE Sector Map — maps instrument series / industry to a clean sector label.
# yfinance returns a "sector" field; we normalise it to these canonical names.
# ---------------------------------------------------------------------------
SECTOR_NORMALISATION: Dict[str, str] = {
    "technology": "IT",
    "information technology": "IT",
    "financial services": "FINANCIALS",
    "banks": "FINANCIALS",
    "banking": "FINANCIALS",
    "nbfc": "FINANCIALS",
    "consumer cyclical": "CONSUMER",
    "consumer defensive": "CONSUMER",
    "fmcg": "CONSUMER",
    "healthcare": "PHARMA",
    "pharmaceuticals": "PHARMA",
    "drug": "PHARMA",
    "energy": "ENERGY",
    "oil": "ENERGY",
    "gas": "ENERGY",
    "utilities": "UTILITIES",
    "power": "UTILITIES",
    "electricity": "UTILITIES",
    "industrials": "INDUSTRIALS",
    "capital goods": "INDUSTRIALS",
    "automobile": "AUTO",
    "auto": "AUTO",
    "basic materials": "METALS",
    "metals": "METALS",
    "mining": "METALS",
    "real estate": "REALTY",
    "realty": "REALTY",
    "communication services": "TELECOM",
    "telecom": "TELECOM",
    "media": "MEDIA",
}

# Macro / weather sector tailwind multipliers — updated dynamically
# based on temperature anomalies and news sentiment.
# Keys match canonical sector labels above.
SECTOR_BASE_TAILWIND: Dict[str, float] = {
    "UTILITIES": 0.6,    # High sensitivity to weather
    "ENERGY":    0.5,
    "PHARMA":    0.4,
    "IT":        0.3,
    "FINANCIALS":0.3,
    "AUTO":      0.2,
    "CONSUMER":  0.2,
    "METALS":    0.2,
    "INDUSTRIALS":0.2,
    "REALTY":    0.1,
    "TELECOM":   0.1,
    "MEDIA":     0.1,
}

# Known NIFTY index constituents (abbreviated; production would fetch from NSE API)
# These sets are used to apply score bonuses.
NIFTY50_SYMBOLS: Set[str] = {
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI",
    "TITAN","WIPRO","NESTLEIND","ULTRACEMCO","POWERGRID","NTPC","TECHM","SUNPHARMA",
    "HCLTECH","ONGC","BAJAJFINSV","TATAMOTORS","ADANIENT","JSWSTEEL","TATASTEEL",
    "COALINDIA","BPCL","GRASIM","DIVISLAB","CIPLA","HDFCLIFE","DRREDDY","M&M",
    "APOLLOHOSP","TATACONSUM","EICHERMOT","SBILIFE","ADANIPORTS","BRITANNIA",
    "BAJAJ-AUTO","HEROMOTOCO","UPL","SHREECEM","INDUSINDBK","LTIM",
}

NIFTY200_SYMBOLS: Set[str] = NIFTY50_SYMBOLS | {
    "AUROPHARMA","BANDHANBNK","BERGEPAINT","BIOCON","BOSCHLTD","CANBK","CHOLAFIN",
    "COLPAL","CONCOR","DABUR","DLF","GODREJCP","GODREJPROP","HAVELLS","HINDZINC",
    "IDFCFIRSTB","IGL","INDIGO","IOC","IRCTC","MCDOWELL-N","MFSL","MOTHERSON",
    "MPHASIS","NAVINFLUOR","NMDC","PAGEIND","PERSISTENT","PETRONET","PIDILITIND",
    "PIIND","PNB","RECLTD","SIEMENS","SRF","TATAPOWER","TORNTPHARM","TRENT",
    "VEDL","VOLTAS","ZEEL","ZYDUSLIFE","ADANIGREEN","ADANIPOWER","ATGL","AWL",
    "BALKRISIND","BATAINDIA","CANFINHOME",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StockMetadata:
    """Rich per-symbol metadata computed during universe selection."""
    symbol: str
    instrument_token: int
    sector: str
    last_price: float
    avg_daily_volume: float       # 20-day average
    avg_daily_turnover_cr: float  # ₹ Crore
    annualised_vol: float
    momentum_20d: float           # % price return over 20 trading days
    rsi_14: float                 # Relative Strength Index (14-day)
    composite_score: float        # Final ranking score ∈ [0, 1]
    liquidity_score: float
    momentum_score: float
    volatility_score: float
    sector_score: float
    in_nifty50: bool
    in_nifty200: bool
    in_nifty500: bool
    last_scored_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    volume_surge_ratio: float = 1.0   # current_vol / 5d_avg_vol; >2.5 = surge

    def to_redis_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_redis_dict(cls, d: Dict[bytes, bytes]) -> "StockMetadata":
        decoded = {k.decode(): v.decode() for k, v in d.items()}
        return cls(
            symbol=decoded["symbol"],
            instrument_token=int(decoded["instrument_token"]),
            sector=decoded["sector"],
            last_price=float(decoded["last_price"]),
            avg_daily_volume=float(decoded["avg_daily_volume"]),
            avg_daily_turnover_cr=float(decoded["avg_daily_turnover_cr"]),
            annualised_vol=float(decoded["annualised_vol"]),
            momentum_20d=float(decoded["momentum_20d"]),
            rsi_14=float(decoded["rsi_14"]),
            composite_score=float(decoded["composite_score"]),
            liquidity_score=float(decoded["liquidity_score"]),
            momentum_score=float(decoded["momentum_score"]),
            volatility_score=float(decoded["volatility_score"]),
            sector_score=float(decoded["sector_score"]),
            in_nifty50=decoded["in_nifty50"] == "True",
            in_nifty200=decoded["in_nifty200"] == "True",
            in_nifty500=decoded["in_nifty500"] == "True",
            last_scored_at=decoded.get("last_scored_at", ""),
            volume_surge_ratio=float(decoded.get("volume_surge_ratio", "1.0")),
        )


@dataclass
class UniverseSnapshot:
    """A point-in-time snapshot of the active universe."""
    symbols: List[str]
    metadata: Dict[str, StockMetadata]
    built_at: datetime
    total_candidates_screened: int
    filter_stats: Dict[str, int]   # how many passed/failed each stage

    @property
    def size(self) -> int:
        return len(self.symbols)


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _compute_rsi(closes: List[float], period: int = 14) -> float:
    """Standard RSI using Wilder's smoothing."""
    if len(closes) < period + 1:
        return 50.0   # neutral default
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _compute_annualised_vol(closes: List[float], trading_days: int = 252) -> float:
    """Log-return realised volatility, annualised."""
    if len(closes) < 3:
        return 0.0
    log_returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if not log_returns:
        return 0.0
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
    return math.sqrt(variance * trading_days)


def _compute_momentum(closes: List[float], lookback: int = 20) -> float:
    """Simple price momentum: (close_t / close_{t-n}) - 1."""
    if len(closes) < lookback + 1:
        return 0.0
    return (closes[-1] / closes[-(lookback + 1)]) - 1.0


# ---------------------------------------------------------------------------
# Scoring functions — each returns a float ∈ [0, 1]
# ---------------------------------------------------------------------------

def _score_liquidity(adv: float, turnover_cr: float) -> float:
    """
    Higher ADV and turnover = higher score.
    Uses log-normalisation against reference levels:
        ADV ref: 1,000,000 shares/day (score = 0.5 at ref)
        Turnover ref: ₹10 Cr/day
    """
    adv_score = min(1.0, math.log1p(adv) / math.log1p(5_000_000))
    turn_score = min(1.0, math.log1p(turnover_cr) / math.log1p(50.0))
    return round((adv_score + turn_score) / 2, 4)


def _score_momentum(momentum_20d: float, rsi: float) -> float:
    """
    Reward stocks with positive but not extreme momentum.
    Penalise overbought (RSI > 70) and oversold (RSI < 30) — poor timing.
    """
    # Momentum component: sigmoid-shaped, centred at 0
    mom_norm = 1.0 / (1.0 + math.exp(-momentum_20d * 10))

    # RSI component: quadratic penalty for extremes; peaks at RSI = 50
    rsi_norm = 1.0 - (abs(rsi - 50) / 50) ** 1.5

    return round((mom_norm * 0.6 + rsi_norm * 0.4), 4)


def _score_volatility(ann_vol: float) -> float:
    """
    Target vol band for mean-reversion alpha: 15–40% annualised.
    Score = 1.0 at vol = 25%; decays to 0 outside the viable range.
    Uses a tent (triangular) function centred at 25%.
    """
    target = 0.25
    low, high = 0.10, 0.90
    if ann_vol < low or ann_vol > high:
        return 0.0
    if ann_vol <= target:
        return round((ann_vol - low) / (target - low), 4)
    else:
        return round((high - ann_vol) / (high - target), 4)


def _score_sector(
    sector: str,
    weather_anomaly: float,
    sentiment_score: float,
    geo_risk: float = 0.0,
) -> float:
    """
    Score based on macro/sector tailwinds:
    • Base tailwind from historical sector-weather correlation
    • Boosted by positive Gemini sentiment
    • Weather anomaly amplifies UTILITIES and ENERGY
    • Geopolitical risk rotates capital: penalises vulnerable sectors,
      boosts defensives (PHARMA, CONSUMER, UTILITIES)

    geo_risk ∈ [0,1]: higher = more geopolitical stress active.
    Sector sensitivity defined in GEO_SECTOR_SENSITIVITY:
        positive sensitivity → geo risk REDUCES the score
        negative sensitivity → geo risk INCREASES the score (defensive rotation)
    """
    from geopolitical import GEO_SECTOR_SENSITIVITY
    base = SECTOR_BASE_TAILWIND.get(sector, 0.2)

    # Weather effect
    weather_boost = 0.0
    if sector in ("UTILITIES", "ENERGY"):
        weather_boost = min(0.3, abs(weather_anomaly) / 10.0)

    # Sentiment effect
    sentiment_boost = sentiment_score * 0.2

    # Geopolitical rotation effect
    # sensitivity > 0: geo risk hurts the sector (subtract)
    # sensitivity < 0: geo risk helps the sector (add)
    sensitivity = GEO_SECTOR_SENSITIVITY.get(sector, 0.20)
    geo_impact  = geo_risk * sensitivity   # positive = reduce score, negative = increase
    geo_boost   = -geo_impact              # flip sign so negative sensitivity adds score

    raw = base + weather_boost + sentiment_boost + geo_boost
    return round(min(1.0, max(0.0, raw)), 4)


# ---------------------------------------------------------------------------
# yfinance data fetcher (synchronous, wrapped in executor by caller)
# ---------------------------------------------------------------------------

def _fetch_yfinance_data(
    symbols_ns: List[str],     # e.g. ["RELIANCE.NS", "TCS.NS"]
    period_days: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch historical OHLCV for a batch of symbols from yfinance.
    Returns dict: {clean_symbol: {closes, volumes, sector, ...}}
    """
    result: Dict[str, Dict[str, Any]] = {}

    # yfinance bulk download is more efficient than individual tickers
    # but less reliable for sector data — we do both.
    period_str = f"{period_days}d"

    for sym_ns in symbols_ns:
        clean = sym_ns.replace(".NS", "")

        # Hard guard: skip any symbol that looks like an index, G-Sec, or bond
        # These should never reach yfinance — this is a safety net only
        if (
            clean.startswith("$")
            or " " in clean
            or clean.endswith("-SG")
            or clean.endswith("-GS")
            or clean.endswith("-GB")
            or clean.endswith("-TB")
            or clean.endswith("-BZ")
            or any(c in clean for c in ["&", "#", "@", "!"])
        ):
            logger.debug("yfinance guard: skipping non-equity symbol %s", clean)
            continue

        try:
            ticker = yf.Ticker(sym_ns)
            hist = ticker.history(period=period_str, auto_adjust=True)

            if hist.empty or len(hist) < 5:
                continue

            closes = hist["Close"].dropna().tolist()
            volumes = hist["Volume"].dropna().tolist()

            if not closes or not volumes:
                continue

            avg_vol = sum(volumes[-20:]) / min(20, len(volumes))
            avg_price = sum(closes[-20:]) / min(20, len(closes))
            avg_turnover_cr = (avg_vol * avg_price) / 1e7   # ₹ Crore

            # Sector info from ticker.info (may be slow — cached by yf)
            info = ticker.info or {}
            raw_sector = (info.get("sector") or info.get("industry") or "").lower()
            sector = SECTOR_NORMALISATION.get(raw_sector, "OTHER")

            result[clean] = {
                "closes": closes,
                "volumes": volumes,
                "avg_daily_volume": avg_vol,
                "avg_daily_turnover_cr": avg_turnover_cr,
                "sector": sector,
                "last_price": closes[-1] if closes else 0.0,
            }
        except Exception as exc:
            logger.debug("yfinance fetch failed for %s: %s", sym_ns, exc)

    return result


# ---------------------------------------------------------------------------
# Main Universe Engine
# ---------------------------------------------------------------------------

class UniverseEngine:
    """
    Orchestrates the full multi-stage universe selection pipeline.

    Public API:
        await engine.run_daily_refresh()         — Stage 1–3 (pre-market)
        await engine.run_intraday_rescore()      — Stage 4 (every 30 min)
        engine.get_active_symbols()              — Current universe list
        engine.get_metadata(symbol)              — StockMetadata for a symbol
        engine.register_change_callback(fn)      — Called when universe changes
        await engine.start_background_tasks()    — Start all scheduler loops
    """

    def __init__(
        self,
        kite,                           # KiteConnect instance (typed as Any to avoid circular)
        redis_client: aioredis.Redis,
        rate_limiter,                   # RateLimiter instance
    ) -> None:
        self._kite          = kite
        self._redis         = redis_client
        self._limiter       = rate_limiter
        self._cfg           = settings.universe
        self._strat_cfg     = settings.strategy

        # In-memory cache for the hot path (avoid Redis round-trip on every tick)
        self._active_symbols: List[str] = []
        self._metadata: Dict[str, StockMetadata] = {}
        self._token_map: Dict[int, str] = {}   # instrument_token → symbol
        self._symbol_to_token: Dict[str, int] = {}

        # Callbacks triggered whenever the active universe changes
        self._change_callbacks: List[Callable[[List[str]], Coroutine]] = []

        # Track what is currently subscribed on WebSocket
        self._subscribed_tokens: Set[int] = set()

        # Sector tailwinds updated externally by the alt-data pipeline
        self._weather_anomaly: float = 0.0
        self._sentiment_score: float = 0.0

        # Macro signals (updated by main loop every strategy cycle)
        self._geo_risk: float = 0.0

        # Last refresh timestamps
        self._last_daily_refresh: Optional[date] = None
        self._last_intraday_rescore: float = 0.0

        logger.info("UniverseEngine initialised.")

    # ------------------------------------------------------------------
    # External signal ingestion
    # ------------------------------------------------------------------

    def update_macro_signals(
        self,
        weather_anomaly: float,
        sentiment_score: float,
        geo_risk: float = 0.0,
    ) -> None:
        """
        Called by main loop to push latest macro data into sector scoring.
        geo_risk is the real GeopoliticalRiskIndex.composite value.
        """
        self._weather_anomaly = weather_anomaly
        self._sentiment_score = sentiment_score
        self._geo_risk        = geo_risk

    def register_change_callback(
        self, fn: Callable[[List[str]], Coroutine]
    ) -> None:
        """Register an async callback fired when the active universe changes."""
        self._change_callbacks.append(fn)

    # ------------------------------------------------------------------
    # Stage 1 + 2 + 3: Daily full refresh
    # ------------------------------------------------------------------

    async def run_daily_refresh(self) -> UniverseSnapshot:
        """
        Full pipeline: fetch all NSE instruments → filter → score → diversify.
        Typically takes 3–8 minutes due to yfinance rate limits.
        Should run at 08:45 IST (before 09:15 market open).
        """
        t_start = time.monotonic()
        logger.info("=" * 60)
        logger.info("Universe: Starting daily full refresh…")
        logger.info("=" * 60)

        filter_stats: Dict[str, int] = {}

        # ---- Step A: Fetch all NSE instruments ----
        all_instruments = await self._fetch_all_nse_instruments()
        filter_stats["total_nse_instruments"] = len(all_instruments)
        logger.info("Fetched %d NSE instruments.", len(all_instruments))

        # ---- Step B: Hard static filters ----
        candidates = self._apply_static_filters(all_instruments)
        filter_stats["after_static_filter"] = len(candidates)
        logger.info("After static filter: %d candidates.", len(candidates))

        # ---- Step C: Fetch historical data for candidates (batched) ----
        scored_stocks = await self._fetch_and_score_candidates(candidates)
        filter_stats["after_scoring"] = len(scored_stocks)
        logger.info("Successfully scored: %d stocks.", len(scored_stocks))

        # ---- Step D: Sector diversification + final selection ----
        final_universe = self._apply_sector_diversification(scored_stocks)
        filter_stats["final_universe"] = len(final_universe)

        # ---- Step E: Persist to Redis + update in-memory state ----
        await self._persist_universe(final_universe)
        previous_symbols = set(self._active_symbols)
        self._active_symbols = [m.symbol for m in final_universe]
        self._metadata       = {m.symbol: m for m in final_universe}
        self._last_daily_refresh = date.today()

        elapsed = time.monotonic() - t_start
        logger.info(
            "Daily refresh complete: %d symbols selected in %.1f s.",
            len(self._active_symbols), elapsed,
        )

        # Fire change callbacks if universe changed
        if set(self._active_symbols) != previous_symbols:
            await self._fire_change_callbacks()

        snapshot = UniverseSnapshot(
            symbols=list(self._active_symbols),
            metadata=dict(self._metadata),
            built_at=datetime.now(timezone.utc),
            total_candidates_screened=filter_stats["total_nse_instruments"],
            filter_stats=filter_stats,
        )
        return snapshot

    async def _fetch_all_nse_instruments(self) -> List[Dict[str, Any]]:
        """Fetch the complete NSE instrument dump from Kite Connect."""
        loop = asyncio.get_running_loop()
        async with self._limiter.request_slot():
            instruments = await loop.run_in_executor(
                None, lambda: self._kite.instruments("NSE")
            )
        return instruments

    def _apply_static_filters(
        self, instruments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Hard filters applied without any API calls:
          - instrument_type == "EQ"    (equity only — no indices, ETFs, etc.)
          - exchange == "NSE"
          - name not blank
          - Symbol must be clean: no $, spaces, or special chars
          - Not a known non-equity suffix: BE, BL, BT, GS, IL, PP, SM, W1
        """
        # Suffixes used by NSE for non-standard equity categories
        # NSE non-equity suffixes to exclude
        EXCLUDED_SUFFIXES = {
            "BE",   # Trade-to-trade (physical settlement)
            "BL",   # Block deal window
            "BT",   # Book transfer
            "GS",   # Government securities
            "SG",   # State government securities / G-Secs
            "ST",   # SME secured term bonds
            "IV",   # SME debentures / NCDs
            "IL",   # Index linked
            "PP",   # Partly paid shares
            "SM",   # Small & medium enterprises
            "W1",   # Warrants
            "GB",   # Government bonds
            "TB",   # Treasury bills
            "IT",   # Infrastructure / bond instruments
            "DB",   # Debentures
            "NCD",  # Non-convertible debentures
        }

        # Additional prefix-based exclusion for SME/bond symbols
        EXCLUDED_PREFIXES = ("$",)
        EXCLUDED_CONTAINS = ("-ST", "-IV", "-IT", "-DB", "-SG", "-NCD")

        filtered = []
        for inst in instruments:
            sym = inst.get("tradingsymbol", "")

            # Skip indices, bonds and synthetic symbols
            if not sym or sym.startswith("$") or " " in sym:
                continue

            # Skip SME bond patterns even without $ prefix (e.g. GIRIRAJ-ST)
            if any(sym.endswith(pat) for pat in EXCLUDED_CONTAINS):
                continue

            # Skip non-equity instrument types
            if inst.get("instrument_type") != "EQ":
                continue

            # Skip non-NSE instruments
            if inst.get("exchange") != "NSE":
                continue

            # Skip blank names
            if not inst.get("name"):
                continue

            # Skip excluded suffixes (trade-to-trade, warrants, etc.)
            suffix = sym.split("-")[-1] if "-" in sym else sym[-2:]
            if suffix in EXCLUDED_SUFFIXES:
                continue

            # Skip symbols with special characters (hyphens mid-symbol, dots, etc.)
            # Valid NSE equity symbols: only uppercase letters, digits, hyphens at end
            clean_sym = sym.replace("-", "")
            if not clean_sym.isalnum():
                continue

            filtered.append(inst)

        return filtered

    async def _fetch_and_score_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[StockMetadata]:
        """
        Fetch yfinance data for all candidates (batched to respect rate limits)
        and compute composite scores.

        Batches of 50 symbols to avoid overwhelming yfinance.
        """
        scored: List[StockMetadata] = []
        symbols = [inst["tradingsymbol"] for inst in candidates]
        # Build lookup: symbol → instrument token
        token_lookup = {
            inst["tradingsymbol"]: inst["instrument_token"]
            for inst in candidates
        }

        batch_size = 50
        total_batches = math.ceil(len(symbols) / batch_size)
        loop = asyncio.get_running_loop()

        for batch_idx in range(total_batches):
            batch = symbols[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            ns_batch = [f"{s}.NS" for s in batch]

            logger.info(
                "Scoring batch %d/%d (%d symbols)…",
                batch_idx + 1, total_batches, len(batch),
            )

            # yfinance is synchronous; run in executor
            raw_data: Dict[str, Dict[str, Any]] = await loop.run_in_executor(
                None, lambda b=ns_batch: _fetch_yfinance_data(b, self._cfg.HISTORICAL_DAYS)
            )

            for sym, data in raw_data.items():
                meta = self._score_stock(sym, data, token_lookup.get(sym, 0))
                if meta is not None:
                    scored.append(meta)

            # Polite delay between batches to avoid yfinance bans
            if batch_idx < total_batches - 1:
                await asyncio.sleep(2.0)

        return scored

    def _score_stock(
        self,
        symbol: str,
        data: Dict[str, Any],
        token: int,
    ) -> Optional[StockMetadata]:
        """
        Apply quantitative filters and compute composite score for one stock.
        Returns None if the stock fails any hard filter.
        """
        closes: List[float] = data["closes"]
        volumes: List[float] = data.get("volumes", [])
        last_price = data["last_price"]
        adv = data["avg_daily_volume"]
        turnover_cr = data["avg_daily_turnover_cr"]
        sector = data["sector"]

        # ---- Hard filters ----
        if not (self._cfg.MIN_PRICE_INR <= last_price <= self._cfg.MAX_PRICE_INR):
            return None
        if adv < self._cfg.MIN_AVG_DAILY_VOLUME:
            return None
        if turnover_cr < self._cfg.MIN_AVG_DAILY_TURNOVER_CR:
            return None

        ann_vol = _compute_annualised_vol(closes)
        if not (self._cfg.MIN_ANNUALISED_VOL <= ann_vol <= self._cfg.MAX_ANNUALISED_VOL):
            return None

        # ---- Technical indicators ----
        momentum_20d = _compute_momentum(closes, 20)
        rsi_14 = _compute_rsi(closes, 14)

        # ---- Component scores ----
        liq_score  = _score_liquidity(adv, turnover_cr)
        mom_score  = _score_momentum(momentum_20d, rsi_14)
        vol_score  = _score_volatility(ann_vol)
        sec_score  = _score_sector(sector, self._weather_anomaly, self._sentiment_score, self._geo_risk)

        # ---- Index membership bonuses ----
        bonus = 0.0
        in_n50  = symbol in NIFTY50_SYMBOLS
        in_n200 = symbol in NIFTY200_SYMBOLS
        if in_n50:
            bonus = self._cfg.NIFTY50_BONUS
        elif in_n200:
            bonus = self._cfg.NIFTY200_BONUS
        else:
            bonus = self._cfg.NIFTY500_BONUS

        # ---- Weighted composite score ----
        composite = (
            liq_score  * self._cfg.WEIGHT_LIQUIDITY  +
            mom_score  * self._cfg.WEIGHT_MOMENTUM   +
            vol_score  * self._cfg.WEIGHT_VOLATILITY +
            sec_score  * self._cfg.WEIGHT_SECTOR_SIGNAL
        ) + bonus

        composite = min(1.0, round(composite, 4))

        return StockMetadata(
            symbol=symbol,
            instrument_token=token,
            sector=sector,
            last_price=last_price,
            avg_daily_volume=adv,
            avg_daily_turnover_cr=turnover_cr,
            annualised_vol=ann_vol,
            momentum_20d=momentum_20d,
            rsi_14=rsi_14,
            composite_score=composite,
            liquidity_score=liq_score,
            momentum_score=mom_score,
            volatility_score=vol_score,
            sector_score=sec_score,
            in_nifty50=in_n50,
            in_nifty200=in_n200,
            in_nifty500=symbol in NIFTY200_SYMBOLS,   # Treat NIFTY200 as proxy for 500
        )

    def _apply_sector_diversification(
        self, candidates: List[StockMetadata]
    ) -> List[StockMetadata]:
        """
        Select the top TARGET_UNIVERSE_SIZE stocks while enforcing sector caps.

        Algorithm:
          1. Sort all candidates by composite_score descending.
          2. Greedily add stocks to the universe.
          3. Skip if sector is already at MAX_STOCKS_PER_SECTOR.
          4. Stop when TARGET_UNIVERSE_SIZE is reached.
        """
        # Sort by score descending
        sorted_candidates = sorted(
            candidates, key=lambda m: m.composite_score, reverse=True
        )

        sector_counts: Dict[str, int] = {}
        selected: List[StockMetadata] = []

        for stock in sorted_candidates:
            if len(selected) >= self._cfg.TARGET_UNIVERSE_SIZE:
                break
            count = sector_counts.get(stock.sector, 0)
            if count >= self._cfg.MAX_STOCKS_PER_SECTOR:
                continue
            sector_counts[stock.sector] = count + 1
            selected.append(stock)

        # Log sector breakdown
        breakdown: Dict[str, int] = {}
        for s in selected:
            breakdown[s.sector] = breakdown.get(s.sector, 0) + 1
        logger.info("Universe sector breakdown: %s", breakdown)

        return selected

    # ------------------------------------------------------------------
    # Stage 4: Intraday dynamic rescoring
    # ------------------------------------------------------------------

    async def run_intraday_rescore(self) -> None:
        """
        Every 30 minutes during market hours:
          1. Fetch current-day volume from Kite for all universe stocks.
          2. Compute volume surge ratio vs. 5-day average.
          3. Stocks with surge > VOLUME_SURGE_MULTIPLIER are bumped up.
          4. OFI trends detected from Redis are used to flag active stocks.
          5. Bottom-ranked stocks replaced by surging outside-universe stocks.
        """
        logger.info("Universe: Running intraday rescore…")
        loop = asyncio.get_running_loop()

        # Fetch intraday volumes for current universe
        exchange_syms = [f"NSE:{s}" for s in self._active_symbols]
        if not exchange_syms:
            return

        try:
            async with self._limiter.request_slot():
                raw_quotes = await loop.run_in_executor(
                    None, lambda: self._kite.quote(exchange_syms)
                )
        except Exception as exc:
            logger.warning("Intraday rescore: quote fetch failed: %s", exc)
            return

        rescored: List[Tuple[str, float]] = []   # (symbol, updated_score)

        for key, quote in raw_quotes.items():
            sym = key.split(":")[1]
            meta = self._metadata.get(sym)
            if not meta:
                continue

            # Volume surge detection
            current_vol = float(quote.get("volume", 0))
            baseline_vol = meta.avg_daily_volume
            surge_ratio = (current_vol / baseline_vol) if baseline_vol > 0 else 1.0
            meta.volume_surge_ratio = round(surge_ratio, 3)

            # OFI boost: strong directional OFI = elevated score
            ofi_raw = await self._redis.get(f"{settings.redis.OFI_KEY_PREFIX}{sym}")
            ofi = float(ofi_raw) if ofi_raw else 0.0
            ofi_boost = min(0.1, abs(ofi) * 0.15)

            # Volume surge boost (capped at +0.15)
            surge_boost = min(0.15, math.log1p(max(0, surge_ratio - 1)) * 0.1)

            updated_score = min(1.0, meta.composite_score + surge_boost + ofi_boost)
            meta.composite_score = round(updated_score, 4)
            rescored.append((sym, updated_score))

        # Re-persist updated scores to Redis
        if rescored:
            pipe = self._redis.pipeline()
            try:
                for sym, score in rescored:
                    pipe.zadd(settings.redis.UNIVERSE_KEY, {sym: score})
                    meta = self._metadata.get(sym)
                    if meta:
                        pipe.hset(
                            f"{settings.redis.UNIVERSE_META_KEY}:{sym}",
                            mapping=meta.to_redis_dict(),
                        )
                await pipe.execute()
            finally:
                await pipe.reset()

        # Re-sort active symbols by updated scores
        self._active_symbols = sorted(
            self._active_symbols,
            key=lambda s: self._metadata[s].composite_score if s in self._metadata else 0,
            reverse=True,
        )

        self._last_intraday_rescore = time.monotonic()
        logger.info(
            "Intraday rescore complete. Top 3: %s",
            [(s, f"{self._metadata[s].composite_score:.3f}")
             for s in self._active_symbols[:3] if s in self._metadata],
        )

    # ------------------------------------------------------------------
    # Redis persistence
    # ------------------------------------------------------------------

    async def _persist_universe(self, universe: List[StockMetadata]) -> None:
        """
        Write universe to Redis:
          • Sorted Set: UNIVERSE_KEY → {symbol: composite_score}
          • Hash: UNIVERSE_META_KEY:SYMBOL → metadata fields
        """
        pipe = self._redis.pipeline()
        try:
            # Clear old universe
            pipe.delete(settings.redis.UNIVERSE_KEY)

            for meta in universe:
                pipe.zadd(settings.redis.UNIVERSE_KEY, {meta.symbol: meta.composite_score})
                pipe.hset(
                    f"{settings.redis.UNIVERSE_META_KEY}:{meta.symbol}",
                    mapping=meta.to_redis_dict(),
                )
                pipe.expire(f"{settings.redis.UNIVERSE_META_KEY}:{meta.symbol}", 86400)

            await pipe.execute()
        finally:
            await pipe.reset()
        logger.debug("Universe persisted to Redis (%d symbols).", len(universe))

    async def _load_from_redis(self) -> bool:
        """
        Attempt to load universe from Redis on startup (if today's data exists).
        Returns True if loaded successfully, False if refresh needed.
        """
        raw = await self._redis.zrange(
            settings.redis.UNIVERSE_KEY, 0, -1, withscores=True
        )
        if not raw:
            return False

        symbols = [item[0].decode() for item in raw]
        meta_map: Dict[str, StockMetadata] = {}

        for sym in symbols:
            raw_meta = await self._redis.hgetall(
                f"{settings.redis.UNIVERSE_META_KEY}:{sym}"
            )
            if raw_meta:
                try:
                    meta_map[sym] = StockMetadata.from_redis_dict(raw_meta)
                except Exception as exc:
                    logger.debug("Could not parse metadata for %s: %s", sym, exc)

        if not meta_map:
            return False

        # Verify it's today's data in IST
        from datetime import timedelta as _td
        ist = timezone(_td(hours=5, minutes=30))
        today_ist = datetime.now(ist).date().isoformat()

        sample_meta = next(iter(meta_map.values()), None)
        if sample_meta:
            scored_date_str = sample_meta.last_scored_at[:10]
            if scored_date_str != today_ist:
                logger.info("Redis universe is from %s (today is %s) — stale, will refresh.", scored_date_str, today_ist)
                return False

        self._active_symbols = symbols
        self._metadata = meta_map
        self._last_daily_refresh = date.today()
        logger.info("Loaded %d symbols from Redis cache.", len(symbols))
        return True

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_active_symbols(self) -> List[str]:
        """Return current active universe as a list, highest-scored first."""
        return list(self._active_symbols)

    def get_instrument_tokens(self) -> List[int]:
        """Return instrument tokens for the active universe (for WebSocket sub)."""
        return [
            self._metadata[s].instrument_token
            for s in self._active_symbols
            if s in self._metadata and self._metadata[s].instrument_token > 0
        ]

    def get_metadata(self, symbol: str) -> Optional[StockMetadata]:
        return self._metadata.get(symbol)

    def get_sector_breakdown(self) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        for sym in self._active_symbols:
            meta = self._metadata.get(sym)
            if meta:
                breakdown[meta.sector] = breakdown.get(meta.sector, 0) + 1
        return breakdown

    # ------------------------------------------------------------------
    # Change notification
    # ------------------------------------------------------------------

    async def _fire_change_callbacks(self) -> None:
        """Notify all registered listeners that the active universe changed."""
        for cb in self._change_callbacks:
            try:
                await cb(list(self._active_symbols))
            except Exception as exc:
                logger.warning("Universe change callback error: %s", exc)

    # ------------------------------------------------------------------
    # Background task schedulers
    # ------------------------------------------------------------------

    async def _daily_refresh_loop(self) -> None:
        """
        Loop that triggers run_daily_refresh() once per day at the configured
        IST time (default 08:45 IST = 03:15 UTC).
        """
        logger.info("Universe daily refresh scheduler started.")
        while True:
            try:
                now_utc = datetime.now(timezone.utc)
                ist_minutes = (now_utc.hour * 60 + now_utc.minute + 330) % (24 * 60)
                target_minutes = (
                    self._cfg.DAILY_REFRESH_HOUR_IST * 60
                    + self._cfg.DAILY_REFRESH_MINUTE_IST
                )

                # Check if a refresh is due today and hasn't run yet
                if (
                    self._last_daily_refresh != date.today()
                    and ist_minutes >= target_minutes
                ):
                    await self.run_daily_refresh()
                elif self._last_daily_refresh is None:
                    # First run ever — do it immediately regardless of time
                    await self.run_daily_refresh()

            except asyncio.CancelledError:
                logger.info("Universe daily refresh loop cancelled.")
                raise
            except Exception as exc:
                logger.error("Universe daily refresh error: %s", exc, exc_info=True)

            await asyncio.sleep(60)   # Check every minute

    async def _intraday_rescore_loop(self) -> None:
        """Trigger intraday rescore every INTRADAY_RESCORE_INTERVAL_MIN minutes."""
        logger.info("Universe intraday rescore scheduler started.")
        interval_s = self._cfg.INTRADAY_RESCORE_INTERVAL_MIN * 60

        while True:
            try:
                now = time.monotonic()
                if now - self._last_intraday_rescore >= interval_s:
                    if self._active_symbols:
                        await self.run_intraday_rescore()
            except asyncio.CancelledError:
                logger.info("Universe intraday rescore loop cancelled.")
                raise
            except Exception as exc:
                logger.error("Intraday rescore error: %s", exc, exc_info=True)

            await asyncio.sleep(60)

    async def initialise(self) -> None:
        """
        Called once at bot startup:
          1. Try to load today's universe from Redis cache.
          2. If not available, run a full daily refresh immediately.

        FIX B-18 (Bug #1): Previously, if run_daily_refresh() raised an exception,
        it propagated silently — the bot continued startup with _active_symbols=[]
        and the strategy loop would loop forever with "Empty universe — waiting for
        initialisation" every 5 seconds, never trading, with no visible error.

        Now: log the full traceback at CRITICAL level, then fall back to the static
        watchlist so the bot can trade at least a handful of liquid stocks while the
        dynamic universe is unavailable. A Telegram alert will surface the failure.
        """
        logger.info("UniverseEngine: initialising…")
        loaded = await self._load_from_redis()
        if not loaded:
            logger.info("No cached universe — running full refresh now (this may take several minutes)…")
            try:
                await self.run_daily_refresh()
            except Exception as exc:
                logger.critical(
                    "UniverseEngine.initialise FAILED: %s\n"
                    "Falling back to static watchlist (%d symbols). "
                    "Universe will retry at next scheduled refresh (08:45 IST).",
                    exc, len(self._strat_cfg.STATIC_WATCHLIST),
                    exc_info=True,
                )
                # Fall back to static watchlist so strategy loop is not permanently stuck
                fallback = list(self._strat_cfg.STATIC_WATCHLIST)
                if not self._active_symbols and fallback:
                    self._active_symbols = fallback
                    logger.warning(
                        "Fallback universe set to static watchlist: %s",
                        self._active_symbols,
                    )
        else:
            logger.info("Using cached universe: %d symbols.", len(self._active_symbols))

    async def start_background_tasks(self) -> List[asyncio.Task]:
        """Spawn all background scheduler tasks. Returns task list for shutdown management."""
        tasks = [
            asyncio.create_task(self._daily_refresh_loop(), name="universe_daily_refresh"),
            asyncio.create_task(self._intraday_rescore_loop(), name="universe_intraday_rescore"),
        ]
        logger.info("UniverseEngine: %d background tasks started.", len(tasks))
        return tasks


# ---------------------------------------------------------------------------
# Frequency Optimiser
# ---------------------------------------------------------------------------

class FrequencyOptimiser:
    """
    Dynamically calculates the optimal strategy poll_interval based on
    current universe size and Zerodha API headroom.

    Design principles:
      • One batch LTP call covers all symbols → cost is 1 req regardless of size.
      • Processing overhead per symbol ≈ POLL_INTERVAL_PER_SYMBOL_MS ms
        (Redis reads, signal computation, conditional order placement).
      • We target consuming at most 30% of the 10 req/s budget for LTP polling,
        reserving the rest for orders and other calls.

    Formula:
        poll_interval = clamp(
            processing_overhead + api_overhead,
            MIN_POLL_INTERVAL,
            MAX_POLL_INTERVAL
        )

        processing_overhead = n_symbols * POLL_INTERVAL_PER_SYMBOL_MS / 1000
        api_overhead        = 1 / (0.3 * MAX_REQUESTS_PER_SECOND)   = 0.333 s
    """

    def __init__(self) -> None:
        self._cfg = settings.strategy

    def compute_interval(self, n_symbols: int) -> float:
        """Return the recommended poll_interval in seconds for n_symbols."""
        if n_symbols == 0:
            return self._cfg.MAX_POLL_INTERVAL_SECONDS

        # Processing overhead
        proc_overhead = n_symbols * (self._cfg.POLL_INTERVAL_PER_SYMBOL_MS / 1000)

        # API budget overhead: 1 LTP call per cycle; reserve 30% of 10 req/s
        api_overhead = 1.0 / (0.30 * settings.kite.MAX_REQUESTS_PER_SECOND)

        raw_interval = proc_overhead + api_overhead

        return round(
            max(self._cfg.MIN_POLL_INTERVAL_SECONDS,
                min(self._cfg.MAX_POLL_INTERVAL_SECONDS, raw_interval)),
            3,
        )

    def describe(self, n_symbols: int) -> str:
        interval = self.compute_interval(n_symbols)
        cycles_per_min = 60 / interval
        return (
            f"n={n_symbols} symbols → interval={interval:.3f}s "
            f"({cycles_per_min:.1f} cycles/min)"
        )
