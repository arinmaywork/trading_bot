"""
config.py  — V2
===============
Centralised configuration with V2 additions.
All secrets loaded from environment variables — never hard-coded.

V2 additions:
  • MLConfig         — ensemble model parameters
  • AgentConfig      — LangGraph agent pipeline settings
  • BussetiConfig    — risk-constrained Kelly parameters
  • FREDConfig       — FRED API for GPR index
"""

import os
from dataclasses import dataclass, field
from typing import List


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Required env var '{key}' not set. Export it before running the bot."
        )
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ---------------------------------------------------------------------------
# Zerodha / Kite Connect
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KiteConfig:
    API_KEY:                str   = field(default_factory=lambda: _require("KITE_API_KEY"))
    API_SECRET:             str   = field(default_factory=lambda: _require("KITE_API_SECRET"))
    ACCESS_TOKEN:           str   = field(default_factory=lambda: _optional("KITE_ACCESS_TOKEN"))
    MAX_REQUESTS_PER_SECOND: int  = 10
    MAX_ORDERS_PER_MINUTE:  int   = 400
    ORDER_FREEZE_LIMIT:     int   = 5_000
    ORDER_VARIETY:          str   = "regular"
    PRODUCT:                str   = "MIS"
    # LIMIT instead of MARKET: Zerodha's API rejects bare market orders
    # without a market_protection param that the kiteconnect library does
    # not expose. Aggressive LIMIT orders (± LIMIT_PRICE_BUFFER_PCT from
    # signal price) fill immediately on liquid NSE stocks and give us
    # explicit slippage control.
    ORDER_TYPE:             str   = "LIMIT"
    EXCHANGE:               str   = "NSE"
    # BUY  → limit = signal_price × (1 + LIMIT_PRICE_BUFFER_PCT)
    # SELL → limit = signal_price × (1 − LIMIT_PRICE_BUFFER_PCT)
    # 0.2 % buffer is enough to absorb 1-second book movement on Nifty50 stocks.
    LIMIT_PRICE_BUFFER_PCT: float = 0.002
    PAPER_TRADE:            bool  = field(
        default_factory=lambda: _optional("PAPER_TRADE", "false").lower() == "true"
    )


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RedisConfig:
    HOST:            str = field(default_factory=lambda: _optional("REDIS_HOST", "127.0.0.1"))
    PORT:            int = field(default_factory=lambda: int(_optional("REDIS_PORT", "6379")))
    DB:              int = 0
    PASSWORD:        str = field(default_factory=lambda: _optional("REDIS_PASSWORD", ""))
    TICK_STREAM:     str = "lob:ticks"
    CANDLE_HASH:     str = "candles:1m"
    OFI_KEY_PREFIX:  str = "ofi:"
    MAX_STREAM_LEN:  int = 100_000
    UNIVERSE_KEY:    str = "universe:active"
    UNIVERSE_META_KEY: str = "universe:meta"


# ---------------------------------------------------------------------------
# Alternative Data — Weather
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeatherConfig:
    API_KEY:               str       = field(default_factory=lambda: _require("OPENWEATHER_API_KEY"))
    BASE_URL:              str       = "https://api.openweathermap.org/data/2.5/weather"
    CITIES:                List[str] = field(default_factory=lambda: [
        "Mumbai", "Delhi", "Chennai", "Hyderabad", "Ahmedabad",
        "Pune", "Bengaluru", "Kolkata", "Jaipur", "Surat",
    ])
    FETCH_INTERVAL_SECONDS: int = 300


# ---------------------------------------------------------------------------
# Financial News (EODHD)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NewsConfig:
    API_KEY:               str = field(default_factory=lambda: _optional("EODHD_API_KEY", ""))
    BASE_URL:              str = "https://eodhd.com/api/news"
    FETCH_INTERVAL_SECONDS: int = 120
    MAX_ARTICLES:          int = 10


# ---------------------------------------------------------------------------
# Google Gemini (LLM) — V2: used by AgentPipeline
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GeminiConfig:
    API_KEY:          str   = field(default_factory=lambda: _require("GEMINI_API_KEY"))
    MODEL:            str   = "gemini-1.5-flash"
    MAX_OUTPUT_TOKENS: int  = 512
    TEMPERATURE:      float = 0.15


# ---------------------------------------------------------------------------
# FRED API — for Caldara-Iacoviello GPR Index
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FREDConfig:
    # FRED API key is optional (CSV endpoint is public)
    API_KEY: str = field(default_factory=lambda: _optional("FRED_API_KEY", ""))
    GPR_REFRESH_INTERVAL_SECONDS: int = 3600   # Monthly data — refresh hourly


# ---------------------------------------------------------------------------
# LangGraph Agent Pipeline  (Enhancement 3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentConfig:
    # Conviction threshold: route to RiskManagerAgent if |score| > this
    HIGH_CONVICTION_THRESHOLD: float = 0.40
    # VIX threshold above which vol regime is "HIGH"
    VIX_HIGH_THRESHOLD:   float = 22.0
    VIX_EXTREME_THRESHOLD: float = 30.0
    # Max concurrent agent invocations (prevents Gemini rate limit)
    MAX_CONCURRENT_RUNS: int = 3
    # Retry on LLM failure
    MAX_RETRIES: int = 2


# ---------------------------------------------------------------------------
# ML Ensemble Signal  (Enhancement 4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MLConfig:
    # Rolling feature window
    FEATURE_WINDOW_SIZE: int  = 200
    MIN_TRAIN_SAMPLES:   int  = 50
    RETRAIN_INTERVAL_S:  int  = 1800   # 30 minutes

    # XGBoost params
    XGB_N_ESTIMATORS: int   = 80
    XGB_MAX_DEPTH:    int   = 4
    XGB_LEARNING_RATE: float = 0.08
    XGB_SUBSAMPLE:    float  = 0.80
    XGB_COLSAMPLE:    float  = 0.80

    # Ridge alpha (regularisation strength)
    RIDGE_ALPHA: float = 1.0

    # Signal scaling: maps raw log-return output to [-1, 1]
    SIGNAL_SCALE: float = 20.0

    # Redis TTL for feature store
    FEATURE_STORE_TTL_S: int = 86400


# ---------------------------------------------------------------------------
# Busseti Risk-Constrained Kelly  (Enhancement 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BussetiConfig:
    # Maximum probability of a ruin event per period
    EPSILON:        float = 0.05    # 5%
    # Minimum acceptable wealth ratio (ruin threshold)
    W_FLOOR:        float = 0.95    # Portfolio must not fall below 95%
    # Bisection iterations (20 → convergence to within 0.001)
    BISECTION_ITERS: int  = 20


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TelegramConfig:
    BOT_TOKEN:           str = field(default_factory=lambda: _require("TELEGRAM_BOT_TOKEN"))
    CHAT_ID:             str = field(default_factory=lambda: _require("TELEGRAM_CHAT_ID"))
    API_BASE:            str = "https://api.telegram.org/bot"
    SEND_TIMEOUT_SECONDS: int = 10


# ---------------------------------------------------------------------------
# Universe Selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UniverseConfig:
    MIN_PRICE_INR:          float = 20.0
    MAX_PRICE_INR:          float = 50_000.0
    MIN_AVG_DAILY_VOLUME:   int   = 200_000
    MIN_AVG_DAILY_TURNOVER_CR: float = 2.0
    MIN_ANNUALISED_VOL:     float = 0.10
    MAX_ANNUALISED_VOL:     float = 0.90
    WEIGHT_LIQUIDITY:       float = 0.35
    WEIGHT_MOMENTUM:        float = 0.25
    WEIGHT_VOLATILITY:      float = 0.20
    WEIGHT_SECTOR_SIGNAL:   float = 0.20
    MAX_STOCKS_PER_SECTOR:  int   = 8
    TARGET_UNIVERSE_SIZE:   int   = 50
    NIFTY50_BONUS:          float = 0.15
    NIFTY200_BONUS:         float = 0.08
    NIFTY500_BONUS:         float = 0.03
    DAILY_REFRESH_HOUR_IST:  int  = 8
    DAILY_REFRESH_MINUTE_IST: int = 45
    INTRADAY_RESCORE_INTERVAL_MIN: int = 30
    VOLUME_SURGE_MULTIPLIER: float = 2.5
    HISTORICAL_DAYS:        int   = 30
    MAX_WEBSOCKET_SUBSCRIPTIONS: int = 200


# ---------------------------------------------------------------------------
# Strategy Parameters (V2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    STATIC_WATCHLIST: List[str] = field(default_factory=lambda: [
        "RELIANCE", "NTPC", "POWERGRID", "TATAPOWER", "ADANIGREEN"
    ])
    USE_DYNAMIC_UNIVERSE: bool = field(
        default_factory=lambda: _optional("USE_DYNAMIC_UNIVERSE", "true").lower() == "true"
    )
    KELLY_FRACTION:              float = 0.50   # Used as fallback only; Busseti is primary
    MAX_POSITION_FRACTION:       float = 0.05
    MIN_ALPHA_THRESHOLD:         float = 0.005  # R-10: raised from 0.001 → 0.5% min signal
    VOL_SPIKE_THRESHOLD:         float = 2.0
    GEOPOLITICAL_RISK_THRESHOLD: float = 0.65
    VWAP_LOOKBACK:               int   = 30
    OFI_STRONG_BUY_THRESHOLD:    float = 0.30
    OFI_STRONG_SELL_THRESHOLD:   float = -0.30
    TOTAL_CAPITAL:               float = field(
        default_factory=lambda: float(_optional("TOTAL_CAPITAL", "500000"))
    )
    MIN_POLL_INTERVAL_SECONDS:   float = 0.5
    MAX_POLL_INTERVAL_SECONDS:   float = 5.0
    POLL_INTERVAL_PER_SYMBOL_MS: float = 20.0

    # ---------------------------------------------------------------------------
    # R-10: Transaction cost awareness (Zerodha NSE-EQ intraday / MIS)
    # ---------------------------------------------------------------------------
    # Zerodha charges ₹20 flat OR 0.03% of order value (whichever is lower)
    # per executed order. For a round trip (buy + sell) = 2 orders = up to ₹40.
    BROKERAGE_PER_ORDER:         float = 20.0    # ₹ per executed order (flat cap)
    BROKERAGE_PCT:               float = 0.0003  # 0.03% — used when < flat cap
    STT_INTRADAY_SELL_RATE:      float = 0.00025 # 0.025% STT on sell-side only (intraday)
    EXCHANGE_CHARGE_RATE:        float = 0.0000345  # NSE turnover charge (both sides)

    # Minimum ₹ value per trade so brokerage stays a small fraction of P&L.
    # Example: ₹2000 position → brokerage ≈ ₹40 → only 2% cost hurdle vs P&L.
    # Trades that can't reach this value are skipped (cost would exceed profit).
    MIN_TRADE_VALUE:             float = field(
        default_factory=lambda: float(_optional("MIN_TRADE_VALUE", "2000"))
    )

    # Forced square-off time (IST). All open MIS positions closed at this time
    # to avoid Zerodha auto square-off charges (₹50+GST per position after 3:20 PM).
    SQUARE_OFF_HOUR_IST:         int   = 15
    SQUARE_OFF_MINUTE_IST:       int   = 15

    # ---------------------------------------------------------------------------
    # R-12: Trailing Stop Loss (TSL) & Risk Management
    # ---------------------------------------------------------------------------
    # TSL_ACTIVATION_PCT: Profit % at which TSL starts trailing (e.g. 0.6%)
    # TSL_CALLBACK_PCT:   % distance from peak price to trigger exit (e.g. 0.3%)
    # HARD_STOP_LOSS_PCT: Max loss from entry price before emergency exit (e.g. 1.0%)
    TSL_ACTIVATION_PCT:   float = 0.006   # 0.6% profit triggers TSL
    TSL_CALLBACK_PCT:     float = 0.003   # 0.3% trailing buffer
    HARD_STOP_LOSS_PCT:   float = 0.012   # 1.2% hard stop

    # ---------------------------------------------------------------------------
    # R-11: CNC / MIS Hybrid Strategy
    # ---------------------------------------------------------------------------
    # Signals with high ML confidence AND generated before CNC_ENTRY_CUTOFF_HOUR
    # are classified as CNC (delivery/swing, held overnight). All others → MIS.
    #
    # CNC benefits: no auto-square risk, no need to trade under time pressure.
    # CNC costs:    higher STT (0.1% both sides vs 0.025% sell-only for MIS),
    #               overnight gap risk, capital locked for multiple days.
    #
    # CNC_CAPITAL_PCT: fraction of TOTAL_CAPITAL reserved for CNC swing positions.
    # The remaining fraction is used for MIS intraday positions.
    CNC_MIN_CONFIDENCE:    float = 0.75    # ML confidence required to qualify as CNC
    CNC_ALPHA_THRESHOLD:   float = 0.008   # Stronger alpha required for overnight risk
    CNC_CAPITAL_PCT:       float = 0.40    # 40% of capital reserved for CNC swing
    CNC_ENTRY_CUTOFF_HOUR: int   = 13      # Only enter CNC positions before 1 PM IST
    CNC_MAX_HOLD_DAYS:     int   = 5       # Auto-exit CNC positions after this many days


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LoggingConfig:
    LEVEL:  str = "INFO"
    FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT: str = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Master settings object  (V2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    kite:     KiteConfig     = field(default_factory=KiteConfig)
    redis:    RedisConfig    = field(default_factory=RedisConfig)
    weather:  WeatherConfig  = field(default_factory=WeatherConfig)
    news:     NewsConfig     = field(default_factory=NewsConfig)
    gemini:   GeminiConfig   = field(default_factory=GeminiConfig)
    fred:     FREDConfig     = field(default_factory=FREDConfig)
    agent:    AgentConfig    = field(default_factory=AgentConfig)
    ml:       MLConfig       = field(default_factory=MLConfig)
    busseti:  BussetiConfig  = field(default_factory=BussetiConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    logging:  LoggingConfig  = field(default_factory=LoggingConfig)


settings = Settings()
