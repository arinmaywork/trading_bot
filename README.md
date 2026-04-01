# SentiStack — NSE High-Frequency Trading Bot

A production-grade, fully asynchronous Python trading system for the Indian National Stock Exchange (NSE) that fuses **limit order book microstructure**, **real-time alternative data**, and **Gemini LLM cognitive sentiment** into a single unified alpha signal.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        main.py (Orchestrator)                  │
│   asyncio event loop  •  graceful shutdown  •  task registry   │
└──────────┬───────────────┬───────────────┬─────────────────────┘
           │               │               │
   ┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
   │ data_        │ │ alternative_ │ │ strategy.py    │
   │ ingestion.py │ │ data.py      │ │                │
   │              │ │              │ │ SentiStack α   │
   │ KiteTicker   │ │ WeatherAPI   │ │ RiskManager    │
   │ (WebSocket)  │ │ EODHD/yfinance│ │ Kelly Criterion│
   │ Redis Stream │ │ Gemini LLM   │ │                │
   │ OHLCV Candles│ │              │ └────────┬───────┘
   │ OFI          │ └──────────────┘          │
   └──────────────┘                    ┌──────▼───────┐
                                       │ execution.py │
                                       │              │
                                       │ OrderExecutor│
                                       │ OrderSlicer  │
                                       │ Telegram Bot │
                                       └──────────────┘
        ┌─────────────────────────────────────────────┐
        │          rate_limiter.py  (shared)           │
        │   10 req/s   •   400 orders/min  (Zerodha)  │
        └─────────────────────────────────────────────┘
```

---

## Module Reference

| File | Module | Key Classes |
|---|---|---|
| `config.py` | Configuration | `Settings`, `KiteConfig`, `StrategyConfig` |
| `rate_limiter.py` | Rate Control | `RateLimiter`, `TokenBucket` |
| `data_ingestion.py` | Module 1 — Microstructure | `AsyncKiteTickerWrapper`, `CandleAggregator`, `RedisStreamWriter` |
| `alternative_data.py` | Module 2 — Alt Data & LLM | `AlternativeDataPipeline`, `GeminiSentimentAnalyser`, `WeatherDataFetcher` |
| `strategy.py` | Module 3 — Signal & Risk | `StrategyEngine`, `RiskManager`, `SignalState` |
| `execution.py` | Module 4 — Execution | `OrderExecutor`, `OrderSlicer`, `TelegramNotifier` |
| `main.py` | Orchestrator | `main()`, `strategy_loop()`, `GracefulShutdown` |

---

## Mathematical Formulas

### Order Flow Imbalance (OFI)
```
OFI = (V_bid_L0 - V_ask_L0) / (V_bid_L0 + V_ask_L0)   ∈ [-1, +1]
```
Where `L0` denotes the best (level-0) bid/ask.

### SentiStack Alpha Signal
```
Alpha_t = ((P_t - VWAP_t) / VWAP_t) × log(1 + |S_t|) × sign(S_t) × f(OFI)
```
| Symbol | Meaning |
|---|---|
| `P_t` | Current last traded price |
| `VWAP_t` | Rolling 30-min volume-weighted average price |
| `S_t` | Gemini sentiment score ∈ [-1, 1] |
| `f(OFI)` | Piecewise LOB confirmation multiplier ∈ [0.5, 1.5] |

### Risk-Constrained Half-Kelly
```
f* = α / σ²            (full Kelly for continuous returns)
f_half = 0.5 × f*      (Half-Kelly conservatism)
Q = floor(f_half × Capital / Price)
```
**Decay conditions** (size → 0):
- `σ_t > 2.0 × σ_baseline`   (volatility spike)
- Geopolitical risk index `> 0.65`

---

## Setup

### 1. Prerequisites
- Python 3.10+
- Redis 7+ (local or remote)
- Zerodha Kite Connect account with live WebSocket access

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
# Zerodha
export KITE_API_KEY="your_api_key"
export KITE_API_SECRET="your_api_secret"
export KITE_ACCESS_TOKEN="your_access_token"   # Regenerate daily

# Redis
export REDIS_HOST="127.0.0.1"
export REDIS_PORT="6379"
export REDIS_PASSWORD=""   # optional

# Google Gemini
export GEMINI_API_KEY="your_gemini_key"

# OpenWeatherMap
export OPENWEATHER_API_KEY="your_owm_key"

# Telegram Bot
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
export TELEGRAM_CHAT_ID="-1001234567890"   # Private channel chat ID

# Optional
export EODHD_API_KEY="your_eodhd_key"   # Falls back to yfinance if absent
export TOTAL_CAPITAL="500000"            # INR (default: 500,000)
```

### 4. Access Token Regeneration
The Kite access token expires daily at 06:00 IST. Use the standard
Kite Connect OAuth flow to regenerate it before market open.

### 5. Run
```bash
python main.py
```

---

## Redis Key Schema

| Key Pattern | Type | Contents |
|---|---|---|
| `lob:ticks` | Stream | Raw tick payloads (symbol, ltp, volume, OFI, LOB) |
| `candles:1m:<SYM>:<YYYYMMDD_HHMM>` | Hash | open, high, low, close, volume, vwap_sum |
| `ofi:<SYMBOL>` | String | Latest OFI float, TTL=60s |

---

## Telegram Notifications

The bot pushes formatted HTML notifications to your private Telegram channel:

```
✅ TRADE EXECUTION
──────────────────────────────
🏷️ Symbol:       NTPC
📊 Action:       🟢 BUY
🔢 Quantity:     1,200 shares
💰 Avg Fill:     ₹385.40
🎯 Alpha Score:  +0.002341
📉 Slippage:     +2.1 bps
──────────────────────────────
🧠 Sentiment:    Excitement
📝 Rationale:    Surprise RBI rate cut expected to...
```

---

## Risk Warnings

> **This software is provided for educational and research purposes.**
> Live trading with real capital carries significant financial risk.
> The bot makes no guarantees of profitability.
> Always test in paper-trading mode first.
> Ensure compliance with SEBI regulations and Zerodha's terms of service.

---

## Extending the Bot

- **Add more symbols**: Update `WATCHLIST` in `config.py`
- **Tune alpha**: Modify `compute_alpha()` in `strategy.py`
- **Add geopolitical feed**: Replace the `geo_risk` proxy in `main.py` with a real index (e.g., GDELT)
- **Limit orders**: Change `ORDER_TYPE` to `"LIMIT"` in `config.py` and pass a price to `place_order`
