# SentiStack V2 — NSE Algo Trading Bot

A production-grade, fully asynchronous Python trading system for the Indian National Stock Exchange (NSE). It fuses **limit order book microstructure**, **ML ensemble signals**, **LLM cognitive sentiment**, and **geopolitical risk monitoring** into a single unified alpha signal, with full cloud deployment on GCP free tier and Telegram-based remote control.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    main.py  (Orchestrator)                       │
│   asyncio event loop  •  graceful shutdown  •  task registry    │
│   TokenManager (headless Telegram OAuth)                         │
└──────┬──────────────┬──────────────┬──────────────┬─────────────┘
       │              │              │              │
┌──────▼──────┐ ┌─────▼──────┐ ┌────▼──────┐ ┌────▼───────────┐
│data_        │ │alternative_ │ │geopolit-  │ │universe.py     │
│ingestion.py │ │data.py      │ │ical.py    │ │                │
│             │ │             │ │           │ │UniverseEngine  │
│KiteTicker   │ │WeatherAPI   │ │India VIX  │ │FrequencyOpt.   │
│(WebSocket)  │ │EODHD/news   │ │USD/INR    │ │Dynamic top-50  │
│Redis Stream │ │Gemini LLM   │ │GPR Index  │ │Intraday rescore│
│OHLCV Candles│ │agent_pipeline│ │           │ │                │
│MLOFI / OFI  │ └─────────────┘ └───────────┘ └────────────────┘
└─────────────┘
       │
┌──────▼──────────────────────────────────────────────┐
│                   strategy.py                        │
│   StrategyEngine  •  RiskManager  •  SignalState     │
│   CNC / MIS hybrid product classification            │
│   Busseti Risk-Constrained Kelly sizing              │
└──────────────────────────┬──────────────────────────┘
                           │
                    ┌──────▼───────┐     ┌────────────────────┐
                    │ execution.py │     │telegram_controller │
                    │              │     │                    │
                    │ OrderExecutor│     │ BotState           │
                    │ LIMIT+buffer │     │ Live commands      │
                    │ square-off   │     │ Error forwarding   │
                    │ CNC/MIS fill │     │ Task heartbeats    │
                    └──────────────┘     └────────────────────┘
```

---

## Features

- **Dynamic Universe** — scores 400+ NSE stocks daily, trades the top 50 by liquidity and momentum
- **ML Ensemble** — XGBoost + Ridge meta-learner predicts 1-min forward returns, retrains every 30 min
- **Gemini Sentiment** — LLM cognitive sentiment from live news headlines (auto-rotates models on quota limits)
- **Geopolitical Risk** — composite GRI from VIX, USD/INR, and keyword conflict signals
- **CNC / MIS Hybrid** — high-confidence signals with early entry become CNC (delivery/swing); rest are MIS (intraday)
- **FIFO P&L Tracking** — per-symbol realized P&L with brokerage and STT costs; accessible via `/pnl`
- **Headless Startup** — no terminal needed; sends Zerodha login URL to Telegram on startup
- **Auto Error Forwarding** — any ERROR/CRITICAL log line is forwarded to Telegram automatically
- **Task Heartbeats** — each background task reports health; visible via `/tasks`
- **Trade Guards** — stop new buys or sells instantly via Telegram without restarting

---

## Module Reference

| File | Responsibility | Key Classes |
|---|---|---|
| `main.py` | Orchestration & startup | `TokenManager`, `strategy_loop`, `GracefulShutdown` |
| `config.py` | Centralised settings | `Settings`, `KiteConfig`, `StrategyConfig` |
| `data_ingestion.py` | LOB & tick processing | `AsyncKiteTickerWrapper`, `CandleAggregator` |
| `ml_signal.py` | Machine learning | `EnsembleSignalEngine`, `FeatureStore` |
| `strategy.py` | Signal fusion & risk | `StrategyEngine`, `RiskManager`, `SignalState` |
| `universe.py` | Stock selection | `UniverseEngine`, `FrequencyOptimiser` |
| `alternative_data.py` | External feeds | `AlternativeDataPipeline`, `WeatherDataFetcher` |
| `geopolitical.py` | Macro risk | `GeopoliticalRiskMonitor`, `GPRSnapshot` |
| `execution.py` | Order management | `OrderExecutor`, `OrderSlicer` |
| `telegram_controller.py` | Remote control | `TelegramController`, `BotState` |
| `telegram_log_handler.py` | Error alerting | `TelegramLogHandler` |
| `logbook.py` | Trade logging & P&L | `Logbook`, `TradeLogRow`, `get_pnl_report` |
| `agent_pipeline.py` | LLM pipeline | `AgentPipeline`, `ModelRotator` |
| `rate_limiter.py` | API compliance | `RateLimiter`, `TokenBucket` |
| `refresh_token.py` | Manual token tool | Interactive + `--check` / `--auto` modes |

---

## Mathematical Foundations

### Multi-Level OFI (MLOFI)
```
MLOFI = Σ_{i=0}^{4}  w_i × (V_bid_i − V_ask_i) / (V_bid_i + V_ask_i)
Weights: [0.40, 0.25, 0.18, 0.10, 0.07]
```

### SentiStack Alpha Signal
```
Alpha_t = ML_Signal × Geopolitical_Multiplier
```
`ML_Signal` is the XGBoost+Ridge meta-learner output ∈ [-1, 1], trained on microstructure, sentiment, and volatility features.

### Busseti Risk-Constrained Kelly
```
f* = argmax E[log(1 + f·r)]
     subject to P(Wealth < 95%) ≤ 5%
```
Solved via bisection on the empirical return distribution stored in Redis.

### CNC / MIS Classification
A signal is classified as **CNC** (delivery) when all conditions hold:
- Signal is not decayed
- Direction is not FLAT
- `confidence ≥ CNC_MIN_CONFIDENCE` (default 0.72)
- `|ml_signal| ≥ CNC_ALPHA_THRESHOLD` (default 0.55)
- Entry time is before `CNC_ENTRY_CUTOFF_HOUR` IST (default 13:00)

All other signals are **MIS** (intraday, squared off by 3:20 PM IST).

---

## Setup

### Option A — GCP Free Tier (recommended for 24/7 running)

**Free tier constraints:** e2-micro only, regions us-central1 / us-east1 / us-west1, 30 GB pd-standard disk. Keep the VM always running — static IP is free while running but billed when stopped.

```bash
# 1. Create VM
gcloud compute instances create sentistack \
  --machine-type=e2-micro \
  --zone=us-central1-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard

# 2. SSH in and run setup
gcloud compute ssh sentistack --zone=us-central1-a
bash deploy/setup-gcp.sh

# 3. Fill in your API keys
nano /opt/sentistack/trading_bot/.env

# 4. Start the bot
sudo systemctl start sentistack
sudo journalctl -u sentistack -f
```

On first start, your Telegram receives the Zerodha login URL. Send `/token <request_token>` and the bot starts automatically.

### Option B — Local / Manual

**Prerequisites:** Python 3.10+, Redis 7+

```bash
git clone https://github.com/arinmaywork/trading_bot.git
cd trading_bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env          # fill in all required keys
python main.py
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `KITE_API_KEY` | ✅ | Zerodha Kite Connect app API key |
| `KITE_API_SECRET` | ✅ | Zerodha Kite Connect app secret |
| `KITE_ACCESS_TOKEN` | — | Leave blank — bot fetches via Telegram on startup |
| `TELEGRAM_BOT_TOKEN` | ✅ | From @BotFather |
| `TELEGRAM_CHAT_ID` | ✅ | Your personal chat ID (from @userinfobot) |
| `GEMINI_API_KEY` | ✅ | Google AI Studio (free tier available) |
| `OPENWEATHER_API_KEY` | ✅ | OpenWeatherMap free tier |
| `EODHD_API_KEY` | — | Optional news feed; falls back to yfinance |
| `FRED_API_KEY` | — | Optional GPR data from FRED |
| `REDIS_HOST` | — | Default: `localhost` (use `redis` for Docker) |
| `REDIS_PORT` | — | Default: `6379` |
| `TOTAL_CAPITAL` | — | INR, default `500000` |
| `PAPER_TRADE` | — | `true` for paper mode, `false` for live |
| `MIN_TRADE_VALUE` | — | Default `2000` INR |

---

## Daily Token Flow (headless server)

The Zerodha access token expires at midnight IST. On each restart, the bot automatically:

1. Checks `.kite_token` cache — if today's token exists and is valid, uses it immediately
2. If no valid token, sends a Telegram message:
   ```
   🔑 SentiStack — Zerodha login required
   1️⃣ Open: https://kite.zerodha.com/connect/login?api_key=...
   2️⃣ Log in, copy request_token from redirect URL
   3️⃣ Send: /token XXXXXXXX
   ```
3. Once you send `/token`, the bot starts up within seconds

For mid-session token refresh (without restart): use `/login` then `/token`.

---

## Telegram Commands

### Trading Control
| Command | Description |
|---|---|
| `/pause` | Pause order execution (monitoring continues) |
| `/resume` | Resume order execution |
| `/stop` | Graceful shutdown |
| `/nobuy` | Block all new BUY entries |
| `/okbuy` | Re-enable BUY entries |
| `/nosell` | Block all new SELL/short entries |
| `/oksell` | Re-enable SELL entries |
| `/mode` | Switch trading mode (FULL / GRI_ONLY / PAPER_MONITOR) |

### Monitoring
| Command | Description |
|---|---|
| `/status` | Full pipeline snapshot (GRI, VIX, sentiment, active guards) |
| `/tasks` | Background task heartbeat status (🟢 < 2 min / 🟡 < 10 min / 🔴 stale) |
| `/pnl` | Today's P&L statement — MIS + CNC sections with net after brokerage & STT |
| `/capital <amount>` | Update paper trade capital (INR) |

### Token Management
| Command | Description |
|---|---|
| `/login` | Get today's Zerodha login URL |
| `/token <request_token>` | Apply new access token without restarting the bot |

---

## Redis Key Schema

| Key Pattern | Type | Contents |
|---|---|---|
| `lob:ticks` | Stream | Raw tick payloads (symbol, ltp, volume, MLOFI, LOB) |
| `candles:1m:<SYM>:<TS>` | Hash | OHLCV + VWAP + MLOFI aggregates |
| `ofi:<SYMBOL>` | String | Latest OFI float, TTL=60s |
| `ml:features:<SYM>` | List | Rolling 200-observation feature window |
| `universe:active` | Set | Currently traded symbols |

---

## Risk Warnings

> **This software is for educational and research purposes.**
> Live trading carries significant financial risk. The bot makes no guarantees of profitability.
> Always run in `PAPER_MONITOR` mode for at least one full trading week before enabling live orders.
> Ensure compliance with SEBI regulations and Zerodha's terms of service.
> The authors accept no responsibility for financial losses.
