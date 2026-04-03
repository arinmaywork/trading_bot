# SentiStack V2 — Full System Documentation

SentiStack is a production-grade, asynchronous trading system for the National Stock Exchange (NSE). It fuses **Limit Order Book (LOB) microstructure**, **ML ensemble signals**, **LLM cognitive sentiment**, **geopolitical risk monitoring**, and **CNC/MIS hybrid product classification** into a unified execution engine deployable on GCP free tier with full Telegram remote control.

**Last updated:** April 2026

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [System Architecture](#2-system-architecture)
3. [Module Reference](#3-module-reference)
4. [Telegram Commands](#4-telegram-commands)
5. [Daily Token Workflow](#5-daily-token-workflow)
6. [Trading Modes](#6-trading-modes)
7. [CNC / MIS Hybrid Strategy](#7-cnc--mis-hybrid-strategy)
8. [Mathematical Foundations](#8-mathematical-foundations)
9. [Infrastructure & Data Schema](#9-infrastructure--data-schema)
10. [GCP Deployment](#10-gcp-deployment)
11. [Known Issues & Fixes](#11-known-issues--fixes)
12. [Best Practices & Risk Warnings](#12-best-practices--risk-warnings)

---

## 1. Quick Start

### Prerequisites
- Python 3.10+
- Redis 7+
- Zerodha Kite Connect API credentials with live WebSocket access
- Telegram bot token (from @BotFather) and your personal chat ID

### Install & Run (local)
```bash
git clone https://github.com/arinmaywork/trading_bot.git
cd trading_bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env       # fill in all required keys (no quotes needed)
python main.py
```

### Install & Run (GCP — recommended)
```bash
# On your GCP VM (e2-micro, us-central1-a)
bash deploy/setup-gcp.sh   # one-time setup
nano /opt/sentistack/trading_bot/.env
sudo systemctl start sentistack
sudo journalctl -u sentistack -f
```

On first start, your **Telegram receives the Zerodha login URL**. Send `/token <request_token>` and the bot starts within seconds. No terminal interaction required.

---

## 2. System Architecture

The bot runs entirely inside a single `asyncio` event loop. All tasks are co-operative coroutines — no threads except the Telegram log handler and Redis I/O.

```
Startup
  └── TokenManager.get_token()
        ├── Check KITE_ACCESS_TOKEN env var
        ├── Check .kite_token cache (today's date)
        └── [headless] _telegram_oauth_flow()  ← sends URL to Telegram, waits for /token

Main Event Loop (asyncio)
  ├── tg_controller.poll_loop()          — Telegram command listener
  ├── universe_daily_refresh()           — rebuilds universe at 08:45 IST
  ├── universe_intraday_rescore()        — rescores every 30 min
  ├── candle_aggregator()                — 1-min OHLCV from Redis tick stream
  ├── alt_data_analysis()                — weather + news + Gemini sentiment
  ├── geopolitical_monitor()             — VIX / USD-INR / GPR updates
  ├── ml_retrain_loop()                  — XGBoost+Ridge retrains every 30 min
  ├── strategy_loop()                    — main signal evaluation cycle
  │     ├── Check no_new_buys / no_new_sells guards
  │     ├── StrategyEngine.evaluate()
  │     ├── CNC/MIS classification
  │     ├── RiskManager Kelly sizing
  │     ├── OrderExecutor.execute()
  │     └── Logbook.log_trade()
  └── sentiment_loop()                   — periodic Gemini analysis
```

**Key design decisions:**
- All market time checks use explicit `timezone(timedelta(hours=5, minutes=30))` — server timezone is irrelevant
- Square-off uses LIMIT orders at ±2% LTP buffer to avoid Zerodha "protection amount 0" error
- Telegram error log handler forwards any ERROR/CRITICAL line automatically (rate-limited to 1/30s)
- Trade guards (`no_new_buys`, `no_new_sells`) are checked each strategy cycle before any signal is acted on

---

## 3. Module Reference

### Core Modules

| File | Responsibility | Key Classes / Functions |
|---|---|---|
| `main.py` | Orchestration, startup, strategy loop | `TokenManager`, `strategy_loop`, `is_market_open`, `GracefulShutdown` |
| `config.py` | Centralised settings (env → dataclass) | `Settings`, `KiteConfig`, `StrategyConfig`, `TelegramConfig` |
| `data_ingestion.py` | LOB tick processing, candle aggregation | `AsyncKiteTickerWrapper`, `CandleAggregator`, `calculate_mlofi` |
| `ml_signal.py` | ML ensemble signal engine | `EnsembleSignalEngine`, `FeatureStore`, `FeatureVector` |
| `strategy.py` | Signal fusion, risk sizing, CNC/MIS | `StrategyEngine`, `RiskManager`, `SignalState`, `busseti_kelly_bisection` |
| `universe.py` | Dynamic stock universe | `UniverseEngine`, `FrequencyOptimiser`, `StockScorer` |
| `alternative_data.py` | External data feeds | `AlternativeDataPipeline`, `WeatherDataFetcher`, `NewsFetcher` |
| `geopolitical.py` | Macro risk monitoring | `GeopoliticalRiskMonitor`, `GPRSnapshot` |
| `execution.py` | Order placement and square-off | `OrderExecutor`, `OrderSlicer`, `TelegramNotifier` |
| `agent_pipeline.py` | LLM sentiment pipeline | `AgentPipeline`, `ModelRotator`, `SentimentAnalyzerAgent` |

### Support Modules

| File | Responsibility |
|---|---|
| `telegram_controller.py` | All Telegram commands, `BotState`, trade guards, `/pnl` |
| `telegram_log_handler.py` | Auto-forwards ERROR/CRITICAL logs to Telegram (rate-limited) |
| `logbook.py` | Daily trade CSV logging, FIFO P&L computation, brokerage/STT cost model |
| `rate_limiter.py` | Token-bucket (10 req/s, 400 orders/min) for Zerodha API compliance |
| `refresh_token.py` | Standalone token refresh tool (`--check`, `--auto`, `--force` modes) |

---

## 4. Telegram Commands

### Trading Control

| Command | Effect |
|---|---|
| `/pause` | Pauses order execution; monitoring and signal evaluation continue |
| `/resume` | Resumes order execution |
| `/stop` | Graceful shutdown of the bot |
| `/nobuy` | Blocks all new BUY entries for the session |
| `/okbuy` | Re-enables BUY entries |
| `/nosell` | Blocks all new SELL/short entries |
| `/oksell` | Re-enables SELL entries |
| `/mode` | Opens mode selector keyboard (FULL / GRI_ONLY / PAPER_MONITOR) |

### Monitoring

| Command | Output |
|---|---|
| `/status` | GRI level, VIX, USD/INR, sentiment score, active trade guards, uptime |
| `/tasks` | Per-task heartbeat: 🟢 < 2 min / 🟡 < 10 min / 🔴 stale |
| `/pnl` | Today's FIFO P&L — MIS trades + CNC trades + grand total net of brokerage & STT |
| `/capital <amount>` | Update paper trade capital (INR) mid-session |

### Token Management

| Command | Usage |
|---|---|
| `/login` | Bot replies with today's Zerodha login URL |
| `/token <request_token>` | Exchange request_token for access_token; hot-swaps into running session |

Both bare request tokens (`/token abc123`) and full redirect URLs (`/token http://127.0.0.1/?request_token=abc123&status=success`) are accepted.

---

## 5. Daily Token Workflow

The Zerodha access token expires at midnight IST every day.

### Headless Server (GCP / systemd)

On every restart, `TokenManager.get_token()` runs this sequence:

```
1. Check KITE_ACCESS_TOKEN in .env → validate via kite.profile()
2. Check .kite_token cache file → validate if generated_date == today
3. If no valid token and stdin is not a tty (headless):
     → Send Telegram message with login URL
     → Long-poll Telegram for /token command
     → Exchange request_token → access_token
     → Save to .kite_token cache
     → Advance TelegramController._offset to skip this update
4. Continue startup
```

### Local / Interactive (terminal)

Same flow except step 3 falls back to the original interactive terminal prompt — browser opens automatically, paste the token.

### Manual Mid-Session Refresh

If the token expires during the trading day without restarting:
1. Send `/login` → bot replies with the login URL
2. Log in, copy the `request_token`
3. Send `/token <request_token>` → bot hot-swaps the token and resumes immediately

---

## 6. Trading Modes

Selected via Telegram at startup or changed mid-session with `/mode`.

| Mode | Orders Sent | Gemini Used | GRI Used |
|---|---|---|---|
| `PAPER_MONITOR` | ❌ None | ✅ | ✅ |
| `GRI_ONLY` | ✅ Live | ❌ | ✅ |
| `FULL` | ✅ Live | ✅ | ✅ |

**Recommended:** Start with `PAPER_MONITOR` for at least one full trading week to observe signal quality before switching to live.

---

## 7. CNC / MIS Hybrid Strategy

Each signal is classified as either CNC (delivery/swing) or MIS (intraday) at signal generation time in `strategy.py`.

### Classification Logic

A signal is marked **CNC** when ALL of the following hold:

| Condition | Default threshold |
|---|---|
| Signal is not decayed | — |
| Direction is not FLAT | — |
| `signal.confidence` ≥ `CNC_MIN_CONFIDENCE` | 0.72 |
| `abs(ml_signal)` ≥ `CNC_ALPHA_THRESHOLD` | 0.55 |
| Entry IST hour < `CNC_ENTRY_CUTOFF_HOUR` | 13 (1 PM IST) |

All other signals are **MIS**.

### Product Type Propagation

`SignalState.product_type` is set in `strategy.py`, read by:
- `execution.py` — passes `product_type` to Kite `place_order()`
- `logbook.py` — stores in trade CSV for P&L separation

### P&L Cost Model

| Cost | MIS | CNC |
|---|---|---|
| STT | 0.025% on sell leg only | 0.1% on both legs |
| Brokerage | ₹20 flat per order (Zerodha) | ₹20 flat per order |
| Exchange charges | 0.00345% (NSE) | 0.00345% (NSE) |

P&L is computed via FIFO matching per symbol. Accessible via `/pnl`.

---

## 8. Mathematical Foundations

### Multi-Level OFI (MLOFI)

Aggregates volume imbalance across 5 LOB levels with decaying weights:

```
MLOFI = Σ_{i=0}^{4}  w_i × (V_bid_i − V_ask_i) / (V_bid_i + V_ask_i)
Weights: [0.40, 0.25, 0.18, 0.10, 0.07]
```

### SentiStack Alpha Signal

```
Alpha_t = ML_Signal × Geopolitical_Multiplier
```

`ML_Signal` is the XGBoost+Ridge meta-learner output ∈ [-1, 1], trained on:
- MLOFI, Aggressive Flow, VWAP deviation
- RSI, realised volatility
- Gemini sentiment score
- GRI composite

### Busseti Risk-Constrained Kelly

```
f* = argmax E[log(1 + f·r)]
     subject to P(Wealth < 95%) ≤ 5%
```

Solved via bisection on the empirical 1-min return distribution stored in Redis (rolling 200 obs per symbol). The bot uses **half-Kelly** by default for additional conservatism.

### Decay Conditions (size → 0)

- Realised volatility `σ_t > 2.0 × σ_baseline`
- GRI composite > 0.65
- Signal confidence below threshold

### Square-Off Pricing

To avoid Zerodha's "Market order cannot be placed with protection amount 0" error, square-off orders use LIMIT type with a 2% buffer from LTP:

```
BUY  square-off price  = round(LTP × 1.02, 1)
SELL square-off price  = round(LTP × 0.98, 1)
```

---

## 9. Infrastructure & Data Schema

### Redis Key Schema

| Key Pattern | Type | TTL | Description |
|---|---|---|---|
| `lob:ticks` | Stream | — | Raw tick payloads from KiteTicker |
| `candles:1m:<SYM>:<TS>` | Hash | 2h | OHLCV + VWAP + MLOFI aggregates |
| `ofi:<SYMBOL>` | String | 60s | Latest OFI float |
| `ml:features:<SYM>` | List | — | Rolling 200-obs labelled feature window |
| `universe:active` | Set | — | Currently traded symbols |
| `universe:scores` | Hash | — | Per-symbol composite scores |

### File Layout

```
/opt/sentistack/trading_bot/
├── main.py                  # entry point
├── config.py
├── strategy.py
├── execution.py
├── telegram_controller.py
├── telegram_log_handler.py
├── logbook.py
├── data_ingestion.py
├── ml_signal.py
├── universe.py
├── alternative_data.py
├── geopolitical.py
├── agent_pipeline.py
├── rate_limiter.py
├── refresh_token.py
├── requirements.txt
├── .env                     # secrets — never commit
├── .kite_token              # daily token cache (auto-managed)
├── venv/
├── logs/
│   └── bot_live.log         # rotating 10 MB log
└── deploy/
    ├── sentistack.service   # systemd unit
    └── setup-gcp.sh         # one-command GCP VM setup
```

### Systemd Service

```ini
[Unit]
Description=SentiStack V2 Trading Bot
After=network-online.target redis-server.service
Requires=redis-server.service

[Service]
Type=simple
User=arinmay_work
WorkingDirectory=/opt/sentistack/trading_bot
EnvironmentFile=/opt/sentistack/trading_bot/.env
ExecStart=/opt/sentistack/trading_bot/venv/bin/python main.py
Restart=on-failure
RestartSec=30s
```

Useful commands:
```bash
sudo systemctl start|stop|restart sentistack
sudo journalctl -u sentistack -f        # live logs
sudo journalctl -u sentistack --since "1 hour ago"
```

---

## 10. GCP Deployment

### Free Tier Constraints (stay within these — zero cost)

| Resource | Free limit | Config used |
|---|---|---|
| VM | 1× e2-micro | e2-micro |
| Region | us-central1, us-east1, us-west1 | us-central1-a |
| Disk | 30 GB pd-standard | 30 GB pd-standard |
| Egress | 1 GB/month to non-Google | ~200 MB typical |
| Static IP | Free while VM running | 1 static IP |

**Critical:** Never stop the VM — a stopped VM's static IP is billed (~$7/month). Restart instead of stop if needed.

### VM Setup Steps

```bash
# 1. Create VM
gcloud compute instances create sentistack \
  --machine-type=e2-micro \
  --zone=us-central1-a \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard \
  --tags=sentistack

# 2. Reserve and attach static IP
gcloud compute addresses create sentistack-ip --region=us-central1
gcloud compute instances delete-access-config sentistack \
  --access-config-name="External NAT" --zone=us-central1-a
gcloud compute instances add-access-config sentistack \
  --zone=us-central1-a \
  --address=$(gcloud compute addresses describe sentistack-ip \
               --region=us-central1 --format='get(address)')

# 3. SSH in
gcloud compute ssh sentistack --zone=us-central1-a

# 4. Run setup script (installs deps, Redis, clones repo, installs systemd service)
bash deploy/setup-gcp.sh
```

### Swap (critical for e2-micro)

e2-micro has only 1 GB RAM. The setup script creates a 2 GB swapfile:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Zerodha IP Whitelist

The bot's static IP must be whitelisted in your Kite Connect app:
1. Go to https://developers.kite.trade → your app → Edit
2. Add `<YOUR_STATIC_IP>` to the redirect/postback URL whitelist
3. The VM's IP is stable as long as the VM keeps running

---

## 11. Known Issues & Fixes

### "Market order cannot be placed with protection amount 0"

Zerodha NSE requires a reference price for market order protection. **Fixed** in `execution.py`: square-off now uses `ORDER_TYPE_LIMIT` with ±2% LTP buffer instead of `ORDER_TYPE_MARKET`.

### Gemini 2.5 Pro quota exhausted (429)

The free tier allows ~50 requests/day on 2.5 Pro. **Fixed** via `ModelRotator` in `agent_pipeline.py`: automatically rotates to Gemini 2.5 Flash on 429, then Flash Lite, with 24h cooldowns per model.

### FRED GPR series 404

FRED renamed the Geopolitical Risk indices. The bot falls back to a neutral GRI baseline (composite = 0.5) when FRED is unavailable. No impact on trading logic.

### Bot crashes on headless server startup

The old interactive OAuth flow called `input()` and `webbrowser.open()`, both failing under systemd. **Fixed**: `TokenManager` detects `not sys.stdin.isatty()` and switches to `_telegram_oauth_flow()`, which polls Telegram for the `/token` command instead.

### StartLimitIntervalSec warning in journalctl

This systemd key belongs in `[Unit]` not `[Service]`. Cosmetic only — does not affect operation. Will be corrected in a future service file update.

---

## 12. Best Practices & Risk Warnings

1. **Paper trade first** — run `PAPER_MONITOR` mode for at least one full trading week before enabling live orders
2. **Capital limits** — never allocate more than you can afford to lose entirely; start with ₹50,000–₹100,000
3. **Rate limits** — do not run multiple bot instances on the same Kite API key simultaneously
4. **Token security** — `.env` and `.kite_token` are chmod 600; never commit them to git
5. **CNC positions** — CNC trades carry overnight risk; monitor them via `/status` and use `/nosell` if you need to hold
6. **Holiday handling** — the bot uses `is_market_open()` (Mon–Fri, 9:15–15:30 IST) but does not check NSE holiday calendars; it will run idle on exchange holidays
7. **Swap exhaustion** — on e2-micro, monitor memory with `free -h`; if swap is consistently full, the bot will slow down
8. **GitHub secrets** — never push `.env` or any file containing API keys; `.gitignore` covers `.env`, `.kite_token`, `.env.sh`

> **Disclaimer:** Trading involves significant financial risk. This software is provided for educational and research purposes. The authors accept no responsibility for trading losses. Always comply with SEBI regulations and Zerodha's terms of service.
