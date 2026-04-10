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
- **ML Ensemble** — XGBoost + Ridge meta-learner predicts 1-min forward returns, retrains every 30 min (5,000-sample rolling window, 7-day Redis TTL)
- **Gemini Sentiment** — LLM cognitive sentiment from live news headlines (auto-rotates models on quota limits)
- **Geopolitical Risk** — composite GRI from VIX, USD/INR, and keyword conflict signals
- **CNC / MIS Hybrid** — high-confidence signals with early entry become CNC (delivery/swing); rest are MIS (intraday)
- **Parametric Risk-Constrained Kelly** — closed-form `f* = μ/σ²` with Busseti VaR cap `(1−W_floor)/(zσ−μ)`; vol-scaled stops
- **Vol-Scaled Stops + Time Stop** — HSL/TSL_activation/TSL_callback = `max(config_floor, k·σ_bar)`; 25-min kill for dead trades
- **Portfolio Loss Budgets (R-13)** — daily −2% / weekly −5% / monthly −8% auto-halt + per-symbol consecutive-loss blacklist
- **Broker-Balance Auto-Sync (R-14)** — polls `kite.margins()` and resizes active capital to match real Zerodha equity (live mode only)
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
| `portfolio_risk.py` | D/W/M loss budgets, blacklist, halts | `PortfolioRiskMonitor`, `RiskBudget` |
| `position_manager.py` | Exit engine — HSL/TSL/time stops | `PositionManager`, `PositionState` |
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

### Busseti Risk-Constrained Kelly (parametric form)
```
σ_bar = σ_annual / √(252 × 375)          # per-minute vol
μ     = |ml_signal| / SIGNAL_SCALE       # per-bar expected edge

f_kelly  = μ / σ_bar²                    # raw Kelly
f_varcap = (1 − W_floor) / (z · σ_bar − μ)   # Busseti VaR cap (95% CI)

f_final  = KELLY_FRACTION × min(f_kelly, f_varcap, 2·MAX_POSITION_FRACTION)
```
The VaR cap enforces `P(Wealth < W_floor) ≤ 5%` with `W_floor = 0.95`, `z = 1.645`.
An empirical bisection sizer runs as a secondary safety cap. The hard
`MAX_POSITION_FRACTION` (default 5%) is the final ceiling.

### Vol-Scaled Stops
```
HSL  = max(HARD_STOP_LOSS_PCT,   3.0 · σ_bar)
TSL_act = max(TSL_ACTIVATION_PCT, 2.5 · σ_bar)
TSL_cb  = max(TSL_CALLBACK_PCT,   1.5 · σ_bar)
```
Stops are floored at config values — never looser than the user-chosen
floor, but wider for volatile names so the bot isn't shaken out of
trades by normal noise. A 25-minute time stop kills trades that drift
without making progress.

### Portfolio Loss Budgets (R-13)
```
day_limit_inr   = DAILY_LOSS_LIMIT_PCT   · active_capital   # default 2%
week_limit_inr  = WEEKLY_LOSS_LIMIT_PCT  · active_capital   # default 5%
month_limit_inr = MONTHLY_LOSS_LIMIT_PCT · active_capital   # default 8%
```
FIFO-matched realized P&L (from `logbook.py` daily CSVs) is compared to
these rupee budgets every `RISK_CHECK_INTERVAL_S` (default 60s).
Escalation is one-way inside each window: DAY → WEEK → MONTH.
DAY/WEEK halts auto-lift when the calendar window closes; MONTH halt
forces `PAPER_MONITOR` mode + manual `/resume`.

### CNC / MIS Classification
A signal is classified as **CNC** (delivery) when all conditions hold:
- Signal is not decayed
- Direction is not FLAT
- `confidence ≥ CNC_MIN_CONFIDENCE` (default 0.72)
- `|ml_signal| ≥ CNC_ALPHA_THRESHOLD` (default 0.55)
- Entry time is before `CNC_ENTRY_CUTOFF_HOUR` IST (default 13:00)

All other signals are **MIS** (intraday, squared off by 3:20 PM IST).

---

## Capital Management

The bot has a single source of truth for active trading capital: `BotState.paper_capital`.
The name is historical — it holds the active capital in both paper and live mode.
Every change (manual via `/capital`, automatic via broker sync, or seeded from
`TOTAL_CAPITAL` at startup) fans out to:

- `RiskManager._capital` → used by the parametric Kelly sizer in `strategy.py`
- `PortfolioRiskMonitor._capital` → re-anchors the daily/weekly/monthly loss budgets
- `settings.strategy.TOTAL_CAPITAL` → updated where possible so status banners reflect truth

### Manual capital control
```
/capital              → show active capital + current loss budgets
/capital 700000       → set active capital to ₹7,00,000
```
Position sizing and D/W/M loss budgets rescale on the next strategy cycle.
Already-open positions keep their original quantities — only new entries are affected.

### Automatic capital sync from Zerodha (R-14)
In **live mode only**, set `AUTO_SYNC_CAPITAL=true` in `.env` to have the bot
poll `kite.margins()` every 5 minutes and resize the active capital to match
the real broker balance.

```
active_capital = min(equity.net × SAFETY_BUFFER, AUTO_SYNC_MAX_CAPITAL)
```

Defaults:
- `AUTO_SYNC_INTERVAL_S     = 300`    (poll every 5 min)
- `AUTO_SYNC_SAFETY_BUFFER  = 0.90`   (use 90% of broker balance; 10% headroom for slippage/costs)
- `AUTO_SYNC_MIN_DELTA_PCT  = 0.01`   (only resize if delta ≥ 1% — prevents churn)
- `AUTO_SYNC_MAX_CAPITAL    = 1e7`    (hard ceiling at ₹1 crore)

On every successful sync the bot posts a Telegram notification showing the
raw broker balance, safety buffer, and the new active capital. The
`PortfolioRiskMonitor` then re-anchors the D/W/M loss budgets on the next
cycle, so your risk limits always move with your actual account size.

**Manual controls:**
- `/balance` — show the raw Zerodha equity balance without changing anything
- `/synccapital` — force an immediate resync (useful after adding/withdrawing funds)

**Paper mode:** Auto-sync is deliberately disabled. Paper capital is whatever
you set via `/capital` or `TOTAL_CAPITAL` — there is no real broker to poll.

### Bootstrap mode for small accounts (Task-1)

The default risk envelope — `MAX_POSITION_FRACTION = 5%` and
`MIN_TRADE_VALUE = ₹2000` — is calibrated for accounts of ₹1 lakh or
more. At ₹5-10 k starting capital the R-10 cost filter rejects every
signal because the Kelly-sized quantity cannot reach the ₹2000
notional floor that keeps brokerage under 2% of expected P&L.

**Bootstrap mode** flips both knobs while the active capital is below a
threshold so the small-capital validation phase can actually trade:

| Knob | Normal | Bootstrap |
|---|---|---|
| Max position fraction | 5% | 35% |
| Min trade value | ₹2000 | ₹500 |
| Capital threshold | — | ₹50,000 (configurable) |

The switch is **automatic**. Every sizing decision reads the active
capital (`BotState.paper_capital`) and picks the right envelope via
`get_effective_position_fraction()` / `get_effective_min_trade_value()`
in `config.py`. The moment the account grows past
`BOOTSTRAP_CAPITAL_THRESHOLD` the bot reverts to normal sizing on the
next trade — no restart, no command.

**Controls:**
- `.env`: `BOOTSTRAP_MODE=true` (default), `BOOTSTRAP_CAPITAL_THRESHOLD=50000`
- Telegram `/status` shows a `🚀 Bootstrap:` line indicating ON/OFF
  plus the active fraction and min trade value.

**⚠️ Warning.** Bootstrap sizing is intentionally aggressive. A 35% per-position
cap means two bad trades can consume most of a ₹5k account. Treat
bootstrap mode as validation only — it exists so you can confirm
signals are firing and orders are placing correctly, not as a
steady-state operating envelope. The right answer to "I want more
positions and smaller drawdowns" is to raise capital past the
threshold, not to raise bootstrap sizing itself.

### Paper ↔ Live toggle (R-15)

The `PAPER_TRADE` env var sets the mode at startup, but you can flip it
at runtime over Telegram without restarting the bot:

| Command | Effect |
|---|---|
| `/tradingmode` | Show current mode (PAPER or LIVE) |
| `/papermode` | Switch to PAPER immediately — always safe |
| `/livemode` | Arm a 60-second LIVE confirmation window |
| `/livemode CONFIRM` | Confirm the switch to LIVE within the window |

The two-step confirmation for `/livemode` is intentional — switching to
live mode means real money is at risk, and a fat-fingered command is
the last thing you want there. The bot waits for an explicit
`/livemode CONFIRM` within 60 seconds; if you don't send it in time,
the arm expires and nothing changes.

Already-open positions are **never** touched by a mode switch — only
future entries are affected. If you switch from LIVE back to PAPER
while holding real Zerodha positions, those positions remain real; the
bot just won't place any new real orders until you run `/livemode`
again.

The override is in-memory only. A restart reverts to whatever
`PAPER_TRADE` is in your `.env`. This is deliberate: if the bot crashes
while in LIVE, you don't want it silently coming back up live when
you expected paper.

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
| `TOTAL_CAPITAL` | — | INR, default `500000` (seed value; may be overridden at runtime) |
| `PAPER_TRADE` | — | `true` for paper mode, `false` for live |
| `MIN_TRADE_VALUE` | — | Default `2000` INR |
| `AUTO_SYNC_CAPITAL` | — | `true` to auto-sync capital from Zerodha balance every 5 min (live mode only) |
| `AUTO_SYNC_MAX_CAPITAL` | — | Hard ceiling on auto-synced capital (INR). Default `10000000` (₹1 Cr) |
| `BOOTSTRAP_MODE` | — | `true` to auto-apply small-capital sizing when active capital is below threshold. Default `true` |
| `BOOTSTRAP_CAPITAL_THRESHOLD` | — | INR cap below which bootstrap sizing activates. Default `50000` |

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
| `/resume` | Resume order execution (also clears any risk halt) |
| `/stop` | Graceful shutdown |
| `/nobuy` | Block all new BUY entries |
| `/okbuy` | Re-enable BUY entries |
| `/nosell` | Block all new SELL/short entries |
| `/oksell` | Re-enable SELL entries |
| `/mode` | Switch analysis mode (FULL / GRI_ONLY / PAPER_MONITOR) |
| `/tradingmode` | Show current PAPER vs LIVE state |
| `/papermode` | Switch to PAPER (simulated fills, always safe) |
| `/livemode` | Arm LIVE mode (requires `/livemode CONFIRM` within 60s) |

### Monitoring
| Command | Description |
|---|---|
| `/status` | Full pipeline snapshot (GRI, VIX, sentiment, active guards) |
| `/tasks` | Background task heartbeat status (🟢 < 2 min / 🟡 < 10 min / 🔴 stale) |
| `/pnl` | Today's P&L statement — MIS + CNC sections with net after brokerage & STT |
| `/risk` | Portfolio loss-budget snapshot (D/W/M + blacklist + active halts) |

### Capital Management
| Command | Description |
|---|---|
| `/capital` | Show current active capital + current D/W/M loss budgets |
| `/capital <amount>` | Set active capital manually; rescales position sizing + risk budgets |
| `/balance` | Show Zerodha equity balance (cash, utilised, net) |
| `/synccapital` | Force immediate capital resync from Zerodha balance |

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
