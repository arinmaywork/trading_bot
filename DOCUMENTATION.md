# SentiStack V2 — Full System Documentation

SentiStack is a production-grade, asynchronous high-frequency trading system for the National Stock Exchange (NSE). It fuses **Limit Order Book (LOB) microstructure**, **ML ensemble signals**, **LLM cognitive sentiment**, and **risk-constrained Kelly gambling** into a unified execution engine.

---

## 1. Quick Start Guide

### Prerequisites
*   **Python 3.10+**
*   **Redis 7+**: Essential for the tick stream, candle aggregation, and ML feature store.
*   **Zerodha Kite Connect API**: Live credentials with WebSocket access.
*   **Google Gemini API Key**: For sentiment analysis (Gemini 1.5 Flash recommended).

### Setup & Launch
1.  **Configure Environment**: Edit `.env.sh` or export the variables:
    ```bash
    export KITE_API_KEY="your_api_key"
    export KITE_API_SECRET="your_api_secret"
    export GEMINI_API_KEY="your_gemini_key"
    export TELEGRAM_BOT_TOKEN="your_bot_token"
    export TELEGRAM_CHAT_ID="your_chat_id"
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Bot**:
    ```bash
    python main.py
    ```
4.  **Authenticate**: The bot will automatically open your browser for Zerodha login. Paste the `request_token` from the redirect URL back into the terminal. Access tokens expire daily at 06:00 IST.

---

## 2. System Architecture

The bot operates on a **Research -> Signal -> Execution** loop within an `asyncio` event loop.

1.  **Data Ingestion (`data_ingestion.py`)**: Streams Level-2 Market Data via `KiteTicker`. Computes real-time MLOFI and Aggressive Flow.
2.  **Cognitive Analysis (`alternative_data.py`, `agent_pipeline.py`)**: Periodically fetches news/weather. Uses **Gemini 1.5 Flash** via LangGraph to generate a global sentiment score.
3.  **Geopolitical Monitoring (`geopolitical.py`)**: Tracks India VIX, USD/INR, and GPR indices to adjust global risk parameters.
4.  **Universe Management (`universe.py`)**: Dynamically selects the top 50 most liquid/volatile stocks at market open and rescores them intraday.
5.  **ML Ensemble (`ml_signal.py`)**: Predicts 1-minute forward log-returns using an XGBoost + Ridge ensemble.
6.  **Strategy & Risk (`strategy.py`)**: Orchestrates signals and applies the **Busseti (2016) Risk-Constrained Kelly** algorithm for position sizing.
7.  **Execution (`execution.py`)**: Handles order slicing and Zerodha API communication.

---

## 3. Module Reference

### Core Modules
| File | Responsibility | Key Classes/Methods |
| :--- | :--- | :--- |
| `main.py` | Orchestration & Lifecycle | `TokenManager`, `strategy_loop`, `main` |
| `data_ingestion.py` | LOB & Tick Processing | `AsyncKiteTickerWrapper`, `calculate_mlofi` |
| `ml_signal.py` | Machine Learning | `EnsembleSignalEngine`, `FeatureVector`, `FeatureStore` |
| `strategy.py` | Signal Fusion & Risk | `StrategyEngine`, `RiskManager`, `busseti_kelly_bisection` |
| `universe.py` | Stock Selection | `UniverseEngine`, `FrequencyOptimiser` |
| `alternative_data.py` | External Feeds | `AlternativeDataPipeline`, `GeminiSentimentAnalyser` |
| `execution.py` | Order Management | `OrderExecutor`, `OrderSlicer` |
| `geopolitical.py` | Macro Risk | `GeopoliticalRiskMonitor`, `GPRSnapshot` |

### Utility Modules
*   `config.py`: Centralised settings using Python dataclasses.
*   `rate_limiter.py`: Token-bucket implementation for Zerodha API compliance (10 req/s, 400 orders/min).
*   `telegram_controller.py`: Interactive bot interface.
*   `logbook.py`: Daily performance and signal logging.

---

## 4. Operational Choices & Interactive Controls

### Trading Modes
Upon startup, the bot presents a **Mode Selector** via Telegram:
*   **`PAPER_MONITOR`**: Read-only mode. Logs signals and logic but places no orders.
*   **`GRI_ONLY`**: Ignores news/sentiment; relies strictly on Geopolitical Risk and Microstructure.
*   **`LIVE_TRADING`**: Full execution of MIS (Intraday) orders on the NSE.

### Telegram Commands
*   `/status`: Returns a live health report (GRI, VIX, Top Alpha symbols, and Gemini status).
*   `/pause` / `/resume`: Instantly halts or starts the strategy evaluation loop.
*   `/mode`: Switch trading modes on-the-fly without restarting the bot.

---

## 5. Mathematical Foundations

### Multi-Level OFI (MLOFI)
Aggregates volume imbalance across 5 LOB levels with decaying weights:
`MLOFI = Σ_{i=0}^{4} w_i × (V_bid_i − V_ask_i) / (V_bid_i + V_ask_i)`
Weights: `[0.40, 0.25, 0.18, 0.10, 0.07]`

### Busseti Risk-Constrained Kelly
Maximises expected log-wealth subject to a ruin constraint:
`f* = argmax E[log(1 + f·r)]`
Constraint: `P(Wealth < 95%) ≤ 5%`
Solved via bisection on the empirical return distribution stored in Redis.

### SentiStack Alpha
`Alpha_t = ML_Signal × Geopolitical_Multiplier`
The `ML_Signal` is the meta-learner output (scaled to [-1, 1]) predicted from microstructure, sentiment, and volatility features.

---

## 6. Infrastructure & Data Schema

### Redis Key Schema
| Key Pattern | Type | Description |
| :--- | :--- | :--- |
| `lob:ticks` | Stream | Raw tick payloads and microstructure metrics. |
| `candles:1m:<SYM>:<TS>` | Hash | OHLCV + VWAP + MLOFI aggregates. |
| `ml:features:<SYM>` | List | Rolling window of 200 labelled feature vectors. |
| `universe:active` | Set | Currently traded symbols. |

### ML Training Pipeline
The `EnsembleSignalEngine` runs a background task (`run_retrain_loop`) that retrains the models every 30 minutes.
*   **Minimum Data**: Requires 50 labelled observations to activate.
*   **Features**: MLOFI, Aggressive Flow, VWAP Dev, Sentiment, RSI, Vol, GRI.
*   **Target**: 1-minute forward log-return.

---

## 7. Best Practices & Risk Warnings

1.  **Training Latency**: The ML model starts in "Fallback Mode" (V1 static logic) until it collects enough data.
2.  **Slippage**: Market orders are used for immediate execution. Monitor slippage reports in Telegram during high volatility.
3.  **Rate Limits**: Zerodha has strict rate limits. The built-in `RateLimiter` ensures compliance, but avoid running multiple instances on the same API key.
4.  **Capital Risk**: This is a high-frequency system. Always test new strategies in `PAPER_MONITOR` mode for at least one full trading week.

> **Disclaimer**: Trading involves significant risk. This software is for educational purposes. The authors are not responsible for financial losses.
