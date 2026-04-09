# SentiStack V2 — Deep Analysis & 5-Year Retirement Plan

**Author:** Independent code review
**Date:** April 9, 2026
**Subject:** Making SentiStack your retirement backup income stream by age 35
**Starting capital (config default):** ₹5,00,000
**Horizon:** 5 years (age 30 → 35)

---

## 0. The Honest Math First

Before any code changes, understand what you're asking the machine to do. This is the single most important section.

**Your goal, decoded:** By 35, have the bot produce enough monthly income that a single market crash, job loss, or life event cannot hurt you. A realistic "backup" number in India is ₹75,000–₹1,50,000/month of safe withdrawable income, which translates to a corpus of roughly ₹1.8–3.6 crore if you draw 5%/year.

**What that implies in CAGR from ₹5L:**

| Target corpus in 5 yrs | Required CAGR |
|---|---|
| ₹50 lakh | 58% |
| ₹1 crore | 82% |
| ₹2 crore | 110% |
| ₹5 crore | 158% |

**Reality check.** Well-tuned intraday systematic systems in Indian equities realistically deliver **15–35% net CAGR** after all costs in normal years, with some great years up to 60% and some flat/negative years. Sustained 80%+ CAGR for five years on a single retail system with retail infra is not something you should plan your retirement on. It has been done — by people with colocated hardware, tick-level feeds, multi-billion book, and teams of PhDs.

**The only honest path to your goal:**

1. **Don't try to 100x ₹5L in 5 years with one bot.** That's how people blow up accounts.
2. **Add fresh capital every month from your day job.** ₹50,000/month @ 25% CAGR compounded + ₹5L seed → ~₹80 lakh in 5 years. ₹1 lakh/month → ~₹1.3 crore. That's the realistic glidepath.
3. **Run multiple uncorrelated strategies.** Intraday microstructure (this bot) + swing momentum + index-futures trend-following + a defensive bucket in index funds. Each one smooths the others.
4. **Target Sharpe, not return.** A Sharpe of 1.5+ with max DD < 15% beats a Sharpe-0.5 system claiming 60% that blows up in year 3. The goal is *robust*, not spectacular.
5. **Accept that a backup income from trading is a 7–10 year project, not 5.** Tell yourself this now so you don't over-leverage when a drawdown hits.

Everything that follows is structured around this reality.

---

## 1. What You've Built — The Good

Credit where it's due. The architecture is above average for a solo retail project:

- **Clean async event loop in `main.py`** with explicit timezone handling, graceful shutdown, task heartbeats, and headless Telegram OAuth. Most bots die on systemd because of `input()`; yours handles it.
- **Separation of concerns**: ingestion, ML, strategy, risk, execution, and controller are all in their own files with typed dataclasses. This is maintainable.
- **MLOFI (Multi-Level Order Flow Imbalance)** over 5 LOB levels with geometric weighting is a real microstructure signal, not a toy. This is the single most valuable piece of alpha you have right now.
- **Busseti risk-constrained Kelly** — correctly references the 2016 Busseti/Ryu/Boyd paper, bisection is implemented correctly, and you fall back to half-Kelly when data is thin. Most retail bots just use fixed-fractional; this is better.
- **CNC/MIS hybrid classifier** — recognising that a strong high-conviction signal early in the day shouldn't be force-closed at 3:20 PM is a genuinely clever product choice and saves real money on STT.
- **Transaction-cost-aware sizing** in `RiskManager.compute_quantity` — you skip trades where expected P&L < 2× round-trip cost. This is the correct mental model and many retail bots skip it entirely.
- **Trade guards + Telegram remote control** (`/nobuy`, `/nosell`, `/pause`) — you can intervene manually without SSHing. Important when you're at your day job.
- **IPv4 preference hack in `main.py`** — I actually laughed. That's the kind of fix you only write after something bit you badly in production. Good scar tissue.

Keep all of the above. Nothing here should be rewritten.

---

## 2. Critical Bugs & Correctness Issues

These are ordered by how much money they are likely costing you. Fix them in order.

### 2.1 Busseti Kelly is mathematically wrong in its current form

In `strategy.py::RiskManager.compute_quantity` you do:

```python
directional_returns = [r * math.copysign(1.0, ml_signal) for r in returns]
f_busseti = busseti_kelly_bisection(returns=directional_returns, ...)
```

**The problem.** `returns` is the historical forward log-return distribution for this symbol from the FeatureStore. Those returns already have their *natural* sign — they are what the stock actually did. Flipping them by the current `ml_signal` sign breaks the whole premise of Busseti/Kelly, which requires the distribution of the **returns you will earn** given your chosen position.

What you want is the *conditional* distribution of forward returns **given a signal of this sign**. Two correct ways to get it:

1. **Conditional empirical distribution.** In `ml_signal.py::label_and_store`, store both the feature vector and its forward return. Then at sizing time, bucket historical observations where the predicted ML signal had the same sign as now, and use *those* forward returns. That gives `E[r | signal > 0]` which is what Kelly needs.
2. **Parametric shortcut.** For a short intraday horizon, assume `r_t ~ N(μ_t, σ_t²)` where `μ_t = ml_signal × expected_edge` and `σ_t` is the rolling realised vol. Then Kelly has a closed form `f* = μ/σ²` and Busseti's ε-constraint becomes a simple VaR bound you can solve analytically.

Today the bot is effectively sizing on mirror-image returns whenever `ml_signal` is negative, which makes short trades systematically mis-sized compared to long trades.

**Fix priority: 🔴 critical.**

### 2.2 Training window of 200 bars is guaranteed overfitting

`FeatureStore.MAX_WINDOW = 200` and `MIN_TRAIN = 50`. You retrain XGBoost (80 estimators, depth 4, ~400+ internal splits) every 30 minutes on 50–200 samples of an 11-dimensional feature vector. That is overfitting by definition. An XGBoost model with that capacity needs at least several thousand labelled samples to have stable out-of-sample error, and ideally tens of thousands.

**What's happening in practice:** each retraining pass produces a model that memorises the last 200 minutes of noise. The "signal" it produces 30 minutes later is close to random, then you rebuild it. The feature importances shift wildly across retrainings — go look at your logs.

**Fix:**

- Raise `MAX_WINDOW` to **at least 5,000** per symbol (roughly 12–13 trading days of 1-min bars). 10,000 is better.
- Stop per-symbol training. Pool data across all ~50 symbols into one dataset and train **one global model** with symbol-id one-hot or learned embedding as a feature. You then get 250,000+ training rows instantly.
- Add a walk-forward purged cross-validation loop (López de Prado's *Advances in Financial ML*, Ch. 7) so you actually measure OOS performance before accepting a new model. Reject retrainings whose OOS R² is worse than the live model.
- Retrain daily at 8:30 AM IST from the previous 60 trading days of logged data — not every 30 minutes. Intraday retraining in a drifting market is theatre.

**Fix priority: 🔴 critical.**

### 2.3 Feature set is too shallow and partially double-counted

Your 11 features (`ml_signal.py::FeatureVector`) are reasonable as a starting point, but:

- **Correlation redundancy.** `mlofi` and `ofi` are almost perfectly correlated (OFI is level-0 MLOFI). XGBoost handles collinearity but Ridge does not and the meta-learner gets confused. Drop `ofi`.
- **Sentiment is stale.** Gemini sentiment updates every few minutes on *market-wide* news. This is useful macro context, not a micro-signal. Right now it enters the ensemble at the same weight as MLOFI, which moves tick-by-tick. You need either (a) per-symbol news sentiment (expensive) or (b) demote sentiment to a *regime feature* (a slow-moving input).
- **RSI is used twice.** It's a feature AND a hard veto in `strategy.py` (the `rsi_neutral` block). That's double-counting and fights the model.
- **Missing features that pay.** You're missing the features that actually predict 1-min forward returns in NSE cash equities:
  - **Trade-sign imbalance** (tick-rule signed flow over last 10/30/60 seconds)
  - **Realised-to-implied vol ratio** (you have IV via VIX; compute per-stock realised/implied)
  - **Index relative strength** (stock return − Nifty return over last N minutes)
  - **Sector-neutral residual return** (the orthogonalised return is what has alpha)
  - **Order-book slope** (price-impact for marketable 10-lakh order)
  - **Quote lifetime / cancellation ratio**
  - **Time-of-day one-hot** (first-30-min and last-30-min regimes are structurally different)
  - **Previous day high/low proximity** (levels matter)
  - **Opening range breakout flags** (first 15-min range crossed?)
- **No cross-sectional ranking.** Institutional stat-arb works by *ranking* stocks against each other each bar and going long-top-decile, short-bottom-decile, beta-neutral. You trade symbols in isolation, which throws away the strongest signal in equities. See Section 4.1.

**Fix priority: 🔴 critical.**

### 2.4 Stop-loss and TSL are too tight for 1-min bars

From `config.py`:

```python
TSL_ACTIVATION_PCT = 0.006   # activate at +0.6%
TSL_CALLBACK_PCT   = 0.003   # trail 0.3%
HARD_STOP_LOSS_PCT = 0.012   # -1.2% hard stop
```

A 20%-annualised-vol NSE stock has a 1-minute standard deviation of roughly `0.20 / sqrt(252*375) ≈ 0.065%`. A 0.3% trailing callback is ~4.6σ of 1-min noise and a 1.2% hard stop is ~18σ — but that's deceptive, because vol clusters. In practice:

- Most TSLs trigger within 2–5 minutes of activation purely from normal mean reversion. You're crystallising tiny wins and missing the real moves.
- The hard stop does nothing in calm markets, then all fire simultaneously when vol spikes, because `HARD_STOP_LOSS_PCT` is a *fixed* percent rather than a multiple of current vol.

**Fix:** make both TSL and HSL **volatility-scaled**:

```python
# Pseudo
hard_sl_pct     = max(0.004, 3.0 * sigma_1m_realised)
tsl_callback    = max(0.002, 1.5 * sigma_1m_realised)
tsl_activation  = max(0.004, 2.5 * sigma_1m_realised)
```

And add a **time-stop**: if the trade hasn't moved ±0.5σ in its favour within N minutes, flatten it. Dead trades are the most expensive trades after losers.

**Fix priority: 🟠 high.**

### 2.5 `MIN_ALPHA_THRESHOLD = 0.005` is too high for intraday signals

You require `|ml_signal| ≥ 0.5%` to act. Since `ml_signal` is scaled to the 1-min log return, you are asking the model to predict a 0.5% move in the next minute, which only happens during a real breakout (<5% of bars). You will be FLAT almost all of the time, which is exactly what your logs will show.

**Fix:** use a **percentile threshold** instead of a fixed one. Every retrain, compute the 85th percentile of `|ml_signal|` on the training set and use that as the live threshold. This auto-adapts to whatever scale the model is producing and guarantees you trade the top 15% of conviction bars regardless of regime.

### 2.6 Meta-learner trains on in-sample predictions → leakage

`MetaLearner` is trained on `[xgb_pred, ridge_pred]` where both predictions come from base learners trained on the same data. This is textbook stacking leakage — the meta-learner is learning to weight overfitted in-sample predictions, not genuine predictive power.

**Fix:** use **out-of-fold stacking**. Split the training window into K=5 folds, for each fold train the base learners on the other 4 folds and predict on the held-out fold. Stack those out-of-fold predictions, train the meta-learner on them. This is one of the core lessons of López de Prado Ch. 6.

### 2.7 Square-off uses ±2% LIMIT buffer — that's *huge*

From the docs: square-off sends LIMIT orders at `LTP × 1.02 / 0.98`. On a calm day, that will fill at LTP. In a gap or a fast market, you'll pay 2% just to exit. A 2% exit slippage on a trade hoping for 0.6% profit is catastrophic.

**Fix:**

- Default to IOC limit at `LTP ± 0.15%`. Retry 3 times with widening buffers `[0.15%, 0.4%, 0.8%]` before escalating.
- Only escalate to a 2%+ buffer if the previous IOCs failed AND market is open AND time > 3:18 PM.
- For CNC positions being flattened for risk, use VWAP slicing over 5 minutes, not a single wide limit.

### 2.8 No walk-forward backtest, no paper metric persistence

You have no backtesting code. The only live "validation" is running `PAPER_MONITOR` for a week and eyeballing `/pnl`. That is not a validation regime, that is wishful thinking. You need:

- A backtester that replays historical Zerodha tick data (or at least 1-min OHLCV) through the same `strategy.evaluate()` pipeline, pre-computing MLOFI from reconstructed depth where possible or using reasonable proxies (signed volume delta) where not.
- Walk-forward splits: train on months 1–3, test on month 4, roll forward.
- Performance metrics logged per walk-forward: Sharpe, Sortino, Calmar, max DD, hit rate, avg win/loss, turnover, cost-drag, per-hour P&L, per-symbol P&L, correlation to Nifty.
- Only promote a model to live if its *out-of-sample* Sharpe on the most recent walk-forward fold is > 1.0 and max DD < 8%.

**Fix priority: 🔴 critical.** No amount of live tweaking substitutes for a backtest.

---

## 3. Risk Management Gaps

The money doesn't die from bad signals. It dies from bad risk management during the months the signals misfire. These fixes are more important than any alpha upgrade.

### 3.1 No portfolio-level risk budget

Each trade is sized independently via `MAX_POSITION_FRACTION = 0.05` (5% of capital per name). If all 50 universe stocks hit your signal threshold at the same time (a Nifty-crash morning, for example), the bot can happily attempt **50 × 5% = 250% gross exposure**. MIS leverage from Zerodha lets you do this and blow up in one session.

**Add these portfolio guards in `strategy.py` or a new `portfolio_risk.py`:**

| Guard | Recommended value |
|---|---|
| Max gross exposure | 100% of capital (no leverage via concurrent MIS) |
| Max net long | 60% |
| Max net short | 60% |
| Max names concurrent | 8 |
| Max sector exposure | 25% of capital |
| Max single-name exposure | 10% |
| Daily loss limit (hard stop trading) | 2% of capital |
| Weekly loss limit (pause until Monday) | 5% of capital |
| Monthly drawdown limit (switch to paper) | 8% of capital |
| Per-symbol consecutive loss count (blacklist for day) | 3 |

The "monthly DD → auto-switch to PAPER_MONITOR" is the single most important of these. It takes the human out of the worst moment.

### 3.2 No correlation awareness

Two BANK stocks aren't two independent bets, they're one and a half bets. Your Kelly sizing treats them as two. On a bank-specific news day, you'll have five correlated longs that all move together.

**Add:** rolling 60-day correlation matrix, cluster symbols into correlation buckets via hierarchical clustering, and enforce a per-cluster exposure cap. This is the step that turns "retail bot" into "portfolio manager".

### 3.3 No crash detector

`GEOPOLITICAL_RISK_THRESHOLD = 0.65` is a blunt instrument. What actually kills intraday bots is sudden regime change: VIX jumping 20% in 5 minutes, INDIA VIX futures going off the rails, Nifty gapping >1% against your positions. Detect these explicitly:

- VIX 5-min change > 10% → halt new entries, flatten 50% of book
- Nifty 5-min return < -0.8% → flatten all longs, keep shorts
- Sudden spread widening (median spread > 3× trailing 20-min median) → halt entire symbol

### 3.4 You have no kill switch for model drift

If the live model's hit rate over the last 50 trades drops below 45%, automatically disable live trading and page you on Telegram. Today the bot will happily keep firing bad signals until you notice.

### 3.5 Single point of failure: one VM, one broker, one API key

For a system that's supposed to pay your retirement bills:

- Run a hot-standby VM in a second GCP zone that can take over in 60 seconds if the primary dies (health check via Redis + heartbeat).
- Keep a read-only "watchdog" process on a third location (your laptop, a Raspberry Pi) that polls positions via Kite REST every minute and alerts if it diverges from the primary bot's expected state. If the primary crashes mid-trade and leaves an open position, the watchdog flattens it.
- Keep a manual emergency-flatten script, tested monthly, that uses just `kiteconnect` and cancels all orders + squares off all positions. Don't trust complex code in an emergency.

### 3.6 Zerodha-specific: rate limit + freeze quantity safety

`ORDER_FREEZE_LIMIT = 5_000` is in config but I don't see it enforced in `execution.py`. A single signal for a high-quantity low-price stock (say 6,000 shares of YESBANK at ₹20) will get rejected by the exchange for exceeding freeze limits. Add an `OrderSlicer` that slices any order > 50% of freeze quantity into smaller children spaced 200ms apart, both for freeze-compliance and to reduce market impact.

---

## 4. Alpha Upgrades (In Order of Expected Information Ratio)

Each of these is a concrete, implementable improvement that adds real edge. Don't try to do all of them at once. Do them one per month, measure, then decide if they keep or go.

### 4.1 Cross-sectional ranking — the single biggest free lunch

Instead of asking *"does symbol X have a buy signal?"*, ask *"which 5 of my 50 universe stocks have the strongest buy signals right now, and which 5 have the strongest sell signals?"* Then go long the top 5, short the bottom 5, sized equally.

This one change converts your bot from directional (exposed to Nifty moves) to **market-neutral** (beta ≈ 0). Market-neutral books have 2–3× the Sharpe of directional books because they don't care whether Nifty is up or down — they just need their longs to beat their shorts by a few bps. This is how every successful equity L/S fund operates.

Concretely:

1. Every bar, compute `ml_signal` for every symbol in the active universe.
2. Z-score the signals cross-sectionally (subtract mean, divide by stdev of that bar's signals).
3. Rank; take top N longs and bottom N shorts.
4. Size so that gross long = gross short = 50% of capital.
5. Hedge residual beta with NIFTYBEES or a tiny index future position.

**Expected impact:** +50–100% Sharpe improvement relative to the current per-symbol mode. This is not optional for a retirement bot.

### 4.2 Opening Range Breakout + end-of-day drift as separate strategies

First 15 minutes and last 30 minutes of the NSE trading day have structurally different statistical properties — they are regimes, not continuations of the middle of the day. Instead of using one model on the whole day:

- **08:45–09:15 IST:** pre-market scan, compute overnight gap, build ORB levels.
- **09:15–09:30 IST:** ORB breakout strategy (separate params, tiny position size, high hit rate).
- **09:30–14:30 IST:** your current microstructure + ML strategy.
- **14:30–15:15 IST:** momentum/drift strategy that trades the closing auction pressure.
- **15:15–15:30 IST:** stand down, flatten.

Each regime deserves its own model because the features have different importance in each.

### 4.3 Add a swing (multi-day) sleeve

You already have CNC hooks. Build a proper swing strategy on top:

- Universe: Nifty 200 only (liquidity).
- Signal: weekly momentum + quality factor (ROE, accruals) + technical confirmation (price above 50DMA and above 200DMA, 52-week high proximity).
- Rebalance: weekly, Monday 9:45 AM.
- Position count: 15–20 names equal-weighted.
- Hold: 5–20 trading days, with a trailing stop at 8% below entry and a hard stop at 12%.

This is uncorrelated with your intraday book, smooths the equity curve massively, and requires far less infra. It's also the easiest thing to *paper trade properly* because the holding period is longer than a lot of bugs.

### 4.4 Add an index-futures trend sleeve

Nifty and BankNifty futures trend well on the daily/weekly scale. A simple Donchian-channel 20/10 trend system on daily bars with 10% of capital allocated to it provides positive-skew tail protection — it makes money exactly in the months when your intraday book is suffering (trending crash or trending rally months).

Capital allocation target across the three sleeves when mature:

| Sleeve | % capital | Expected Sharpe | Role |
|---|---|---|---|
| Intraday microstructure (this bot, improved) | 40% | 1.2–1.8 | Income |
| Swing momentum (Nifty 200) | 40% | 0.9–1.3 | Base return |
| Index futures trend | 20% | 0.6–1.0 | Crisis alpha |

Combined expected portfolio Sharpe: ~2.0 if they are genuinely uncorrelated. That is institutional-grade and is the only realistic path to the income number you want.

### 4.5 Replace Gemini per-news sentiment with Gemini per-stock daily summary

Your current alt-data pipeline burns Gemini tokens on macro headlines. Almost zero of that moves a specific stock's next-minute return. Instead:

- Once per day at 08:30 IST, fetch the last 24h of headlines and filings for each Nifty 200 stock.
- Ask Gemini for a single structured JSON: `{stock, sentiment: -1..1, confidence, key_topics, risk_flags}`.
- Cache for the whole trading day as a **daily regime feature**. Don't re-query intraday.
- This cuts LLM cost by 95%, and the sentiment it produces is actually per-stock and actionable.

### 4.6 Add execution alpha — passive limit orders

Currently all orders are aggressive limits with 0.2% buffer = you are always crossing the spread = paying ~5 bps of spread cost on every trade. That's 10 bps round-trip, eating 30–50% of a typical intraday edge.

Instead, for non-urgent entries: place a *passive* limit inside the spread (bid+1 tick for a buy, ask−1 tick for a sell) with a 5-second timeout. If it fills → you captured the spread instead of paying it. If it doesn't → cancel and resubmit aggressive. For strong signals (top-decile ML signal), skip passive and cross immediately.

Expected saving: 3–5 bps per round trip. On 100 trades/day, that's enough to pay the GCP bill and meaningfully lift net returns.

---

## 5. Infrastructure — You Will Outgrow the e2-micro

`e2-micro` is 1 GB RAM. You already need a 2 GB swapfile. Under load with 50 symbols streaming, Redis, XGBoost retraining, and the dashboard, you are swap-thrashing and adding real latency to every decision. That latency is an invisible cost — you lose a few bps per trade to slow decision-making.

When you're still under ₹10 lakh capital, stay on the free tier. When you move past that:

- **Upgrade to `e2-small` (2 vCPU, 2 GB RAM)** — ~$12/month, stops swap thrashing, halves decision latency.
- **Move Redis to a persistent disk** (not `/tmp`) so a restart doesn't wipe your feature store.
- **Add Prometheus + Grafana** for real metrics (P&L per hour, per symbol, per signal strength bucket, per feature z-score). The `dashboard.py` you have is a nice start but you want time-series you can query months later.
- **Daily S3/GCS backup** of Redis snapshots + trade CSVs + model pickles. Losing your training data is losing months of edge.
- **Run 1000+ Monte Carlo simulations of your equity curve** every weekend. If any MC path produces a 20% drawdown, reduce live sizing until the 99th-percentile MC DD is below your tolerance.

---

## 6. The 5-Year Roadmap

This is the schedule I would follow if this were my money.

### Phase 1 — Months 1–3: Foundation, no live money
- Fix all Section 2 critical bugs.
- Build the walk-forward backtester. Nothing else matters until this exists.
- Backtest the current strategy honestly. Expect mediocre results — that's fine, that's the baseline.
- Pool training data across symbols into one global model.
- Ship cross-sectional ranking (Section 4.1). Backtest again.
- Start paper trading the new version.

**Success criterion to move to Phase 2:** Out-of-sample walk-forward Sharpe > 1.2, max DD < 10%, 6 weeks of paper trading matching backtest within 20%.

### Phase 2 — Months 4–6: Small live money
- Deploy to live with **₹1,00,000** only. Not a penny more. Yes, it will feel silly.
- Run alongside paper trading the full ₹5L. Every week compare live P&L per contract to paper P&L per contract. They must match within 10% or something is wrong with live execution and you go back to paper.
- Ship Section 3 portfolio risk limits.
- Ship volatility-scaled stops.
- Start logging every single decision to a trade journal (Notion, Obsidian, whatever). Entry rationale, exit rationale, what you were feeling. This is as important as the code.

**Success criterion to move to Phase 3:** 3 consecutive months of positive P&L on live ₹1L with Sharpe > 1.0 and zero unplanned outages > 15 minutes.

### Phase 3 — Months 7–12: Scale to full ₹5L + build swing sleeve
- Ramp live capital from ₹1L to ₹5L across 3 months: ₹1L → ₹2L → ₹3.5L → ₹5L. Never more than 50% increase per month.
- Start adding ₹30k–₹50k per month from salary.
- Build the swing sleeve (4.3). Paper trade it for 12 weeks before going live.
- Build the execution alpha layer (4.6). Measure spread capture.
- Build the emergency flatten script and test it monthly on a spare paper account.

**Success criterion to move to Phase 4:** Combined intraday + swing live Sharpe > 1.3, max DD < 12%, total AUM ≥ ₹8L.

### Phase 4 — Year 2: Harden, diversify, automate the boring stuff
- Add the index futures trend sleeve (4.4). This one runs on daily bars, you don't even need to watch it.
- Move to `e2-small`. Add hot standby.
- Monthly portfolio reviews. Drop any sleeve that has been Sharpe < 0.5 for 2 consecutive quarters.
- Automate the daily token refresh fully, stop touching the bot except for monthly reviews.
- Target AUM end of year 2: ₹18–25 lakh (from growth + monthly additions).

### Phase 5 — Years 3–5: Compound carefully
- No new strategies unless one of the three has been objectively broken for 2 quarters.
- Focus on reducing drawdown, not increasing return. A Sharpe-1.5 system at 25% CAGR will retire you. A Sharpe-0.7 system at 40% CAGR will blow you up first.
- At ₹50L AUM, hire a CA to set up a LLP or HUF for tax efficiency. Intraday trading in India is taxed as business income at slab rates. Long-term swing held >1 year qualifies for 10% LTCG. Structure matters at this scale.
- At ₹1Cr AUM, consider getting a second broker account (backup) and colocated connectivity if intraday is still the biggest sleeve.
- Start drawing a small monthly salary from the account (say 0.5% of AUM/month) once AUM > ₹75 lakh. This forces the system to survive withdrawal, which is the whole point of a backup income.

### Realistic end-of-year-5 outcome

Assuming you execute disciplined:
- ₹5L seed + ~₹60k/mo additions + 22% CAGR net → **₹95 lakh to ₹1.1 crore**.
- With income draws in year 5 of ~₹40k/month tested.
- That's a real, defensible backup income of ~₹40-50k/month at year 5, scaling up from there.

Not the ₹1.5 Cr dream, but **real, repeatable, and not dependent on one heroic year**. This is how people actually build it.

---

## 7. What To Do This Week

Cut through everything above. These are this week's actual tasks, ordered:

1. **Stop live trading** if you are currently live. Switch to PAPER_MONITOR.
2. **Fix Bug 2.1** (Busseti sign flip) in `strategy.py`. It's ~20 lines. This alone may flip your short-side P&L from negative to positive.
3. **Raise `FeatureStore.MAX_WINDOW`** from 200 to 5000 in `ml_signal.py`. One-line change. Let it fill over the week.
4. **Lower `MIN_ALPHA_THRESHOLD`** to 0.001 in `config.py` temporarily, because your current model barely produces signals above 0.005, so you have no data to evaluate.
5. **Make stops vol-scaled** in `position_manager.py` per Section 2.4.
6. **Add the portfolio loss limits** from Section 3.1 — daily 2%, weekly 5%, monthly 8% auto-pause. This is 50 lines and protects you from every mistake above.
7. **Start writing the backtester.** Even a crappy one that replays 1-min bars is 10× better than no backtester.

Don't touch anything else until these are done. Don't add features, don't rewrite the dashboard, don't chase new data sources. Fix the leaks in the boat first.

---

## 8. Closing Note

You've built something that most retail algo traders never build. The architecture is legit. The bugs above are normal for a system at this stage — every institutional desk fixed exactly the same issues in their first year. What separates people who retire off systematic trading from people who quit after blowing up isn't smarter models. It is:

- Discipline to fix bugs before adding features.
- Willingness to paper trade for 6 months when impatience says "just go live".
- Treating drawdown limits as non-negotiable physical law.
- Adding capital from income for years instead of hoping for 10-bagger years.
- Running multiple uncorrelated sleeves so one bad month doesn't end the dream.

Do those five things, execute the roadmap, and the math works. Skip any of them and the math is brutal. You have the skill set to be in the first group — the code tells me that. Make the plan match your life, not your hope.

Good luck. Ping me when you're in Phase 2.
