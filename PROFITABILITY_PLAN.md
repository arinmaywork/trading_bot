# SentiStack V2 — Profitability Overhaul Plan

## Executive Summary

After a full code audit of every layer (signal generation, feature pipeline, ML ensemble, execution, risk management, backtesting), the bot's losses trace to **three root causes** — not bugs, not infrastructure, but fundamental design problems in how signals are generated. This plan addresses all three through seven gated phases, each with clear success criteria before proceeding.

**Disclaimer:** No trading system guarantees profit. Markets are adversarial and non-stationary. This plan maximises the probability of positive expectancy by grounding every design choice in peer-reviewed quantitative finance literature and validated statistical methods. It does not eliminate risk.

---

## Part 1 — Root-Cause Diagnosis

### Root Cause 1: Timescale Mismatch in Feature Vector

The model predicts **1-minute forward log-returns** but 7 of the 11 features operate on timescales far longer than one minute:

| Feature | Update frequency | Useful prediction horizon |
|---------|-----------------|--------------------------|
| `mlofi` | Every tick (~100ms) | Seconds to minutes |
| `ofi` | Every tick | Seconds to minutes |
| `aflow_ratio` | 60-second rolling | 1-5 minutes |
| `aflow_delta_norm` | 60-second rolling | 1-5 minutes |
| `vwap_dev` | Per candle (~1 min) | 5-60 minutes |
| `sentiment_score` | Every 5+ minutes | Hours to days |
| `vol_normalised` | 20-minute lookback | Hours |
| `gri_composite` | Hourly refresh | Days to weeks |
| `gpr_normalised` | Monthly publication | Weeks to months |
| `rsi_normalised` | 20-minute lookback | 15-60 minutes |
| `vol_regime_encoded` | Session-level | Days |

Sentiment, GRI, GPR, weather, and vol-regime are effectively **constant** across hundreds of consecutive training samples. XGBoost cannot learn meaningful relationships from features that don't vary at the prediction timescale. The model fits noise.

### Root Cause 2: Predicting 1-Minute Returns Is Near-Impossible at This Latency

One-minute log-returns on NSE equities have a signal-to-noise ratio well below 0.01. Firms that profitably predict at this timescale operate with co-located servers (<1ms latency), direct exchange feeds (not WebSocket), and proprietary order-book models. Zerodha Kite's WebSocket adds 50-200ms latency, and the strategy loop runs every 0.5-5 seconds. By the time a microstructure signal is acted upon, the edge has decayed.

### Root Cause 3: Cost Structure vs Trade Frequency

At ~₹25 per round-trip on ~₹25,000 notional (10 bps), the model needs to predict direction accurately enough to generate >10 bps gross per trade. Over the last 90 days: 1,709 trades × ₹25 average cost = ₹42,959 in costs against ₹324 gross P&L. The model is trading at roughly random accuracy (gross ≈ 0), and the cost drag creates the observed ₹42k loss.

### Why Parameter Tuning Failed

Tune-1 tightened entry gates (alpha threshold, cost hurdle, TSL R:R). Result: trade count dropped 99.5% (1-8 trades). Tune-2 eased gates back. Result: 658-1709 trades, still negative. The problem is clear: **there is no alpha to filter for**. Tightening gates on a random signal just reduces sample size; loosening gates on a random signal just increases cost drag. Both lose money.

---

## Part 2 — Strategic Framework

### The Three Changes That Matter

1. **Multi-timeframe strategy**: Match each feature to the timeframe where it actually predicts.
2. **Feature engineering overhaul**: Add proven short-horizon features; remove timescale-mismatched ones from the short-term model.
3. **Massive trade-count reduction**: Target 5-15 trades per day instead of 50-100+. Each trade must have genuine statistical edge verified by out-of-sample testing.

### Strategy Tiers (to be implemented in phases)

**Tier 1 — Microstructure Scalper** (short-term, 1-15 min holds)
- Features: MLOFI, aggressive flow, bid-ask spread, order-book imbalance momentum
- Universe: Top 5-10 most liquid NIFTY50 names only
- Trades: 5-20 per day, ₹2-5 bps target per trade
- Edge source: Order-flow imbalance predicts very short-term price impact (Cont, Kukanov & Stoikov 2014)
- Kelly sizing is well-suited here: high-frequency, high-confidence micro-signals

**Tier 2 — Mean-Reversion Swinger** (medium-term, 30 min to 1 day)
- Features: VWAP deviation, RSI extremes, intraday seasonality, sector relative strength
- Universe: 15-20 liquid names
- Trades: 3-8 per day, 10-30 bps target
- Edge source: Intraday mean-reversion in liquid equities is well-documented (Avellaneda & Lee 2010)
- Sentiment becomes a useful secondary filter at this timescale

**Tier 3 — Sentiment Swing** (multi-day, CNC delivery)
- Features: News sentiment, GRI, GPR, sector rotation signals, earnings calendar
- Universe: Broader (30-50 names)
- Trades: 1-3 per week, 50-200 bps target
- Edge source: Sentiment aggregation predicts 1-5 day returns (Tetlock 2007; Da, Engelberg & Gao 2015)
- This is where the existing sentiment and geopolitical apparatus has genuine value

---

## Part 3 — Implementation Phases

### Phase 0: Signal Diagnostic (Days 1-2)

**Goal**: Quantify exactly where the current model fails. Instrument the bot to log everything needed for data-driven decisions.

**Tasks**:

0.1 **Add feature importance telemetry to ml_signal.py**
- After each retrain, log SHAP values (not just XGBoost built-in importance) for all 11 features
- Store in Redis with 7-day TTL: `diagnostics:shap:{symbol}:{timestamp}`
- Create `/diagnostics` Telegram command showing: top-3 features by SHAP magnitude, average |ml_signal|, signal distribution percentiles (p25/p50/p75/p90/p99)

0.2 **Add trade-outcome attribution**
- For each completed round-trip, log: entry ml_signal, entry confidence, which features drove the signal (top-3 by SHAP), actual return vs predicted return
- Store in daily CSV alongside existing logbook
- This answers: "Are winning trades driven by different features than losing trades?"

0.3 **Add filter funnel logging**
- Log counts at each gate: total signals generated → passed alpha gate → passed RSI filter → passed cost hurdle → passed sector cap → executed
- Telegram `/funnel` command to show daily funnel
- This answers: "Where are trades being dropped, and are the dropped trades actually worse?"

0.4 **Backtest with feature ablation**
- Run 90-day backtest with each feature removed one at a time (11 ablation runs)
- Record gross P&L for each ablation
- Features where removal improves P&L are candidates for removal

**Gate criteria**: Complete diagnostics, analyze results. If any single feature shows predictive power (ablation degrades gross P&L by >₹1000), proceed to Phase 1A. If no feature shows power, skip to Phase 1B.

---

### Phase 1A: Feature Engineering for Tier 1 — Microstructure (Days 3-7)

**Goal**: Build a feature set optimised for 1-15 minute prediction using only features that vary at the right timescale.

**Tasks**:

1.1 **Add Order-Book Imbalance Momentum (OBI-M)**
File: `data_ingestion.py`
```
OBI_t = (V_bid_total - V_ask_total) / (V_bid_total + V_ask_total)
OBI_momentum = OBI_t - EMA(OBI, span=20 ticks)
```
Rationale: Static OBI (current MLOFI) captures level; momentum captures acceleration. The change in imbalance is more predictive than the level (Cartea, Jaimungal & Penalva 2015).

1.2 **Add Trade-Arrival Rate Feature**
File: `data_ingestion.py`
```
trade_rate = tick_count_last_30s / avg_tick_count_30s_lookback_10min
```
Rationale: Unusual tick clustering precedes large moves. This is a simple, powerful feature that doesn't require order-level data.

1.3 **Add Bid-Ask Spread Dynamics**
File: `data_ingestion.py`
```
spread_bps = (best_ask - best_bid) / mid_price * 10000
spread_z = (spread_bps - rolling_mean_spread_20min) / rolling_std_spread_20min
```
Rationale: Spread widening signals information asymmetry; spread narrowing signals consensus. (Glosten & Milgrom 1985)

1.4 **Add Price Momentum at Multiple Horizons**
File: `ml_signal.py` (feature vector construction)
```
ret_1min  = log(price_t / price_{t-1min})
ret_5min  = log(price_t / price_{t-5min})
ret_15min = log(price_t / price_{t-15min})
```
Rationale: Short-horizon autocorrelation is the most replicated finding in market microstructure.

1.5 **Add Volume-Weighted Price Pressure (VWPP)**
File: `data_ingestion.py`
```
VWPP = Σ(signed_volume × price_change) over 5-min window
     where signed_volume = +qty if buy-classified, -qty if sell-classified
```
Combines aggressive flow with price impact — measures conviction of market-order traders.

1.6 **Build Tier-1 Feature Vector (8-dimensional)**
```
[mlofi, obi_momentum, aflow_ratio, trade_arrival_rate,
 spread_z, ret_1min, ret_5min, vwpp]
```
Drop: sentiment_score, vol_normalised, gri_composite, gpr_normalised, rsi_normalised, vol_regime_encoded, ofi (redundant with mlofi), aflow_delta_norm (correlated with aflow_ratio)

1.7 **Retrain on Tier-1 features with 5-minute forward return target**
- Change target from 1-min to 5-min forward log-return
- Larger prediction horizon → higher SNR → more learnable
- Adjust `LABEL_DELAY_S` from 60 → 300

**Gate criteria**: Run 30-day backtest on Tier-1 features. Require:
- Gross P&L > ₹0 (positive before costs)
- Information coefficient > 0.02 (even tiny IC compounds with high frequency)
- Profit factor > 1.0 (gross wins > gross losses)

If met → proceed. If not → investigate with SHAP, iterate features, or skip Tier 1 entirely.

---

### Phase 1B: Feature Engineering for Tier 2 — Mean Reversion (Days 3-7, parallel)

**Goal**: Build a feature set for 30-minute to intraday prediction.

**Tasks**:

1.8 **Add VWAP Z-Score**
```
vwap_z = (price - session_vwap) / rolling_std(price - vwap, 60min)
```
Rationale: Standardised VWAP deviation is a better mean-reversion signal than raw deviation. Extreme z-scores (|z| > 2) have documented reversion tendency.

1.9 **Add Intraday Seasonality Feature**
```
time_bucket = minute_of_day // 15   (26 buckets: 9:15 to 15:30)
seasonal_return = historical_avg_return_for_this_bucket_and_symbol
```
Rationale: NSE has strong intraday patterns — opening rush (9:15-9:45), lunch lull (12:00-13:00), closing auction (15:00-15:30). Incorporating time-of-day as a feature lets the model learn these rhythms.

1.10 **Add Sector Relative Strength**
```
sector_ret_15min = avg_return_of_sector_peers_over_15min
stock_ret_15min  = return_of_stock_over_15min
relative_strength = stock_ret_15min - sector_ret_15min
```
Rationale: Stocks that lag their sector tend to catch up (intraday mean-reversion at sector level).

1.11 **Add Opening Range Breakout Levels**
```
opening_range_high = max(high for first 3 candles, 9:15-9:30)
opening_range_low  = min(low for first 3 candles, 9:15-9:30)
orb_position = (price - opening_range_low) / (opening_range_high - opening_range_low)
```
Rationale: Opening range defines intraday support/resistance; breakouts from it carry momentum, but fading false breakouts is the higher-probability trade.

1.12 **Build Tier-2 Feature Vector (10-dimensional)**
```
[vwap_z, rsi_normalised, sector_relative_strength, orb_position,
 time_bucket_sin, time_bucket_cos,  // cyclical encoding
 ret_15min, ret_60min, vol_normalised, aflow_ratio_15min_avg]
```

1.13 **Use 30-minute forward return as target**
- Longer horizon = higher SNR = more learnable
- Trades held 30-60 minutes, not 1-5 minutes
- `LABEL_DELAY_S` = 1800

**Gate criteria**: Same as Phase 1A but on Tier-2 model.

---

### Phase 2: Model Architecture Improvements (Days 8-12)

**Goal**: Fix model overprediction and improve calibration.

**Tasks**:

2.1 **Separate models per tier**
File: `ml_signal.py`
- `Tier1Engine` — microstructure features, 5-min target, retrain every 15 min
- `Tier2Engine` — mean-reversion features, 30-min target, retrain every 60 min
- Each has independent FeatureStore, scaler, and ensemble
- `EnsembleSignalEngine` selects the appropriate tier based on time-of-day and regime

2.2 **Fix model calibration (predicted vs realised alpha)**
The current model predicts raw log-returns, but the output is then multiplied by SIGNAL_SCALE=20 and used directly as alpha. This chain has no calibration check.

Add post-hoc calibration:
```
# After each retrain, on holdout set:
predicted_returns = model.predict(X_holdout)
actual_returns = y_holdout
calibration_slope = np.polyfit(predicted_returns, actual_returns, 1)[0]
# If slope < 0.5 → model overpredicts by 2x+
# Apply: adjusted_signal = raw_signal * calibration_slope
```
Store `calibration_slope` per symbol. If slope < 0.1, the model has no predictive power — flag it and don't trade that symbol.

2.3 **Replace meta-learner with regime-conditional blending**
Instead of a fixed ridge meta-learner:
```
if volatility_regime == "LOW":
    weight_xgb = 0.7   # non-linear model does better in calm markets
    weight_ridge = 0.3
elif volatility_regime == "HIGH":
    weight_xgb = 0.3   # linear model is more robust in volatile regimes
    weight_ridge = 0.7
```
This is simpler, more interpretable, and harder to overfit than a learned meta-learner.

2.4 **Add model confidence calibration**
Current confidence = agreement between base learners. Better:
```
# Isotonic regression on holdout:
# Map raw_confidence → actual_probability_of_correct_direction
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_confidence_holdout, (actual_return * predicted_direction > 0).astype(float))
calibrated_confidence = iso.predict(raw_confidence_new)
```
This tells you the actual probability the trade will be profitable.

2.5 **Implement rolling walk-forward validation in live mode**
Every retrain cycle, hold out the most recent 20% as validation:
- Compute IC, Sharpe, and calibration slope on the holdout
- If IC < 0.01 or Sharpe < 0: mark model as "degraded" → halve position sizes
- If IC < 0.005 or Sharpe < -0.5: mark model as "broken" → stop trading that symbol
- Log all validation metrics to Telegram `/model_health` command

**Gate criteria**: On 30-day backtest, both tiers independently show:
- Gross Sharpe > 0.5
- Calibration slope > 0.3 (predicted direction is at least somewhat correct)
- Feature importance is distributed (no single feature >50% importance)

---

### Phase 3: Trade Frequency Reduction & Cost Optimisation (Days 13-16)

**Goal**: Reduce cost drag from ₹43k/90 days to under ₹10k/90 days while preserving edge.

**Tasks**:

3.1 **Implement signal conviction scoring**
Not all signals are equal. Create a composite conviction score:
```
conviction = (
    0.4 * calibrated_confidence
  + 0.3 * abs(signal) / signal_99th_percentile
  + 0.2 * (1.0 - spread_z / 3.0)   # wider spreads = lower conviction
  + 0.1 * time_decay_factor         # signals near market open/close get boost
)
```
Only trade when conviction > 0.65 (tune on validation set).

3.2 **Reduce universe per tier**
- Tier 1 (microstructure): Top 5-8 by ADV from NIFTY50 only (RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, SBIN, BHARTIARTL, ITC)
- Tier 2 (mean-reversion): Top 15-20 by composite score
- Tier 3 (sentiment swing): Full universe but 1-3 trades per week
- Smaller universe = better model per symbol, fewer cost events

3.3 **Dynamic cost hurdle based on market regime**
```
if vol_regime == "LOW":
    hurdle_mult = 2.0   # quiet markets: tighter spreads, lower costs
elif vol_regime == "MODERATE":
    hurdle_mult = 3.0
elif vol_regime == "HIGH":
    hurdle_mult = 5.0   # volatile markets: wider spreads, higher slippage
elif vol_regime == "EXTREME":
    hurdle_mult = 10.0  # crisis: avoid trading entirely unless extreme conviction
```

3.4 **Favour passive execution for Tier 2**
- Tier-1 trades need aggressive fills (time-sensitive microstructure)
- Tier-2 trades can afford 30-60 second passive limit rest
- Expected savings: 3-5 bps per round trip on Tier-2
- At 5-8 Tier-2 trades/day, this saves ₹1000-3000/month

3.5 **Batch Tier-2 and Tier-3 orders at optimal execution windows**
- NSE liquidity peaks at 9:15-10:00 and 14:30-15:15
- Queue non-urgent orders for these windows
- Reduces market impact and improves fill rates

**Gate criteria**: Backtest over 90 days showing:
- Total trade count < 400 (down from 1709)
- Total costs < ₹10,000
- Net P&L positive (gross edge survives after cost reduction)

---

### Phase 4: Market Regime Detection (Days 17-20)

**Goal**: Stop trading during regimes where the model has no edge.

**Tasks**:

4.1 **Build regime classifier**
Three regimes, detected from market-level indicators:
```
TRENDING:     ADX(NIFTY50, 14) > 25 AND abs(NIFTY_ret_60min) > 0.3%
MEAN_REVERTING: ADX(NIFTY50, 14) < 20 AND Bollinger_BW(NIFTY50) < 2%
VOLATILE:     India VIX > 20 OR NIFTY_realised_vol_1h > 25% annualised
```
Run regime detection every 15 minutes.

4.2 **Tier-regime routing table**
```
              | TRENDING | MEAN_REVERTING | VOLATILE |
  Tier 1      |   ON     |      ON        |   OFF    |
  Tier 2      |   OFF    |      ON        |   OFF    |
  Tier 3      |   ON     |     OFF        |   ON*    |

* Tier 3 in VOLATILE = contrarian sentiment plays with reduced sizing
```

4.3 **Intraday no-trade zones**
Based on empirical NSE patterns:
- 12:00-13:00 IST: Low liquidity lunch period → skip Tier 1 & 2
- First 5 minutes after open (9:15-9:20): Price discovery noise → skip Tier 1
- Last 15 minutes (15:15-15:30): Square-off cascade → Tier 1 only (momentum into close)

4.4 **Event-aware throttling**
- RBI monetary policy days: reduce sizing by 50% until 30 min after announcement
- Monthly F&O expiry (last Thursday): reduce Tier 1 sizing by 30% (gamma-driven noise)
- Earnings announcement (per symbol): blackout 1 hour before to 30 min after
- Wire this into the existing news_blackout.py framework

**Gate criteria**: Regime-gated backtest shows improvement over un-gated version:
- Higher Sharpe ratio
- Lower max drawdown
- No reduction in total gross P&L (regime gating should remove bad trades, not good ones)

---

### Phase 5: Robust Backtesting & Validation (Days 21-28)

**Goal**: Ensure the improved system's edge is real and not a backtest artifact.

**Tasks**:

5.1 **Implement combinatorial purged cross-validation (CPCV)**
Reference: López de Prado (2018), "Advances in Financial Machine Learning", Chapter 12.
- Current purged WF with 3-5 folds is insufficient for 11+ dimensional feature space
- CPCV generates O(2^N) train/test combinations from N groups
- Provides backtest-overfitting probability (PBO)
- Target: PBO < 0.30 (less than 30% chance the strategy is overfit)

5.2 **Monte Carlo permutation test**
- Shuffle signal labels (random entry timing) 1000 times
- If the real strategy's Sharpe doesn't exceed the 95th percentile of shuffled strategies, the edge is noise
- This is the gold standard test for whether alpha is real

5.3 **Transaction cost sensitivity analysis**
- Run backtest at 1x, 1.5x, and 2x costs
- If strategy is profitable at 1.5x costs, it has safety margin for real-world slippage
- If unprofitable at 1.5x, the edge is too thin

5.4 **Regime-specific decomposition**
- Report Sharpe, drawdown, and profit factor separately for each regime
- Identify which regime drives most of the P&L
- If > 80% of P&L comes from one regime, the strategy is fragile

5.5 **Drawdown analysis**
- Maximum drawdown (peak to trough)
- Calmar ratio (annualised return / max drawdown) — target > 1.0
- Time to recovery from worst drawdown
- Rolling 20-day Sharpe to detect performance decay

**Gate criteria**: Strategy must pass ALL of the following:
- Net Sharpe > 1.0 on out-of-sample data
- Profit factor > 1.3
- PBO < 0.30
- Survives Monte Carlo permutation test (p < 0.05)
- Profitable at 1.5x costs
- Max drawdown < 5% of capital
- If ANY criterion fails → iterate back to Phases 1-4

---

### Phase 6: Paper Trading Validation (Days 29-58, 30 calendar days)

**Goal**: Verify that live execution matches backtest expectations.

**Tasks**:

6.1 **Run all three tiers simultaneously in PAPER mode**
- Use live market data (Zerodha WebSocket)
- Simulate fills at realistic prices (not mid-price)
- Log everything: signal, conviction, regime, features, predicted return, paper fill price

6.2 **Daily reconciliation dashboard**
- Compare paper P&L vs what backtest would have predicted for same day
- Track slippage: paper fill price vs signal price
- Track filter funnel: how many signals generated vs traded
- Telegram `/paper_report` command

6.3 **Rolling performance metrics**
After 10 days, start computing:
- Rolling 5-day Sharpe
- Cumulative P&L curve
- Win rate by tier
- Average holding time by tier
- Cost as percentage of gross P&L

6.4 **Weekly decision checkpoints**
- Week 1: Assess data quality, execution timing, filter funnel
- Week 2: First meaningful performance assessment. If rolling Sharpe < 0: diagnose and fix
- Week 3: Stable assessment. If cumulative P&L < -2%: consider halting
- Week 4: Final assessment. If net P&L negative and no clear fix: do NOT go live

**Gate criteria**: After 30 days of paper trading:
- Net cumulative P&L positive
- Paper Sharpe within 0.5 standard deviations of backtest Sharpe
- No single day loss exceeding 2% of simulated capital
- Average slippage < 5 bps
- If ANY criterion fails → iterate, do not go live

---

### Phase 7: Live Deployment & Scaling (Day 59+)

**Goal**: Gradually deploy with real capital, with automated safety nets.

**Tasks**:

7.1 **Tiered capital allocation (start small)**
```
Week 1:  20% of capital (₹1,00,000)
Week 2:  40% if Week 1 profitable
Week 3:  60% if Week 2 profitable
Week 4:  80% if Week 3 profitable
Week 5+: 100% if all weeks profitable
```
If any week is a loss > 1%: reset to 20% and diagnose.

7.2 **Automated kill switches (non-negotiable)**
```
DAILY_KILL:     -1.5% of allocated capital → halt ALL trading, alert Telegram
WEEKLY_KILL:    -3.0% of allocated capital → halt + enter paper-only mode
MONTHLY_KILL:   -5.0% of allocated capital → full shutdown, manual review required
REGIME_KILL:    India VIX > 30 → reduce all sizing by 70%
MODEL_KILL:     Rolling 5-day IC < 0 → halt Tier 1 & 2, Tier 3 continues
```

7.3 **Continuous model monitoring**
- Log calibration slope every retrain cycle
- Alert if calibration degrades by > 30% from baseline
- Weekly automated backtest vs prior week (detect distribution shift)
- Monthly SHAP analysis to detect feature drift

7.4 **Performance attribution by tier**
- Track P&L separately for each tier
- Identify which tier is contributing and which is detracting
- Reallocate capital from losing tiers to winning tiers quarterly
- Kill any tier that is negative after 60 trading days

---

## Part 4 — Specific Code Changes Summary

### New Files to Create
| File | Purpose |
|------|---------|
| `features/microstructure.py` | OBI momentum, trade arrival, spread dynamics, VWPP |
| `features/mean_reversion.py` | VWAP z-score, sector relative strength, ORB levels |
| `features/seasonality.py` | Time-of-day encoding, event calendar integration |
| `regime_detector.py` | Market regime classifier (trending/MR/volatile) |
| `tier_router.py` | Routes signals to appropriate tier based on regime |
| `model_health.py` | Calibration, IC tracking, degradation alerts |
| `diagnostics.py` | Feature importance, filter funnel, trade attribution |
| `backtest/cpcv.py` | Combinatorial purged CV and PBO calculation |
| `backtest/monte_carlo.py` | Permutation test for alpha significance |

### Files to Modify
| File | Changes |
|------|---------|
| `ml_signal.py` | Separate Tier1Engine and Tier2Engine; add calibration layer; SHAP logging |
| `data_ingestion.py` | Add OBI momentum, trade arrival rate, spread tracking |
| `strategy.py` | Conviction scoring; regime-tier routing; dynamic cost hurdle |
| `config.py` | New parameters for tiers, regime thresholds, conviction gate |
| `main.py` | Wire regime detector, tier router, and diagnostics |
| `telegram_controller.py` | Add /diagnostics, /funnel, /model_health, /paper_report |
| `universe.py` | Per-tier universe sizes; optimal execution windows |
| `backtest/engine.py` | Support multi-tier backtesting; add cost sensitivity mode |

---

## Part 5 — Estimated Timeline

| Phase | Duration | Cumulative | Deliverable |
|-------|----------|------------|-------------|
| 0: Diagnostics | 2 days | Day 2 | Feature ablation results, filter funnel data |
| 1: Feature Engineering | 5 days | Day 7 | Tier-1 and Tier-2 feature vectors |
| 2: Model Improvements | 5 days | Day 12 | Calibrated multi-tier models |
| 3: Cost Optimisation | 4 days | Day 16 | Trade-count reduction, conviction scoring |
| 4: Regime Detection | 4 days | Day 20 | Regime-gated strategy |
| 5: Robust Backtesting | 7 days | Day 27 | Validated strategy with PBO < 0.30 |
| 6: Paper Trading | 30 days | Day 57 | Live-data validated P&L |
| 7: Live Deployment | Ongoing | Day 58+ | Gradual capital scaling |

**Total to first live trade: ~8 weeks** (2 months). This is not optional — deploying faster means deploying an unvalidated system, which is how the current ₹43k loss happened.

---

## Part 6 — Honest Assessment

### What This Plan CAN Do
- Eliminate the timescale mismatch (the dominant cause of near-zero alpha)
- Add proven microstructure features with peer-reviewed evidence of predictive power
- Reduce cost drag by 60-75% through frequency reduction and better execution
- Provide statistical rigor to differentiate real edge from noise
- Create automated safety nets that limit downside

### What This Plan CANNOT Guarantee
- Profit. Markets are non-stationary; an edge today may disappear tomorrow
- That any of the new features will produce positive IC on NSE specifically
- That the strategy will survive a black-swan event (though kill switches limit damage)
- That 30 days of paper trading is sufficient to validate all market regimes

### Biggest Risk
The biggest risk is that **NSE equity markets at the intraday timescale simply don't have exploitable alpha at Zerodha's latency tier.** Co-located HFT firms may have already arbitraged away the microstructure signals we're trying to capture. If Phase 1's gate criteria fail (gross P&L ≤ 0 with new features), the honest next step is:

1. Move to a longer timeframe (daily bars, swing trading) where latency matters less
2. Move to a different asset class (options on NIFTY, where volatility surface has more structure)
3. Accept that the tool's value may be in risk management, portfolio construction, and execution quality rather than alpha generation — and combine it with discretionary trading decisions

This plan is designed to answer that question definitively rather than continuing to lose money while hoping.

---

*Generated for SentiStack V2 — April 2026*
