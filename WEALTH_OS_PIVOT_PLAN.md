# SentiStack → Wealth OS: The Pivot Plan

**Date:** July 3, 2026
**Owner:** Arinmay
**Inputs:** Capital ₹5–15L + ₹50k+/month savings · Recommend-and-approve autonomy · CAS import for MF data · Priorities: portfolio health + SIP/goal automation

---

## 1. The Goal, Stated Honestly

"Never think about money again" is not a trading problem. It is a **wealth-engine problem** with four levers, in order of impact:

1. **Savings rate** — your ₹50k+/month is worth more than any alpha. It is guaranteed return.
2. **Cost & tax drag** — expense ratios, churn, STT, LTCG mistakes. Fully controllable.
3. **Asset allocation & discipline** — being invested correctly and not panic-selling. Mostly controllable.
4. **Alpha (stock picking, timing)** — the smallest, least reliable lever. Your own live logs prove it: 1,629 order legs churning ₹3L of turnover on ₹10k capital, bleeding costs on ₹150 trades.

The old bot spent 100% of its effort on lever 4. Wealth OS inverts that: it automates levers 1–3 ruthlessly and treats lever 4 as a small, strictly-capped satellite.

**The math that actually works** (13% blended net CAGR, a defensible equity-heavy long-term assumption for India, not a promise):

| Horizon | ₹10L seed + ₹50k/month | Monthly income at 4% SWP |
|---|---|---|
| 5 years | ~₹62 lakh | — (accumulation) |
| 10 years | ~₹1.55 crore | ~₹52,000 |
| 15 years | ~₹3.4 crore | ~₹1,13,000 |

Raise the monthly contribution as income grows and the 15-year number crosses ₹5 crore. This is the real "passive income": the portfolio itself, systematically managed, eventually paying you via SWP. The tool's job is to make sure nothing leaks and nothing requires your attention.

---

## 2. What Happens to SentiStack

| Component | Verdict | Why |
|---|---|---|
| `telegram_controller.py`, log handler, heartbeats | **KEEP** | Becomes the approval + notification interface |
| `backtest/` (engine, cost model, walk-forward) | **KEEP** | Validates the swing sleeve in Phase 4; cost model reused for every recommendation |
| `portfolio_risk.py` (loss budgets, sector caps) | **KEEP, adapt** | Re-anchored to total net worth instead of intraday capital |
| `logbook.py`, FIFO P&L, Zerodha cost fidelity | **KEEP** | Powers XIRR and tax computation |
| `config.py`, deploy scripts, GCP setup, token flow | **KEEP** | Same infra, new brain |
| `rate_limiter.py` | **KEEP** | Kite API compliance |
| Intraday loop: tick ingestion, 1-min ML ensemble, MLOFI, Gemini intraday sentiment, geopolitical multipliers, aggressive execution | **FREEZE** | Archive to `legacy/` branch. Not deleted — retired. The cost math at your capital cannot work |

Nothing is wasted. Roughly 60% of the codebase carries over.

---

## 3. Wealth OS Architecture

```
┌────────────────────────────────────────────────────────┐
│                 Telegram (approve / ask / digest)       │
└──────────────┬─────────────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────┐
│                    Recommendation Engine                │
│   every action = a card: WHAT, WHY, IMPACT, [APPROVE]  │
└──┬──────────┬──────────────┬──────────────┬────────────┘
   │          │              │              │
┌──▼───────┐ ┌▼────────────┐ ┌▼───────────┐ ┌▼───────────┐
│Portfolio │ │Analytics    │ │Goal & SIP  │ │Swing Sleeve│
│Aggregator│ │Engine       │ │Engine      │ │(Phase 4)   │
│          │ │             │ │            │ │            │
│Kite      │ │XIRR, alloc, │ │Goal corpus │ │Nifty-200   │
│holdings  │ │MF overlap,  │ │math, SIP   │ │momentum +  │
│(free API)│ │expense drag,│ │optimizer,  │ │quality,    │
│CAS parser│ │concentration│ │rebalance   │ │weekly,     │
│(casparser│ │benchmark vs │ │bands, tax  │ │≤20% capped │
│ CAMS/KFin│ │Nifty/index  │ │harvesting  │ │via backtest│
└──┬───────┘ └─────────────┘ └────────────┘ └────────────┘
   │
┌──▼──────────────────────────────────────────────────────┐
│  SQLite store: holdings, transactions, NAVs, goals,     │
│  recommendations, approvals  (replaces Redis hot path)  │
│  Free data: AMFI NAV feed, NSE bhavcopy, yfinance EOD   │
└─────────────────────────────────────────────────────────┘
```

**Key design decisions:**

- **Recommend-and-approve everywhere.** No order fires without your Telegram tap. Kite Connect **Personal** (free tier) covers holdings, positions, funds, and order placement — we don't need the paid market-data tier because a portfolio manager runs on EOD data, which is free (AMFI daily NAVs, NSE bhavcopy, yfinance).
- **CAS import for mutual funds.** Monthly CAMS/KFintech Consolidated Account Statement PDF → parsed by the open-source `casparser` library → covers every MF folio you own regardless of platform (Kuvera, Coin, direct). You forward the CAS email PDF; the tool does the rest.
- **EOD cadence, not ticks.** The system thinks once a day after market close and once a week in depth. This eliminates the infra strain, latency costs, and daily token urgency of the old bot.
- **Running cost: ~₹0.** Free Kite Personal API, free EOD data, free GCP e2-micro (now ample, since no tick firehose), free casparser.

---

## 4. What Wealth OS Actually Does

### 4.1 Portfolio Health (Priority 1)
- Unified net-worth view: Zerodha equity + all MFs + cash, updated daily.
- **XIRR** per holding, per goal, and total — the only honest return number.
- **Asset allocation** vs your target (equity/debt/gold/international) with drift alerts.
- **MF overlap analysis** — two funds holding the same 40 stocks is one fund with double fees.
- **Cost audit** — expense-ratio drag in ₹/year, regular-vs-direct plan detection, exit-load awareness.
- **Concentration & risk** — single-stock, single-sector, single-AMC exposure caps (reusing `portfolio_risk.py` logic).
- Weekly Telegram digest; monthly deep report.

### 4.2 Goal & SIP Automation (Priority 2)
- Define goals (emergency fund, house, financial independence) with target corpus, horizon, and priority.
- Required-SIP math per goal; surplus allocator for the ₹50k+/month.
- **Rebalancing recommendations** using 5/25 bands (act only when an asset class drifts ±5 absolute or ±25 relative percent) — direct new SIP money to the underweight class first so rebalancing rarely needs selling (no tax event).
- **Tax intelligence:** harvest the ₹1.25L/year LTCG exemption systematically; tax-loss harvesting candidates in December–March; flag STCG-vs-LTCG timing before any recommended sell.
- Annual glide-path shift toward debt as each goal approaches.

### 4.3 Monitoring & Alerts
- Daily EOD check (5 minutes of compute): NAV updates, corporate actions on your stocks, drift, upcoming SIP dates, fund manager exits / rating downgrades on your funds.
- Alert only when action is useful. Silence is the default. "Never think about money" means the tool thinks instead.

### 4.4 Swing/Stock Sleeve — Phase 4, capped, evidence-gated
Your "look for stocks & buy/sell when necessary" requirement, done responsibly:
- Universe: Nifty 200. Signal: 6/12-month momentum + quality filter (ROE, low accruals, price above 200DMA). Rebalance monthly. 10–15 names.
- **Hard cap: 20% of total portfolio.** The core 80% stays in index/MF.
- **Gate:** goes live only after the existing walk-forward backtester shows OOS Sharpe > 1.0 on 5+ years of real (not synthetic) EOD data, then 3 months of paper recommendations tracked against reality.
- Recommendations arrive as Telegram cards with thesis + position size + stop; you approve; the tool places the order (or you do it manually in Kite).

---

## 5. Build Phases

| Phase | Timeline | Deliverable | Done when |
|---|---|---|---|
| **0. Freeze** | Week 1 | Intraday bot to `legacy/` branch, live trading off | Repo restructured, nothing can fire an order |
| **1. Data foundation** | Weeks 1–3 | Kite holdings sync + casparser CAS import + SQLite store + `/portfolio` `/networth` commands | Your real, complete portfolio renders in Telegram |
| **2. Analytics** | Weeks 3–6 | XIRR, allocation, overlap, cost audit, health score, weekly digest | First monthly report matches your manual records within 1% |
| **3. Goals & recommendations** | Weeks 6–10 | Goal engine, SIP allocator, rebalance bands, tax module, approve/reject workflow | First approved recommendation executed end-to-end |
| **4. Swing sleeve** | Months 3–6 | Momentum+quality screen through the existing backtester, paper first | Backtest gate passed + 3 months paper tracking |
| **5. Steady state** | Ongoing | Quarterly reviews, annual tax pack, glide-path shifts | You touch it < 15 min/week |

Phases 1–3 are pure software against your own data — zero market risk while building. We can start Phase 0+1 in the next session.

---

## 6. Risk Rules (Non-Negotiable, Inherited & Adapted)

- No leverage, no F&O, no intraday. Ever, in this system.
- Swing sleeve ≤ 20% of portfolio; single stock ≤ 5%; single sector ≤ 25% (reuses existing sector-cap code).
- Every sell recommendation shows tax impact before you approve.
- If the swing sleeve underperforms its index benchmark over any rolling 2-year window, it auto-recommends folding back into index funds.
- Emergency fund (6 months expenses, liquid/FD) is tracked as Goal #1 and must be full before surplus goes to equity.

---

## 7. What This Is Not

No system removes market risk. A 30–40% equity drawdown will happen at least once in your horizon; the tool's job in that year is to stop you from selling and to keep the SIPs firing — that is when most of the eventual corpus is actually bought. And I'm not a financial advisor; treat the return assumptions as planning inputs, not guarantees.

---

## 8. Decision Needed From You

1. Approve the freeze of intraday live trading (Phase 0).
2. Confirm target asset allocation to start with (default proposal: 65% equity / 20% debt / 10% gold / 5% cash until goals are defined in Phase 3).
3. When ready: forward your latest CAS PDF (CAMS "detailed" statement) so Phase 1 has real data on day one.

Then we build.
