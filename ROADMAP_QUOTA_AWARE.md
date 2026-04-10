# SentiStack V2 — Quota-Aware Improvement Roadmap

Each task below is sized to fit comfortably inside one Claude session. Each
ends at a committable checkpoint so nothing is half-built if a limit hits.

## How to use this plan

Every task has:
- **Scope** — what gets built
- **Files touched** — blast radius
- **Session budget** — rough context/time estimate
- **Pause checkpoints** — safe stop points
- **Resume prompt** — paste verbatim in a fresh session to continue

**Rule of thumb:** If the "approaching usage limit" warning appears mid-task,
stop at the next pause checkpoint, commit WIP, and resume later.

---

## Week 1 — Small-capital enablement + foundation

### Task 1 — BOOTSTRAP_MODE for ₹5-10k trading
**Why first:** Without this, the paper → small-capital rollout is blocked.

**Scope:**
- Add `BOOTSTRAP_MODE`, `BOOTSTRAP_CAPITAL_THRESHOLD` (₹50k), `BOOTSTRAP_MAX_POSITION_FRACTION` (0.35), `BOOTSTRAP_MIN_TRADE_VALUE` (₹500) to `StrategyConfig`.
- Add `get_effective_position_fraction()` / `get_effective_min_trade_value()` helpers in config.py that switch on active capital.
- Replace direct reads of `MAX_POSITION_FRACTION` / `MIN_TRADE_VALUE` in `strategy.py`, `risk.py`, `position_manager.py` with the helpers.
- Telegram `/status` shows "Bootstrap: ON" when active.
- Auto-revert when capital grows past threshold.
- Update README and `.env.example`.

**Files touched:** `config.py`, `strategy.py`, `risk.py`, `position_manager.py`, `telegram_controller.py`, `README.md`, `.env.example`

**Session budget:** ~40% of one session

**Pause checkpoints:**
1. config.py + helpers compile cleanly
2. All call sites replaced, every file passes `python -m py_compile`
3. README + .env.example updated

**Resume prompt:**
> Resume SentiStack V2 Task 1 (BOOTSTRAP_MODE). Read ROADMAP_QUOTA_AWARE.md Task 1 section, then check git log and config.py to see which pause checkpoint was reached. Continue from the next unchecked checkpoint.

---

### Task 2 — Walk-forward backtester scaffold (part 1/2)
**Why:** Biggest long-term lever — validates every future change against history.

**Scope (scaffold + data loader only):**
- New `backtest/engine.py`: purged walk-forward splitter, event-driven replay skeleton.
- New `backtest/data_loader.py`: Zerodha historical minute bars via `kite.historical_data()` with on-disk parquet cache.
- New `backtest/cost_model.py`: mirror logbook.py cost function exactly.
- No strategy wiring yet — dummy always-flat strategy for end-to-end run.
- CLI: `python -m backtest.engine --symbol RELIANCE --start 2024-01-01 --end 2024-06-30`

**Files touched:** `backtest/{__init__.py, engine.py, data_loader.py, cost_model.py}`

**Session budget:** ~70% of one session

**Pause checkpoints:**
1. data_loader.py caches one symbol
2. cost_model.py unit test matches logbook.py
3. engine.py runs dummy strategy end-to-end

**Resume prompt:**
> Resume SentiStack V2 Task 2 (backtester scaffold). Read ROADMAP_QUOTA_AWARE.md Task 2 section. Check backtest/ directory, see which files exist and compile, continue from the next checkpoint. Strategy wiring is Task 3 — do not start it here.

---

### Task 3 — Walk-forward backtester strategy adapter (part 2/2)
**Scope:**
- `backtest/strategy_adapter.py`: wrap live `StrategyEngine` for historical bars.
- Plug in `risk.py` sizing and `portfolio_risk.py` budgets.
- Per-trade CSV + summary stats (Sharpe, max DD, win rate, profit factor, net of costs).
- Purged k-fold walk-forward with 20% embargo.
- Sample run on 6 months × 5 NSE symbols → `backtest/reports/`.

**Files touched:** `backtest/strategy_adapter.py`, minor edits to `backtest/engine.py`

**Session budget:** ~80% of one session

**Pause checkpoints:**
1. Adapter runs one bar through StrategyEngine without crash
2. Full walk-forward completes on one symbol
3. Multi-symbol report generated

**Resume prompt:**
> Resume SentiStack V2 Task 3 (backtester strategy adapter). Read ROADMAP_QUOTA_AWARE.md Task 3 section. Read backtest/engine.py and strategy_adapter.py for current state. Continue from the next checkpoint.

---

## Week 2 — Alpha + sizing improvements

### Task 4 — Self-calibrating alpha threshold + confidence-weighted Kelly
**Scope:**
- `strategy.py`: rolling 90th-percentile alpha threshold (200-bar window per symbol), replacing fixed `MIN_ALPHA_THRESHOLD`.
- `risk.py`: multiply Kelly fraction by `min(1.0, ml_confidence / 0.7)`, floor 0.3.
- Config flags to toggle (default ON), fallback to static values if disabled.
- Backtest A/B report.

**Files touched:** `strategy.py`, `risk.py`, `config.py`, `README.md`

**Session budget:** ~50% of one session + verification run

**Pause checkpoints:**
1. Rolling threshold computes correctly on sample data
2. Confidence-weighted Kelly matches spec
3. A/B backtest complete

**Resume prompt:**
> Resume SentiStack V2 Task 4 (alpha threshold + confidence Kelly). Read ROADMAP_QUOTA_AWARE.md Task 4 section. Read strategy.py and risk.py for current state. Continue and produce the A/B backtest report.

---

### Task 5 — Sector correlation cap + intraday MTM drawdown stop ✅ COMPLETE
**Scope:**
- Extend `portfolio_risk.py`: sector map from `data/nse_sector_map.csv`; max 30% per sector.
- Intraday MTM stop: unrealized + realized < -1.5% capital → halt new entries, auto-clear next open.
- Wire into strategy_loop pre-trade checks.
- Telegram `/risk` shows sector exposure breakdown.

**Files touched:** `portfolio_risk.py`, `main.py`, `telegram_controller.py`, `data/nse_sector_map.csv` (new), `README.md`

**Session budget:** ~60% of one session

**Pause checkpoints:**
1. Sector exposure calc correct on mock portfolio
2. MTM stop triggers on simulated drawdown
3. `/risk` panel renders new fields

**Resume prompt:**
> Resume SentiStack V2 Task 5 (sector cap + MTM stop). Read ROADMAP_QUOTA_AWARE.md Task 5 section. Read portfolio_risk.py; check data/ for nse_sector_map.csv. Continue from the next checkpoint.

---

## Week 3 — Execution alpha + observability

### Task 6 — Passive limit orders with market fallback ✅ COMPLETE
**Scope:**
- `execution.py`: `place_passive_limit()` at mid ± spread/4, 90-sec TTL, fallback to market.
- Behind `EXECUTION_PASSIVE_MODE` flag (default OFF).
- Slippage logging.
- A/B via backtester.

**Files touched:** `execution.py`, `config.py`, `main.py`

**Session budget:** ~70% of one session

**Pause checkpoints:**
1. Limit order path compiles, paper dry run works
2. Fallback-to-market path tested
3. A/B result documented

**Resume prompt:**
> Resume SentiStack V2 Task 6 (passive limit orders). Read ROADMAP_QUOTA_AWARE.md Task 6 section. Read execution.py for current place_passive_limit state. Continue from the next checkpoint.

---

### Task 7 — Slippage monitor + daily Telegram digest
**Scope:**
- `monitor.py` (new): rolling slippage from logbook CSVs, alert if degrading vs 10-day baseline.
- 3:30 PM IST Telegram digest (trades, win rate, P&L, unrealized, biggest winner/loser, slippage bps, cost drag).
- Scheduler fires from existing strategy_loop clock.

**Files touched:** `monitor.py` (new), `telegram_controller.py`, `main.py`, `README.md`

**Session budget:** ~50% of one session

**Pause checkpoints:**
1. Slippage computation matches manual check on sample day
2. Digest renders correctly
3. Scheduled firing wired in

**Resume prompt:**
> Resume SentiStack V2 Task 7 (slippage + daily digest). Read ROADMAP_QUOTA_AWARE.md Task 7 section. Check for monitor.py and the digest block in telegram_controller.py. Continue from the next checkpoint.

---

## Week 4 — Fidelity + polish

### Task 8 — Full Zerodha cost fidelity (R-16)
**Scope:**
- Add GST (18% on brokerage+exchange), stamp duty (0.003% buy), SEBI turnover (₹10/crore) to `logbook.py`, `strategy.py` cost filter, and `backtest/cost_model.py`.
- Verify against a real Zerodha contract note.

**Files touched:** `logbook.py`, `strategy.py`, `backtest/cost_model.py`, `README.md`

**Session budget:** ~30% of one session

**Resume prompt:**
> Resume SentiStack V2 Task 8 (cost fidelity). Read ROADMAP_QUOTA_AWARE.md Task 8 section. Read logbook.py cost section. Add GST/stamp/SEBI lines and confirm parity with a Zerodha contract note.

---

### Task 9 — News-event blackout
**Scope:**
- 5-minute per-symbol cooldown after a high-impact sentiment headline.
- Config flag; `/status` shows active blackouts.

**Files touched:** `sentiment.py` (or equivalent), `strategy.py`, `telegram_controller.py`

**Session budget:** ~30% of one session

**Resume prompt:**
> Resume SentiStack V2 Task 9 (news blackout). Read ROADMAP_QUOTA_AWARE.md Task 9 section. Read the sentiment module and strategy.py entry gate. Add the 5-minute per-symbol cooldown.

---

## Weekly cadence

Recommended pairing (2 sittings per week):
- **Week 1:** Task 1 (half) + Task 2 (full)
- **Week 2:** Task 3 (full) + Task 4 (half)
- **Week 3:** Task 5 (half) + Task 6 (full)
- **Week 4:** Task 7 (half) + Task 8 (half) + Task 9 (half) — batched

## Universal pause protocol

If any session hits the quota warning:
1. Stop at the next numbered pause checkpoint in the active task.
2. Run `python -m py_compile` on every touched file — must be clean.
3. Commit with message `WIP Task N checkpoint M — <brief>`.
4. Do not push unless the checkpoint is end-to-end working.
5. In the next session, paste the task's resume prompt. Claude will read files, verify state, and continue.

## Task status

- [x] Task 1 — BOOTSTRAP_MODE
- [x] Task 2 — Backtester scaffold
- [x] Task 3 — Backtester strategy adapter
- [x] Task 4 — Alpha threshold + confidence Kelly
- [x] Task 5 — Sector cap + MTM stop
- [x] Task 6 — Passive limit orders
- [ ] Task 7 — Slippage monitor + daily digest
- [ ] Task 8 — Cost fidelity (R-16)
- [ ] Task 9 — News blackout
