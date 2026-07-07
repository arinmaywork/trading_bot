# Wealth OS — Quota-Aware Build Roadmap

Session-sized tasks, each ending at a committable checkpoint. Same protocol as
ROADMAP_QUOTA_AWARE.md: if a usage-limit warning appears, stop at the next
checkpoint, commit WIP, resume later with the task's resume prompt.

**Architecture reference:** WEALTH_OS_PIVOT_PLAN.md
**Runtime target:** Oracle Cloud free tier (Ubuntu ARM), systemd, Telegram control.
**All user inputs (CAS PDFs, config) flow through Telegram — no SSH needed day-to-day.**

---

## T0 — Freeze + scaffold ✅ COMPLETE (2026-07-03)
- `PAPER_TRADE=true` in `.env.sh` — live trading frozen.
- `wealth_os/` package created; `data/cas/` for uploaded statements.
- This roadmap written.

## T1 — CAS import via Telegram + SQLite + portfolio commands ✅ COMPLETE (2026-07-03)
- `wealth_os/db.py` — SQLite store (mf_holdings, mf_transactions, equity_holdings, meta).
- `wealth_os/cas_import.py` — casparser wrapper; CAS PDF → db.
- `wealth_os/bot.py` — standalone long-polling Telegram bot (aiohttp, no new heavy deps):
  upload CAS PDF (caption = password, or reply when prompted), `/portfolio`, `/networth`, `/help`.
- `wealth_os/main.py` — entrypoint `python -m wealth_os.main`.
- `deploy/setup-oracle.sh` — one-shot Oracle VM setup + systemd unit.

**Resume prompt:** Resume Wealth OS T1. Read WEALTH_OS_ROADMAP.md T1 and wealth_os/*.py; check which files exist and compile; continue from the first missing piece.

## T2 — Zerodha holdings sync ✅ COMPLETE (2026-07-03)
- `wealth_os/kite_sync.py` — raw Kite REST via aiohttp (no kiteconnect/twisted dep);
  reuses legacy `.kite_token` cache format; holdings + margins → db.
- Bot: `/sync`, `/stocks`, `/login`, `/token <request_token>` (token msg auto-deleted);
  `/networth` shows MF + equity + broker cash. Auth failure auto-triggers login flow.
- Mocked sync + token-staleness tests pass.

## T3 — Daily refresh + digest ✅ COMPLETE (2026-07-03)
- `wealth_os/nav_fetch.py` — AMFI NAVAll.txt feed (free), ISIN-matched to held schemes.
- Equity prices refresh via Kite holdings during digest (no market-data subscription
  needed); if token stale, digest notes it instead of failing.
- 18:30 IST daily: NAV refresh → equity sync → networth snapshot → digest card
  (net worth, day change vs last snapshot, MF/equity/cash split, top equity movers
  by ₹ day-impact, active-SIP heads-up). `/digest` on demand, `/refresh` for NAVs.
- Parser + snapshot day-change verified against mocked AMFI feed.

## T4 — Analytics engine ✅ COMPLETE (2026-07-03)
- `wealth_os/analytics.py` — dependency-free XIRR (bisection; verified against
  hand-computed and Excel-style cases to <0.01%); per-scheme + total MF XIRR from
  CAS transactions; keyword scheme classifier (equity/debt/gold/intl/hybrid);
  allocation vs target (default 65/20/10/5, override via meta `target_alloc`);
  flags: scheme >25% of MF, AMC >40%, regular plans, stock >10% of net worth.
- `/health` renders the full card. Hybrid schemes split 65/35 equity/debt.
- Deferred: MF holdings-overlap analysis (needs scheme portfolio disclosures —
  fold into T8 research stack). Equity XIRR needs tradebook import → T6.

## T5 — Goals, SIP allocator, rebalance recommendations ✅ COMPLETE (2026-07-03)
- `wealth_os/goals.py` — required-SIP math (verified by FV round-trip), corpus
  waterfall by priority, monthly surplus allocator, 5/25 bands with
  new-money-first logic (sell only if gap > ~6 months of surplus, with tax warning).
- Commands: `/goal add Name 25L 5 [prio]`, `/goal del`, `/goals` (funded bars),
  `/plan`, `/rebalance`, `/recs`, `/surplus`, `/target 65 20 10 5`.
- Recommendations table + inline ✅/❌ buttons; decisions logged with timestamp;
  card edits in place to show APPROVED/REJECTED. Owner-locked callbacks.

## T6 — Tax module ✅ COMPLETE (2026-07-03)
- `wealth_os/tax.py` — FIFO lot engine (hand-verified) over MF CAS txns + equity
  tradebook; FY bucketing (Apr–Mar); LTCG 12.5% >₹1.25L, STCG 20%, non-equity → slab flag.
- `/tax`: FY realized gains + est. tax + equity XIRR (from tradebook).
- `/harvest`: LTCG-exemption harvesting (per-lot, fills remaining room) +
  tax-loss candidates with LT/ST label and 30-day rebuy warning.
- Zerodha Console tradebook CSV import via Telegram upload (idempotent on trade_id);
  unlocks equity XIRR deferred from T4.
- Caveats (documented in cards): no pre-2018 grandfathering, no split/bonus adj.

## T7 — Oracle hardening ✅ COMPLETE (2026-07-03)
- `wealth_os/backup.py` — sqlite online-backup API → gzip → data/backups/
  (keep 14); nightly 23:00 IST copy sent to your Telegram chat (survives VM loss;
  restore = gunzip → data/wealth.db). `/backup` on demand.
- Heartbeat file touched every poll loop; `wealthos-watchdog.timer` (5 min)
  restarts the service if inactive or heartbeat stale >10 min.
- Journald capped at 200M (log rotation). `/status`: uptime, db size, token
  freshness, last import/sync/backup, pending recs.

## T8 — Swing sleeve ✅ BUILT (2026-07-03) — 🔒 VALIDATION PENDING ON VM
- `wealth_os/swing.py` — risk-adjusted 12-1 momentum on Nifty 200 (NSE list at
  runtime, Nifty-50 fallback); filters: >200DMA + positive momentum; top 15
  equal-weight, exit rank>30 buffer; 25bps/side costs; 20% sleeve cap.
- Monthly-rebalance backtester (pandas/yfinance, lazy-imported — core bot stays
  lean). Mechanics verified on synthetic data; sandbox has no Yahoo/NSE access,
  so the REAL-data run happens on the VM: `/backtest` or `python -m wealth_os.swing`.
- Three-stage gate enforced in code: LOCKED → (real backtest Sharpe>1.0,
  |maxDD|<25%) → PAPER 90 days → LIVE recommendations with approval cards.
  `/screen` refuses to emit live picks before then.
- Quality-factor caveat: ROE/accruals need paid fundamentals; trend + low-vol
  ranking is the free-data approximation (documented in module docstring).

---

## T9 — Hardening & versatility pass ✅ (2026-07-05)
- Fixed: Telegram 4096-char limit (messages now chunk on line boundaries);
  generic handler surfaces real error type+message; /portfolio footer label.
- Fixed (data coherence): Kuvera imports now carry ISINs over from prior CAS
  imports via a scheme-name normaliser (`analytics.norm_scheme`), and XIRR /
  tax price lookups match across CAS↔Kuvera naming — a Kuvera snapshot no
  longer silently breaks NAV refresh, XIRR, or /harvest.
- New: send a `wealth_*.db.gz` backup to the bot → confirm-button restore
  (safety backup taken first); `/trend` net-worth sparkline from snapshots;
  equity XIRR (tradebook) shown in /health.

## T10 — Wealth pass: AI advisory + quality veto + behavioral guardrails ✅ (2026-07-05)
- `advisor.py` — `/ask <question>`: Gemini (free tier, REST) answers over a
  compact JSON snapshot of the user's own data. STRICTLY advisory — system
  prompt forbids buy/sell calls; signals remain systematic-only.
  Needs GEMINI_API_KEY in .env.sh (optional GEMINI_MODEL).
- `quality.py` — screener.in quality-screen CSV upload → veto list; /screen
  marks momentum picks outside it ⚠️ VETO. Fundamentals used to avoid
  garbage, not to pick winners (deliberate design).
- Digest guardrails: CRASH PROTOCOL (drawdown ≥10% from 1y peak → the
  pre-committed hold-and-buy message) + annual April SIP step-up nudge.
- CSV router: tradebook → quality screen → clear error.

## T11 — Expenses: burn rate, savings rate, surplus truth-check ✅ (2026-07-06)
- `expense.py` — column-flexible CSV import from any expense app (date/amount
  autodetect, income rows skipped, idempotent). Deliberately NOT a budgeting
  app: no category budgets/nagging.
- `/spend`: 3-mo avg burn, month-to-date, top categories, savings rate
  (needs `/income`), ⚠️ when configured /surplus drifts >15% from reality,
  emergency-goal resize suggestion (6 × real burn). Advisor context includes
  spending. CSV router order: tradebook → expenses → screener.

## Status: ALL TASKS BUILT (T0–T11). Remaining human steps:
1. VM: `git pull && pip install -r wealth_os/requirements.txt && sudo bash deploy/setup-oracle.sh`
2. Send CAS PDF + tradebook CSV to the bot; /sync; set /goals /surplus /target
3. Run /backtest → let the gate decide if the swing sleeve ever goes live

---

## Operating notes
- Legacy intraday bot stays on `main` untouched but frozen (PAPER_TRADE=true).
  Wealth OS runs as its own systemd service (`wealthos.service`); the two never run together.
- Secrets live only in `.env.sh` on the VM (gitignored). CAS PDFs land in `data/cas/` (gitignore it).
- Telegram is the single control surface: statements, tokens, approvals, reports.
