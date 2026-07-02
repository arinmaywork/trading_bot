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

## T6 — Tax module
- LTCG ₹1.25L/yr harvest recommendations; tax-loss harvesting (Dec–Mar); STCG/LTCG
  timing check on every sell recommendation; FY summary for ITR.
- Checkpoints: (1) FIFO lot engine on mf_transactions + equity, (2) harvest candidates, (3) FY report.

**Resume prompt:** Resume Wealth OS T6 (tax). Read WEALTH_OS_ROADMAP.md T6. Build tax.py using FIFO lots from db transactions.

## T7 — Oracle hardening
- Nightly SQLite backup to Oracle Object Storage (or GitHub private artifact);
  watchdog systemd timer; log rotation; `/backup` `/status` commands.

**Resume prompt:** Resume Wealth OS T7 (hardening). Read deploy/setup-oracle.sh; add backup timer + watchdog.

## T8 — Swing sleeve (GATED)
- Nifty-200 momentum+quality monthly screen; validate through existing `backtest/`
  walk-forward on ≥5y real EOD data; paper recommendations 3 months; hard 20% cap.
- Does not start until T1–T5 are live and the backtest gate passes.

**Resume prompt:** Resume Wealth OS T8 (swing sleeve). Read WEALTH_OS_PIVOT_PLAN.md §4.4 + backtest/engine.py. Build the screen; run walk-forward on real data first.

---

## Operating notes
- Legacy intraday bot stays on `main` untouched but frozen (PAPER_TRADE=true).
  Wealth OS runs as its own systemd service (`wealthos.service`); the two never run together.
- Secrets live only in `.env.sh` on the VM (gitignored). CAS PDFs land in `data/cas/` (gitignore it).
- Telegram is the single control surface: statements, tokens, approvals, reports.
