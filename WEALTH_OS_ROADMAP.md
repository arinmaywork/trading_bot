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

## T2 — Zerodha holdings sync
- `wealth_os/kite_sync.py` — reuse `.kite_token` flow from old bot; pull `kite.holdings()`
  + `kite.positions()` into equity_holdings; `/sync` command; `/login` `/token` reuse.
- `/networth` shows combined MF + equity + cash (kite.margins).
- Checkpoints: (1) holdings fetch works with cached token, (2) db populated, (3) /networth combined.

**Resume prompt:** Resume Wealth OS T2 (Kite sync). Read WEALTH_OS_ROADMAP.md T2, wealth_os/db.py, and the TokenManager in main.py (legacy). Build kite_sync.py; wire /sync into bot.py.

## T3 — Daily refresh + digest
- `wealth_os/nav_fetch.py` — AMFI daily NAV feed (free) for MF; yfinance EOD for stocks.
- Scheduled daily job (18:30 IST): refresh values, snapshot networth to db, Telegram digest
  (net worth, day change, top movers, upcoming SIPs).
- Checkpoints: (1) NAV refresh for held ISINs, (2) snapshot table, (3) digest card.

**Resume prompt:** Resume Wealth OS T3 (daily refresh). Read WEALTH_OS_ROADMAP.md T3 + db.py. Build nav_fetch.py and the digest scheduler in bot.py.

## T4 — Analytics engine
- `wealth_os/analytics.py` — XIRR (per scheme, per asset class, total) from mf_transactions;
  asset allocation vs target; MF overlap (via scheme portfolio disclosures); expense-ratio audit
  (regular vs direct detection); concentration flags.
- `/health` command → full report card; monthly deep report.
- Checkpoints: (1) XIRR matches hand-check, (2) allocation vs target, (3) /health renders.

**Resume prompt:** Resume Wealth OS T4 (analytics). Read WEALTH_OS_ROADMAP.md T4 + db.py schema. Build analytics.py; verify XIRR against a hand-computed folio.

## T5 — Goals, SIP allocator, rebalance recommendations
- `wealth_os/goals.py` — goals table, required-SIP math, surplus allocator (₹50k+/mo),
  5/25 rebalance bands, new-money-first rebalancing.
- Recommendation cards with APPROVE/REJECT inline buttons; approved actions logged.
  (Execution stays manual or via Kite order placement — free personal API.)
- Checkpoints: (1) goal math, (2) drift detection, (3) approval workflow round-trip.

**Resume prompt:** Resume Wealth OS T5 (goals). Read WEALTH_OS_ROADMAP.md T5 + analytics.py. Build goals.py + inline-button approval flow in bot.py.

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
