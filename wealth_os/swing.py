"""Wealth OS — T8 swing sleeve: risk-adjusted momentum on Nifty 200.

Strategy (parameter-light, monthly):
  Universe   : Nifty 200 (NSE archives CSV; fallback: embedded Nifty 50)
  Filters    : price > 200DMA  AND  12-1 month momentum > 0
  Rank       : momentum_12_1 / annualised vol (risk-adjusted momentum)
  Portfolio  : top 15 equal-weight; exit only when rank falls below 30 (buffer
               cuts churn); monthly rebalance on first trading day
  Costs      : 25 bps per side (delivery STT both sides + charges + slippage)
  Cap        : sleeve ≤ 20% of net worth — enforced at recommendation time

GATE (non-negotiable, from WEALTH_OS_PIVOT_PLAN §4.4):
  /screen stays locked until `python -m wealth_os.swing` (or /backtest) shows
  Sharpe > 1.0 and maxDD < 25% on ≥5y of REAL EOD data, then 90 days of paper
  recommendations. Gate state lives in db meta 'swing_gate' / 'swing_paper_start'.

Heavy deps (pandas, yfinance) are imported lazily — only when screening or
backtesting, so the core bot stays lean.

Quality caveat: true quality factors (ROE, accruals) need paid fundamental
data; the trend filter + low-vol ranking is the free-data approximation.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime

from . import db

log = logging.getLogger("wealth_os.swing")

TOP_N, BUFFER_RANK = 15, 30
COST_PER_SIDE = 0.0025
SLEEVE_CAP = 0.20
GATE_SHARPE, GATE_MAXDD, PAPER_DAYS = 1.0, 0.25, 90

NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
FALLBACK_50 = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "ITC", "BHARTIARTL",
    "LT", "SBIN", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "BAJFINANCE", "M&M",
    "SUNPHARMA", "MARUTI", "HCLTECH", "TITAN", "TATAMOTORS", "NTPC",
    "ULTRACEMCO", "POWERGRID", "TATASTEEL", "ASIANPAINT", "COALINDIA",
    "BAJAJFINSV", "GRASIM", "ONGC", "NESTLEIND", "ADANIPORTS", "JSWSTEEL",
    "HINDALCO", "DRREDDY", "TECHM", "CIPLA", "WIPRO", "SBILIFE", "EICHERMOT",
    "BPCL", "DIVISLAB", "BRITANNIA", "TATACONSUM", "APOLLOHOSP", "HEROMOTOCO",
    "INDUSINDBK", "HDFCLIFE", "BAJAJ-AUTO", "SHRIRAMFIN", "LTIM", "TRENT",
]


# ── Universe + data (lazy heavy imports) ─────────────────────────────

def _universe() -> list[str]:
    try:
        import io
        import urllib.request
        req = urllib.request.Request(NIFTY200_URL,
                                     headers={"User-Agent": "Mozilla/5.0"})
        text = urllib.request.urlopen(req, timeout=15).read().decode()
        import csv as _csv
        syms = [r["Symbol"].strip() for r in _csv.DictReader(io.StringIO(text))
                if r.get("Symbol")]
        if len(syms) > 100:
            log.info("universe: Nifty %d from NSE", len(syms))
            return syms
    except Exception as e:
        log.warning("NSE universe fetch failed (%s) — fallback Nifty 50", e)
    return list(FALLBACK_50)


def _download_closes(symbols: list[str], years: int):
    """Daily adjusted closes DataFrame [date × symbol]. Requires yfinance."""
    import pandas as pd
    import yfinance as yf
    tickers = [s + ".NS" for s in symbols]
    raw = yf.download(tickers, period=f"{years}y", interval="1d",
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
    raw.columns = [str(c).replace(".NS", "") for c in raw.columns]
    return raw.dropna(axis=1, thresh=int(len(raw) * 0.7))


# ── Signal math (operates on a closes DataFrame) ─────────────────────

def _scores(closes, asof=None):
    """Risk-adjusted 12-1 momentum + trend filter at `asof` (or latest)."""
    df = closes.loc[:asof] if asof is not None else closes
    if len(df) < 260:
        return {}
    px = df.iloc[-1]
    mom = df.iloc[-22] / df.iloc[-252] - 1.0          # 12-1m momentum
    dma200 = df.iloc[-200:].mean()
    daily = df.iloc[-252:].pct_change()
    vol = daily.std() * (252 ** 0.5)
    out = {}
    for sym in df.columns:
        if any(x != x for x in (px[sym], mom[sym], dma200[sym], vol[sym])):
            continue
        if px[sym] > dma200[sym] and mom[sym] > 0 and vol[sym] > 0:
            out[sym] = mom[sym] / vol[sym]
    return out


# ── Backtest (monthly, buffer exit, costs) ───────────────────────────

def backtest(closes=None, years: int = 6) -> dict:
    import pandas as pd
    if closes is None:
        closes = _download_closes(_universe(), years)
    month_ends = closes.groupby(pd.Grouper(freq="ME")).tail(1).index
    holdings: set = set()
    rets, dates, turnover_total = [], [], 0.0
    for i in range(12, len(month_ends) - 1):
        t, t1 = month_ends[i], month_ends[i + 1]
        scores = _scores(closes, asof=t)
        if not scores:
            holdings = set()
            rets.append(0.0)
            dates.append(t1)
            continue
        ranked = sorted(scores, key=scores.get, reverse=True)
        keep = holdings & set(ranked[:BUFFER_RANK])
        new = [s for s in ranked[:TOP_N] if s not in keep]
        target = (list(keep) + new)[:TOP_N] if (keep or new) else []
        turnover = (len(set(target) ^ holdings) / max(len(target), 1)) if target else 0
        holdings = set(target)
        if not holdings:
            rets.append(0.0)
            dates.append(t1)
            continue
        px_t, px_t1 = closes.loc[t], closes.loc[t1]
        period = [(px_t1[s] / px_t[s] - 1.0) for s in holdings
                  if px_t[s] == px_t[s] and px_t1[s] == px_t1[s]]
        gross = sum(period) / len(period) if period else 0.0
        rets.append(gross - turnover * COST_PER_SIDE * 2)
        turnover_total += turnover
        dates.append(t1)

    s = pd.Series(rets, index=dates)
    if len(s) < 24:
        return {"error": f"only {len(s)} usable months — need ≥24"}
    n_yrs = len(s) / 12
    curve = (1 + s).cumprod()
    cagr = curve.iloc[-1] ** (1 / n_yrs) - 1
    sharpe = (s.mean() / s.std()) * (12 ** 0.5) if s.std() > 0 else 0.0
    maxdd = float((curve / curve.cummax() - 1).min())
    yearly = {str(y): float((1 + g).prod() - 1)
              for y, g in s.groupby(s.index.year)}
    return {"months": len(s), "cagr": float(cagr), "sharpe": float(sharpe),
            "maxdd": maxdd, "yearly": yearly,
            "avg_monthly_turnover": turnover_total / max(len(s), 1),
            "universe": len(closes.columns)}


def run_and_gate(years: int = 6) -> dict:
    """Real-data backtest → write gate to meta. Called by CLI / /backtest."""
    res = backtest(years=years)
    if "error" in res:
        return res
    passed = res["sharpe"] > GATE_SHARPE and abs(res["maxdd"]) < GATE_MAXDD
    gate = {**{k: res[k] for k in ("months", "cagr", "sharpe", "maxdd", "universe")},
            "passed": passed, "at": datetime.now().isoformat(timespec="seconds")}
    db.set_meta("swing_gate", json.dumps(gate))
    if passed and not db.get_meta("swing_paper_start"):
        db.set_meta("swing_paper_start", date.today().isoformat())
    return {**res, "passed": passed}


# ── Live screen (gated) ──────────────────────────────────────────────

def gate_status() -> dict:
    raw = db.get_meta("swing_gate")
    gate = json.loads(raw) if raw else None
    paper_start = db.get_meta("swing_paper_start")
    paper_days = ((date.today() - date.fromisoformat(paper_start)).days
                  if paper_start else None)
    return {"gate": gate, "paper_days": paper_days,
            "live_ok": bool(gate and gate.get("passed") and paper_days is not None
                            and paper_days >= PAPER_DAYS)}


def screen() -> list[dict]:
    closes = _download_closes(_universe(), 2)
    scores = _scores(closes)
    ranked = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
    sleeve = db.networth()["total"] * SLEEVE_CAP
    per_name = sleeve / TOP_N if ranked else 0
    px = closes.iloc[-1]
    return [{"symbol": s, "score": scores[s], "price": float(px[s]),
             "alloc": per_name, "qty": int(per_name / px[s]) if px[s] else 0}
            for s in ranked]


def report_card(res: dict) -> str:
    if "error" in res:
        return f"❌ Backtest: {res['error']}"
    ylines = "\n".join(f"   {y}: {r:+.1%}" for y, r in sorted(res["yearly"].items()))
    verdict = ("✅ GATE PASSED — 90-day paper phase starts now"
               if res.get("passed") else
               f"❌ GATE FAILED (need Sharpe>{GATE_SHARPE}, |maxDD|<{GATE_MAXDD:.0%})"
               " — sleeve stays locked; the core portfolio doesn't need it.")
    return (f"<b>🧪 Swing Backtest — {res['months']} months, real EOD,"
            f" {res['universe']} symbols</b>\n\n"
            f"CAGR: <b>{res['cagr']:+.1%}</b> | Sharpe: <b>{res['sharpe']:.2f}</b>"
            f" | MaxDD: {res['maxdd']:.1%}\n"
            f"Avg monthly turnover: {res['avg_monthly_turnover']:.0%}\n\n"
            f"<b>Per year</b>\n{ylines}\n\n{verdict}\n\n"
            "<i>Costs 25bps/side included. Past ≠ future.</i>")


if __name__ == "__main__":  # python -m wealth_os.swing → run validation on VM
    import pprint
    pprint.pprint(run_and_gate())
