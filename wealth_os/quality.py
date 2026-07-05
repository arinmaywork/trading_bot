"""Wealth OS — fundamental quality veto for the swing sleeve.

Philosophy: fundamentals at retail are for AVOIDING garbage, not finding
winners. You maintain a quality screen on screener.in (e.g. ROE > 12,
Debt/Equity < 1, promoter pledging < 5%, interest coverage > 3), export it
as CSV, and send it to the bot. Swing picks NOT on that list get a ⚠️ veto
mark — the momentum engine proposes, the quality list disposes.

The CSV parser is deliberately flexible: it looks for an NSE-code/symbol
column, else falls back to the Name column.
"""
from __future__ import annotations

import csv
import json
from datetime import date

from . import db

META_KEY = "quality_universe"


def import_screener_csv(path: str) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty CSV")
        cols = {c.lower().strip(): c for c in reader.fieldnames}
        sym_col = next((cols[k] for k in cols
                        if "nse" in k or "symbol" in k or "ticker" in k), None)
        name_col = next((cols[k] for k in cols if "name" in k), None)
        if not sym_col and not name_col:
            raise ValueError("no symbol/name column found — is this a"
                             " screener.in export?")
        symbols, names = set(), set()
        for row in reader:
            if sym_col and (s := (row.get(sym_col) or "").strip().upper()):
                symbols.add(s)
            if name_col and (nm := (row.get(name_col) or "").strip()):
                names.add(nm.lower())
    if not symbols and not names:
        raise ValueError("no rows found in the screen export")
    db.set_meta(META_KEY, json.dumps({
        "symbols": sorted(symbols), "names": sorted(names),
        "date": date.today().isoformat(),
    }))
    return {"symbols": len(symbols), "names": len(names)}


def universe() -> dict | None:
    raw = db.get_meta(META_KEY)
    return json.loads(raw) if raw else None


def is_approved(symbol: str) -> bool | None:
    """True/False against the quality list; None if no list uploaded."""
    u = universe()
    if not u:
        return None
    if u["symbols"]:
        return symbol.upper() in set(u["symbols"])
    return any(symbol.lower() in n or n.startswith(symbol.lower())
               for n in u["names"])


# ── Automatic fundamentals via yfinance (no manual export needed) ────
# Rules: ROE ≥ 12%, positive profit margin, Debt/Equity < 1.5× (skipped
# for financials, where leverage is the business model). Missing data →
# abstain (None), never veto: absence of evidence isn't evidence of junk.

CACHE_KEY = "quality_auto_cache"
CACHE_DAYS = 30  # fundamentals move quarterly

ROE_MIN, PM_MIN, DE_MAX = 0.12, 0.0, 150.0  # yfinance D/E is in percent


def _fetch_info(symbol: str) -> dict:  # patchable in tests
    import yfinance as yf
    return yf.Ticker(f"{symbol}.NS").info or {}


def _judge(info: dict) -> tuple[bool | None, str]:
    roe, de = info.get("returnOnEquity"), info.get("debtToEquity")
    pm, sector = info.get("profitMargins"), info.get("sector") or ""
    if roe is None and pm is None:
        return None, "no data"
    reasons = []
    if roe is not None and roe < ROE_MIN:
        reasons.append(f"ROE {roe:.0%}")
    if pm is not None and pm <= PM_MIN:
        reasons.append("loss-making")
    if (de is not None and de > DE_MAX
            and "financial" not in sector.lower()):
        reasons.append(f"D/E {de / 100:.1f}x")
    if reasons:
        return False, ", ".join(reasons)
    return True, f"ROE {roe:.0%}" if roe is not None else "ok"


def auto_quality(symbols: list[str]) -> dict[str, dict]:
    """{symbol: {ok: bool|None, why: str}} — cached ~30 days per symbol."""
    cache = json.loads(db.get_meta(CACHE_KEY) or "{}")
    cutoff = date.today().toordinal() - CACHE_DAYS
    out, dirty = {}, False
    for sym in symbols:
        hit = cache.get(sym)
        if hit and hit.get("ord", 0) >= cutoff:
            out[sym] = {"ok": hit["ok"], "why": hit["why"]}
            continue
        try:
            ok, why = _judge(_fetch_info(sym))
        except Exception as e:
            ok, why = None, f"fetch failed ({str(e)[:40]})"
        out[sym] = {"ok": ok, "why": why}
        cache[sym] = {"ok": ok, "why": why, "ord": date.today().toordinal()}
        dirty = True
    if dirty:
        db.set_meta(CACHE_KEY, json.dumps(cache))
    return out
