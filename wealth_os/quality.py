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
