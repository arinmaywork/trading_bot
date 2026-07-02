"""Wealth OS — daily MF NAV refresh from the free AMFI feed.

AMFI publishes all-India NAVs daily (~23:00 IST) as a semicolon-separated
text file. No API key, no cost. We match on ISIN against held schemes.
Line format:
  Scheme Code;ISIN Div Payout/ISIN Growth;ISIN Div Reinvestment;Scheme Name;NAV;Date
"""
from __future__ import annotations

import logging
from datetime import datetime

import aiohttp

from . import db

log = logging.getLogger("wealth_os.nav")

AMFI_URL = "https://www.amfiindia.com/spages/NAVAll.txt"


async def refresh_mf_navs(session: aiohttp.ClientSession) -> dict:
    """Update nav/value for all held schemes. Returns summary."""
    held = {r["isin"] for r in db.mf_holdings() if r["isin"]}
    if not held:
        return {"updated": 0, "held": 0, "nav_date": None}

    async with session.get(AMFI_URL) as r:
        text = await r.text()

    updates: list[dict] = []
    for line in text.splitlines():
        parts = line.split(";")
        if len(parts) != 6:
            continue
        _, isin1, isin2, _, nav_s, date_s = (p.strip() for p in parts)
        isin = isin1 if isin1 in held else (isin2 if isin2 in held else None)
        if not isin:
            continue
        try:
            nav = float(nav_s)
            nav_date = datetime.strptime(date_s, "%d-%b-%Y").strftime("%Y-%m-%d")
        except ValueError:
            continue
        updates.append({"isin": isin, "nav": nav, "nav_date": nav_date})

    n = db.update_mf_navs(updates)
    nav_date = max((u["nav_date"] for u in updates), default=None)
    log.info("AMFI refresh: %d/%d schemes updated (as of %s)", n, len(held), nav_date)
    return {"updated": n, "held": len(held), "nav_date": nav_date}
