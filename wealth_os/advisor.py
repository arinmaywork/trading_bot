"""Wealth OS — /ask: AI advisory layer over YOUR portfolio data.

Design rules (deliberate):
- ADVISORY ONLY. The LLM explains, summarises, answers questions about your
  own data. It never generates buy/sell signals — those come only from the
  systematic engines (rebalance bands, tax rules, gated swing screen).
- Uses Gemini free tier via plain REST (no SDK). Set GEMINI_API_KEY in
  .env.sh; optionally GEMINI_MODEL (default gemini-2.5-flash).
"""
from __future__ import annotations

import json
import os

import aiohttp

from . import analytics, db, goals

SYSTEM = (
    "You are the analytical assistant inside Arinmay's personal portfolio "
    "manager (Indian markets: NSE equities + mutual funds, INR). Answer his "
    "question using the portfolio context provided. Rules: be concise and "
    "concrete; use ₹ and Indian tax rules (LTCG 12.5% >₹1.25L/FY, STCG 20% "
    "on equity); NEVER recommend specific stocks or funds to buy or sell — "
    "if asked, explain that trade recommendations come only from the "
    "systematic engines (/rebalance, /harvest, /screen) with approval "
    "buttons; you may explain concepts, analyse the existing portfolio, and "
    "discuss trade-offs. You are not a SEBI-registered advisor and should "
    "say so if giving anything close to advice."
)


def build_context() -> str:
    """Compact snapshot of everything the bot knows — fed to the LLM."""
    n = db.networth()
    alloc = analytics.allocation()
    x = analytics.mf_xirr()
    ctx = {
        "net_worth": round(n["total"]),
        "split": {"mf": round(n["mf"]), "equity": round(n["equity"]),
                  "cash": round(n["cash"])},
        "allocation_pct": {
            k: round(v / alloc["total"] * 100, 1)
            for k, v in alloc["buckets"].items() if alloc["total"] and v},
        "target_pct": {k: round(v * 100) for k, v in
                       analytics.target_allocation().items()},
        "mf_xirr_pct": round(x["total"] * 100, 1) if x["total"] is not None else None,
        "top_mf": [{"scheme": h["scheme"][:40], "value": round(h["value"] or 0)}
                   for h in db.mf_holdings()[:8]],
        "stocks": [{"sym": r["symbol"], "value": round(r["value"] or 0)}
                   for r in db.equity_holdings()[:10]],
        "goals": [{"name": g["name"], "funded_pct": round(g["funded_pct"] * 100),
                   "needs_sip": round(g["req_sip"])}
                  for g in goals.goal_status()],
        "monthly_surplus": round(goals.monthly_surplus()),
        "flags": analytics.flags()[:5],
        "trend": [{"d": r["date"], "v": round(r["total"])}
                  for r in db.snapshots_recent(10)],
    }
    return json.dumps(ctx, ensure_ascii=False)


async def ask(session: aiohttp.ClientSession, question: str) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return ("GEMINI_API_KEY not set. Get a free key at aistudio.google.com,"
                " add it to /opt/wealthos/.env.sh, restart the service.")
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    prompt = (f"{SYSTEM}\n\n--- PORTFOLIO CONTEXT (JSON) ---\n{build_context()}"
              f"\n\n--- QUESTION ---\n{question}")
    async with session.post(url, json={
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 800},
    }) as r:
        data = await r.json()
    if r.status != 200:
        err = (data.get("error") or {}).get("message", str(r.status))
        return (f"Gemini error: {err[:200]}\n"
                "(If the model name is stale, set GEMINI_MODEL in .env.sh)")
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        return "Gemini returned an empty response — try rephrasing."
