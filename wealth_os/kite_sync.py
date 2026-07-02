"""Wealth OS — Zerodha holdings sync via raw Kite REST (no kiteconnect dep).

Why raw REST: the kiteconnect package pulls in twisted+pyOpenSSL for its
WebSocket ticker, which Wealth OS never uses. Holdings/margins/session are
three plain HTTPS calls. Kite Connect Personal (free) covers all of them.

Token cache reuses the legacy .kite_token format:
    {"access_token": ..., "generated_date": "YYYY-MM-DD"}
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import aiohttp
import pytz

from . import db

IST = pytz.timezone("Asia/Kolkata")
TOKEN_CACHE = Path(__file__).resolve().parent.parent / ".kite_token"
API = "https://api.kite.trade"


class KiteAuthError(Exception):
    """Token missing/stale — caller should trigger the login flow."""


def _api_key() -> str:
    key = os.environ.get("KITE_API_KEY")
    if not key:
        raise KiteAuthError("KITE_API_KEY not set")
    return key


def login_url() -> str:
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={_api_key()}"


def cached_token() -> str | None:
    try:
        data = json.loads(TOKEN_CACHE.read_text())
    except (OSError, ValueError):
        return None
    gen = str(data.get("generated_date", ""))[:10]
    today = datetime.now(IST).strftime("%Y-%m-%d")
    return data.get("access_token") if gen == today else None


def save_token(access_token: str) -> None:
    TOKEN_CACHE.write_text(json.dumps({
        "access_token": access_token,
        "generated_date": datetime.now(IST).strftime("%Y-%m-%d"),
    }))
    try:
        TOKEN_CACHE.chmod(0o600)
    except OSError:
        pass


async def _get(session: aiohttp.ClientSession, path: str, token: str) -> dict:
    headers = {
        "X-Kite-Version": "3",
        "Authorization": f"token {_api_key()}:{token}",
    }
    async with session.get(f"{API}{path}", headers=headers) as r:
        data = await r.json()
    if r.status == 403 or data.get("error_type") == "TokenException":
        raise KiteAuthError(data.get("message", "token rejected"))
    if data.get("status") != "success":
        raise RuntimeError(f"Kite {path}: {data.get('message', r.status)}")
    return data["data"]


async def exchange_request_token(session: aiohttp.ClientSession,
                                 request_token: str) -> str:
    secret = os.environ.get("KITE_API_SECRET")
    if not secret:
        raise KiteAuthError("KITE_API_SECRET not set")
    key = _api_key()
    checksum = hashlib.sha256(f"{key}{request_token}{secret}".encode()).hexdigest()
    async with session.post(
        f"{API}/session/token",
        headers={"X-Kite-Version": "3"},
        data={"api_key": key, "request_token": request_token, "checksum": checksum},
    ) as r:
        data = await r.json()
    if data.get("status") != "success":
        raise KiteAuthError(data.get("message", "session exchange failed"))
    token = data["data"]["access_token"]
    save_token(token)
    return token


async def sync(session: aiohttp.ClientSession) -> dict:
    """Pull holdings + cash into the DB."""
    token = cached_token()
    if not token:
        raise KiteAuthError("No valid Kite token for today")

    raw = await _get(session, "/portfolio/holdings", token)
    margins = await _get(session, "/user/margins", token)

    rows = []
    for h in raw:
        qty = (h.get("quantity") or 0) + (h.get("t1_quantity") or 0)
        if qty <= 0:
            continue
        ltp = h.get("last_price") or 0
        rows.append({
            "symbol": h.get("tradingsymbol", "?"),
            "qty": qty,
            "avg_price": h.get("average_price") or 0,
            "ltp": ltp,
            "value": qty * ltp,
        })
    cash = float(((margins.get("equity") or {}).get("net")) or 0)
    db.replace_equity(rows, cash)
    return {
        "positions": len(rows),
        "value": sum(r["value"] for r in rows),
        "pnl": sum((r["ltp"] - r["avg_price"]) * r["qty"] for r in rows),
        "cash": cash,
    }
