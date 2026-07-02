"""Wealth OS — FIFO lot engine, capital-gains tax, LTCG harvesting.

Rules implemented (FY 2024-25 onward, equity-oriented assets):
  LT threshold 365 days · LTCG 12.5% above ₹1.25L/FY exemption · STCG 20%.
Non-equity buckets (debt/gold/intl) are reported but taxed at slab — flagged,
not computed. Caveats: no pre-2018 grandfathering, no split/bonus adjustment.

Equity trade history comes from the Zerodha Console tradebook CSV
(uploaded via Telegram). MF history comes from CAS transactions.
"""
from __future__ import annotations

import csv
import html
from datetime import date, datetime

from . import analytics, db

LT_DAYS = 365
LTCG_RATE, STCG_RATE, LTCG_EXEMPT = 0.125, 0.20, 125_000.0


def fy_label(d: date) -> str:
    y = d.year if d.month >= 4 else d.year - 1
    return f"{y}-{(y + 1) % 100:02d}"


def current_fy() -> str:
    return fy_label(date.today())


def _d(s: str) -> date | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


# ── FIFO engine ──────────────────────────────────────────────────────

def fifo(events: list[tuple[date, float, float]]) -> tuple[list, list]:
    """events: (date, units [+buy/-sell], price) sorted here by date.
    Returns (open_lots [{date, units, price}], realized [{...}])."""
    lots: list[dict] = []
    realized: list[dict] = []
    for d, units, price in sorted(events, key=lambda e: e[0]):
        if units > 0:
            lots.append({"date": d, "units": units, "price": price})
            continue
        remaining = -units
        while remaining > 1e-6 and lots:
            lot = lots[0]
            take = min(lot["units"], remaining)
            realized.append({
                "sell_date": d, "buy_date": lot["date"], "units": take,
                "buy_price": lot["price"], "sell_price": price,
                "gain": take * (price - lot["price"]),
                "days": (d - lot["date"]).days,
            })
            lot["units"] -= take
            remaining -= take
            if lot["units"] <= 1e-6:
                lots.pop(0)
    return lots, realized


def _mf_events() -> dict[str, list]:
    out: dict[str, list] = {}
    for t in db.mf_transactions_all():
        units, nav, d = t["units"], t["nav"], _d(t["date"])
        if not units or d is None:
            continue
        price = nav if nav else (abs(t["amount"] or 0) / abs(units))
        out.setdefault(t["scheme"], []).append((d, units, price))
    return out


def _equity_events() -> dict[str, list]:
    out: dict[str, list] = {}
    for t in db.equity_trades_all():
        d = _d(t["trade_date"])
        if d is None or not t["quantity"]:
            continue
        sign = 1 if str(t["trade_type"]).lower().startswith("b") else -1
        out.setdefault(t["symbol"], []).append((d, sign * t["quantity"], t["price"]))
    return out


def _is_equity_taxed(name: str, is_stock: bool) -> bool:
    if is_stock:
        return True
    return analytics.classify(name) in ("equity", "hybrid")


# ── Realized gains per FY ────────────────────────────────────────────

def realized_gains(fy: str | None = None) -> dict:
    fy = fy or current_fy()
    agg = {"ltcg": 0.0, "stcg": 0.0, "slab_gain": 0.0, "rows": []}
    for is_stock, events_by_name in ((False, _mf_events()), (True, _equity_events())):
        for name, events in events_by_name.items():
            _, realized = fifo(events)
            for r in realized:
                if fy_label(r["sell_date"]) != fy:
                    continue
                eq = _is_equity_taxed(name, is_stock)
                bucket = ("ltcg" if r["days"] >= LT_DAYS else "stcg") if eq else "slab_gain"
                agg[bucket] += r["gain"]
                agg["rows"].append({**r, "name": name, "bucket": bucket})
    return agg


# ── Unrealized lots + harvesting ─────────────────────────────────────

def _current_prices() -> dict[str, float]:
    prices = {}
    for h in db.mf_holdings():
        if h["nav"]:
            prices[h["scheme"]] = h["nav"]
    for r in db.equity_holdings():
        if r["ltp"]:
            prices[r["symbol"]] = r["ltp"]
    return prices


def unrealized_lots() -> list[dict]:
    """Open FIFO lots with unrealized gain, LT/ST split, per asset."""
    prices = _current_prices()
    today = date.today()
    out = []
    for is_stock, events_by_name in ((False, _mf_events()), (True, _equity_events())):
        for name, events in events_by_name.items():
            price = prices.get(name)
            if not price:
                continue
            open_lots, _ = fifo(events)
            for lot in open_lots:
                days = (today - lot["date"]).days
                out.append({
                    "name": name, "is_stock": is_stock,
                    "units": lot["units"], "buy_price": lot["price"],
                    "price": price, "days": days,
                    "lt": days >= LT_DAYS,
                    "equity_taxed": _is_equity_taxed(name, is_stock),
                    "gain": lot["units"] * (price - lot["price"]),
                })
    return out


def harvest_report() -> dict:
    """LTCG-exemption harvesting + tax-loss candidates for the current FY."""
    fy = current_fy()
    realized = realized_gains(fy)
    room = max(0.0, LTCG_EXEMPT - realized["ltcg"])
    lots = unrealized_lots()

    gains, filled = [], 0.0
    for lot in sorted((l for l in lots if l["lt"] and l["equity_taxed"] and l["gain"] > 0),
                      key=lambda l: l["gain"], reverse=True):
        if filled >= room:
            break
        take_gain = min(lot["gain"], room - filled)
        units = lot["units"] * (take_gain / lot["gain"])
        gains.append({**lot, "harvest_units": units, "harvest_gain": take_gain})
        filled += take_gain

    losses = sorted((l for l in lots if l["gain"] < 0), key=lambda l: l["gain"])
    return {"fy": fy, "realized_ltcg": realized["ltcg"], "realized_stcg": realized["stcg"],
            "room": room, "gain_candidates": gains, "filled": filled,
            "loss_candidates": losses[:8]}


# ── Tradebook CSV import ─────────────────────────────────────────────

def import_tradebook_csv(path: str) -> dict:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            if not r.get("trade_id") or not r.get("symbol"):
                continue
            rows.append({
                "trade_id": r["trade_id"], "symbol": r["symbol"],
                "isin": r.get("isin", ""), "trade_date": r.get("trade_date", "")[:10],
                "trade_type": r.get("trade_type", "").lower(),
                "quantity": float(r.get("quantity") or 0),
                "price": float(r.get("price") or 0),
            })
    if not rows:
        raise ValueError("No trades found — is this a Zerodha Console tradebook CSV?")
    added = db.insert_equity_trades(rows)
    return {"parsed": len(rows), "added": added}


def equity_xirr() -> float | None:
    """XIRR over the tradebook + current equity value (needs full history)."""
    flows = []
    for t in db.equity_trades_all():
        d = _d(t["trade_date"])
        if d is None:
            continue
        amt = (t["quantity"] or 0) * (t["price"] or 0)
        buy = str(t["trade_type"]).lower().startswith("b")
        flows.append((d, -amt if buy else amt))
    if not flows:
        return None
    val = db.networth()["equity"]
    if val > 0:
        flows.append((date.today(), val))
    return analytics.xirr(flows)


# ── Cards ────────────────────────────────────────────────────────────

def tax_card() -> str:
    fy = current_fy()
    r = realized_gains(fy)
    lines = [f"<b>🧾 Tax — FY {fy}</b>\n",
             f"Realized LTCG: ₹{r['ltcg']:,.0f} (exempt up to ₹{LTCG_EXEMPT:,.0f},"
             f" then {LTCG_RATE:.1%})",
             f"Realized STCG: ₹{r['stcg']:,.0f} (@ {STCG_RATE:.0%})"]
    ltcg_tax = max(0.0, r["ltcg"] - LTCG_EXEMPT) * LTCG_RATE
    stcg_tax = max(0.0, r["stcg"]) * STCG_RATE
    lines.append(f"Est. tax due: <b>₹{ltcg_tax + stcg_tax:,.0f}</b>")
    if r["slab_gain"]:
        lines.append(f"Non-equity gains (slab rate): ₹{r['slab_gain']:,.0f}")
    x = equity_xirr()
    if x is not None:
        lines.append(f"\nEquity XIRR (tradebook): <b>{x:+.1%}</b>")
    lines.append(f"\n{len(r['rows'])} sale(s) this FY. /harvest for opportunities.")
    lines.append("<i>Estimates only — confirm with your CA. No grandfathering/"
                 "split adjustments.</i>")
    return "\n".join(lines)


def harvest_card() -> str:
    h = harvest_report()
    lines = [f"<b>🌾 Harvest — FY {h['fy']}</b>\n",
             f"LTCG exemption room: <b>₹{h['room']:,.0f}</b>"
             f" (used ₹{h['realized_ltcg']:,.0f} of ₹{LTCG_EXEMPT:,.0f})"]
    if h["gain_candidates"]:
        lines.append("\n<b>Harvest LT gains tax-free</b> (sell + rebuy resets cost basis)")
        for c in h["gain_candidates"][:6]:
            lines.append(f"• {html.escape(c['name'][:40])}: sell {c['harvest_units']:,.1f}u"
                         f" → book ₹{c['harvest_gain']:,.0f} gain, ₹0 tax")
        lines.append(f"Total harvestable: ₹{h['filled']:,.0f}"
                     f" → saves ₹{h['filled'] * LTCG_RATE:,.0f} future tax")
    else:
        lines.append("No LT gain lots to harvest right now.")
    if h["loss_candidates"]:
        lines.append("\n<b>Loss harvesting</b> (offset gains; mind 30-day rebuy gap)")
        for c in h["loss_candidates"][:5]:
            term = "LT" if c["lt"] else "ST"
            lines.append(f"• {html.escape(c['name'][:40])}: {term} loss ₹{-c['gain']:,.0f}")
    lines.append("\n<i>Execute manually or ask me to prep orders — always approve first.</i>")
    return "\n".join(lines)
