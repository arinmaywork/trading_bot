"""Wealth OS — analytics engine: XIRR, allocation, health flags.

XIRR: bisection on annualised rate, no external deps. Cashflow convention:
money you put in = negative, money you get back (or still hold) = positive.
Sign is derived from transaction units (units>0 → purchase → outflow).

Equity XIRR needs the Zerodha tradebook (not in holdings API) — deferred to
the tax task (T6); until then equity shows simple unrealised return.
"""
from __future__ import annotations

import html
import json
from datetime import date, datetime

from . import db

# ── XIRR ─────────────────────────────────────────────────────────────

def xirr(flows: list[tuple[date, float]]) -> float | None:
    """Annualised IRR via bisection. Returns None if undefined."""
    if len(flows) < 2:
        return None
    flows = sorted(flows)
    if not (any(cf < 0 for _, cf in flows) and any(cf > 0 for _, cf in flows)):
        return None
    t0 = flows[0][0]

    def npv(rate: float) -> float:
        return sum(cf / (1.0 + rate) ** ((d - t0).days / 365.0) for d, cf in flows)

    lo, hi = -0.9999, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            break
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def _parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _txn_flows(rows) -> list[tuple[date, float]]:
    """CAS transactions → signed cashflows. Skips unit-less rows (STT etc.)."""
    flows = []
    for t in rows:
        units, amount = t["units"], t["amount"]
        if not units or not amount:
            continue
        d = _parse_date(t["date"])
        if d is None:
            continue
        # units>0: bought (cash out, negative); units<0: redeemed (cash in)
        flows.append((d, -abs(amount) if units > 0 else abs(amount)))
    return flows


def mf_xirr() -> dict:
    """Total + per-scheme XIRR from CAS transactions and current values."""
    today = date.today()
    txns = db.mf_transactions_all()
    values = {}  # scheme → current value
    for h in db.mf_holdings():
        values[h["scheme"]] = values.get(h["scheme"], 0) + (h["value"] or 0)

    by_scheme: dict[str, list] = {}
    for t in txns:
        by_scheme.setdefault(t["scheme"], []).append(t)

    per_scheme, all_flows = [], []
    for scheme, rows in by_scheme.items():
        flows = _txn_flows(rows)
        val = values.get(scheme, 0)
        if val > 0:
            flows.append((today, val))
        all_flows.extend(flows)
        invested = sum(-cf for _, cf in flows if cf < 0)
        if invested < 1000:  # skip noise
            continue
        r = xirr(flows)
        if r is not None:
            per_scheme.append({"scheme": scheme, "xirr": r,
                               "value": val, "invested": invested})
    per_scheme.sort(key=lambda x: x["xirr"], reverse=True)
    return {"total": xirr(all_flows), "schemes": per_scheme}


# ── Asset allocation ─────────────────────────────────────────────────

# checked in order — first keyword hit wins
_BUCKETS: list[tuple[str, tuple[str, ...]]] = [
    ("international", ("nasdaq", "s&p", "us equity", "global", "international",
                       "world", "feeder", "fof us", "china", "japan")),
    ("gold", ("gold", "silver", "commodit")),
    ("debt", ("liquid", "overnight", "money market", "ultra short", "low duration",
              "short duration", "gilt", "bond", "debt", "corporate", "credit risk",
              "banking & psu", "banking and psu", "floater", "floating",
              "arbitrage", "fixed maturity", "medium duration", "dynamic bond")),
    ("hybrid", ("hybrid", "balanced", "multi asset", "equity savings",
                "aggressive", "conservative", "asset allocator")),
    ("equity", ("nifty", "sensex", "index", "flexi cap", "flexicap", "large cap",
                "largecap", "mid cap", "midcap", "small cap", "smallcap", "elss",
                "tax saver", "focused", "value", "contra", "dividend yield",
                "multi cap", "multicap", "large & mid", "equity", "bluechip",
                "momentum", "quality", "alpha", "next 50", "midcap 150")),
]

DEFAULT_TARGET = {"equity": 0.65, "debt": 0.20, "gold": 0.10, "cash": 0.05}


def classify(scheme_name: str) -> str:
    low = scheme_name.lower()
    for bucket, keys in _BUCKETS:
        if any(k in low for k in keys):
            return bucket
    return "other"


def allocation() -> dict:
    """Bucket → value. Direct stocks count as equity; hybrid split 65/35."""
    buckets = {"equity": 0.0, "debt": 0.0, "gold": 0.0,
               "international": 0.0, "cash": 0.0, "other": 0.0}
    for h in db.mf_holdings():
        b = classify(h["scheme"] or "")
        v = h["value"] or 0
        if b == "hybrid":
            buckets["equity"] += 0.65 * v
            buckets["debt"] += 0.35 * v
        else:
            buckets[b] += v
    n = db.networth()
    buckets["equity"] += n["equity"]
    buckets["cash"] += n["cash"]
    total = sum(buckets.values())
    return {"buckets": buckets, "total": total}


def target_allocation() -> dict:
    raw = db.get_meta("target_alloc")
    if raw:
        try:
            return json.loads(raw)
        except ValueError:
            pass
    return dict(DEFAULT_TARGET)


# ── Flags ────────────────────────────────────────────────────────────

def flags() -> list[str]:
    out = []
    mfs = db.mf_holdings()
    mf_total = sum(h["value"] or 0 for h in mfs)
    if mf_total > 0:
        for h in mfs:
            pct = (h["value"] or 0) / mf_total
            if pct > 0.25:
                out.append(f"⚠️ {h['scheme'][:40]} is {pct:.0%} of your MF book")
        amc: dict[str, float] = {}
        for h in mfs:
            amc[h["amc"] or "?"] = amc.get(h["amc"] or "?", 0) + (h["value"] or 0)
        for name, v in amc.items():
            if v / mf_total > 0.40:
                out.append(f"⚠️ {v / mf_total:.0%} of MF money with one AMC ({name[:30]})")
        regular = [h for h in mfs if "direct" not in (h["scheme"] or "").lower()]
        if regular:
            v = sum(h["value"] or 0 for h in regular)
            out.append(f"💸 {len(regular)} regular plan(s) worth ₹{v:,.0f} — "
                       "direct plans save ~0.5-1%/yr")
    eqs = db.equity_holdings()
    eq_total = sum(r["value"] or 0 for r in eqs)
    if eq_total > 0:
        n = db.networth()
        for r in eqs:
            if (r["value"] or 0) / max(n["total"], 1) > 0.10:
                out.append(f"⚠️ {r['symbol']} is "
                           f"{(r['value'] or 0) / n['total']:.0%} of total net worth")
    return out


# ── Health card ──────────────────────────────────────────────────────

def health_card() -> str:
    n = db.networth()
    if n["total"] <= 0:
        return "No data yet — import a CAS PDF and run /sync first."
    x = mf_xirr()
    alloc = allocation()
    target = target_allocation()

    lines = ["<b>💊 Portfolio Health</b>\n",
             f"Net worth: <b>₹{n['total']:,.0f}</b>"]
    if x["total"] is not None:
        lines.append(f"MF portfolio XIRR: <b>{x['total']:+.1%}</b>")

    lines.append("\n<b>Allocation vs target</b>")
    b, total = alloc["buckets"], alloc["total"]
    merged_eq = b["equity"] + b["international"]
    view = {"equity": merged_eq, "debt": b["debt"], "gold": b["gold"],
            "cash": b["cash"]}
    for k, v in view.items():
        cur = v / total if total else 0
        tgt = target.get(k, 0)
        drift = cur - tgt
        mark = "✅" if abs(drift) < 0.05 else "🔶"
        lines.append(f"{mark} {k.title()}: {cur:.0%} (target {tgt:.0%}, "
                     f"{drift:+.0%})")
    if b["other"] > 0:
        lines.append(f"❓ Unclassified: ₹{b['other']:,.0f}")
    if b["international"] > 0:
        lines.append(f"   (incl. international {b['international'] / total:.0%})")

    if x["schemes"]:
        best, worst = x["schemes"][0], x["schemes"][-1]
        lines.append("\n<b>Schemes by XIRR</b>")
        lines.append(f"🏆 {html.escape(best['scheme'][:42])}: {best['xirr']:+.1%}")
        if worst is not best:
            lines.append(f"🐌 {html.escape(worst['scheme'][:42])}: {worst['xirr']:+.1%}")

    fl = flags()
    if fl:
        lines.append("\n<b>Flags</b>")
        lines.extend(html.escape(f) if "<" in f else f for f in fl)
    else:
        lines.append("\n✅ No concentration or cost flags")
    lines.append("\n<i>Equity XIRR needs tradebook import — coming in T6.</i>")
    return "\n".join(lines)
