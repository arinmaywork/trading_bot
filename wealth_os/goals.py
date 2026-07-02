"""Wealth OS — goal math, monthly SIP allocator, 5/25 rebalance bands.

Corpus is allocated to goals waterfall-style by priority (goal 1 fills first).
Rebalancing prefers directing new money to underweights (no tax event);
selling is only recommended when new money can't close the gap in ~6 months.
"""
from __future__ import annotations

from datetime import date, datetime

from . import analytics, db

DEFAULT_RETURN = 0.12  # planning assumption, overridable via meta 'expected_return'
DEFAULT_SURPLUS = 50_000.0


def expected_return() -> float:
    return float(db.get_meta("expected_return", str(DEFAULT_RETURN)))


def monthly_surplus() -> float:
    return float(db.get_meta("monthly_surplus", str(DEFAULT_SURPLUS)))


def required_sip(target: float, months: int, annual_r: float | None = None) -> float:
    """Monthly SIP (end-of-month) to reach `target` in `months`."""
    if months <= 0:
        return target
    r = (1 + (annual_r if annual_r is not None else expected_return())) ** (1 / 12) - 1
    if r == 0:
        return target / months
    return target * r / ((1 + r) ** months - 1)


def fv_lumpsum(amount: float, months: int, annual_r: float | None = None) -> float:
    r = (annual_r if annual_r is not None else expected_return())
    return amount * (1 + r) ** (months / 12)


def _months_left(target_date: str) -> int:
    try:
        td = datetime.strptime(target_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return 0
    days = (td - date.today()).days
    return max(0, round(days / 30.44))


def goal_status() -> list[dict]:
    """Waterfall current net worth across goals by priority."""
    corpus = db.networth()["total"]
    out = []
    for g in db.list_goals():
        alloc = min(corpus, g["target_amount"])
        corpus -= alloc
        months = _months_left(g["target_date"])
        # what today's allocated corpus grows to by the goal date
        fv_existing = fv_lumpsum(alloc, months)
        gap = max(0.0, g["target_amount"] - fv_existing)
        out.append({
            "name": g["name"], "target": g["target_amount"],
            "date": g["target_date"], "priority": g["priority"],
            "allocated": alloc, "months": months,
            "funded_pct": alloc / g["target_amount"] if g["target_amount"] else 0,
            "req_sip": required_sip(gap, months) if gap > 0 else 0.0,
        })
    return out


def monthly_plan() -> dict:
    """Split the monthly surplus across goals by priority."""
    surplus = monthly_surplus()
    remaining = surplus
    rows = []
    status = goal_status()
    for g in status:
        take = min(remaining, g["req_sip"])
        rows.append({**g, "sip": take})
        remaining -= take
    return {"surplus": surplus, "rows": rows, "unallocated": remaining}


# ── Rebalance (5/25 bands) ───────────────────────────────────────────

def rebalance_recos() -> list[dict]:
    """Bands: act when |drift| ≥ 5pp absolute OR ≥ 25% of target (relative)."""
    alloc = analytics.allocation()
    target = analytics.target_allocation()
    total = alloc["total"]
    if total <= 0:
        return []
    b = alloc["buckets"]
    view = {"equity": b["equity"] + b["international"], "debt": b["debt"],
            "gold": b["gold"], "cash": b["cash"]}
    recos = []
    surplus = monthly_surplus()
    for k, tgt in target.items():
        cur = view.get(k, 0) / total
        drift = cur - tgt
        if abs(drift) < 0.05 and (tgt == 0 or abs(drift) / tgt < 0.25):
            continue
        amt = abs(drift) * total
        if drift < 0:  # underweight → new money first
            months = amt / surplus if surplus else 99
            recos.append({
                "kind": "rebalance",
                "title": f"Underweight {k}: direct new money there",
                "detail": (f"{k.title()} is {cur:.0%} vs target {tgt:.0%} "
                           f"(gap ₹{amt:,.0f}). Route the next "
                           f"{max(1, round(months))} month(s) of surplus "
                           f"(₹{surplus:,.0f}/mo) into {k} before anything else. "
                           "No selling, no tax event."),
            })
        else:  # overweight
            months = amt / surplus if surplus else 99
            if months <= 6:
                detail = (f"{k.title()} is {cur:.0%} vs target {tgt:.0%} "
                          f"(₹{amt:,.0f} over). Fixable without selling: pause "
                          f"new money into {k} for ~{max(1, round(months))} month(s) "
                          "and direct surplus to underweights.")
            else:
                detail = (f"{k.title()} is {cur:.0%} vs target {tgt:.0%} "
                          f"(₹{amt:,.0f} over — too large for new money alone). "
                          f"Consider selling ₹{amt:,.0f} of {k} into underweights. "
                          "⚠️ Check tax impact before approving (T6 will automate).")
            recos.append({"kind": "rebalance",
                          "title": f"Overweight {k}", "detail": detail})
    return recos
