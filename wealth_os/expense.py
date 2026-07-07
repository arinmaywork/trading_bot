"""Wealth OS — expenses: import from any expense-app CSV, judge the numbers
that matter for wealth (burn, savings rate, surplus truth, emergency cover).

Deliberately NOT a budgeting app: no per-category budgets or nagging. Your
expense app captures; this module answers four questions:
  1. What do I actually spend per month?         (3-month average burn)
  2. What is my savings rate?                    (needs /income set)
  3. Is my configured surplus honest?            (income − burn vs /surplus)
  4. Is my emergency fund sized right?           (6 × burn vs Emergency goal)

CSV parsing is column-flexible: finds date + amount columns by name; uses a
type/income column to skip income rows if present, else treats every row as
an expense. Import is idempotent.
"""
from __future__ import annotations

import csv
from datetime import date, datetime

from . import db, goals

_DATE_KEYS = ("date", "time", "day")
_AMT_KEYS = ("amount", "debit", "inr", "amt", "cost", "value")
_CAT_KEYS = ("category", "tag", "type of", "group")
_NOTE_KEYS = ("note", "description", "narration", "remark", "payee", "merchant")
_TYPE_KEYS = ("type", "income/expense", "flow", "transaction type")

_DATE_FMTS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y",
              "%d-%b-%Y", "%b %d, %Y", "%Y/%m/%d", "%d.%m.%Y")


def _find_col(cols: dict, keys) -> str | None:
    for k in keys:
        for low, orig in cols.items():
            if k in low:
                return orig
    return None


def _parse_date(s: str) -> str | None:
    s = (s or "").strip()[:20]
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s.split(" ")[0] if " " in s and ":" in s
                                     else s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:  # ISO with time
        return datetime.fromisoformat(s.replace("Z", "")).strftime("%Y-%m-%d")
    except ValueError:
        return None


def import_expenses_csv(path: str) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty CSV")
        cols = {c.lower().strip(): c for c in reader.fieldnames}
        t_col = _find_col(cols, _TYPE_KEYS)
        if t_col:  # don't let the type column masquerade as date/amount
            cols = {k: v for k, v in cols.items() if v != t_col}
        d_col = _find_col(cols, _DATE_KEYS)
        a_col = _find_col(cols, _AMT_KEYS)
        if not d_col or not a_col:
            raise ValueError("need date + amount columns — not an expense export")
        c_col = _find_col(cols, _CAT_KEYS)
        n_col = _find_col(cols, _NOTE_KEYS)

        rows, skipped_income = [], 0
        for r in reader:
            d = _parse_date(r.get(d_col, ""))
            try:
                amt = abs(float(str(r.get(a_col, "")).replace(",", "")
                                .replace("₹", "").strip()))
            except ValueError:
                continue
            if d is None or amt == 0:
                continue
            if t_col and "income" in (r.get(t_col) or "").lower():
                skipped_income += 1
                continue
            rows.append((d, amt, (r.get(c_col) or "").strip() if c_col else "",
                         (r.get(n_col) or "").strip()[:80] if n_col else ""))
    if not rows:
        raise ValueError("no expense rows recognised")
    with db.connect() as con:
        before = con.execute("SELECT COUNT(*) c FROM expenses").fetchone()["c"]
        con.executemany("INSERT OR IGNORE INTO expenses VALUES (?,?,?,?)", rows)
        after = con.execute("SELECT COUNT(*) c FROM expenses").fetchone()["c"]
    return {"parsed": len(rows), "added": after - before,
            "skipped_income": skipped_income,
            "from": min(r[0] for r in rows), "to": max(r[0] for r in rows)}


# ── Analysis ─────────────────────────────────────────────────────────

def monthly_burn() -> dict:
    """Average of the last 3 complete months + current month so far."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT substr(date,1,7) ym, SUM(amount) total FROM expenses"
            " GROUP BY ym ORDER BY ym DESC LIMIT 13").fetchall()
    if not rows:
        return {"avg3": None, "this_month": None, "months": 0}
    cur_ym = date.today().strftime("%Y-%m")
    complete = [r["total"] for r in rows if r["ym"] != cur_ym]
    this_m = next((r["total"] for r in rows if r["ym"] == cur_ym), 0.0)
    avg3 = sum(complete[:3]) / min(3, len(complete)) if complete else None
    return {"avg3": avg3, "this_month": this_m, "months": len(rows),
            "history": [(r["ym"], r["total"]) for r in rows[:6]]}


def top_categories(ym: str | None = None) -> list:
    ym = ym or date.today().strftime("%Y-%m")
    with db.connect() as con:
        return con.execute(
            "SELECT COALESCE(NULLIF(category,''),'uncategorised') c,"
            " SUM(amount) total FROM expenses WHERE substr(date,1,7)=?"
            " GROUP BY c ORDER BY total DESC LIMIT 5", (ym,)).fetchall()


def spend_card() -> str:
    b = monthly_burn()
    if not b["months"]:
        return ("No expense data yet — export CSV from your expense app and"
                " send it here. Re-send monthly; duplicates are ignored.")
    lines = ["<b>💸 Spending</b>\n"]
    if b["avg3"]:
        lines.append(f"Monthly burn (3-mo avg): <b>₹{b['avg3']:,.0f}</b>")
    lines.append(f"This month so far: ₹{b['this_month']:,.0f}")
    for ym, total in b.get("history", [])[1:4]:
        lines.append(f"   {ym}: ₹{total:,.0f}")
    cats = top_categories()
    if cats:
        lines.append("\n<b>This month's top categories</b>")
        lines.extend(f"• {r['c'][:24]}: ₹{r['total']:,.0f}" for r in cats)

    income = float(db.get_meta("monthly_income", "0") or 0)
    surplus = goals.monthly_surplus()
    if income and b["avg3"]:
        rate = (income - b["avg3"]) / income
        implied = income - b["avg3"]
        lines.append(f"\nIncome: ₹{income:,.0f} → <b>savings rate {rate:.0%}</b>"
                     f" (implied surplus ₹{implied:,.0f}/mo)")
        if abs(implied - surplus) > 0.15 * max(surplus, 1):
            lines.append(f"⚠️ /surplus is set to ₹{surplus:,.0f} but reality says"
                         f" ₹{implied:,.0f} — update it (/surplus {implied:,.0f})"
                         " so /plan and goal math stay honest.")
    elif b["avg3"]:
        lines.append("\nSet /income 150000 to get savings rate + surplus check.")

    if b["avg3"]:
        target = 6 * b["avg3"]
        em = next((g for g in db.list_goals()
                   if "emergency" in g["name"].lower()), None)
        if em and em["target_amount"] < target * 0.9:
            lines.append(f"\n🛟 Emergency goal is ₹{em['target_amount']:,.0f} but"
                         f" 6 months of your real burn is ₹{target:,.0f} —"
                         f" consider /goal add {em['name']} "
                         f"{target / 1e5:.1f}L 1 1 to resize it.")
        elif not em:
            lines.append(f"\n🛟 No Emergency goal yet. 6 months of burn ="
                         f" ₹{target:,.0f}: /goal add Emergency"
                         f" {target / 1e5:.1f}L 1 1")
    return "\n".join(lines)
