"""Wealth OS — CAMS/KFintech CAS PDF importer (via casparser)."""
from __future__ import annotations

import casparser

from . import db


class CASImportError(Exception):
    pass


def import_cas(pdf_path: str, password: str) -> dict:
    """Parse a CAS PDF and load it into the DB. Returns a summary dict."""
    try:
        data = casparser.read_cas_pdf(pdf_path, password, output="dict")
    except Exception as e:  # bad password, wrong file, etc.
        raise CASImportError(str(e)) from e

    # casparser 0.8.x returns a pydantic model despite output="dict" — normalise
    if not isinstance(data, dict):
        if hasattr(data, "model_dump"):      # pydantic v2
            data = data.model_dump()
        elif hasattr(data, "dict"):          # pydantic v1
            data = data.dict()
        else:
            raise CASImportError(f"unexpected casparser output: {type(data).__name__}")
    if "folios" not in data:
        raise CASImportError(
            "This looks like an NSDL/CDSL e-CAS (demat statement). Send the "
            "CAMS/KFintech *mutual fund* CAS instead (camsonline.com → CAS).")

    holdings: list[dict] = []
    transactions: list[dict] = []

    for folio in data.get("folios", []):
        folio_no = str(folio.get("folio", "")).strip()
        amc = folio.get("amc", "")
        for s in folio.get("schemes", []):
            val = s.get("valuation") or {}
            holdings.append({
                "folio": folio_no,
                "amc": amc,
                "scheme": s.get("scheme", ""),
                "isin": s.get("isin"),
                "units": _f(s.get("close")),
                "nav": _f(val.get("nav")),
                "nav_date": str(val.get("date", "")),
                "value": _f(val.get("value")),
            })
            for t in s.get("transactions", []):
                transactions.append({
                    "folio": folio_no,
                    "scheme": s.get("scheme", ""),
                    "date": str(t.get("date", "")),
                    "description": t.get("description", ""),
                    "amount": _f(t.get("amount")),
                    "units": _f(t.get("units")),
                    "nav": _f(t.get("nav")),
                    "balance_units": _f(t.get("balance")),
                    "txn_type": t.get("type", ""),
                })

    period = data.get("statement_period") or {}
    period_str = f"{period.get('from', '?')} → {period.get('to', '?')}"
    db.replace_mf_data(holdings, transactions, period_str)

    active = [h for h in holdings if (h["units"] or 0) > 0.001]
    return {
        "folios": len(data.get("folios", [])),
        "schemes_total": len(holdings),
        "schemes_active": len(active),
        "transactions": len(transactions),
        "value": sum(h["value"] or 0 for h in active),
        "period": period_str,
    }


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
