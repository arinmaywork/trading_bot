"""Wealth OS — Kuvera 'Portfolio Holding Statement' PDF importer.

Kuvera's export is a holdings *snapshot* (no transaction history, no ISINs),
so it feeds /portfolio, /networth and allocation — but not XIRR, tax, or
AMFI NAV auto-refresh (those need ISINs + transactions from a CAMS CAS).
The import summary says so explicitly.
"""
from __future__ import annotations

from datetime import date

from . import analytics, db

HEADER_MARKERS = ("scheme name", "folio number")
COLS = 10  # scheme, category, folio, invested, value, units, nav, avg_nav, ret%, xirr%


class KuveraImportError(Exception):
    pass


def _f(x) -> float | None:
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def import_kuvera_pdf(path: str) -> dict:
    try:
        import pdfplumber
    except ImportError as e:
        raise KuveraImportError("pdfplumber not installed on server") from e

    rows: list[list] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    for r in table:
                        if r and len(r) >= COLS:
                            rows.append([(c or "").strip() for c in r])
    except Exception as e:
        raise KuveraImportError(f"could not read PDF: {e}") from e

    header_seen = any(
        all(m in " ".join(x.lower().replace("\n", " ") for x in r if x)
            for m in HEADER_MARKERS)
        for r in rows
    )
    if not header_seen:
        raise KuveraImportError("not a Kuvera Portfolio Holding Statement")

    # carry ISINs over from any previous CAS import (matched by normalised
    # name) so AMFI NAV refresh and XIRR keep working after a Kuvera import
    known_isins = {analytics.norm_scheme(h["scheme"]): h["isin"]
                   for h in db.mf_holdings() if h["isin"]}

    today = date.today().isoformat()
    holdings, invested_total, isins_carried = [], 0.0, 0
    for r in rows:
        scheme, _cat, folio = r[0].replace("\n", " "), r[1], r[2]
        units, nav, value = _f(r[5]), _f(r[6]), _f(r[4])
        if not scheme or not folio or units is None or not folio[:1].isdigit():
            continue  # header/blank rows
        isin = known_isins.get(analytics.norm_scheme(scheme))
        isins_carried += 1 if isin else 0
        holdings.append({
            "folio": folio, "amc": scheme.split()[0], "scheme": scheme,
            "isin": isin, "units": units, "nav": nav,
            "nav_date": today, "value": value,
        })
        invested_total += _f(r[3]) or 0.0

    if not holdings:
        raise KuveraImportError("no holdings rows found in the statement")

    db.replace_mf_data(holdings, [], f"Kuvera snapshot {today}")
    return {
        "schemes": len(holdings),
        "value": sum(h["value"] or 0 for h in holdings),
        "invested": invested_total,
        "isins_carried": isins_carried,
    }
