"""Wealth OS — SQLite persistence layer.

Single small DB at data/wealth.db. CAS imports are authoritative snapshots:
each import replaces mf_holdings and upserts transactions (idempotent).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "wealth.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS mf_holdings (
    folio      TEXT NOT NULL,
    amc        TEXT,
    scheme     TEXT NOT NULL,
    isin       TEXT,
    units      REAL,
    nav        REAL,
    nav_date   TEXT,
    value      REAL,
    updated_at TEXT,
    PRIMARY KEY (folio, scheme)
);
CREATE TABLE IF NOT EXISTS mf_transactions (
    folio         TEXT,
    scheme        TEXT,
    date          TEXT,
    description   TEXT,
    amount        REAL,
    units         REAL,
    nav           REAL,
    balance_units REAL,
    txn_type      TEXT,
    UNIQUE (folio, scheme, date, description, amount, units)
);
CREATE TABLE IF NOT EXISTS equity_holdings (
    symbol     TEXT PRIMARY KEY,
    qty        REAL,
    avg_price  REAL,
    ltp        REAL,
    value      REAL,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS networth_snapshots (
    date         TEXT PRIMARY KEY,
    mf_value     REAL,
    equity_value REAL,
    cash         REAL,
    total        REAL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_meta(key: str, value: str) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def get_meta(key: str, default: str | None = None) -> str | None:
    with connect() as con:
        row = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def replace_mf_data(holdings: list[dict], transactions: list[dict], statement_period: str) -> None:
    """CAS import: holdings snapshot replaces previous; transactions upsert (idempotent)."""
    now = _now()
    with connect() as con:
        con.execute("DELETE FROM mf_holdings")
        con.executemany(
            "INSERT OR REPLACE INTO mf_holdings VALUES (:folio,:amc,:scheme,:isin,:units,:nav,:nav_date,:value,'%s')" % now,
            holdings,
        )
        con.executemany(
            "INSERT OR IGNORE INTO mf_transactions VALUES "
            "(:folio,:scheme,:date,:description,:amount,:units,:nav,:balance_units,:txn_type)",
            transactions,
        )
        con.execute(
            "INSERT INTO meta VALUES ('last_cas_import',?),('cas_statement_period',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (now, statement_period),
        )


def mf_holdings() -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT * FROM mf_holdings WHERE units > 0.001 ORDER BY value DESC"
        ).fetchall()


def equity_holdings() -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT * FROM equity_holdings ORDER BY value DESC").fetchall()


def networth() -> dict:
    with connect() as con:
        mf = con.execute("SELECT COALESCE(SUM(value),0) v FROM mf_holdings WHERE units>0.001").fetchone()["v"]
        eq = con.execute("SELECT COALESCE(SUM(value),0) v FROM equity_holdings").fetchone()["v"]
    return {"mf": mf, "equity": eq, "total": mf + eq}
