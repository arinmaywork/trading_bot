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
CREATE TABLE IF NOT EXISTS equity_trades (
    trade_id   TEXT PRIMARY KEY,
    symbol     TEXT,
    isin       TEXT,
    trade_date TEXT,
    trade_type TEXT,
    quantity   REAL,
    price      REAL
);
CREATE TABLE IF NOT EXISTS goals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT UNIQUE NOT NULL,
    target_amount REAL NOT NULL,
    target_date   TEXT NOT NULL,
    priority      INTEGER DEFAULT 5,
    created_at    TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT,
    kind       TEXT,
    title      TEXT,
    detail     TEXT,
    status     TEXT DEFAULT 'pending',
    decided_at TEXT
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


def replace_equity(rows: list[dict], cash: float) -> None:
    now = _now()
    with connect() as con:
        con.execute("DELETE FROM equity_holdings")
        con.executemany(
            "INSERT INTO equity_holdings VALUES (:symbol,:qty,:avg_price,:ltp,:value,'%s')" % now,
            rows,
        )
        con.execute(
            "INSERT INTO meta VALUES ('cash',?),('last_equity_sync',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(cash), now),
        )


def update_mf_navs(updates: list[dict]) -> int:
    """updates: [{isin, nav, nav_date}] → refresh nav + value on held schemes."""
    with connect() as con:
        cur = con.executemany(
            "UPDATE mf_holdings SET nav=:nav, nav_date=:nav_date, "
            "value=units*:nav, updated_at='%s' WHERE isin=:isin" % _now(),
            updates,
        )
        return cur.rowcount if cur.rowcount != -1 else len(updates)


def snapshot_networth(date_ist: str) -> dict:
    """Persist today's networth breakdown; returns it with day-change vs previous."""
    n = networth()
    with connect() as con:
        prev = con.execute(
            "SELECT * FROM networth_snapshots WHERE date < ? ORDER BY date DESC LIMIT 1",
            (date_ist,),
        ).fetchone()
        con.execute(
            "INSERT OR REPLACE INTO networth_snapshots VALUES (?,?,?,?,?)",
            (date_ist, n["mf"], n["equity"], n["cash"], n["total"]),
        )
    n["prev_total"] = prev["total"] if prev else None
    n["prev_date"] = prev["date"] if prev else None
    return n


def recent_sips(days: int = 45) -> list[sqlite3.Row]:
    """Schemes with SIP purchases in the last N days → for digest heads-up."""
    with connect() as con:
        return con.execute(
            "SELECT scheme, MAX(date) AS last_date, amount FROM mf_transactions "
            "WHERE txn_type LIKE '%SIP%' AND date >= date('now', ?) "
            "GROUP BY scheme ORDER BY last_date",
            (f"-{days} days",),
        ).fetchall()


def mf_holdings() -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT * FROM mf_holdings WHERE units > 0.001 ORDER BY value DESC"
        ).fetchall()


def insert_equity_trades(rows: list[dict]) -> int:
    with connect() as con:
        before = con.execute("SELECT COUNT(*) c FROM equity_trades").fetchone()["c"]
        con.executemany(
            "INSERT OR IGNORE INTO equity_trades VALUES "
            "(:trade_id,:symbol,:isin,:trade_date,:trade_type,:quantity,:price)",
            rows,
        )
        after = con.execute("SELECT COUNT(*) c FROM equity_trades").fetchone()["c"]
    return after - before


def equity_trades_all() -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT * FROM equity_trades ORDER BY symbol, trade_date"
        ).fetchall()


def add_goal(name: str, amount: float, target_date: str, priority: int = 5) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO goals (name,target_amount,target_date,priority,created_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "target_amount=excluded.target_amount, target_date=excluded.target_date, "
            "priority=excluded.priority",
            (name, amount, target_date, priority, _now()),
        )


def delete_goal(name: str) -> int:
    with connect() as con:
        return con.execute("DELETE FROM goals WHERE name=?", (name,)).rowcount


def list_goals() -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT * FROM goals ORDER BY priority, target_date"
        ).fetchall()


def add_recommendation(kind: str, title: str, detail: str) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO recommendations (created_at,kind,title,detail) VALUES (?,?,?,?)",
            (_now(), kind, title, detail),
        )
        return cur.lastrowid


def decide_recommendation(rec_id: int, status: str) -> sqlite3.Row | None:
    with connect() as con:
        con.execute(
            "UPDATE recommendations SET status=?, decided_at=? WHERE id=?",
            (status, _now(), rec_id),
        )
        return con.execute(
            "SELECT * FROM recommendations WHERE id=?", (rec_id,)).fetchone()


def pending_recommendations() -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT * FROM recommendations WHERE status='pending' ORDER BY id"
        ).fetchall()


def mf_transactions_all() -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute(
            "SELECT * FROM mf_transactions ORDER BY scheme, date"
        ).fetchall()


def equity_holdings() -> list[sqlite3.Row]:
    with connect() as con:
        return con.execute("SELECT * FROM equity_holdings ORDER BY value DESC").fetchall()


def networth() -> dict:
    with connect() as con:
        mf = con.execute("SELECT COALESCE(SUM(value),0) v FROM mf_holdings WHERE units>0.001").fetchone()["v"]
        eq = con.execute("SELECT COALESCE(SUM(value),0) v FROM equity_holdings").fetchone()["v"]
    cash = float(get_meta("cash", "0") or 0)
    return {"mf": mf, "equity": eq, "cash": cash, "total": mf + eq + cash}
