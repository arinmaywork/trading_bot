"""Wealth OS — SQLite backups.

Strategy: consistent snapshot via sqlite3's online backup API → gzip →
data/backups/ (keep last 14) → nightly copy sent to your Telegram chat,
so the data survives even total VM loss. Restore = drop the file back
as data/wealth.db (gunzip first).
"""
from __future__ import annotations

import gzip
import sqlite3
from datetime import datetime
from pathlib import Path

from . import db

BACKUP_DIR = Path(__file__).resolve().parent.parent / "data" / "backups"
KEEP = 14


def make_backup() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw = BACKUP_DIR / f"wealth_{stamp}.db"
    gz = raw.with_suffix(".db.gz")

    src = db.connect()
    try:
        dst = sqlite3.connect(raw)
        with dst:
            src.backup(dst)
        dst.close()
    finally:
        src.close()

    with open(raw, "rb") as f_in, gzip.open(gz, "wb") as f_out:
        f_out.write(f_in.read())
    raw.unlink()

    _prune()
    db.set_meta("last_backup", datetime.now().isoformat(timespec="seconds"))
    return gz


def _prune() -> None:
    backups = sorted(BACKUP_DIR.glob("wealth_*.db.gz"))
    for old in backups[:-KEEP]:
        old.unlink(missing_ok=True)
