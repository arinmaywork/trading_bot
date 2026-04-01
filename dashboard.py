"""
dashboard.py — SentiStack V2 Web Dashboard
===========================================
Real-time browser-based monitoring dashboard for the trading bot.

Run:
    uvicorn dashboard:app --host 0.0.0.0 --port 8765 --reload

Then open:  http://localhost:8765

Features:
  - Live status cards: GRI, VIX, USD/INR, Sentiment, Active Symbols, Trades
  - GRI breakdown panel with component progress bars
  - Real-time signal feed (WebSocket, filterable)
  - Recent trades table
  - Live log console with level filtering, per-line copy, and search
  - Pause / Resume bot control via Redis bot:cmd key
  - Auto-reconnecting WebSocket client
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ── FastAPI / Starlette ──────────────────────────────────────────────────────
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

# ── Redis ────────────────────────────────────────────────────────────────────
try:
    import redis.asyncio as aioredis
    _REDIS_OK = True
except ImportError:
    _REDIS_OK = False

logger = logging.getLogger("dashboard")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BOT_DIR = Path(__file__).parent
_LOG_DIR  = _BOT_DIR / "logs"
_LOG_FILE = _LOG_DIR / "bot_live.log"

# IST
_IST = timezone(timedelta(hours=5, minutes=30))

def _today_str() -> str:
    return datetime.now(_IST).strftime("%Y%m%d")

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MAX_LOG_LINES      = 500          # how many log lines to keep in memory
MAX_SIGNALS        = 200          # signal history for dashboard
MAX_TRADES         = 100          # trade history for dashboard
POLL_INTERVAL_SEC  = 2.0          # Redis / file poll interval


# ═══════════════════════════════════════════════════════════════════════════════
# Connection Manager
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    def __init__(self) -> None:
        self._active: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._active.add(ws)
        logger.info("WS client connected  (total=%d)", len(self._active))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._active.discard(ws)
        logger.info("WS client disconnected (total=%d)", len(self._active))

    async def broadcast(self, data: dict) -> None:
        msg = json.dumps(data)
        dead: List[WebSocket] = []
        async with self._lock:
            clients = list(self._active)
        for ws in clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._active.discard(ws)

    @property
    def count(self) -> int:
        return len(self._active)


# ═══════════════════════════════════════════════════════════════════════════════
# State Store  (in-memory snapshot refreshed by background poller)
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardState:
    def __init__(self) -> None:
        self.bot_state:  Dict[str, str] = {}
        self.gri_state:  Dict[str, Any] = {}
        self.signals:    List[Dict]     = []
        self.trades:     List[Dict]     = []
        self.log_lines:  List[Dict]     = []
        self._log_offset: int           = 0      # byte offset in bot_live.log
        self._sig_offset: int           = 0      # rows already read from signals CSV
        self._trd_offset: int           = 0      # rows already read from trades CSV
        self._last_sig_date: str        = ""
        self._last_trd_date: str        = ""

    # ── Redis helpers ──────────────────────────────────────────────────────────

    async def refresh_redis(self, redis: Any) -> bool:
        """Pull bot:state and geo:risk_index from Redis.  Returns True if changed."""
        changed = False
        try:
            raw_bot = await redis.hgetall("bot:state")
            if raw_bot:
                decoded = {
                    (k.decode() if isinstance(k, bytes) else k):
                    (v.decode() if isinstance(v, bytes) else v)
                    for k, v in raw_bot.items()
                }
                if decoded != self.bot_state:
                    self.bot_state = decoded
                    changed = True

            raw_gri = await redis.hgetall("geo:risk_index")
            if raw_gri:
                decoded_gri = {
                    (k.decode() if isinstance(k, bytes) else k):
                    (v.decode() if isinstance(v, bytes) else v)
                    for k, v in raw_gri.items()
                }
                if decoded_gri != self.gri_state:
                    self.gri_state = decoded_gri
                    changed = True
        except Exception as exc:
            logger.debug("Redis refresh error: %s", exc)
        return changed

    # ── CSV helpers ────────────────────────────────────────────────────────────

    def _read_csv_tail(
        self,
        path: Path,
        headers: List[str],
        offset: int,
        max_rows: int,
    ) -> tuple[List[Dict], int]:
        """Read new rows from a CSV file starting at byte offset."""
        if not path.exists():
            return [], offset
        rows: List[Dict] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(offset)
                content = f.read()
                new_offset = offset + len(content.encode("utf-8"))
                reader = csv.DictReader(
                    content.splitlines(),
                    fieldnames=None,  # auto-detect from first line if offset==0
                )
                # If we're at offset 0 the first line is the header — skip it
                # If we're mid-file there's no header line
                lines = content.splitlines()
                if not lines:
                    return [], offset
                start_line = 0
                if offset == 0 and lines[0].startswith("timestamp"):
                    start_line = 1   # skip header row
                for line in lines[start_line:]:
                    if not line.strip():
                        continue
                    try:
                        parts = next(csv.reader([line]))
                        if len(parts) >= len(headers):
                            rows.append(dict(zip(headers, parts)))
                    except StopIteration:
                        pass
            return rows[-max_rows:], new_offset
        except Exception as exc:
            logger.debug("CSV read error (%s): %s", path.name, exc)
            return [], offset

    def refresh_signals(self) -> bool:
        today = _today_str()
        if today != self._last_sig_date:
            self._last_sig_date = today
            self._sig_offset = 0
            self.signals = []
        path = _LOG_DIR / f"signals_{today}.csv"
        from logbook import SIGNAL_HEADERS  # noqa: PLC0415
        new_rows, self._sig_offset = self._read_csv_tail(
            path, SIGNAL_HEADERS, self._sig_offset, MAX_SIGNALS
        )
        if new_rows:
            self.signals = (self.signals + new_rows)[-MAX_SIGNALS:]
            return True
        return False

    def refresh_trades(self) -> bool:
        today = _today_str()
        if today != self._last_trd_date:
            self._last_trd_date = today
            self._trd_offset = 0
            self.trades = []
        path = _LOG_DIR / f"trades_{today}.csv"
        from logbook import TRADE_HEADERS  # noqa: PLC0415
        new_rows, self._trd_offset = self._read_csv_tail(
            path, TRADE_HEADERS, self._trd_offset, MAX_TRADES
        )
        if new_rows:
            self.trades = (self.trades + new_rows)[-MAX_TRADES:]
            return True
        return False

    # ── Log file tailer ────────────────────────────────────────────────────────

    _LEVEL_RE = re.compile(r"\|(DEBUG|INFO|WARNING|ERROR|CRITICAL)\|")

    def refresh_logs(self) -> bool:
        if not _LOG_FILE.exists():
            return False
        try:
            with open(_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._log_offset)
                chunk = f.read(65536)   # max 64 KB per poll
                self._log_offset = f.tell()
            if not chunk:
                return False
            new_entries: List[Dict] = []
            for raw_line in chunk.splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                m = self._LEVEL_RE.search(raw_line)
                level = m.group(1) if m else "INFO"
                # parse timestamp from pipe-delimited format: HH:MM:SS.mmm|LEVEL|logger|msg
                parts = raw_line.split("|", 3)
                ts = parts[0] if parts else ""
                msg = parts[3] if len(parts) >= 4 else raw_line
                new_entries.append({"ts": ts, "level": level, "msg": msg, "raw": raw_line})
            if new_entries:
                self.log_lines = (self.log_lines + new_entries)[-MAX_LOG_LINES:]
                return True
        except Exception as exc:
            logger.debug("Log tail error: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="SentiStack Dashboard", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_manager = ConnectionManager()
_state   = DashboardState()
_redis: Optional[Any] = None


@app.on_event("startup")
async def _startup() -> None:
    global _redis
    if _REDIS_OK:
        try:
            _redis = aioredis.from_url(REDIS_URL, decode_responses=False)
            await _redis.ping()
            logger.info("Dashboard connected to Redis at %s", REDIS_URL)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — dashboard in log-only mode", exc)
            _redis = None
    asyncio.create_task(_poll_loop(), name="dashboard_poller")


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _redis:
        await _redis.aclose()


# ── Background poller ──────────────────────────────────────────────────────────

async def _poll_loop() -> None:
    """Polls Redis, CSV files, and log file every POLL_INTERVAL_SEC seconds.
    Broadcasts any changed data to all WebSocket clients."""
    while True:
        try:
            payload: Dict[str, Any] = {}

            if _redis:
                if await _state.refresh_redis(_redis):
                    payload["bot_state"] = _state.bot_state
                    payload["gri_state"] = _state.gri_state

            if _state.refresh_signals():
                payload["signals"] = _state.signals[-50:]   # last 50 for initial batch

            if _state.refresh_trades():
                payload["trades"] = _state.trades[-20:]

            if _state.refresh_logs():
                payload["logs"] = _state.log_lines[-100:]   # send up to 100 new lines

            if payload and _manager.count > 0:
                payload["type"] = "update"
                payload["ts"] = datetime.now(_IST).isoformat()
                await _manager.broadcast(payload)

        except Exception as exc:
            logger.debug("Poll loop error: %s", exc)

        await asyncio.sleep(POLL_INTERVAL_SEC)


# ═══════════════════════════════════════════════════════════════════════════════
# REST Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status() -> JSONResponse:
    return JSONResponse({
        "bot_state":  _state.bot_state,
        "gri_state":  _state.gri_state,
        "ws_clients": _manager.count,
    })


@app.get("/api/signals")
async def api_signals(limit: int = 50) -> JSONResponse:
    return JSONResponse(_state.signals[-limit:])


@app.get("/api/trades")
async def api_trades(limit: int = 20) -> JSONResponse:
    return JSONResponse(_state.trades[-limit:])


@app.get("/api/logs")
async def api_logs(limit: int = 200) -> JSONResponse:
    return JSONResponse(_state.log_lines[-limit:])


@app.post("/api/command")
async def api_command(body: dict) -> JSONResponse:
    cmd = body.get("cmd", "").strip().lower()
    if cmd not in ("pause", "resume"):
        raise HTTPException(status_code=400, detail="cmd must be 'pause' or 'resume'")
    if not _redis:
        raise HTTPException(status_code=503, detail="Redis not available")
    await _redis.set("bot:cmd", cmd, ex=60)
    return JSONResponse({"ok": True, "cmd": cmd})


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await _manager.connect(ws)
    try:
        # Send full current state on connect
        await ws.send_text(json.dumps({
            "type":      "init",
            "bot_state": _state.bot_state,
            "gri_state": _state.gri_state,
            "signals":   _state.signals[-50:],
            "trades":    _state.trades[-20:],
            "logs":      _state.log_lines[-200:],
            "ts":        datetime.now(_IST).isoformat(),
        }))
        # Keep alive — just drain any pings from client
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await _manager.disconnect(ws)


# ═══════════════════════════════════════════════════════════════════════════════
# Embedded HTML Dashboard
# ═══════════════════════════════════════════════════════════════════════════════

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SentiStack V2 — Live Dashboard</title>
<style>
/* ── Reset & Base ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:        #0d1117;
  --bg2:       #161b22;
  --bg3:       #21262d;
  --border:    #30363d;
  --text:      #e6edf3;
  --muted:     #8b949e;
  --green:     #3fb950;
  --red:       #f85149;
  --yellow:    #d29922;
  --blue:      #58a6ff;
  --purple:    #bc8cff;
  --orange:    #ffa657;
  --cyan:      #39d353;
  --accent:    #1f6feb;
  font-size: 13px;
}
body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; min-height: 100vh; }

/* ── Topbar ────────────────────────────────────────────────────────────── */
#topbar {
  position: sticky; top: 0; z-index: 100;
  background: var(--bg2); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px;
  padding: 8px 16px; height: 48px;
}
#topbar .brand { font-size: 15px; font-weight: 700; letter-spacing: .5px; color: var(--blue); }
#topbar .brand span { color: var(--text); font-weight: 400; font-size: 12px; }
#status-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--muted); transition: background .3s; }
#status-dot.live { background: var(--green); box-shadow: 0 0 6px var(--green); animation: pulse 2s infinite; }
#status-dot.paused { background: var(--yellow); }
#status-dot.offline { background: var(--red); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
#mode-badge {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 4px; padding: 2px 8px; font-size: 11px;
  color: var(--orange); font-weight: 600; letter-spacing: 1px;
}
.top-spacer { flex: 1; }
#btn-pause, #btn-resume {
  padding: 4px 12px; border-radius: 6px; border: 1px solid var(--border);
  cursor: pointer; font-size: 12px; font-weight: 600; transition: .15s;
}
#btn-pause  { background: #3d1f00; color: var(--orange); border-color: var(--orange); }
#btn-resume { background: #0d2a0d; color: var(--green);  border-color: var(--green); display:none; }
#btn-pause:hover  { background: var(--orange); color: #000; }
#btn-resume:hover { background: var(--green);  color: #000; }
#clock { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; min-width: 90px; text-align:right; }
#ws-indicator { font-size: 11px; color: var(--muted); }
#ws-indicator.connected { color: var(--green); }
#ws-indicator.disconnected { color: var(--red); }

/* ── Layout ────────────────────────────────────────────────────────────── */
#main { padding: 12px 16px; display: flex; flex-direction: column; gap: 12px; }

/* ── Metric Cards ──────────────────────────────────────────────────────── */
#cards { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
@media(max-width: 1200px) { #cards { grid-template-columns: repeat(3, 1fr); } }
@media(max-width: 700px)  { #cards { grid-template-columns: repeat(2, 1fr); } }
.card {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; display: flex; flex-direction: column; gap: 4px;
}
.card-label  { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; }
.card-value  { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.card-sub    { font-size: 11px; color: var(--muted); }
.val-green   { color: var(--green); }
.val-red     { color: var(--red); }
.val-yellow  { color: var(--yellow); }
.val-blue    { color: var(--blue); }
.val-orange  { color: var(--orange); }
.val-purple  { color: var(--purple); }

/* ── Middle row ────────────────────────────────────────────────────────── */
#mid-row { display: grid; grid-template-columns: 340px 1fr; gap: 12px; }
@media(max-width: 900px) { #mid-row { grid-template-columns: 1fr; } }

/* ── Panel base ────────────────────────────────────────────────────────── */
.panel {
  background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
  display: flex; flex-direction: column; overflow: hidden;
}
.panel-header {
  padding: 10px 14px; border-bottom: 1px solid var(--border);
  font-size: 12px; font-weight: 600; letter-spacing: .5px; color: var(--muted);
  display: flex; align-items: center; gap: 8px;
}
.panel-header .ph-badge {
  margin-left: auto; background: var(--bg3); border-radius: 4px;
  padding: 1px 6px; font-size: 10px; color: var(--text);
}
.panel-body { padding: 12px; flex: 1; overflow-y: auto; }

/* ── GRI panel ─────────────────────────────────────────────────────────── */
#gri-panel .gri-composite {
  font-size: 36px; font-weight: 700; margin-bottom: 4px;
}
.gri-level-badge {
  display: inline-block; border-radius: 4px; padding: 1px 8px;
  font-size: 11px; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;
}
.gri-comp-row { display: flex; flex-direction: column; gap: 8px; }
.gri-comp-item label { font-size: 11px; color: var(--muted); display: flex; justify-content: space-between; }
.gri-bar-track {
  height: 6px; background: var(--bg3); border-radius: 3px; margin-top: 3px; overflow: hidden;
}
.gri-bar-fill { height: 100%; border-radius: 3px; transition: width .4s; }
.gri-keywords { margin-top: 10px; }
.gri-kw-title { font-size: 10px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing:.8px; }
.gri-kw-list  { display: flex; flex-wrap: wrap; gap: 4px; }
.gri-kw-chip  {
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 3px; padding: 1px 6px; font-size: 10px; color: var(--text);
}
.gri-headline {
  margin-top: 8px; background: var(--bg3); border-radius: 4px;
  padding: 6px 8px; font-size: 11px; color: var(--muted);
  border-left: 3px solid var(--border); white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}

/* ── Signal feed ────────────────────────────────────────────────────────── */
#signals-panel { flex: 1; }
.sig-filters { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.filter-btn {
  background: var(--bg3); border: 1px solid var(--border); color: var(--muted);
  border-radius: 4px; padding: 2px 10px; font-size: 11px; cursor: pointer; transition: .15s;
}
.filter-btn.active, .filter-btn:hover { background: var(--accent); border-color: var(--accent); color: #fff; }
.sig-list { display: flex; flex-direction: column; gap: 4px; max-height: 280px; overflow-y: auto; }
.sig-row {
  display: grid; grid-template-columns: 80px 50px 90px 70px 70px 1fr;
  gap: 6px; align-items: center;
  background: var(--bg3); border-radius: 4px; padding: 5px 8px;
  font-size: 11px; border-left: 3px solid transparent;
}
.sig-row.BUY  { border-color: var(--green); }
.sig-row.SELL { border-color: var(--red); }
.sig-row.FLAT { border-color: var(--muted); }
.sig-dir { font-weight: 700; }
.sig-dir.BUY  { color: var(--green); }
.sig-dir.SELL { color: var(--red); }
.sig-dir.FLAT { color: var(--muted); }
.sig-alpha  { font-variant-numeric: tabular-nums; }
.sig-symbol { font-weight: 600; }
.sig-rationale { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Trades table ───────────────────────────────────────────────────────── */
#bottom-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media(max-width: 900px) { #bottom-row { grid-template-columns: 1fr; } }
.trd-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.trd-table th {
  text-align: left; padding: 4px 8px; color: var(--muted);
  border-bottom: 1px solid var(--border); font-weight: 600;
  font-size: 10px; text-transform: uppercase; letter-spacing: .6px;
}
.trd-table td { padding: 5px 8px; border-bottom: 1px solid var(--border); }
.trd-table tr:last-child td { border-bottom: none; }
.trd-table tr:hover td { background: var(--bg3); }
.trd-status-ok  { color: var(--green); font-size: 14px; }
.trd-status-err { color: var(--red);   font-size: 14px; }
.trd-dir-buy    { color: var(--green); font-weight: 700; }
.trd-dir-sell   { color: var(--red);   font-weight: 700; }
.trd-paper      { color: var(--yellow); font-size: 10px; }
.trd-live-mode  { color: var(--green);  font-size: 10px; }

/* ── Log Console ────────────────────────────────────────────────────────── */
#log-console-panel { grid-column: 1 / -1; }
.log-toolbar {
  display: flex; gap: 6px; align-items: center; padding: 8px 14px;
  border-bottom: 1px solid var(--border); flex-wrap: wrap;
}
.log-filter-btn {
  background: var(--bg3); border: 1px solid var(--border); color: var(--muted);
  border-radius: 4px; padding: 2px 10px; font-size: 11px; cursor: pointer; transition: .15s;
}
.log-filter-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }
.log-filter-btn.WARN.active  { background: #5a3800; border-color: var(--yellow); color: var(--yellow); }
.log-filter-btn.ERROR.active { background: #3d0000; border-color: var(--red);    color: var(--red); }
#log-search {
  background: var(--bg3); border: 1px solid var(--border); color: var(--text);
  border-radius: 4px; padding: 3px 8px; font-size: 11px; width: 180px; outline: none;
}
#log-search:focus { border-color: var(--accent); }
.log-spacer { flex: 1; }
.log-action-btn {
  background: var(--bg3); border: 1px solid var(--border); color: var(--muted);
  border-radius: 4px; padding: 3px 10px; font-size: 11px; cursor: pointer; transition:.15s;
}
.log-action-btn:hover { border-color: var(--blue); color: var(--blue); }
#log-autoscroll-cb { accent-color: var(--accent); }
.log-autoscroll-lbl { font-size: 11px; color: var(--muted); display:flex; align-items:center; gap:4px; }
#log-body {
  font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
  font-size: 11px; line-height: 1.6;
  max-height: 320px; overflow-y: auto;
  padding: 4px 0;
}
.log-line {
  display: flex; align-items: flex-start; gap: 8px; padding: 1px 8px;
  transition: background .1s;
}
.log-line:hover { background: var(--bg3); }
.log-line:hover .log-copy-btn { opacity: 1; }
.log-ts    { color: var(--muted); flex-shrink: 0; min-width: 95px; }
.log-lvl   { flex-shrink: 0; min-width: 50px; font-weight: 700; font-size: 10px; }
.log-msg   { flex: 1; white-space: pre-wrap; word-break: break-all; }
.log-copy-btn {
  opacity: 0; cursor: pointer; background: none; border: none; color: var(--muted);
  font-size: 13px; padding: 0 4px; line-height: 1; transition: color .1s; flex-shrink: 0;
}
.log-copy-btn:hover { color: var(--blue); }
.lvl-DEBUG    { color: var(--muted); }
.lvl-INFO     { color: var(--blue); }
.lvl-WARNING  { color: var(--yellow); }
.lvl-ERROR    { color: var(--red); }
.lvl-CRITICAL { color: var(--red); font-weight: 700; text-decoration: underline; }

/* ── Empty state ────────────────────────────────────────────────────────── */
.empty-state {
  padding: 24px; text-align: center; color: var(--muted); font-size: 12px;
}

/* ── Scrollbars ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* ── Toast ──────────────────────────────────────────────────────────────── */
#toast {
  position: fixed; bottom: 20px; right: 20px; z-index: 9999;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 16px; font-size: 12px;
  opacity: 0; transition: opacity .3s; pointer-events: none;
}
#toast.show { opacity: 1; }
</style>
</head>
<body>

<!-- ── Topbar ──────────────────────────────────────────────────────────────── -->
<div id="topbar">
  <div class="brand">SentiStack <span>V2</span></div>
  <div id="status-dot" class="offline" title="Bot status"></div>
  <div id="mode-badge">--</div>
  <div id="ws-indicator" class="disconnected">● WS</div>
  <div class="top-spacer"></div>
  <button id="btn-pause"  onclick="sendCmd('pause')">⏸ Pause</button>
  <button id="btn-resume" onclick="sendCmd('resume')">▶ Resume</button>
  <div id="clock">--:--:-- IST</div>
</div>

<!-- ── Main Content ────────────────────────────────────────────────────────── -->
<div id="main">

  <!-- Metric cards -->
  <div id="cards">
    <div class="card" id="card-gri">
      <div class="card-label">GRI Composite</div>
      <div class="card-value val-yellow" id="cv-gri">--</div>
      <div class="card-sub" id="cs-gri-level">--</div>
    </div>
    <div class="card" id="card-vix">
      <div class="card-label">India VIX</div>
      <div class="card-value val-orange" id="cv-vix">--</div>
      <div class="card-sub" id="cs-vix">pts</div>
    </div>
    <div class="card">
      <div class="card-label">USD / INR</div>
      <div class="card-value val-blue" id="cv-usdinr">--</div>
      <div class="card-sub" id="cs-usdinr">--</div>
    </div>
    <div class="card">
      <div class="card-label">Sentiment</div>
      <div class="card-value val-purple" id="cv-sentiment">--</div>
      <div class="card-sub" id="cs-sentiment-class">--</div>
    </div>
    <div class="card">
      <div class="card-label">Active Symbols</div>
      <div class="card-value val-cyan" id="cv-symbols">--</div>
      <div class="card-sub" id="cs-symbols">top symbols</div>
    </div>
    <div class="card">
      <div class="card-label">Trades Today</div>
      <div class="card-value val-green" id="cv-trades">--</div>
      <div class="card-sub" id="cs-signals">-- signals</div>
    </div>
  </div>

  <!-- Mid row: GRI breakdown + Signals -->
  <div id="mid-row">

    <!-- GRI Panel -->
    <div class="panel" id="gri-panel">
      <div class="panel-header">⚠ Geopolitical Risk Index</div>
      <div class="panel-body">
        <div class="gri-composite val-yellow" id="gri-big">--</div>
        <span class="gri-level-badge" id="gri-level-badge">--</span>
        <div class="gri-comp-row">
          <div class="gri-comp-item">
            <label><span>GDELT Conflict</span><span id="gv-gdelt">--</span></label>
            <div class="gri-bar-track"><div class="gri-bar-fill" id="gb-gdelt" style="width:0%;background:#f85149"></div></div>
          </div>
          <div class="gri-comp-item">
            <label><span>India VIX</span><span id="gv-vix">--</span></label>
            <div class="gri-bar-track"><div class="gri-bar-fill" id="gb-vix" style="width:0%;background:#ffa657"></div></div>
          </div>
          <div class="gri-comp-item">
            <label><span>USD/INR Stress</span><span id="gv-usdinr">--</span></label>
            <div class="gri-bar-track"><div class="gri-bar-fill" id="gb-usdinr" style="width:0%;background:#d29922"></div></div>
          </div>
          <div class="gri-comp-item">
            <label><span>RSS Sentiment</span><span id="gv-rss">--</span></label>
            <div class="gri-bar-track"><div class="gri-bar-fill" id="gb-rss" style="width:0%;background:#58a6ff"></div></div>
          </div>
        </div>
        <div class="gri-keywords">
          <div class="gri-kw-title">Active Keywords</div>
          <div class="gri-kw-list" id="gri-kw-list"><span class="gri-kw-chip" style="color:var(--muted)">--</span></div>
        </div>
        <div class="gri-headline" id="gri-headline">No headline available</div>
      </div>
    </div>

    <!-- Signal Feed -->
    <div class="panel" id="signals-panel">
      <div class="panel-header">
        ⚡ Signal Feed
        <span class="ph-badge" id="sig-count">0</span>
        <div class="sig-filters" style="margin-left:8px;">
          <button class="filter-btn active" data-dir="ALL"  onclick="setSigFilter(this,'ALL')">All</button>
          <button class="filter-btn"        data-dir="BUY"  onclick="setSigFilter(this,'BUY')">Buy</button>
          <button class="filter-btn"        data-dir="SELL" onclick="setSigFilter(this,'SELL')">Sell</button>
          <button class="filter-btn"        data-dir="FLAT" onclick="setSigFilter(this,'FLAT')">Flat</button>
          <button class="filter-btn"        data-dir="ACT"  onclick="setSigFilter(this,'ACT')">Actionable</button>
        </div>
      </div>
      <div class="panel-body" style="padding:8px;">
        <div id="sig-list" class="sig-list">
          <div class="empty-state">Waiting for signals…</div>
        </div>
      </div>
    </div>

  </div><!-- /mid-row -->

  <!-- Bottom row: Trades + Session Stats -->
  <div id="bottom-row">

    <!-- Trades -->
    <div class="panel">
      <div class="panel-header">
        📋 Recent Trades
        <span class="ph-badge" id="trd-count">0</span>
      </div>
      <div class="panel-body" style="padding:0; overflow-x:auto;">
        <table class="trd-table">
          <thead>
            <tr>
              <th></th><th>Time</th><th>Symbol</th><th>Dir</th>
              <th>Qty</th><th>Fill ₹</th><th>Slip bps</th><th>Mode</th>
            </tr>
          </thead>
          <tbody id="trd-body">
            <tr><td colspan="8" class="empty-state">No trades yet</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Session Stats -->
    <div class="panel">
      <div class="panel-header">📊 Session Stats</div>
      <div class="panel-body">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
          <div>
            <div class="card-label">Cycle #</div>
            <div class="card-value val-blue" id="ss-cycle">--</div>
          </div>
          <div>
            <div class="card-label">Bot Mode</div>
            <div class="card-value val-orange" id="ss-mode">--</div>
          </div>
          <div>
            <div class="card-label">Signals Today</div>
            <div class="card-value val-purple" id="ss-signals">--</div>
          </div>
          <div>
            <div class="card-label">Last Update</div>
            <div class="card-value" id="ss-last-update" style="font-size:13px;color:var(--muted)">--</div>
          </div>
        </div>
        <div style="margin-top:14px;">
          <div class="card-label" style="margin-bottom:6px;">Top Symbols</div>
          <div id="ss-top-symbols" style="display:flex;flex-wrap:wrap;gap:4px;">
            <span style="color:var(--muted);font-size:11px;">--</span>
          </div>
        </div>
      </div>
    </div>

  </div><!-- /bottom-row -->

  <!-- Log Console -->
  <div class="panel" id="log-console-panel">
    <div class="panel-header">
      🖥 Log Console
      <span class="ph-badge" id="log-count">0</span>
    </div>
    <div class="log-toolbar">
      <button class="log-filter-btn active" data-lvl="ALL"     onclick="setLogFilter(this,'ALL')">All</button>
      <button class="log-filter-btn"        data-lvl="INFO"    onclick="setLogFilter(this,'INFO')">INFO</button>
      <button class="log-filter-btn WARN"   data-lvl="WARNING" onclick="setLogFilter(this,'WARNING')">WARN</button>
      <button class="log-filter-btn ERROR"  data-lvl="ERROR"   onclick="setLogFilter(this,'ERROR')">ERROR</button>
      <input id="log-search" type="text" placeholder="Search logs…" oninput="renderLogs()">
      <div class="log-spacer"></div>
      <label class="log-autoscroll-lbl">
        <input type="checkbox" id="log-autoscroll-cb" checked> Auto-scroll
      </label>
      <button class="log-action-btn" onclick="copyAllLogs()">📋 Copy All</button>
      <button class="log-action-btn" onclick="clearLogs()">🗑 Clear</button>
    </div>
    <div id="log-body"></div>
  </div>

</div><!-- /main -->

<div id="toast"></div>

<!-- ═══════════════════════════════════════════════════════════════════════════
     JavaScript
     ═══════════════════════════════════════════════════════════════════════ -->
<script>
'use strict';

// ── State ───────────────────────────────────────────────────────────────────
let _signals  = [];
let _trades   = [];
let _logs     = [];
let _sigFilter  = 'ALL';
let _logFilter  = 'ALL';
let _ws = null;
let _wsRetries  = 0;
const MAX_WS_DELAY = 16000;

// ── Clock ───────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  const ist = new Date(now.toLocaleString('en-US', {timeZone:'Asia/Kolkata'}));
  const hh = String(ist.getHours()).padStart(2,'0');
  const mm = String(ist.getMinutes()).padStart(2,'0');
  const ss = String(ist.getSeconds()).padStart(2,'0');
  document.getElementById('clock').textContent = `${hh}:${mm}:${ss} IST`;
}
setInterval(updateClock, 1000);
updateClock();

// ── Toast ───────────────────────────────────────────────────────────────────
let _toastTimer = null;
function toast(msg, dur=2200) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove('show'), dur);
}

// ── WebSocket ────────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url   = `${proto}://${location.host}/ws`;
  _ws = new WebSocket(url);

  _ws.onopen = () => {
    _wsRetries = 0;
    document.getElementById('ws-indicator').className = 'connected';
    document.getElementById('ws-indicator').textContent = '● WS';
  };

  _ws.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      handleMessage(data);
    } catch(e) { console.error('WS parse error', e); }
  };

  _ws.onclose = () => {
    document.getElementById('ws-indicator').className = 'disconnected';
    document.getElementById('ws-indicator').textContent = '● WS';
    const delay = Math.min(1000 * Math.pow(2, _wsRetries++), MAX_WS_DELAY);
    setTimeout(connectWS, delay);
  };

  _ws.onerror = () => _ws.close();
}

function handleMessage(data) {
  if (data.bot_state) updateBotState(data.bot_state);
  if (data.gri_state) updateGRI(data.gri_state);
  if (data.signals)   { _signals = data.signals; renderSignals(); }
  if (data.trades)    { _trades  = data.trades;  renderTrades(); }
  if (data.logs)      { appendLogs(data.logs); }
}

// ── Bot State cards ──────────────────────────────────────────────────────────
function updateBotState(s) {
  // Status dot
  const dot = document.getElementById('status-dot');
  if (s.running === '1' && s.paused === '0') {
    dot.className = 'live';
  } else if (s.paused === '1') {
    dot.className = 'paused';
  } else {
    dot.className = 'offline';
  }

  // Pause / resume buttons
  if (s.paused === '1') {
    document.getElementById('btn-pause').style.display  = 'none';
    document.getElementById('btn-resume').style.display = 'inline-block';
  } else {
    document.getElementById('btn-pause').style.display  = 'inline-block';
    document.getElementById('btn-resume').style.display = 'none';
  }

  // Mode badge
  const mode = (s.mode || '--').toUpperCase();
  const mb = document.getElementById('mode-badge');
  mb.textContent = mode;
  mb.style.color = mode === 'LIVE' ? 'var(--green)' : 'var(--orange)';

  // Cards
  setText('cv-sentiment',    fmtFloat(s.sentiment, 4));
  setText('cs-sentiment-class', s.sentiment_class || '--');
  setText('cv-symbols', s.active_symbols || '--');
  setText('cv-trades',  s.trades_today   || '0');
  setText('cs-signals', (s.signals_today || '0') + ' signals');

  // Colour-code sentiment
  const sentVal = parseFloat(s.sentiment);
  document.getElementById('cv-sentiment').className =
    'card-value ' + (sentVal > 0.1 ? 'val-green' : sentVal < -0.1 ? 'val-red' : 'val-yellow');

  // Session stats
  setText('ss-cycle',   s.cycle   || '--');
  setText('ss-mode',    mode);
  setText('ss-signals', s.signals_today || '--');
  const lu = s.last_update ? s.last_update.slice(11,19) : '--';
  setText('ss-last-update', lu);

  // Top symbols chips
  try {
    const syms = JSON.parse(s.top_symbols || '[]');
    const container = document.getElementById('ss-top-symbols');
    container.innerHTML = syms.length ? syms.map(sym =>
      `<span style="background:var(--bg3);border:1px solid var(--border);border-radius:4px;
       padding:2px 8px;font-size:11px;color:var(--text)">${sym}</span>`
    ).join('') : '<span style="color:var(--muted);font-size:11px;">--</span>';
  } catch(e) {}
}

// ── GRI ──────────────────────────────────────────────────────────────────────
function updateGRI(g) {
  // Main composite
  const composite = parseFloat(g.composite || 0);
  setText('cv-gri',   fmtFloat(g.composite, 3));
  setText('cs-gri-level', g.level || '--');
  setText('gri-big',  fmtFloat(g.composite, 3));

  // Level badge
  const lb = document.getElementById('gri-level-badge');
  lb.textContent = (g.level || '--').toUpperCase();
  const levelColors = {
    'LOW':      ['#0d2a0d','var(--green)','var(--green)'],
    'MODERATE': ['#2a2000','var(--yellow)','var(--yellow)'],
    'ELEVATED': ['#3d1f00','var(--orange)','var(--orange)'],
    'HIGH':     ['#3d0000','var(--red)','var(--red)'],
    'EXTREME':  ['#5a0000','#ff4444','#ff4444'],
  };
  const [bg, col, brd] = levelColors[g.level?.toUpperCase()] || ['var(--bg3)','var(--muted)','var(--border)'];
  lb.style.cssText = `background:${bg};color:${col};border:1px solid ${brd};
    display:inline-block;border-radius:4px;padding:1px 8px;
    font-size:11px;font-weight:700;letter-spacing:1px;margin-bottom:12px;`;

  // Colour-code GRI card value
  document.getElementById('cv-gri').className =
    'card-value ' + (composite > 0.7 ? 'val-red' : composite > 0.5 ? 'val-orange' : composite > 0.3 ? 'val-yellow' : 'val-green');

  // Components
  setBar('gdelt',  g.gdelt_score   || g.gdelt   || 0);
  setBar('vix',    g.vix_score     || g.vix     || 0);
  setBar('usdinr', g.usdinr_score  || g.usdinr  || 0);
  setBar('rss',    g.rss_score     || g.rss     || 0);

  // VIX card
  if (g.india_vix !== undefined) {
    setText('cv-vix', fmtFloat(g.india_vix, 1));
    const vixVal = parseFloat(g.india_vix);
    document.getElementById('cv-vix').className =
      'card-value ' + (vixVal > 25 ? 'val-red' : vixVal > 18 ? 'val-orange' : 'val-green');
  }

  // USD/INR card
  if (g.usdinr_rate !== undefined) {
    setText('cv-usdinr', fmtFloat(g.usdinr_rate, 2));
    const usdVal = parseFloat(g.usdinr_rate);
    document.getElementById('cv-usdinr').className =
      'card-value ' + (usdVal > 91 ? 'val-red' : usdVal > 88 ? 'val-orange' : 'val-blue');
    setText('cs-usdinr', usdVal > 92 ? '⚠ Stress' : usdVal > 88 ? '↑ Elevated' : '~ Stable');
  }

  // Keywords
  try {
    const kws = JSON.parse(g.active_keywords || '[]');
    const container = document.getElementById('gri-kw-list');
    container.innerHTML = kws.length ? kws.slice(0,12).map(k =>
      `<span class="gri-kw-chip">${k}</span>`
    ).join('') : '<span class="gri-kw-chip" style="color:var(--muted)">None</span>';
  } catch(e) {}

  // Headline
  if (g.top_headline) {
    document.getElementById('gri-headline').textContent = g.top_headline;
  }
}

function setBar(key, rawVal) {
  const val = Math.min(Math.max(parseFloat(rawVal) || 0, 0), 1);
  setText('gv-' + key, val.toFixed(3));
  const fill = document.getElementById('gb-' + key);
  if (fill) fill.style.width = (val * 100) + '%';
}

// ── Signals ──────────────────────────────────────────────────────────────────
function setSigFilter(btn, f) {
  _sigFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderSignals();
}

function renderSignals() {
  let rows = [..._signals].reverse();
  if (_sigFilter === 'ACT') {
    rows = rows.filter(s => s.is_actionable === 'True' || s.is_actionable === true);
  } else if (_sigFilter !== 'ALL') {
    rows = rows.filter(s => (s.direction || '').toUpperCase() === _sigFilter);
  }

  document.getElementById('sig-count').textContent = rows.length;
  const container = document.getElementById('sig-list');

  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">No signals match filter</div>';
    return;
  }

  container.innerHTML = rows.slice(0, 80).map(s => {
    const dir = (s.direction || '').toUpperCase();
    const alpha = parseFloat(s.alpha || 0);
    const alphaCol = alpha > 0 ? 'var(--green)' : alpha < 0 ? 'var(--red)' : 'var(--muted)';
    const rationale = (s.rationale || '').replace(/</g,'&lt;');
    const ts = (s.timestamp || '').slice(11, 19);
    return `<div class="sig-row ${dir}">
      <span class="log-ts">${ts}</span>
      <span class="sig-dir ${dir}">${dir}</span>
      <span class="sig-symbol">${s.symbol || '--'}</span>
      <span class="sig-alpha" style="color:${alphaCol}">${alpha >= 0 ? '+' : ''}${alpha.toFixed(5)}</span>
      <span style="color:var(--muted)">${s.gri_level || '--'}</span>
      <span class="sig-rationale" title="${rationale}">${rationale}</span>
    </div>`;
  }).join('');
}

// ── Trades ────────────────────────────────────────────────────────────────────
function renderTrades() {
  const rows = [..._trades].reverse().slice(0, 25);
  document.getElementById('trd-count').textContent = _trades.length;
  const tbody = document.getElementById('trd-body');

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No trades yet</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map(t => {
    const ok = t.success === 'True' || t.success === true;
    const dir = (t.direction || '').toUpperCase();
    const ts  = (t.timestamp || '').slice(11, 19);
    const slip = parseFloat(t.slippage_bps || 0);
    const slipCol = slip > 5 ? 'var(--red)' : slip > 2 ? 'var(--yellow)' : 'var(--muted)';
    return `<tr>
      <td><span class="${ok ? 'trd-status-ok' : 'trd-status-err'}">${ok ? '✓' : '✗'}</span></td>
      <td style="color:var(--muted)">${ts}</td>
      <td style="font-weight:600">${t.symbol || '--'}</td>
      <td class="${dir === 'BUY' ? 'trd-dir-buy' : 'trd-dir-sell'}">${dir}</td>
      <td>${t.qty || '--'}</td>
      <td>₹${fmtFloat(t.fill_price, 2)}</td>
      <td style="color:${slipCol}">${slip.toFixed(1)}</td>
      <td><span class="${t.mode === 'LIVE' ? 'trd-live-mode' : 'trd-paper'}">${t.mode || '--'}</span></td>
    </tr>`;
  }).join('');
}

// ── Logs ──────────────────────────────────────────────────────────────────────
function setLogFilter(btn, lvl) {
  _logFilter = lvl;
  document.querySelectorAll('.log-filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderLogs();
}

function appendLogs(newLines) {
  _logs = [..._logs, ...newLines].slice(-500);
  renderLogs();
}

function renderLogs() {
  const search  = (document.getElementById('log-search').value || '').toLowerCase();
  let filtered  = _logs;

  if (_logFilter !== 'ALL') {
    filtered = filtered.filter(l => l.level === _logFilter);
  }
  if (search) {
    filtered = filtered.filter(l =>
      (l.msg || '').toLowerCase().includes(search) ||
      (l.raw || '').toLowerCase().includes(search)
    );
  }

  document.getElementById('log-count').textContent = filtered.length;

  const body    = document.getElementById('log-body');
  const wasAtBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 30;

  body.innerHTML = filtered.slice(-300).map((l, i) => {
    const lvl  = l.level || 'INFO';
    const msg  = (l.msg  || l.raw || '').replace(/</g,'&lt;');
    const raw  = (l.raw  || '').replace(/\\/g,'\\\\').replace(/'/g,"\\'");
    return `<div class="log-line">
      <span class="log-ts">${l.ts || ''}</span>
      <span class="log-lvl lvl-${lvl}">${lvl}</span>
      <span class="log-msg">${msg}</span>
      <button class="log-copy-btn" title="Copy line" onclick="copyLine('${raw}')">📋</button>
    </div>`;
  }).join('');

  const cb = document.getElementById('log-autoscroll-cb');
  if (cb.checked && wasAtBottom) {
    body.scrollTop = body.scrollHeight;
  }
}

function copyLine(text) {
  navigator.clipboard.writeText(text).then(() => toast('Line copied!')).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta);
    toast('Line copied!');
  });
}

function copyAllLogs() {
  const text = _logs.map(l => l.raw || l.msg || '').join('\n');
  navigator.clipboard.writeText(text).then(() => toast('All logs copied!')).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta);
    toast('All logs copied!');
  });
}

function clearLogs() {
  _logs = [];
  renderLogs();
  toast('Log console cleared');
}

// ── Bot control ───────────────────────────────────────────────────────────────
async function sendCmd(cmd) {
  try {
    const res = await fetch('/api/command', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({cmd}),
    });
    const data = await res.json();
    if (data.ok) {
      toast(`Command sent: ${cmd}`);
    } else {
      toast(`Error: ${data.detail || 'unknown'}`);
    }
  } catch(e) {
    toast('Failed to send command (Redis offline?)');
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function fmtFloat(val, dp) {
  const n = parseFloat(val);
  return isNaN(n) ? '--' : n.toFixed(dp);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
connectWS();

// Fallback: fetch initial state via REST if WS takes too long
setTimeout(async () => {
  if (_signals.length === 0) {
    try {
      const [status, sigs, trd, logs] = await Promise.all([
        fetch('/api/status').then(r=>r.json()),
        fetch('/api/signals').then(r=>r.json()),
        fetch('/api/trades').then(r=>r.json()),
        fetch('/api/logs').then(r=>r.json()),
      ]);
      if (status.bot_state) updateBotState(status.bot_state);
      if (status.gri_state) updateGRI(status.gri_state);
      _signals = sigs;  renderSignals();
      _trades  = trd;   renderTrades();
      appendLogs(logs);
    } catch(e) {}
  }
}, 1500);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(_HTML)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8765, reload=False, log_level="info")
