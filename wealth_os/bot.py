"""Wealth OS — Telegram bot (long-polling, aiohttp only, owner-locked).

Flow for CAS import:
  1. Send the CAS PDF to the bot. Put the password in the caption, OR
  2. send it without caption and reply with the password when prompted.
The password message is deleted from the chat after use.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp
import pytz

from . import (advisor, analytics, backup, db, goals, kite_sync, nav_fetch,
               quality, swing, tax)
from .cas_import import CASImportError, import_cas
from .kite_sync import KiteAuthError
from .kuvera_import import KuveraImportError, import_kuvera_pdf

IST = pytz.timezone("Asia/Kolkata")
DIGEST_HOUR, DIGEST_MIN = 18, 30    # daily digest at 18:30 IST
BACKUP_HOUR, BACKUP_MIN = 23, 0     # nightly backup at 23:00 IST
HEARTBEAT = Path(__file__).resolve().parent.parent / "data" / "heartbeat"

log = logging.getLogger("wealth_os.bot")

CAS_DIR = Path(__file__).resolve().parent.parent / "data" / "cas"

HELP = (
    "<b>💰 Wealth OS</b>\n\n"
    "📄 Send your CAMS/KFintech CAS PDF (caption = password if protected)\n"
    "     or a Kuvera Portfolio Holding Statement PDF\n"
    "📄 Send your Zerodha Console tradebook CSV for equity tax data\n"
    "📄 Send a screener.in quality-screen CSV to veto weak swing picks\n\n"
    "/ask &lt;question&gt; — AI answers about YOUR portfolio (advisory only)\n"
    "/tax — FY capital gains + est. tax + equity XIRR\n"
    "/harvest — LTCG-exemption + tax-loss opportunities\n"
    "/portfolio — mutual fund holdings\n"
    "/stocks — Zerodha equity holdings\n"
    "/sync — refresh holdings + cash from Zerodha\n"
    "/networth — total across MF + equity + cash\n"
    "/health — XIRR, allocation vs target, risk flags\n"
    "/goals — goal progress | /goal add Name 25L 5 [prio]\n"
    "/plan — split monthly surplus across goals\n"
    "/rebalance — 5/25 drift check → approval cards\n"
    "/recs — pending recommendations\n"
    "/surplus 60000 — set monthly investable surplus\n"
    "/target 65 20 10 5 — set eq/debt/gold/cash %\n"
    "/refresh — pull latest MF NAVs from AMFI\n"
    "/digest — today's summary (auto-fires 18:30 IST daily)\n"
    "/backtest — validate swing sleeve on real EOD data (slow)\n"
    "/screen — swing picks (locked until gate + 90d paper)\n"
    "/trend — net worth history sparkline\n"
    "/status — service health, db, last backup\n"
    "/backup — snapshot db + send it here (send a backup back to restore)\n"
    "/login — get today's Zerodha login URL\n"
    "/token &lt;request_token&gt; — apply new Kite token\n"
    "/help — this message"
)


class WealthBot:
    def __init__(self, token: str, owner_chat_id: int):
        self.base = f"https://api.telegram.org/bot{token}"
        self.owner = int(owner_chat_id)
        self._pending_pdf: str | None = None      # path awaiting password
        self._pending_restore: str | None = None  # backup file awaiting confirm
        self._session: aiohttp.ClientSession | None = None

    # ── Telegram plumbing ────────────────────────────────────────────
    async def api(self, method: str, **params):
        async with self._session.post(f"{self.base}/{method}", json=params) as r:
            data = await r.json()
        if not data.get("ok"):
            log.warning("telegram %s failed: %s", method, data.get("description"))
        return data.get("result")

    async def send(self, text: str):
        # Telegram rejects messages >4096 chars — chunk on line boundaries
        chunks, cur = [], ""
        for line in text.split("\n"):
            if len(cur) + len(line) + 1 > 3900:
                chunks.append(cur)
                cur = line
            else:
                cur = f"{cur}\n{line}" if cur else line
        chunks.append(cur)
        for chunk in chunks:
            await self.api("sendMessage", chat_id=self.owner, text=chunk,
                           parse_mode="HTML", disable_web_page_preview=True)

    async def _download(self, file_id: str, dest: Path) -> Path:
        info = await self.api("getFile", file_id=file_id)
        url = f"{self.base.replace('/bot', '/file/bot')}/{info['file_path']}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with self._session.get(url) as r:
            dest.write_bytes(await r.read())
        return dest

    # ── Main loop ────────────────────────────────────────────────────
    async def run(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=90))
        offset = 0
        self._started = datetime.now(IST)
        await self.send("💰 Wealth OS online. /help for commands.")
        digest_task = asyncio.create_task(self._digest_loop())
        backup_task = asyncio.create_task(self._backup_loop())
        try:
            while True:
                try:
                    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
                    HEARTBEAT.touch()
                except OSError:
                    pass
                try:
                    updates = await self.api(
                        "getUpdates", offset=offset, timeout=50,
                        allowed_updates=["message", "callback_query"])
                    for u in updates or []:
                        offset = u["update_id"] + 1
                        try:
                            if cq := u.get("callback_query"):
                                await self._handle_callback(cq)
                            else:
                                await self._handle(u.get("message") or {})
                        except Exception as e:
                            log.exception("handler error")
                            await self.send(f"⚠️ Error ({type(e).__name__}): "
                                            f"{html.escape(str(e)[:200])}")
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(5)
        finally:
            digest_task.cancel()
            backup_task.cancel()
            await self._session.close()

    # ── Handlers ─────────────────────────────────────────────────────
    async def _handle(self, msg: dict):
        chat_id = (msg.get("chat") or {}).get("id")
        if chat_id != self.owner:
            return  # owner-locked

        if doc := msg.get("document"):
            await self._on_document(doc, msg.get("caption"))
            return

        text = (msg.get("text") or "").strip()
        if not text:
            return
        if text.startswith("/start") or text.startswith("/help"):
            await self.send(HELP)
        elif text.startswith("/portfolio"):
            await self.send(self._portfolio_card())
        elif text.startswith("/stocks"):
            await self.send(self._stocks_card())
        elif text.startswith("/networth"):
            await self.send(self._networth_card())
        elif text.startswith("/sync"):
            await self._sync_kite()
        elif text.startswith("/ask"):
            q = text[4:].strip()
            if not q:
                await self.send("Usage: /ask why is my equity allocation high?")
            else:
                await self.send("🤔 Thinking…")
                ctx_answer = await advisor.ask(self._session, q)
                await self.send(ctx_answer)
        elif text.startswith("/goal "):
            await self._goal_cmd(text)
        elif text.startswith("/goals"):
            await self.send(self._goals_card())
        elif text.startswith("/plan"):
            await self.send(self._plan_card())
        elif text.startswith("/rebalance"):
            await self._rebalance_cmd()
        elif text.startswith("/recs"):
            await self._recs_cmd()
        elif text.startswith("/surplus"):
            await self._set_meta_cmd(text, "monthly_surplus",
                                     "Monthly surplus set to ₹{:,.0f}")
        elif text.startswith("/target"):
            await self._target_cmd(text)
        elif text.startswith("/backtest"):
            if getattr(self, "_backtest_task", None) and not self._backtest_task.done():
                await self.send("🧪 A backtest is already running — results will"
                                " arrive here when it finishes.")
            else:
                # background task: must NOT block the update loop, or the
                # heartbeat goes stale and the watchdog kills us mid-run
                self._backtest_task = asyncio.create_task(self._run_backtest())
        elif text.startswith("/screen"):
            await self._screen_cmd()
        elif text.startswith("/trend"):
            await self.send(self._trend_card())
        elif text.startswith("/status"):
            await self.send(self._status_card())
        elif text.startswith("/backup"):
            await self._do_backup(announce=True)
        elif text.startswith("/tax"):
            card = await asyncio.get_running_loop().run_in_executor(None, tax.tax_card)
            await self.send(card)
        elif text.startswith("/harvest"):
            card = await asyncio.get_running_loop().run_in_executor(None, tax.harvest_card)
            await self.send(card)
        elif text.startswith("/health"):
            card = await asyncio.get_running_loop().run_in_executor(
                None, analytics.health_card)
            await self.send(card)
        elif text.startswith("/refresh"):
            await self._refresh_navs()
        elif text.startswith("/digest"):
            await self._daily_digest()
        elif text.startswith("/login"):
            await self._send_login_url()
        elif text.startswith("/token"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                await self.send("Usage: /token <request_token>")
            else:
                await self.api("deleteMessage", chat_id=self.owner,
                               message_id=msg["message_id"])
                await self._apply_token(parts[1].strip())
        elif self._pending_pdf:
            # treat as CAS password; remove it from chat history
            await self.api("deleteMessage", chat_id=self.owner,
                           message_id=msg["message_id"])
            await self._import(self._pending_pdf, text)
        else:
            await self.send("Unknown command. /help")

    async def _on_document(self, doc: dict, caption: str | None):
        name = doc.get("file_name", "file")
        if name.lower().endswith(".db.gz"):
            path = await self._download(doc["file_id"], CAS_DIR.parent / "restore" / name)
            self._pending_restore = str(path)
            await self.api(
                "sendMessage", chat_id=self.owner, parse_mode="HTML",
                text=f"🗄 Restore database from <b>{html.escape(name)}</b>?\n"
                     "Current data is backed up first, then REPLACED by this file.",
                reply_markup={"inline_keyboard": [[
                    {"text": "♻️ Restore", "callback_data": "restore:yes"},
                    {"text": "Cancel", "callback_data": "restore:no"},
                ]]})
            return
        if name.lower().endswith(".csv"):
            path = await self._download(doc["file_id"], CAS_DIR / name)
            loop = asyncio.get_running_loop()
            try:
                s = await loop.run_in_executor(
                    None, tax.import_tradebook_csv, str(path))
                await self.send(f"✅ Tradebook imported: {s['parsed']} trades parsed,"
                                f" {s['added']} new.\n/tax and /harvest now cover equity.")
                return
            except Exception:
                pass  # not a tradebook — try screener.in quality export
            try:
                s = await loop.run_in_executor(
                    None, quality.import_screener_csv, str(path))
                await self.send(
                    f"✅ Quality screen imported: {s['symbols']} symbols"
                    f" ({s['names']} names).\n/screen picks outside this list"
                    " will be marked ⚠️ VETO. Re-upload monthly to keep it fresh.")
            except Exception as e:
                await self.send("❌ CSV not recognised as Zerodha tradebook or"
                                f" screener.in export: {html.escape(str(e)[:150])}")
            return
        if not name.lower().endswith(".pdf"):
            await self.send("Send a CAS PDF or a Zerodha tradebook CSV.")
            return
        await self.send(f"📥 Receiving <b>{html.escape(name)}</b>…")
        path = await self._download(doc["file_id"], CAS_DIR / name)
        await self._import(str(path), (caption or "").strip())

    async def _import(self, path: str, password: str):
        """Try CAS first, then Kuvera; only ask for a password if the PDF
        is actually encrypted and none was given. Any unexpected failure is
        reported with its real message — never a silent 'check logs'."""
        try:
            await self._import_inner(path, password)
        except Exception as e:
            log.exception("statement import failed")
            self._pending_pdf = None
            await self.send(f"❌ Import error ({type(e).__name__}):"
                            f" {html.escape(str(e)[:300])}")

    async def _import_inner(self, path: str, password: str):
        await self.send("⏳ Parsing statement…")
        loop = asyncio.get_running_loop()
        try:
            summary = await loop.run_in_executor(None, import_cas, path, password)
        except CASImportError as cas_err:
            # not a CAS (or locked) — try Kuvera holding statement
            try:
                s = await loop.run_in_executor(None, import_kuvera_pdf, path)
            except KuveraImportError:
                msg = str(cas_err).lower()
                if "password" in msg or "decrypt" in msg or "encrypted" in msg:
                    self._pending_pdf = path
                    await self.send("🔒 This PDF is password-protected. Reply with"
                                    " the password (usually your PAN, uppercase).")
                else:
                    self._pending_pdf = None
                    await self.send(
                        f"❌ Couldn't parse: {html.escape(str(cas_err)[:150])}\n\n"
                        "Supported: CAMS/KFintech CAS (detailed) and Kuvera"
                        " Portfolio Holding Statement.")
                return
            self._pending_pdf = None
            ret = (s["value"] / s["invested"] - 1) if s["invested"] else 0
            carried = s.get("isins_carried", 0)
            isin_note = (
                f"🔗 {carried}/{s['schemes']} schemes matched to ISINs from your"
                " CAS — NAV refresh + XIRR keep working for those."
                if carried else
                "⚠️ Snapshot only — no ISINs/transactions, so NAV auto-refresh,"
                " XIRR and /tax stay limited. For full power send the CAMS CAS"
                " (camsonline.com → detailed statement).")
            await self.send(
                "✅ <b>Kuvera snapshot imported</b>\n"
                f"Schemes: {s['schemes']} | Invested: ₹{s['invested']:,.0f}\n"
                f"<b>Value: ₹{s['value']:,.0f}</b> ({ret:+.1%} absolute)\n\n"
                f"{isin_note}\n\n/portfolio for details")
            return
        self._pending_pdf = None
        await self.send(
            "✅ <b>CAS imported</b>\n"
            f"Period: {summary['period']}\n"
            f"Folios: {summary['folios']} | Active schemes: {summary['schemes_active']}"
            f" (of {summary['schemes_total']})\n"
            f"Transactions: {summary['transactions']}\n"
            f"<b>MF value: ₹{summary['value']:,.0f}</b>\n\n/portfolio for details"
        )

    # ── Kite sync ────────────────────────────────────────────────────
    async def _send_login_url(self):
        try:
            await self.send(
                "🔑 <b>Zerodha login</b>\n"
                f"1️⃣ Open: {kite_sync.login_url()}\n"
                "2️⃣ Log in, copy <code>request_token</code> from the redirect URL\n"
                "3️⃣ Send: /token XXXXXXXX"
            )
        except KiteAuthError as e:
            await self.send(f"❌ {html.escape(str(e))}")

    async def _apply_token(self, request_token: str):
        try:
            await kite_sync.exchange_request_token(self._session, request_token)
        except Exception as e:
            await self.send(f"❌ Token exchange failed: {html.escape(str(e)[:200])}")
            return
        await self.send("✅ Kite token saved for today.")
        await self._sync_kite()

    async def _sync_kite(self):
        await self.send("⏳ Syncing from Zerodha…")
        try:
            s = await kite_sync.sync(self._session)
        except KiteAuthError:
            await self._send_login_url()
            return
        except Exception as e:
            await self.send(f"❌ Sync failed: {html.escape(str(e)[:200])}")
            return
        sign = "🟢" if s["pnl"] >= 0 else "🔴"
        await self.send(
            "✅ <b>Zerodha synced</b>\n"
            f"Stocks: {s['positions']} | Value: ₹{s['value']:,.0f}\n"
            f"{sign} Unrealised P&L: ₹{s['pnl']:,.0f}\n"
            f"Cash: ₹{s['cash']:,.0f}\n\n/networth for the full picture"
        )

    # ── Swing sleeve (gated) ─────────────────────────────────────────
    async def _run_backtest(self):
        await self.send("🧪 Backtest started: downloading ~6y EOD for the"
                        " universe, then walk-forward. Takes 5–20 min on this"
                        " VM — I'll ping progress and post the report here."
                        " The bot stays responsive meanwhile.")
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, swing.run_and_gate)
        mins = 0
        try:
            while True:
                try:
                    res = await asyncio.wait_for(asyncio.shield(fut), timeout=300)
                    break
                except asyncio.TimeoutError:
                    mins += 5
                    if mins >= 45:
                        await self.send("❌ Backtest exceeded 45 min — likely a"
                                        " data-download hang. Try again later or"
                                        " run on the VM: venv/bin/python -m wealth_os.swing")
                        return
                    await self.send(f"🧪 …still running ({mins} min). Normal on"
                                    " a small VM — Yahoo throttles bulk downloads.")
            await self.send(swing.report_card(res))
        except Exception as e:
            await self.send(f"❌ Backtest failed ({type(e).__name__}):"
                            f" {html.escape(str(e)[:250])}\n"
                            "If it's an import error: venv/bin/pip install"
                            " yfinance pandas && restart.")

    async def _screen_cmd(self):
        st = swing.gate_status()
        if not st["gate"]:
            await self.send("🔒 Swing sleeve locked — no validation run yet.\n"
                            "Run /backtest first (real data, Sharpe>1.0 required).")
            return
        if not st["gate"].get("passed"):
            await self.send(f"🔒 Swing sleeve locked — last backtest failed the gate"
                            f" (Sharpe {st['gate']['sharpe']:.2f},"
                            f" maxDD {st['gate']['maxdd']:.0%}).")
            return
        mode = ("LIVE — cap 20% of net worth" if st["live_ok"] else
                f"PAPER — day {st['paper_days']}/{swing.PAPER_DAYS}; track, don't buy")
        await self.send(f"⏳ Screening universe… ({mode})")
        try:
            picks = await asyncio.get_running_loop().run_in_executor(None, swing.screen)
        except Exception as e:
            await self.send(f"❌ Screen failed: {html.escape(str(e)[:200])}")
            return
        if not picks:
            await self.send("No names pass the filters right now (weak tape) —"
                            " that is information, not a bug. Stay in the core.")
            return
        # quality layer: manual screener list wins if uploaded, else
        # fundamentals are fetched automatically (yfinance, 30-day cache)
        manual = quality.universe() is not None
        auto = {}
        if not manual:
            auto = await asyncio.get_running_loop().run_in_executor(
                None, quality.auto_quality, [p["symbol"] for p in picks])
        lines = [f"<b>📋 Swing Screen — {mode}</b>\n"]
        vetoed = 0
        for p in picks:
            if manual:
                ok = quality.is_approved(p["symbol"])
                why = "not in quality screen"
            else:
                a = auto.get(p["symbol"], {})
                ok, why = a.get("ok"), a.get("why", "")
            mark = (f"  ⚠️ VETO ({html.escape(why)})" if ok is False
                    else f"  ✓ {html.escape(why)}" if ok else "  · no fundamentals data")
            vetoed += 1 if ok is False else 0
            lines.append(f"• <b>{html.escape(p['symbol'])}</b> @ ₹{p['price']:,.1f}"
                         f" | {p['qty']} sh (₹{p['alloc']:,.0f})"
                         f" | score {p['score']:.2f}{mark}")
        if vetoed:
            lines.append(f"\n⚠️ {vetoed} pick(s) failed quality checks —"
                         " skip those, momentum in weak businesses is fragile.")
        src = ("your screener.in list" if manual
               else "auto fundamentals (ROE≥12%, profitable, D/E<1.5×; banks exempt from D/E)")
        lines.append(f"<i>Quality source: {src}</i>")
        lines.append("\nEqual-weight, exit on rank>30 at monthly review."
                     " Approve any trade explicitly — nothing is automatic.")
        await self.send("\n".join(lines))
        if st["live_ok"]:
            await self._send_reco(
                "swing", "Monthly swing rebalance",
                f"{len(picks)} names, ₹{picks[0]['alloc'] * len(picks):,.0f} sleeve"
                " (20% cap). Details in /screen above. Approve to proceed with"
                " manual/Kite orders.")

    # ── Backup + status ──────────────────────────────────────────────
    async def send_document(self, path: Path, caption: str = ""):
        form = aiohttp.FormData()
        form.add_field("chat_id", str(self.owner))
        if caption:
            form.add_field("caption", caption)
        form.add_field("document", path.read_bytes(), filename=path.name)
        async with self._session.post(f"{self.base}/sendDocument", data=form) as r:
            data = await r.json()
        if not data.get("ok"):
            log.warning("sendDocument failed: %s", data.get("description"))
        return data.get("ok", False)

    async def _backup_loop(self):
        while True:
            now = datetime.now(IST)
            target = now.replace(hour=BACKUP_HOUR, minute=BACKUP_MIN,
                                 second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            await asyncio.sleep((target - now).total_seconds())
            try:
                await self._do_backup()
            except Exception:
                log.exception("nightly backup failed")
                await self.send("⚠️ Nightly backup failed — check logs.")

    async def _do_backup(self, announce: bool = False):
        path = await asyncio.get_running_loop().run_in_executor(
            None, backup.make_backup)
        size_kb = path.stat().st_size / 1024
        ok = await self.send_document(
            path, caption=f"🗄 Wealth OS backup {path.name} ({size_kb:.0f} KB). "
                          "Keep this — restore by placing it back as data/wealth.db")
        if announce and not ok:
            await self.send("⚠️ Backup created locally but Telegram upload failed.")

    async def _do_restore(self, gz_path: str):
        import gzip
        import shutil
        safety = None
        try:
            safety = await asyncio.get_running_loop().run_in_executor(
                None, backup.make_backup)
            with gzip.open(gz_path, "rb") as f_in, open(db.DB_PATH, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            n = db.networth()  # validates the restored db opens + has schema
        except Exception as e:
            await self.send(f"❌ Restore failed ({type(e).__name__}):"
                            f" {html.escape(str(e)[:200])}\n"
                            f"Pre-restore safety backup: {safety.name if safety else '—'}")
            return
        await self.send(f"♻️ <b>Database restored.</b> Net worth in restored data:"
                        f" ₹{n['total']:,.0f}\n"
                        f"(previous db saved as {safety.name})\n/status to verify")

    def _trend_card(self) -> str:
        rows = db.snapshots_recent(30)
        if len(rows) < 2:
            return ("Not enough history yet — snapshots accumulate with each"
                    " daily digest. Check back in a few days.")
        rows = list(reversed(rows))  # oldest → newest
        vals = [r["total"] for r in rows]
        lo, hi = min(vals), max(vals)
        blocks = "▁▂▃▄▅▆▇█"
        spark = "".join(
            blocks[int((v - lo) / (hi - lo) * 7)] if hi > lo else blocks[0]
            for v in vals)
        d = vals[-1] - vals[0]
        pct = d / vals[0] * 100 if vals[0] else 0
        return (
            f"<b>📈 Net Worth — last {len(vals)} snapshots</b>\n\n"
            f"<code>{spark}</code>\n"
            f"{rows[0]['date']}: ₹{vals[0]:,.0f}\n"
            f"{rows[-1]['date']}: ₹{vals[-1]:,.0f}\n"
            f"{'🟢' if d >= 0 else '🔴'} {'+' if d >= 0 else ''}₹{d:,.0f} ({pct:+.1f}%)"
        )

    def _status_card(self) -> str:
        now = datetime.now(IST)
        up = now - getattr(self, "_started", now)
        days, rem = divmod(int(up.total_seconds()), 86400)
        hrs, rem = divmod(rem, 3600)
        db_size = db.DB_PATH.stat().st_size / 1024 if db.DB_PATH.exists() else 0
        n = db.networth()
        token_ok = kite_sync.cached_token() is not None
        return (
            "<b>🩺 Status</b>\n\n"
            f"Uptime: {days}d {hrs}h {rem // 60}m\n"
            f"DB: {db_size:,.0f} KB | Net worth tracked: ₹{n['total']:,.0f}\n"
            f"Kite token today: {'✅' if token_ok else '❌ (/login)'}\n"
            f"Last CAS import: {(db.get_meta('last_cas_import') or 'never')[:10]}\n"
            f"Last equity sync: {(db.get_meta('last_equity_sync') or 'never')[:10]}\n"
            f"Last backup: {(db.get_meta('last_backup') or 'never')[:16]}\n"
            f"Pending recommendations: {len(db.pending_recommendations())}\n"
            f"Digest 18:30 IST | Backup 23:00 IST | Watchdog: systemd timer"
        )

    # ── Goals, plan, rebalance, approvals ────────────────────────────
    @staticmethod
    def _parse_amt(s: str) -> float:
        s = s.strip().lower().replace(",", "").replace("₹", "")
        mult = 1.0
        if s.endswith("cr"):
            mult, s = 1e7, s[:-2]
        elif s.endswith("l"):
            mult, s = 1e5, s[:-1]
        elif s.endswith("k"):
            mult, s = 1e3, s[:-1]
        return float(s) * mult

    async def _goal_cmd(self, text: str):
        # /goal add Name 25L 5 [priority]   |   /goal del Name
        parts = text.split()
        try:
            if parts[1] == "del":
                n = db.delete_goal(parts[2])
                await self.send("🗑 Deleted." if n else "No goal by that name.")
                return
            if parts[1] != "add":
                raise ValueError
            name, amount, years = parts[2], self._parse_amt(parts[3]), float(parts[4])
            prio = int(parts[5]) if len(parts) > 5 else 5
            target_date = (datetime.now(IST) +
                           timedelta(days=round(years * 365.25))).strftime("%Y-%m-%d")
            db.add_goal(name, amount, target_date, prio)
            sip = goals.required_sip(amount, round(years * 12))
            await self.send(
                f"🎯 Goal <b>{html.escape(name)}</b>: ₹{amount:,.0f} by {target_date}"
                f" (priority {prio})\nRequired SIP from zero: ₹{sip:,.0f}/mo"
                f" @ {goals.expected_return():.0%} — /goals for funded status")
        except (IndexError, ValueError):
            await self.send("Usage: /goal add House 25L 5 [priority]\n"
                            "       /goal del House")

    def _goals_card(self) -> str:
        st = goals.goal_status()
        if not st:
            return ("No goals yet. Start with an emergency fund:\n"
                    "/goal add Emergency 3L 1 1")
        lines = ["<b>🎯 Goals</b> (corpus waterfall by priority)\n"]
        for g in st:
            bar = "█" * round(g["funded_pct"] * 10) + "░" * (10 - round(g["funded_pct"] * 10))
            lines.append(
                f"<b>{html.escape(g['name'])}</b> — ₹{g['target']:,.0f} by {g['date']}\n"
                f"{bar} {g['funded_pct']:.0%} funded"
                + (f" | needs ₹{g['req_sip']:,.0f}/mo" if g["req_sip"] > 0 else " ✅"))
        return "\n".join(lines)

    def _plan_card(self) -> str:
        p = goals.monthly_plan()
        if not p["rows"]:
            return "No goals to plan for — /goal add first."
        lines = [f"<b>📆 Monthly Plan — ₹{p['surplus']:,.0f} surplus</b>\n"]
        for r in p["rows"]:
            if r["sip"] > 0:
                lines.append(f"→ ₹{r['sip']:,.0f} to <b>{html.escape(r['name'])}</b>")
        if p["unallocated"] > 0:
            lines.append(f"→ ₹{p['unallocated']:,.0f} unallocated — goals fully"
                         " covered; consider raising targets or equity")
        lines.append("\nRoute per /rebalance underweights. /surplus to adjust.")
        return "\n".join(lines)

    async def _rebalance_cmd(self):
        recos = goals.rebalance_recos()
        if not recos:
            await self.send("✅ Allocation within 5/25 bands — nothing to do.")
            return
        for r in recos:
            await self._send_reco(r["kind"], r["title"], r["detail"])

    async def _send_reco(self, kind: str, title: str, detail: str):
        rec_id = db.add_recommendation(kind, title, detail)
        await self.api(
            "sendMessage", chat_id=self.owner, parse_mode="HTML",
            text=f"💡 <b>#{rec_id} {html.escape(title)}</b>\n\n{html.escape(detail)}",
            reply_markup={"inline_keyboard": [[
                {"text": "✅ Approve", "callback_data": f"rec:approved:{rec_id}"},
                {"text": "❌ Reject", "callback_data": f"rec:rejected:{rec_id}"},
            ]]})

    async def _handle_callback(self, cq: dict):
        if (cq.get("from") or {}).get("id") != self.owner:
            return
        data = cq.get("data") or ""
        parts = data.split(":")
        if parts[0] == "restore":
            await self.api("answerCallbackQuery", callback_query_id=cq["id"])
            if parts[1] == "yes" and self._pending_restore:
                await self._do_restore(self._pending_restore)
            else:
                await self.send("Restore cancelled.")
            self._pending_restore = None
            return
        if len(parts) == 3 and parts[0] == "rec":
            _, status, rec_id = parts
            rec = db.decide_recommendation(int(rec_id), status)
            await self.api("answerCallbackQuery", callback_query_id=cq["id"],
                           text=f"Recommendation {status}")
            if rec and (m := cq.get("message")):
                mark = "✅ APPROVED" if status == "approved" else "❌ REJECTED"
                await self.api(
                    "editMessageText", chat_id=self.owner,
                    message_id=m["message_id"], parse_mode="HTML",
                    text=f"💡 <b>#{rec['id']} {html.escape(rec['title'])}</b>\n\n"
                         f"{html.escape(rec['detail'])}\n\n<b>{mark}</b>")

    async def _recs_cmd(self):
        pend = db.pending_recommendations()
        if not pend:
            await self.send("No pending recommendations.")
            return
        for r in pend:
            await self.api(
                "sendMessage", chat_id=self.owner, parse_mode="HTML",
                text=f"💡 <b>#{r['id']} {html.escape(r['title'])}</b>\n\n"
                     f"{html.escape(r['detail'])}",
                reply_markup={"inline_keyboard": [[
                    {"text": "✅ Approve", "callback_data": f"rec:approved:{r['id']}"},
                    {"text": "❌ Reject", "callback_data": f"rec:rejected:{r['id']}"},
                ]]})

    async def _set_meta_cmd(self, text: str, key: str, ok_fmt: str):
        try:
            val = self._parse_amt(text.split()[1])
        except (IndexError, ValueError):
            await self.send(f"Usage: /{key.split('_')[1]} 60000")
            return
        db.set_meta(key, str(val))
        await self.send(ok_fmt.format(val))

    async def _target_cmd(self, text: str):
        try:
            eq, debt, gold, cash = (float(x) for x in text.split()[1:5])
            assert abs(eq + debt + gold + cash - 100) < 0.01
        except (ValueError, AssertionError, IndexError):
            await self.send("Usage: /target 65 20 10 5 (must sum to 100)")
            return
        db.set_meta("target_alloc", json.dumps(
            {"equity": eq / 100, "debt": debt / 100,
             "gold": gold / 100, "cash": cash / 100}))
        await self.send(f"🎯 Target set: {eq:.0f}/{debt:.0f}/{gold:.0f}/{cash:.0f}"
                        " eq/debt/gold/cash. Run /rebalance to check drift.")

    # ── Daily refresh + digest ───────────────────────────────────────
    async def _digest_loop(self):
        while True:
            now = datetime.now(IST)
            target = now.replace(hour=DIGEST_HOUR, minute=DIGEST_MIN,
                                 second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            await asyncio.sleep((target - now).total_seconds())
            try:
                await self._daily_digest()
            except Exception:
                log.exception("daily digest failed")

    async def _refresh_navs(self):
        try:
            s = await nav_fetch.refresh_mf_navs(self._session)
        except Exception as e:
            await self.send(f"❌ AMFI refresh failed: {html.escape(str(e)[:150])}")
            return None
        await self.send(
            f"✅ NAVs refreshed: {s['updated']}/{s['held']} schemes"
            f" (as of {s['nav_date'] or '?'})")
        return s

    async def _daily_digest(self):
        # 1. refresh MF NAVs (best effort)
        nav_note = ""
        try:
            s = await nav_fetch.refresh_mf_navs(self._session)
            nav_note = f"NAVs: {s['updated']}/{s['held']} as of {s['nav_date'] or '?'}"
        except Exception as e:
            nav_note = f"⚠️ NAV refresh failed ({str(e)[:60]})"
        # 2. refresh equity via Kite if today's token is alive (best effort)
        eq_note = ""
        try:
            await kite_sync.sync(self._session)
        except KiteAuthError:
            eq_note = "⚠️ Equity prices stale — /login to refresh"
        except Exception as e:
            eq_note = f"⚠️ Equity sync failed ({str(e)[:60]})"
        # 3. snapshot + day change
        today = datetime.now(IST).strftime("%Y-%m-%d")
        n = db.snapshot_networth(today)
        lines = [f"<b>🌆 Daily Digest — {today}</b>\n",
                 f"<b>Net worth: ₹{n['total']:,.0f}</b>"]
        if n["prev_total"]:
            d = n["total"] - n["prev_total"]
            pct = d / n["prev_total"] * 100
            lines.append(f"{'🟢' if d >= 0 else '🔴'} {'+' if d >= 0 else ''}₹{d:,.0f}"
                         f" ({pct:+.2f}%) since {n['prev_date']}")
        lines.append(f"\nMF ₹{n['mf']:,.0f} | Equity ₹{n['equity']:,.0f}"
                     f" | Cash ₹{n['cash']:,.0f}")
        # 4. equity day movers
        movers = json.loads(db.get_meta("equity_movers", "[]") or "[]")
        if movers:
            lines.append("\n<b>Movers today</b>")
            for m in movers[:4]:
                lines.append(f"{'🟢' if m['day_pnl'] >= 0 else '🔴'} "
                             f"{html.escape(m['symbol'])} ₹{m['day_pnl']:,.0f}"
                             f" ({m['day_pct']:+.1f}%)")
        # 5. SIP heads-up (inferred from CAS transaction history)
        sips = db.recent_sips()
        if sips:
            lines.append(f"\n📆 {len(sips)} active SIP(s) detected —"
                         " keep balance funded")
        # 6. CRASH PROTOCOL — the pre-committed message for the bad year.
        #    Written now, while calm, so future-you reads it while scared.
        snaps = db.snapshots_recent(365)
        if snaps:
            peak = max(r["total"] for r in snaps)
            dd = (n["total"] - peak) / peak if peak else 0
            if dd <= -0.10:
                lines.append(
                    f"\n🛡 <b>Drawdown protocol active ({dd:.0%} from peak)</b>\n"
                    "This is the part of the plan we agreed on in advance:\n"
                    "1. SIPs keep firing — this month's units are the cheapest"
                    " you've been offered in a while.\n"
                    "2. Selling now converts a temporary loss into a permanent"
                    " one. Every -20% in Nifty's history recovered; the only"
                    " investors hurt were the ones who sold into it.\n"
                    "3. If you feel the urge to act: /rebalance — if bands"
                    " trigger, it will tell you to BUY equity cheap, not sell.\n"
                    "4. Check /trend monthly, not daily, until this passes.")
        # 7. Annual step-up nudge (first week of April — new FY, appraisals)
        now = datetime.now(IST)
        if (now.month == 4 and now.day <= 7
                and db.get_meta("stepup_nudged") != str(now.year)):
            db.set_meta("stepup_nudged", str(now.year))
            cur = goals.monthly_surplus()
            lines.append(
                f"\n📈 <b>Annual step-up time.</b> Surplus is ₹{cur:,.0f}/mo —"
                f" raising it 10% (/surplus {cur * 1.1:,.0f}) adds more to your"
                " final corpus than almost any return improvement. New FY,"
                " new increment: pay yourself first.")
        for note in (nav_note, eq_note):
            if note:
                lines.append(f"\n{note}")
        await self.send("\n".join(lines))

    # ── Cards ────────────────────────────────────────────────────────
    def _portfolio_card(self) -> str:
        rows = db.mf_holdings()
        if not rows:
            return "No MF holdings yet. Send me your CAS PDF (see /help)."
        total = sum(r["value"] or 0 for r in rows)
        lines = [f"<b>📊 Mutual Funds — ₹{total:,.0f}</b>\n"]
        for r in rows[:25]:
            pct = (r["value"] or 0) / total * 100 if total else 0
            lines.append(
                f"• {html.escape((r['scheme'] or '?')[:48])}\n"
                f"   ₹{(r['value'] or 0):,.0f}  ({pct:.1f}%)  "
                f"{(r['units'] or 0):,.2f}u @ ₹{(r['nav'] or 0):,.2f}"
            )
        if len(rows) > 25:
            lines.append(f"…and {len(rows) - 25} more")
        nav_date = rows[0]["nav_date"] if rows else "?"
        lines.append(f"\nNAVs as of {nav_date} (last import/refresh)")
        return "\n".join(lines)

    def _stocks_card(self) -> str:
        rows = db.equity_holdings()
        if not rows:
            return "No equity holdings yet — run /sync."
        total = sum(r["value"] or 0 for r in rows)
        lines = [f"<b>📈 Zerodha Equity — ₹{total:,.0f}</b>\n"]
        for r in rows[:30]:
            pnl = ((r["ltp"] or 0) - (r["avg_price"] or 0)) * (r["qty"] or 0)
            sign = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                f"• <b>{html.escape(r['symbol'])}</b> {r['qty']:.0f} @ ₹{r['avg_price']:,.2f}"
                f" → ₹{r['ltp']:,.2f}  {sign} ₹{pnl:,.0f}"
            )
        sync_at = db.get_meta("last_equity_sync", "never")
        lines.append(f"\nSynced: {sync_at[:16]}")
        return "\n".join(lines)

    def _networth_card(self) -> str:
        n = db.networth()
        cas = db.get_meta("last_cas_import", "never") or "never"
        eq_sync = db.get_meta("last_equity_sync", "never") or "never"
        return (
            "<b>💼 Net Worth</b>\n\n"
            f"Mutual funds: ₹{n['mf']:,.0f}\n"
            f"Equity (Zerodha): ₹{n['equity']:,.0f}\n"
            f"Cash (broker): ₹{n['cash']:,.0f}\n"
            f"<b>Total: ₹{n['total']:,.0f}</b>\n\n"
            f"Last CAS import: {cas[:10]} | Last sync: {eq_sync[:10]}"
        )
