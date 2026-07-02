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

from . import analytics, db, kite_sync, nav_fetch
from .cas_import import CASImportError, import_cas
from .kite_sync import KiteAuthError

IST = pytz.timezone("Asia/Kolkata")
DIGEST_HOUR, DIGEST_MIN = 18, 30  # daily digest at 18:30 IST

log = logging.getLogger("wealth_os.bot")

CAS_DIR = Path(__file__).resolve().parent.parent / "data" / "cas"

HELP = (
    "<b>💰 Wealth OS</b>\n\n"
    "📄 Send your CAMS/KFintech CAS PDF to import mutual funds\n"
    "     (caption = PDF password, usually your PAN)\n\n"
    "/portfolio — mutual fund holdings\n"
    "/stocks — Zerodha equity holdings\n"
    "/sync — refresh holdings + cash from Zerodha\n"
    "/networth — total across MF + equity + cash\n"
    "/health — XIRR, allocation vs target, risk flags\n"
    "/refresh — pull latest MF NAVs from AMFI\n"
    "/digest — today's summary (auto-fires 18:30 IST daily)\n"
    "/login — get today's Zerodha login URL\n"
    "/token &lt;request_token&gt; — apply new Kite token\n"
    "/help — this message"
)


class WealthBot:
    def __init__(self, token: str, owner_chat_id: int):
        self.base = f"https://api.telegram.org/bot{token}"
        self.owner = int(owner_chat_id)
        self._pending_pdf: str | None = None  # path awaiting password
        self._session: aiohttp.ClientSession | None = None

    # ── Telegram plumbing ────────────────────────────────────────────
    async def api(self, method: str, **params):
        async with self._session.post(f"{self.base}/{method}", json=params) as r:
            data = await r.json()
        if not data.get("ok"):
            log.warning("telegram %s failed: %s", method, data.get("description"))
        return data.get("result")

    async def send(self, text: str):
        await self.api("sendMessage", chat_id=self.owner, text=text,
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
        await self.send("💰 Wealth OS online. /help for commands.")
        digest_task = asyncio.create_task(self._digest_loop())
        try:
            while True:
                try:
                    updates = await self.api("getUpdates", offset=offset,
                                             timeout=50, allowed_updates=["message"])
                    for u in updates or []:
                        offset = u["update_id"] + 1
                        try:
                            await self._handle(u.get("message") or {})
                        except Exception:
                            log.exception("handler error")
                            await self.send("⚠️ Error handling that — check logs.")
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    await asyncio.sleep(5)
        finally:
            digest_task.cancel()
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
        if not name.lower().endswith(".pdf"):
            await self.send("I only understand CAS PDFs for now.")
            return
        await self.send(f"📥 Receiving <b>{html.escape(name)}</b>…")
        path = await self._download(doc["file_id"], CAS_DIR / name)
        if caption:
            await self._import(str(path), caption.strip())
        else:
            self._pending_pdf = str(path)
            await self.send("🔑 Reply with the PDF password (usually your PAN, uppercase).")

    async def _import(self, path: str, password: str):
        await self.send("⏳ Parsing CAS…")
        try:
            summary = await asyncio.get_running_loop().run_in_executor(
                None, import_cas, path, password)
        except CASImportError as e:
            await self.send(f"❌ Parse failed: {html.escape(str(e)[:200])}\n"
                            "Wrong password? Send the PDF again.")
            self._pending_pdf = None
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
        lines.append(f"\nNAVs as of {nav_date} (CAS import)")
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
