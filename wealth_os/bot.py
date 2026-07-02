"""Wealth OS — Telegram bot (long-polling, aiohttp only, owner-locked).

Flow for CAS import:
  1. Send the CAS PDF to the bot. Put the password in the caption, OR
  2. send it without caption and reply with the password when prompted.
The password message is deleted from the chat after use.
"""
from __future__ import annotations

import asyncio
import html
import logging
from pathlib import Path

import aiohttp

from . import db
from .cas_import import CASImportError, import_cas

log = logging.getLogger("wealth_os.bot")

CAS_DIR = Path(__file__).resolve().parent.parent / "data" / "cas"

HELP = (
    "<b>💰 Wealth OS</b>\n\n"
    "📄 Send your CAMS/KFintech CAS PDF to import mutual funds\n"
    "     (caption = PDF password, usually your PAN)\n\n"
    "/portfolio — mutual fund holdings\n"
    "/networth — total across MF + equity\n"
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
        elif text.startswith("/networth"):
            await self.send(self._networth_card())
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

    def _networth_card(self) -> str:
        n = db.networth()
        last = db.get_meta("last_cas_import", "never")
        return (
            "<b>💼 Net Worth</b>\n\n"
            f"Mutual funds: ₹{n['mf']:,.0f}\n"
            f"Equity (Zerodha): ₹{n['equity']:,.0f}"
            f"{'' if n['equity'] else '  (sync coming in T2)'}\n"
            f"<b>Total: ₹{n['total']:,.0f}</b>\n\n"
            f"Last CAS import: {last[:10]}"
        )
