"""
telegram_controller.py — V2
============================
Telegram Bot command handler + live status broadcaster for SentiStack V2.

Features:
  1. /mode command — choose trading mode interactively at startup
  2. /status — show current bot state on demand
  3. /stop / /pause / /resume — control trading
  4. Periodic status broadcasts every 30 min during market hours
  5. Clean human-readable trade notifications

Telegram inline keyboard flow:
  Bot starts → sends mode selection keyboard → user picks → bot confirms → trading begins

Commands:
  /start   — show welcome + mode selector
  /mode    — change trading mode mid-session
  /status  — full pipeline status snapshot
  /pause   — pause order execution (monitoring continues)
  /resume  — resume order execution
  /stop    — graceful shutdown
  /help    — list all commands
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional

import aiohttp

from config import settings

logger = logging.getLogger(__name__)

_IST = timezone(timedelta(hours=5, minutes=30))


def _ist_now() -> datetime:
    return datetime.now(_IST)


def _fmt_time() -> str:
    return _ist_now().strftime("%d %b %Y  %H:%M:%S IST")


# ---------------------------------------------------------------------------
# Trading Mode
# ---------------------------------------------------------------------------

class TradingMode(Enum):
    FULL          = "full"           # sentiment + GRI + ML (requires Gemini)
    GRI_ONLY      = "gri_only"       # GRI + microstructure, no Gemini
    PAPER_MONITOR = "paper_monitor"  # paper trade but don't place orders


MODE_DESCRIPTIONS = {
    TradingMode.FULL: (
        "🧠 <b>Full Mode</b>\n"
        "Uses Gemini sentiment + GRI + XGBoost ML.\n"
        "<i>Requires Gemini API quota. Best signal quality.</i>"
    ),
    TradingMode.GRI_ONLY: (
        "⚡ <b>GRI-Only Mode</b>\n"
        "Trades on geopolitical risk + microstructure (MLOFI/OFI).\n"
        "<i>No Gemini needed. Works even when quota is exhausted.</i>"
    ),
    TradingMode.PAPER_MONITOR: (
        "👁 <b>Monitor Only</b>\n"
        "Runs full pipeline but suppresses all order execution.\n"
        "<i>Safe observation mode.</i>"
    ),
}


# ---------------------------------------------------------------------------
# Bot State (shared with main loop)
# ---------------------------------------------------------------------------

@dataclass
class BotState:
    mode:           TradingMode  = TradingMode.GRI_ONLY
    paused:         bool         = False
    mode_confirmed: bool         = False     # True once user has chosen mode
    start_time:     datetime     = field(default_factory=_ist_now)

    # Live metrics (updated by strategy_loop)
    last_gri:       float        = 0.0
    last_gri_level: str          = "UNKNOWN"
    last_vix:       float        = 0.0
    last_usdinr:    float        = 0.0
    last_sentiment: float        = 0.0
    last_sentiment_class: str    = "Neutral"
    last_alpha_mult: float       = 1.0
    last_kelly_mult: float       = 1.0
    signals_today:  int          = 0
    trades_today:   int          = 0
    top_symbols:    list         = field(default_factory=list)
    active_symbols: int          = 0
    gemini_working: bool         = False
    last_update:    str          = ""

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.last_update = _fmt_time()


# ---------------------------------------------------------------------------
# Telegram Controller
# ---------------------------------------------------------------------------

class TelegramController:
    """
    Handles Telegram bot commands and periodic status updates.
    Integrates with the main strategy loop via shared BotState.
    """

    def __init__(self, state: BotState) -> None:
        cfg = settings.telegram
        self._token    = cfg.BOT_TOKEN
        self._chat_id  = str(cfg.CHAT_ID)
        self._base_url = f"{cfg.API_BASE}{self._token}"
        self._timeout  = aiohttp.ClientTimeout(total=10)
        self._state    = state
        self._offset   = 0
        self._mode_chosen_event = asyncio.Event()
        # B-18 FIX: replace SystemExit with a proper asyncio Event so /stop
        # goes through GracefulShutdown instead of bypassing it with SystemExit.
        self._stop_event = asyncio.Event()
        if state.mode_confirmed:
            self._mode_chosen_event.set()

    @property
    def stop_requested(self) -> bool:
        """True if the Telegram /stop command has been received."""
        return self._stop_event.is_set()

    async def wait_for_stop(self) -> None:
        """Await until /stop is received via Telegram."""
        await self._stop_event.wait()

    # ── HTTP helpers ──────────────────────────────────────────────────────

    async def _post(self, method: str, payload: Dict) -> Optional[Dict]:
        url = f"{self._base_url}/{method}"
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as s:
                async with s.post(url, json=payload) as r:
                    if r.status == 200:
                        return await r.json()
        except Exception as exc:
            logger.debug("Telegram post failed: %s", exc)
        return None

    async def send(self, text: str, reply_markup: Optional[Dict] = None) -> None:
        payload: Dict = {
            "chat_id":    self._chat_id,
            "text":       text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        await self._post("sendMessage", payload)

    async def _answer_callback(self, callback_id: str, text: str = "") -> None:
        await self._post("answerCallbackQuery",
                         {"callback_query_id": callback_id, "text": text})

    # ── Mode selector keyboard ────────────────────────────────────────────

    def _mode_keyboard(self) -> Dict:
        return {
            "inline_keyboard": [
                [{"text": "🧠 Full (Gemini + GRI + ML)",
                  "callback_data": "mode_full"}],
                [{"text": "⚡ GRI-Only (No Gemini needed)",
                  "callback_data": "mode_gri_only"}],
                [{"text": "👁 Monitor Only (no orders)",
                  "callback_data": "mode_paper_monitor"}],
            ]
        }

    async def send_mode_selector(self) -> None:
        gemini_status = "✅ Gemini quota available" if self._state.gemini_working \
                        else "⚠️ Gemini quota exhausted — GRI-Only recommended"
        text = (
            f"🤖 <b>SentiStack V2 — Choose Trading Mode</b>\n"
            f"{'─' * 34}\n"
            f"🕐 <b>Started:</b> {_fmt_time()}\n"
            f"🌐 <b>Gemini:</b> {gemini_status}\n\n"
            f"Please select how you want the bot to trade today:\n"
        )
        await self.send(text, reply_markup=self._mode_keyboard())

    # ── Status message ────────────────────────────────────────────────────

    def _status_text(self, prefix: str = "📊") -> str:
        s = self._state
        mode_labels = {
            TradingMode.FULL:          "🧠 Full (Gemini+GRI+ML)",
            TradingMode.GRI_ONLY:      "⚡ GRI-Only",
            TradingMode.PAPER_MONITOR: "👁 Monitor Only",
        }
        paused_str = "⏸ PAUSED" if s.paused else "▶️ RUNNING"
        gri_emoji  = "🟢" if s.last_gri < 0.30 else ("🟡" if s.last_gri < 0.50 else "🔴")
        sent_emoji = {"Fear": "😨", "Excitement": "🚀", "Neutral": "😐"}.get(
            s.last_sentiment_class, "😐"
        )
        gemini_str = "✅ Working" if s.gemini_working else "❌ Quota exhausted"
        elapsed = _ist_now() - s.start_time
        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
        mins = rem // 60

        top = ", ".join(s.top_symbols[:3]) if s.top_symbols else "loading..."

        return (
            f"{prefix} <b>SentiStack V2 — Status</b>\n"
            f"{'─' * 34}\n"
            f"🕐 <b>Time:</b>        {_fmt_time()}\n"
            f"⏱ <b>Uptime:</b>      {hours}h {mins}m\n"
            f"{'─' * 34}\n"
            f"⚙️ <b>Mode:</b>        {mode_labels[s.mode]}\n"
            f"🎮 <b>State:</b>       {paused_str}\n"
            f"🔢 <b>Universe:</b>    {s.active_symbols} symbols\n"
            f"🏆 <b>Top 3:</b>       {top}\n"
            f"{'─' * 34}\n"
            f"🌐 <b>GRI:</b>         {gri_emoji} {s.last_gri:.3f} ({s.last_gri_level})\n"
            f"📈 <b>VIX:</b>         {s.last_vix:.1f}\n"
            f"💱 <b>USD/INR:</b>     {s.last_usdinr:.2f}\n"
            f"⚡ <b>α mult:</b>      {s.last_alpha_mult:.2f}×\n"
            f"🎲 <b>Kelly mult:</b>  {s.last_kelly_mult:.2f}×\n"
            f"{'─' * 34}\n"
            f"🧠 <b>Gemini:</b>      {gemini_str}\n"
            f"{sent_emoji} <b>Sentiment:</b>   {s.last_sentiment_class} "
            f"({s.last_sentiment:+.3f})\n"
            f"{'─' * 34}\n"
            f"📡 <b>Signals today:</b> {s.signals_today}\n"
            f"💼 <b>Trades today:</b>  {s.trades_today}\n"
            f"🕐 <b>Last update:</b>   {s.last_update}\n"
        )

    # ── Command handlers ──────────────────────────────────────────────────

    async def _handle_command(self, text: str, chat_id: str) -> None:
        cmd = text.split()[0].lower().replace("@", "").split("@")[0]

        if cmd in ("/start", "/help"):
            await self.send(
                # B-13 FIX: was a plain string literal "{'─' * 28}" not an f-string
                f"🤖 <b>SentiStack V2 Commands</b>\n"
                f"{'─' * 28}\n"
                "/mode — Change trading mode\n"
                "/status — Full pipeline status\n"
                "/pause — Pause order execution\n"
                "/resume — Resume order execution\n"
                "/stop — Graceful shutdown\n"
                "/help — This message\n"
            )
            if not self._state.mode_confirmed:
                await self.send_mode_selector()

        elif cmd == "/mode":
            await self.send_mode_selector()

        elif cmd == "/status":
            await self.send(self._status_text())

        elif cmd == "/pause":
            self._state.paused = True
            await self.send("⏸ <b>Trading paused.</b> Monitoring continues. Send /resume to restart.")

        elif cmd == "/resume":
            self._state.paused = False
            await self.send("▶️ <b>Trading resumed.</b>")

        elif cmd == "/stop":
            await self.send("🛑 <b>Shutdown requested.</b> Bot will stop gracefully.")
            # B-18 FIX: raising SystemExit inside an async coroutine bypasses
            # GracefulShutdown and leaves open positions uncleared.
            # Instead, set the stop event — main() can await it alongside SIGINT.
            self._stop_event.set()

    async def _handle_callback(self, callback: Dict) -> None:
        data    = callback.get("data", "")
        cb_id   = callback["id"]
        chat_id = str(callback["message"]["chat"]["id"])

        mode_map = {
            "mode_full":          TradingMode.FULL,
            "mode_gri_only":      TradingMode.GRI_ONLY,
            "mode_paper_monitor": TradingMode.PAPER_MONITOR,
        }
        if data in mode_map:
            chosen = mode_map[data]
            self._state.mode          = chosen
            self._state.mode_confirmed = True
            self._mode_chosen_event.set()

            await self._answer_callback(cb_id, "Mode selected!")
            await self.send(
                f"✅ <b>Mode set: {chosen.value.replace('_', ' ').title()}</b>\n\n"
                f"{MODE_DESCRIPTIONS[chosen]}\n\n"
                f"🚀 Bot is now trading in this mode.\n"
                f"Send /status anytime for a live update."
            )
            logger.info("Trading mode set to: %s", chosen.value)

    # ── Polling loop ──────────────────────────────────────────────────────

    async def poll_loop(self) -> None:
        """Long-poll Telegram for updates (commands + button presses)."""
        logger.info("TelegramController: polling started.")
        while True:
            try:
                result = await self._post("getUpdates", {
                    "offset": self._offset,
                    "timeout": 20,
                    "allowed_updates": ["message", "callback_query"],
                })
                if result and result.get("ok"):
                    for update in result.get("result", []):
                        self._offset = update["update_id"] + 1
                        if "message" in update:
                            msg  = update["message"]
                            text = msg.get("text", "")
                            cid  = str(msg["chat"]["id"])
                            if text.startswith("/"):
                                await self._handle_command(text, cid)
                        elif "callback_query" in update:
                            await self._handle_callback(update["callback_query"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Telegram poll error: %s", exc)
                await asyncio.sleep(5)

    # ── Status broadcast loop ─────────────────────────────────────────────

    async def status_broadcast_loop(self, interval_s: int = 1800) -> None:
        """Send status update every 30 min during market hours."""
        await asyncio.sleep(300)  # wait 5 min for first strategy cycle to complete
        while True:
            try:
                now = _ist_now()
                is_market = (
                    now.weekday() < 5
                    and now.replace(hour=9, minute=15) <= now <= now.replace(hour=15, minute=30)
                )
                if is_market:
                    await self.send(self._status_text("🔔"))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Telegram broadcast error: %s", exc)
            await asyncio.sleep(interval_s)

    # ── Wait for mode selection ───────────────────────────────────────────

    async def wait_for_mode(self, timeout_s: int = 120) -> TradingMode:
        """
        Wait up to timeout_s for the user to choose a mode via Telegram.
        Falls back to GRI_ONLY if no response.
        """
        try:
            await asyncio.wait_for(self._mode_chosen_event.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "No mode selected in %ds — defaulting to GRI_ONLY.", timeout_s
            )
            self._state.mode           = TradingMode.GRI_ONLY
            self._state.mode_confirmed = True
            await self.send(
                f"⏱ <b>No response in {timeout_s}s.</b>\n"
                f"Defaulting to ⚡ <b>GRI-Only mode</b>.\n"
                f"Send /mode to change anytime."
            )
        return self._state.mode
