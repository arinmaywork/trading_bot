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
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from enum import Enum
from pathlib import Path
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
        "👁 <b>Silent Observer</b>\n"
        "Runs full analysis pipeline but places NO orders and logs NO trades.\n"
        "<i>Use on holidays or for pure signal research. /pnl will show nothing.</i>\n"
        "<i>For simulated fills with P&amp;L tracking, use GRI-Only or Full with PAPER_TRADE=true in .env.</i>"
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

    # Token refresh state
    token_valid:    bool         = True   # set False on TokenException
    token_refresh_event: asyncio.Event = field(default_factory=asyncio.Event)

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
    paper_capital:  float        = field(
        default_factory=lambda: float(
            __import__("os").getenv("TOTAL_CAPITAL", "500000")
        )
    )
    # ── Trading guards set via Telegram ──────────────────────────────────
    no_new_buys:    bool         = False   # /nobuy  — block new BUY entries
    no_new_sells:   bool         = False   # /nosell — block new SELL/short entries
    # ── R-13: Portfolio risk budget state (set by portfolio_risk.py) ────
    risk_halt_scope:  str        = "NONE"     # NONE | DAY | WEEK | MONTH
    risk_halt_reason: str        = ""
    risk_halt_until:  Optional[datetime] = None   # when new entries may resume
    risk_day_pnl:     float      = 0.0
    risk_week_pnl:    float      = 0.0
    risk_month_pnl:   float      = 0.0
    risk_day_limit:   float      = 0.0        # absolute ₹ limit (cached for /risk)
    risk_week_limit:  float      = 0.0
    risk_month_limit: float      = 0.0
    risk_blacklist:   set        = field(default_factory=set)  # symbols halted today
    risk_last_check:  str        = ""
    # ── Task-5: Sector exposure + intraday MTM drawdown stop ────────────
    risk_sector_exposure: dict   = field(default_factory=dict)  # sector -> ₹ notional
    risk_sector_cap_inr:  float  = 0.0      # absolute ₹ per-sector cap (cached)
    risk_sector_blocked:  set    = field(default_factory=set)   # sectors at/over cap
    risk_mtm_pnl:         float  = 0.0      # realised + unrealised (₹)
    risk_mtm_limit:       float  = 0.0      # absolute ₹ stop (INTRADAY_MTM_STOP_PCT × cap)
    risk_mtm_stop_active: bool   = False    # True while intraday MTM stop is firing
    # ── R-15: Runtime paper-trade toggle arming (60-sec confirm window) ──
    live_mode_arm_until: Optional[datetime] = None
    # ── Task health — updated by each background task every cycle ────────
    task_heartbeats: dict        = field(default_factory=dict)
    # { task_name: {"last_beat": float, "cycles": int, "errors": int} }

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
        # Kite reference — populated by main.py via set_kite()
        self._kite             = None
        self._api_secret       = ""
        self._token_cache_file = Path(__file__).parent / ".kite_token"
        # Logbook reference — populated by main.py via set_logbook()
        self._logbook          = None
        # PortfolioRiskMonitor reference — set via set_portfolio_risk()
        self._portfolio_risk   = None
        # Broker-balance fetcher — awaitable set by main.py via set_margin_fetcher()
        self._margin_fetcher   = None
        # ModelRotator reference — populated by main.py via set_rotator()
        self._rotator          = None

    # ── Token hot-swap ────────────────────────────────────────────────────

    def set_kite(self, kite: Any, api_secret: str,
                 token_cache_file: Optional[Path] = None) -> None:
        """
        Register the live KiteConnect instance so the /login and /token
        commands can refresh the access token without restarting the bot.

        Call this once from main.py after the initial authentication:
            telegram_controller.set_kite(kite, settings.kite.API_SECRET)
        """
        self._kite             = kite
        self._api_secret       = api_secret
        self._token_cache_file = token_cache_file or (
            Path(__file__).parent / ".kite_token"
        )

    def set_logbook(self, logbook: Any) -> None:
        """
        Register the Logbook instance so /pnl can pull today's trade data.
        Call this once from main.py after the logbook is initialised.
        """
        self._logbook = logbook

    def set_rotator(self, rotator: Any) -> None:
        """
        Register the ModelRotator singleton so /resetquota can clear
        stale Gemini quota state without restarting the bot.
        Call once from main.py: tg_controller.set_rotator(_rotator)
        """
        self._rotator = rotator

    def set_portfolio_risk(self, monitor: Any) -> None:
        """
        Register the PortfolioRiskMonitor so /risk can pull a fresh budget.
        Call once from main.py.
        """
        self._portfolio_risk = monitor

    def set_margin_fetcher(self, fetcher) -> None:
        """
        Register an async callable that returns Zerodha's margin dict
        (kite.margins() schema). Used by /synccapital and /balance.
        """
        self._margin_fetcher = fetcher

    def _save_token_cache(self, token: str) -> None:
        try:
            self._token_cache_file.write_text(
                json.dumps({"access_token": token,
                            "generated_date": date.today().isoformat()}, indent=2)
            )
            self._token_cache_file.chmod(0o600)
        except OSError as exc:
            logger.warning("Could not write token cache: %s", exc)

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
        # Show Gemini quota state honestly so the user can pick the right mode
        gri_val = self._state.last_gri
        gri_emoji = "🟢" if gri_val < 0.30 else ("🟡" if gri_val < 0.50 else "🔴")
        if self._state.gemini_working:
            gemini_status = "✅ Gemini quota available — Full mode recommended"
        else:
            gemini_status = "⚠️ Gemini quota exhausted — GRI-Only recommended"
        text = (
            f"🤖 <b>SentiStack V2 — Choose Trading Mode</b>\n"
            f"{'─' * 34}\n"
            f"🕐 <b>Started:</b> {_fmt_time()}\n"
            f"🌐 <b>GRI:</b> {gri_emoji} {gri_val:.3f} ({self._state.last_gri_level})\n"
            f"🧠 <b>Gemini:</b> {gemini_status}\n\n"
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
        try:
            from config import is_paper_trade as _ipt
            trade_mode_str = "📝 PAPER (simulated)" if _ipt() else "💸 LIVE (real money)"
        except Exception:
            trade_mode_str = "?"
        # Task-1: surface bootstrap-mode state (active iff capital < threshold)
        try:
            from config import (
                is_bootstrap_active as _iba,
                get_effective_position_fraction as _gpf,
                get_effective_min_trade_value as _gmtv,
            )
            if _iba(s.paper_capital):
                bootstrap_str = (
                    f"🚀 ON (frac={_gpf(s.paper_capital):.0%}, "
                    f"min=₹{_gmtv(s.paper_capital):.0f})"
                )
            else:
                bootstrap_str = "💤 OFF (normal sizing)"
        except Exception:
            bootstrap_str = "?"
        gri_emoji  = "🟢" if s.last_gri < 0.30 else ("🟡" if s.last_gri < 0.50 else "🔴")
        sent_emoji = {"Fear": "😨", "Excitement": "🚀", "Neutral": "😐"}.get(
            s.last_sentiment_class, "😐"
        )
        # Bug fix: in GRI-only mode Gemini is intentionally not used — show that
        # clearly instead of the alarming "Quota exhausted" message.
        if s.mode == TradingMode.GRI_ONLY:
            gemini_str = "⚡ Not used (GRI-synthetic mode)"
        elif s.gemini_working:
            gemini_str = "✅ Working"
        else:
            gemini_str = "❌ Quota exhausted"
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
            f"💹 <b>Trading:</b>     {trade_mode_str}\n"
            f"🚀 <b>Bootstrap:</b>   {bootstrap_str}\n"
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
            f"{'─' * 34}\n"
            f"🚦 <b>Guards:</b>        "
            f"{'🚫BUY ' if s.no_new_buys else ''}{'🚫SELL ' if s.no_new_sells else ''}{'None' if not s.no_new_buys and not s.no_new_sells else ''}\n"
            f"🕐 <b>Last update:</b>   {s.last_update}\n"
        )

    # ── Command handlers ──────────────────────────────────────────────────

    async def _handle_command(self, text: str, chat_id: str) -> None:
        cmd = text.split()[0].lower().replace("@", "").split("@")[0]

        if cmd in ("/start", "/help"):
            await self.send(
                f"🤖 <b>SentiStack V2 Commands</b>\n"
                f"{'─' * 28}\n"
                "/mode — Change trading mode (FULL / GRI_ONLY / MONITOR)\n"
                "/tradingmode — Show PAPER vs LIVE\n"
                "/papermode — Switch to PAPER (simulated fills)\n"
                "/livemode — Arm LIVE mode (requires /livemode CONFIRM)\n"
                "/status — Full pipeline status\n"
                "/pause — Pause order execution\n"
                "/resume — Resume order execution\n"
                "/stop — Graceful shutdown\n"
                f"{'─' * 28}\n"
                "🚦 <b>Trade Guards</b>\n"
                "/nobuy — Block all new BUY orders\n"
                "/okbuy — Re-enable BUY orders\n"
                "/nosell — Block all new SELL/short orders\n"
                "/oksell — Re-enable SELL orders\n"
                f"{'─' * 28}\n"
                "🔧 <b>Health &amp; Monitoring</b>\n"
                "/tasks — Background task heartbeat status\n"
                "/resetquota — Clear stale Gemini quota cooldowns\n"
                f"{'─' * 28}\n"
                "📊 <b>P&amp;L &amp; Capital</b>\n"
                "/pnl — Today's P&amp;L statement (MIS + CNC)\n"
                "/risk — Portfolio loss-budget (D/W/M + blacklist)\n"
                "/balance — Show Zerodha equity balance\n"
                "/synccapital — Resync capital from Zerodha balance\n"
                "/capital &lt;amount&gt; — Set capital manually (shows budgets)\n"
                f"{'─' * 28}\n"
                "🔑 <b>Daily Token Refresh (no restart needed)</b>\n"
                "/login — Get today's Zerodha login URL\n"
                "/token &lt;request_token&gt; — Apply new token\n"
                "  Or paste the full redirect URL after /token\n"
                f"{'─' * 28}\n"
                "/help — This message\n"
            )
            if not self._state.mode_confirmed:
                await self.send_mode_selector()

        elif cmd == "/mode":
            await self.send_mode_selector()

        elif cmd == "/tradingmode":
            # R-15: Show current paper/live status
            from config import is_paper_trade as _ipt
            mode_str = "📝 <b>PAPER</b> (simulated fills)" if _ipt() \
                else "💸 <b>LIVE</b> (real money on Zerodha)"
            await self.send(
                f"🔄 <b>Current trading mode</b>\n"
                f"{'─' * 28}\n"
                f"{mode_str}\n\n"
                f"Switch: <code>/papermode</code> or <code>/livemode</code>"
            )

        elif cmd == "/papermode":
            # R-15: Force paper trade (always safe — simulates fills, no real orders)
            try:
                from config import set_paper_trade_override, is_paper_trade as _ipt
                if _ipt():
                    await self.send(
                        "ℹ️ Already in <b>PAPER</b> mode. "
                        "No change."
                    )
                else:
                    set_paper_trade_override(True)
                    self._state.live_mode_arm_until = None
                    await self.send(
                        "📝 <b>Switched to PAPER mode</b>\n"
                        f"{'─' * 28}\n"
                        "All new orders will be simulated with realistic slippage. "
                        "No real trades will hit Zerodha.\n\n"
                        "Already-open real positions are NOT touched — only "
                        "new entries. Use <code>/tradingmode</code> to confirm."
                    )
                    logger.warning("R-15 Paper mode forced via Telegram.")
            except Exception as exc:
                logger.error("/papermode failed: %s", exc, exc_info=True)
                await self.send(f"❌ Mode switch failed: <code>{exc}</code>")

        elif cmd == "/livemode":
            # R-15: Arm / confirm live-mode switch (two-step for safety)
            try:
                from config import set_paper_trade_override, is_paper_trade as _ipt
                parts = text.strip().split(None, 1)
                arg   = parts[1].strip().upper() if len(parts) > 1 else ""

                if not _ipt() and arg != "CONFIRM":
                    await self.send(
                        "ℹ️ Already in <b>LIVE</b> mode. No change.\n"
                        "Switch back with <code>/papermode</code>."
                    )
                elif arg == "CONFIRM":
                    # Confirmation attempt — must be within the arm window
                    now = _ist_now()
                    armed_until = self._state.live_mode_arm_until
                    if armed_until is None or now > armed_until:
                        self._state.live_mode_arm_until = None
                        await self.send(
                            "⏱ <b>Live-mode arm expired or not started.</b>\n"
                            "Send <code>/livemode</code> first, then "
                            "<code>/livemode CONFIRM</code> within 60 seconds."
                        )
                    else:
                        set_paper_trade_override(False)
                        self._state.live_mode_arm_until = None
                        await self.send(
                            "💸 <b>SWITCHED TO LIVE MODE</b>\n"
                            f"{'─' * 28}\n"
                            "⚠️ Real orders will now be sent to Zerodha.\n"
                            "⚠️ Real money is at risk.\n\n"
                            "Safeguards still active:\n"
                            "• R-13 portfolio loss budgets (D/W/M)\n"
                            "• Consecutive-loss blacklist\n"
                            "• Vol-scaled stops + time stop\n"
                            "• /pause, /nobuy, /nosell guards\n\n"
                            "Revert with <code>/papermode</code> at any time."
                        )
                        logger.critical(
                            "R-15 LIVE MODE ENABLED via Telegram — real orders "
                            "will now be sent to Zerodha."
                        )
                else:
                    # First step — arm the confirmation window
                    arm_until = _ist_now() + timedelta(seconds=60)
                    self._state.live_mode_arm_until = arm_until
                    await self.send(
                        "⚠️ <b>Arming LIVE mode — confirmation required</b>\n"
                        f"{'─' * 28}\n"
                        "This will switch the bot from PAPER (simulated) to "
                        "<b>LIVE</b> (real money on Zerodha).\n\n"
                        "✅ Before confirming, check:\n"
                        "• Zerodha token is valid (<code>/status</code>)\n"
                        "• Capital is correct (<code>/capital</code>)\n"
                        "• Loss budgets look sane (<code>/risk</code>)\n"
                        "• Broker balance is as expected (<code>/balance</code>)\n\n"
                        "🔐 To confirm, send within 60 seconds:\n"
                        "<code>/livemode CONFIRM</code>"
                    )
                    logger.warning("R-15 Live mode armed — awaiting /livemode CONFIRM.")
            except Exception as exc:
                logger.error("/livemode failed: %s", exc, exc_info=True)
                await self.send(f"❌ Mode switch failed: <code>{exc}</code>")

        elif cmd == "/status":
            await self.send(self._status_text())

        elif cmd == "/pause":
            self._state.paused = True
            await self.send("⏸ <b>Trading paused.</b> Monitoring continues. Send /resume to restart.")

        elif cmd == "/resume":
            # R-13: also clear any active risk halt on manual resume
            prev_scope = self._state.risk_halt_scope
            self._state.paused = False
            self._state.risk_halt_scope  = "NONE"
            self._state.risk_halt_reason = ""
            self._state.risk_halt_until  = None
            self._state.no_new_buys      = False
            self._state.no_new_sells     = False
            if self._state.mode == TradingMode.PAPER_MONITOR and prev_scope == "MONTH":
                self._state.mode = TradingMode.GRI_ONLY
            extra = ""
            if prev_scope and prev_scope != "NONE":
                extra = f"\n🔓 Risk halt cleared: <code>{prev_scope}</code>"
                logger.warning("Manual /resume — cleared risk halt scope=%s", prev_scope)
            await self.send(f"▶️ <b>Trading resumed.</b>{extra}")

        elif cmd == "/login":
            if self._kite is None:
                await self.send(
                    "⚠️ Kite not initialised yet — please wait for bot startup to complete."
                )
            else:
                login_url = self._kite.login_url()
                await self.send(
                    f"🔑 <b>Zerodha Login — Daily Token Refresh</b>\n"
                    f"{'─' * 34}\n"
                    f"1️⃣ Open this link and log in:\n"
                    f"<code>{login_url}</code>\n\n"
                    f"2️⃣ After login you'll be redirected to a URL like:\n"
                    f"<code>http://127.0.0.1/?request_token=XXXXXXXX&amp;status=success</code>\n\n"
                    f"3️⃣ Send the token to this chat:\n"
                    f"<code>/token XXXXXXXX</code>\n"
                    f"(or paste the full redirect URL after /token)\n\n"
                    f"⏱ The token expires at midnight — refresh each morning before 9:15 AM IST."
                )

        elif cmd == "/token":
            parts = text.strip().split(None, 1)
            raw = parts[1].strip() if len(parts) > 1 else ""
            # Support both bare request_token and full redirect URL
            if "request_token=" in raw:
                raw = raw.split("request_token=")[1].split("&")[0].strip()
            if not raw:
                await self.send(
                    "⚠️ Usage: <code>/token &lt;request_token&gt;</code>\n"
                    "Send /login to get the login URL first."
                )
            elif self._kite is None:
                await self.send(
                    "⏳ <b>Bot is still starting up</b> — the /token command during startup is "
                    "handled automatically.\n"
                    "Once running, you can use /login + /token to refresh the daily token.",
                )
            else:
                await self.send("🔄 Exchanging request_token for access token…")
                try:
                    loop = asyncio.get_running_loop()
                    session = await loop.run_in_executor(
                        None,
                        lambda rt=raw: self._kite.generate_session(
                            rt, api_secret=self._api_secret
                        )
                    )
                    new_token = session["access_token"]
                    # Hot-swap into the running kite instance
                    self._kite.set_access_token(new_token)
                    self._save_token_cache(new_token)
                    # Signal the strategy loop that the token is fresh
                    self._state.token_valid = True
                    self._state.token_refresh_event.set()
                    self._state.token_refresh_event.clear()
                    logger.info("Access token hot-swapped successfully via Telegram /token.")
                    await self.send(
                        f"✅ <b>Token refreshed successfully!</b>\n"
                        f"{'─' * 34}\n"
                        f"🕐 {_fmt_time()}\n"
                        f"Trading will resume immediately.\n"
                        f"Send /status to confirm."
                    )
                except Exception as exc:
                    logger.error("Token refresh via Telegram failed: %s", exc)
                    await self.send(
                        f"❌ <b>Token refresh failed</b>\n"
                        f"Error: <code>{exc}</code>\n\n"
                        f"Make sure you copied the correct request_token "
                        f"(it's valid for only ~2 minutes after login). "
                        f"Send /login to try again."
                    )

        elif cmd == "/nobuy":
            self._state.no_new_buys = True
            await self.send(
                "🚫 <b>New BUY orders blocked.</b>\n"
                "The bot will continue monitoring and can still close existing SELL positions.\n"
                "Send /okbuy to re-enable BUY orders."
            )
            logger.info("no_new_buys flag SET via Telegram.")

        elif cmd == "/okbuy":
            self._state.no_new_buys = False
            await self.send("✅ <b>BUY orders re-enabled.</b> Bot will resume buying on strong signals.")
            logger.info("no_new_buys flag CLEARED via Telegram.")

        elif cmd == "/nosell":
            self._state.no_new_sells = True
            await self.send(
                "🚫 <b>New SELL/short orders blocked.</b>\n"
                "Send /oksell to re-enable SELL orders."
            )
            logger.info("no_new_sells flag SET via Telegram.")

        elif cmd == "/oksell":
            self._state.no_new_sells = False
            await self.send("✅ <b>SELL orders re-enabled.</b>")
            logger.info("no_new_sells flag CLEARED via Telegram.")

        elif cmd == "/tasks":
            # ── Background task health status ─────────────────────────────
            hb = self._state.task_heartbeats
            if not hb:
                await self.send("⏳ No task heartbeats recorded yet — bot may still be starting up.")
            else:
                now_ts = time.time()
                lines  = [
                    "🔧 <b>Background Task Health</b>",
                    f"{'─' * 32}",
                ]
                for name, info in sorted(hb.items()):
                    age     = now_ts - info.get("last_beat", 0)
                    cycles  = info.get("cycles", 0)
                    errors  = info.get("errors", 0)
                    if age < 120:
                        icon = "🟢"
                    elif age < 600:
                        icon = "🟡"
                    else:
                        icon = "🔴"
                    age_str = (
                        f"{int(age)}s ago" if age < 3600
                        else f"{int(age/3600)}h ago"
                    )
                    lines.append(
                        f"{icon} <b>{name}</b>\n"
                        f"   Last beat: {age_str} | Cycles: {cycles} | Errors: {errors}"
                    )
                lines.append(f"{'─' * 32}")
                lines.append(f"🕐 <i>{_fmt_time()}</i>")
                await self.send("\n".join(lines))

        elif cmd == "/pnl":
            # ── Daily P&L statement ───────────────────────────────────────
            if self._logbook is None:
                await self.send("⚠️ Logbook not yet initialised — please try again in a moment.")
            else:
                try:
                    report = self._logbook.get_pnl_report()
                    # Telegram message limit is ~4096 chars; split if needed
                    if len(report) <= 4000:
                        await self.send(f"<pre>{report}</pre>")
                    else:
                        chunk_size = 3900
                        for i in range(0, len(report), chunk_size):
                            await self.send(f"<pre>{report[i:i+chunk_size]}</pre>")
                            await asyncio.sleep(0.3)
                except Exception as exc:
                    logger.error("P&L report error: %s", exc, exc_info=True)
                    await self.send(f"❌ Could not generate P&L report: <code>{exc}</code>")

        elif cmd == "/capital":
            # ── Update active trading capital (paper OR live) ─────────────
            parts = text.strip().split(None, 1)
            raw   = parts[1].strip() if len(parts) > 1 else ""
            # If called without an argument, just report the current value.
            if not raw:
                try:
                    from portfolio_risk import settings as _s
                    _day_lim  = _s.strategy.DAILY_LOSS_LIMIT_PCT   * self._state.paper_capital
                    _week_lim = _s.strategy.WEEKLY_LOSS_LIMIT_PCT  * self._state.paper_capital
                    _mon_lim  = _s.strategy.MONTHLY_LOSS_LIMIT_PCT * self._state.paper_capital
                    await self.send(
                        f"💰 <b>Active capital</b>\n"
                        f"{'─' * 28}\n"
                        f"Current: <b>₹{self._state.paper_capital:,.2f}</b>\n"
                        f"Day limit:   −₹{_day_lim:,.0f}\n"
                        f"Week limit:  −₹{_week_lim:,.0f}\n"
                        f"Month limit: −₹{_mon_lim:,.0f}\n\n"
                        f"To change: <code>/capital &lt;amount&gt;</code>"
                    )
                except Exception as exc:
                    await self.send(
                        f"💰 Current capital: <b>₹{self._state.paper_capital:,.2f}</b>\n"
                        f"To change: <code>/capital &lt;amount&gt;</code>"
                    )
            else:
                # Strip currency symbols and commas (e.g. "₹5,00,000" → "500000")
                raw = raw.replace("₹", "").replace(",", "").strip()
                try:
                    new_capital = float(raw)
                    if new_capital <= 0:
                        raise ValueError("Capital must be positive")
                    old_capital = self._state.paper_capital
                    self._state.paper_capital = new_capital
                    # Compute new loss budgets so the user can sanity-check
                    try:
                        from portfolio_risk import settings as _s
                        _day_lim  = _s.strategy.DAILY_LOSS_LIMIT_PCT   * new_capital
                        _week_lim = _s.strategy.WEEKLY_LOSS_LIMIT_PCT  * new_capital
                        _mon_lim  = _s.strategy.MONTHLY_LOSS_LIMIT_PCT * new_capital
                        budget_block = (
                            f"\n📐 <b>New loss budgets</b>\n"
                            f"Day:   −₹{_day_lim:,.0f}\n"
                            f"Week:  −₹{_week_lim:,.0f}\n"
                            f"Month: −₹{_mon_lim:,.0f}\n"
                        )
                    except Exception:
                        budget_block = ""
                    await self.send(
                        f"✅ <b>Active capital updated</b>\n"
                        f"{'─' * 28}\n"
                        f"Old: ₹{old_capital:,.2f}\n"
                        f"New: <b>₹{new_capital:,.2f}</b>\n"
                        f"{budget_block}\n"
                        f"Takes effect on next strategy cycle. "
                        f"Position sizing, Kelly fraction, and D/W/M loss "
                        f"budgets all rescale automatically."
                    )
                    logger.warning(
                        "Active capital updated to ₹%.2f via Telegram (was ₹%.2f).",
                        new_capital, old_capital,
                    )
                except (ValueError, IndexError):
                    await self.send(
                        "⚠️ Usage: <code>/capital &lt;amount&gt;</code>\n"
                        "Example: <code>/capital 700000</code>\n"
                        "Strips ₹ and commas automatically.\n"
                        "Call without an argument to see current value."
                    )

        elif cmd == "/balance":
            # ── R-14: Show raw Zerodha broker balance ────────────────────
            if self._margin_fetcher is None:
                await self.send(
                    "⚠️ Broker fetcher not wired. "
                    "Available only in live mode after startup."
                )
            else:
                try:
                    margins = await self._margin_fetcher()
                    equity  = (margins or {}).get("equity", {}) or {}
                    avail   = equity.get("available", {}) or {}
                    used    = equity.get("utilised", {}) or {}
                    net     = float(equity.get("net") or avail.get("live_balance") or 0)
                    cash    = float(avail.get("cash") or 0)
                    opening = float(avail.get("opening_balance") or 0)
                    debits  = float(used.get("debits") or 0)
                    await self.send(
                        f"🏦 <b>Zerodha equity balance</b>\n"
                        f"{'─' * 28}\n"
                        f"Opening:    ₹{opening:,.2f}\n"
                        f"Cash:       ₹{cash:,.2f}\n"
                        f"Utilised:   ₹{debits:,.2f}\n"
                        f"Net:        <b>₹{net:,.2f}</b>\n\n"
                        f"Active bot capital: ₹{self._state.paper_capital:,.2f}"
                    )
                except Exception as exc:
                    logger.error("/balance failed: %s", exc, exc_info=True)
                    await self.send(f"❌ Broker fetch failed: <code>{exc}</code>")

        elif cmd == "/synccapital":
            # ── R-14: Force immediate capital sync from broker ───────────
            if self._margin_fetcher is None:
                await self.send(
                    "⚠️ Broker fetcher not wired. "
                    "Auto-sync is only available in live mode."
                )
            else:
                try:
                    from portfolio_risk import settings as _s
                    margins = await self._margin_fetcher()
                    equity  = (margins or {}).get("equity", {}) or {}
                    avail   = equity.get("available", {}) or {}
                    raw_bal = float(
                        equity.get("net")
                        or avail.get("live_balance")
                        or avail.get("cash")
                        or 0
                    )
                    if raw_bal <= 0:
                        await self.send(
                            "⚠️ Broker reported zero balance — nothing to sync. "
                            "Check Zerodha login / funds."
                        )
                    else:
                        buffer  = float(getattr(_s.strategy, "AUTO_SYNC_SAFETY_BUFFER", 0.90))
                        ceiling = float(getattr(_s.strategy, "AUTO_SYNC_MAX_CAPITAL", 1e7))
                        new_cap = min(raw_bal * buffer, ceiling)
                        old_cap = self._state.paper_capital
                        self._state.paper_capital = new_cap
                        await self.send(
                            f"🔄 <b>Manual capital sync from Zerodha</b>\n"
                            f"{'─' * 28}\n"
                            f"Broker balance: ₹{raw_bal:,.2f}\n"
                            f"Safety buffer:  {buffer:.0%}\n"
                            f"Ceiling:        ₹{ceiling:,.0f}\n"
                            f"─────────────\n"
                            f"Old capital: ₹{old_cap:,.2f}\n"
                            f"New capital: <b>₹{new_cap:,.2f}</b>\n\n"
                            f"Position sizing + D/W/M loss budgets "
                            f"rescale on the next strategy cycle."
                        )
                        logger.warning(
                            "Manual capital sync: raw=₹%.2f × %.2f → ₹%.2f (was ₹%.2f)",
                            raw_bal, buffer, new_cap, old_cap,
                        )
                except Exception as exc:
                    logger.error("/synccapital failed: %s", exc, exc_info=True)
                    await self.send(f"❌ Sync failed: <code>{exc}</code>")

        elif cmd == "/risk":
            # ── R-13: Portfolio loss-budget report ───────────────────────
            try:
                from portfolio_risk import format_risk_report
                if self._portfolio_risk is not None:
                    try:
                        self._portfolio_risk.check(self._state)
                    except Exception as exc:
                        logger.warning("/risk: monitor.check failed: %s", exc)
                report = format_risk_report(self._state)
                await self.send(report)
            except Exception as exc:
                logger.error("/risk error: %s", exc, exc_info=True)
                await self.send(f"❌ Could not render risk report: <code>{exc}</code>")

        elif cmd == "/resetquota":
            # ── Gemini quota emergency reset ─────────────────────────────────
            if self._rotator is None:
                await self.send(
                    "⚠️ <b>ModelRotator not wired.</b>\n"
                    "Call tg_controller.set_rotator(_rotator) in main.py."
                )
            else:
                summary = self._rotator.reset_all_quota_state()
                await self.send(
                    f"🔄 <b>Gemini quota state cleared.</b>\n"
                    f"<pre>{summary}</pre>\n"
                    f"All models are now available. The bot will retry Gemini on the next sentiment cycle."
                )

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
