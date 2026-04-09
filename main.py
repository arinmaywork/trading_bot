"""
main.py
=======
Entry point — SentiStack NSE Trading Bot (with Dynamic Universe + Frequency Optimiser)

Full startup sequence:
  0. Interactive Zerodha token auth (browser + terminal paste, cached daily).
  1. Redis connection check.
  2. UniverseEngine.initialise() — loads or builds the active trading universe.
  3. WebSocket subscription to the active universe instrument tokens.
  4. All background tasks launched:
       • universe_daily_refresh    — rebuilds universe at 08:45 IST daily
       • universe_intraday_rescore — rescores every 30 min during market hours
       • candle_aggregator         — 1-min OHLCV from Redis tick stream
       • alt_data_analysis         — weather + news + Gemini sentiment loop
  5. Main strategy loop:
       • Frequency dynamically adjusted by FrequencyOptimiser.
       • On universe change → WebSocket resubscription happens automatically.
  6. SIGINT/SIGTERM → graceful shutdown of all tasks.
"""

# ---------------------------------------------------------------------------
# R-09 FIX: Force IPv4 for all outbound connections.
# Home ISPs rotate IPv6 prefixes periodically, causing PermissionException
# when the new IPv6 is not in the Kite API whitelist. IPv4 addresses are far
# more stable. This patch reorders DNS results so IPv4 (AF_INET) is always
# tried first; IPv6 is kept as a fallback in case no IPv4 route exists.
# Must run before any network-touching import (kiteconnect, aiohttp, etc.).
# ---------------------------------------------------------------------------
import socket as _socket
_orig_getaddrinfo = _socket.getaddrinfo

def _ipv4_preferred(host, port, family=0, type=0, proto=0, flags=0):
    results = _orig_getaddrinfo(host, port, family, type, proto, flags)
    ipv4 = [r for r in results if r[0] == _socket.AF_INET]
    ipv6 = [r for r in results if r[0] != _socket.AF_INET]
    return ipv4 + ipv6

_socket.getaddrinfo = _ipv4_preferred
# ---------------------------------------------------------------------------

import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
import webbrowser
from datetime import datetime, date, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import redis.asyncio as aioredis
from kiteconnect import KiteConnect
from kiteconnect import exceptions as KiteExceptions

from alternative_data import AlternativeDataPipeline
from logbook import Logbook
from telegram_controller import TelegramController, BotState, TradingMode
from telegram_log_handler import TelegramLogHandler
from agent_pipeline import AgentPipeline
from config import settings, is_paper_trade, set_paper_trade_override
from data_ingestion import AsyncKiteTickerWrapper, CandleAggregator
from execution import OrderExecutor, TelegramNotifier
from rate_limiter import RateLimiter
from strategy import RiskManager, StrategyEngine, TradeDirection
from ml_signal import EnsembleSignalEngine
from geopolitical import GeopoliticalRiskMonitor
from universe import FrequencyOptimiser, UniverseEngine
from position_manager import PositionManager
from portfolio_risk import PortfolioRiskMonitor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.logging.LEVEL, logging.INFO),
    format=settings.logging.FORMAT,
    datefmt=settings.logging.DATEFMT,
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Dashboard: also write logs to a rotating file so the web UI can tail it
_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "bot_live.log",
    maxBytes=10 * 1024 * 1024,   # 10 MB
    backupCount=3,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d|%(levelname)s|%(name)s|%(message)s",
    datefmt="%H:%M:%S",
))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("main")

TOKEN_CACHE_FILE = Path(__file__).parent / ".kite_token"


# ===========================================================================
# TOKEN MANAGER  (unchanged from previous version)
# ===========================================================================

class TokenManager:
    def __init__(self, api_key: str, api_secret: str) -> None:
        self._api_key    = api_key
        self._api_secret = api_secret
        self._kite       = KiteConnect(api_key=api_key)
        self._tg_offset  = 0   # Telegram update offset consumed during headless startup

    @property
    def tg_offset(self) -> int:
        """Last Telegram update_id + 1 consumed during headless startup polling."""
        return self._tg_offset

    def _load_cached_token(self) -> Optional[str]:
        if not TOKEN_CACHE_FILE.exists():
            return None
        try:
            data = json.loads(TOKEN_CACHE_FILE.read_text())
            if data.get("generated_date") == date.today().isoformat():
                token = data.get("access_token", "")
                if token:
                    logger.info("Cached token found (generated today).")
                    return token
            else:
                logger.info("Cached token is stale — will re-auth.")
        except Exception as exc:
            logger.warning("Token cache read error: %s", exc)
        return None

    def _save_token(self, token: str) -> None:
        try:
            TOKEN_CACHE_FILE.write_text(
                json.dumps({"access_token": token,
                            "generated_date": date.today().isoformat()}, indent=2)
            )
            TOKEN_CACHE_FILE.chmod(0o600)
            logger.info("Token saved to %s", TOKEN_CACHE_FILE)
        except OSError as exc:
            logger.warning("Could not save token: %s", exc)

    def _validate_token(self, token: str) -> bool:
        self._kite.set_access_token(token)
        try:
            p = self._kite.profile()
            logger.info("Logged in as %s (%s).", p.get("user_name"), p.get("user_id"))
            return True
        except KiteExceptions.TokenException:
            return False
        except Exception as exc:
            logger.warning("Validation call failed (%s) — proceeding anyway.", exc)
            return True

    def _run_oauth_flow(self) -> str:
        login_url = self._kite.login_url()
        print()
        print("=" * 65)
        print("  ZERODHA LOGIN REQUIRED")
        print("=" * 65)
        print()
        print("  After login, copy the request_token from the redirect URL:")
        print("  http://127.0.0.1/?request_token=XXXXXXXX&status=success")
        print()
        print(f"  Login URL:\n  {login_url}\n")
        try:
            webbrowser.open(login_url)
            print("  Browser opened automatically.\n")
        except Exception:
            print("  Please open the URL above manually.\n")

        for attempt in range(1, 4):
            try:
                raw = input("  Paste request_token → ").strip()
            except (EOFError, KeyboardInterrupt):
                raise RuntimeError("Login cancelled.")
            if "request_token=" in raw:
                raw = raw.split("request_token=")[1].split("&")[0].strip()
            if not raw:
                print(f"  Empty input (attempt {attempt}/3).")
                continue
            try:
                session = self._kite.generate_session(raw, api_secret=self._api_secret)
                print("  ✅ Access token obtained.\n")
                return session["access_token"]
            except KiteExceptions.TokenException as exc:
                print(f"  ❌ Invalid token: {exc} (attempt {attempt}/3)\n")
            except Exception as exc:
                print(f"  ❌ Error: {exc} (attempt {attempt}/3)\n")

        raise RuntimeError("Authentication failed after 3 attempts.")

    def _telegram_oauth_flow(self) -> str:
        """
        Headless-server OAuth flow.
        Sends the Zerodha login URL to the configured Telegram chat and then
        long-polls for a /token <request_token> command.  No stdin, no browser.
        Stores the highest Telegram update_id seen in self._tg_offset so that
        the TelegramController's poll loop can skip these already-processed
        messages when it starts up.
        """
        import requests as _requests

        login_url = self._kite.login_url()
        bot_tok   = settings.telegram.BOT_TOKEN
        chat_id   = str(settings.telegram.CHAT_ID)
        api_base  = f"https://api.telegram.org/bot{bot_tok}"

        msg = (
            "🔑 <b>SentiStack — Zerodha login required</b>\n"
            "─────────────────────────────────────────\n"
            "1️⃣  Open this link and log in:\n"
            f"<code>{login_url}</code>\n\n"
            "2️⃣  After login you'll be redirected to:\n"
            "<code>http://127.0.0.1/?request_token=XXXXXXXX&amp;status=success</code>\n\n"
            "3️⃣  Copy <b>only the request_token value</b> and send:\n"
            "<code>/token XXXXXXXX</code>\n\n"
            "⚠️  Do this before 9:15 AM IST — token expires at midnight."
        )
        try:
            _requests.post(
                f"{api_base}/sendMessage",
                json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=15,
            )
            logger.info("Headless startup: login URL sent to Telegram. Waiting for /token …")
        except Exception as exc:
            logger.warning("Could not send Telegram message: %s", exc)

        # Long-poll until we receive /token <request_token>
        offset = 0
        while True:
            try:
                resp = _requests.get(
                    f"{api_base}/getUpdates",
                    params={
                        "offset":          offset,
                        "timeout":         30,
                        "allowed_updates": '["message"]',
                    },
                    timeout=40,
                )
                data = resp.json()
            except Exception as exc:
                logger.warning("Telegram poll error: %s — retrying in 5 s", exc)
                time.sleep(5)
                continue

            for update in data.get("result", []):
                uid  = update["update_id"]
                offset = uid + 1          # advance window
                if offset > self._tg_offset:
                    self._tg_offset = offset

                msg_obj = update.get("message", {})
                text    = msg_obj.get("text", "").strip()

                if not text.startswith("/token"):
                    continue

                parts = text.split(None, 1)
                raw   = parts[1].strip() if len(parts) > 1 else ""
                # Accept full redirect URL as well as bare request_token
                if "request_token=" in raw:
                    raw = raw.split("request_token=")[1].split("&")[0].strip()

                if not raw:
                    # Prompt again
                    try:
                        _requests.post(
                            f"{api_base}/sendMessage",
                            json={
                                "chat_id":    chat_id,
                                "text":       "⚠️ Please send the request_token: <code>/token XXXXXXXX</code>",
                                "parse_mode": "HTML",
                            },
                            timeout=10,
                        )
                    except Exception:
                        pass
                    continue

                logger.info("Headless startup: received request_token via Telegram — exchanging…")
                try:
                    session = self._kite.generate_session(raw, api_secret=self._api_secret)
                    new_token = session["access_token"]
                    try:
                        _requests.post(
                            f"{api_base}/sendMessage",
                            json={
                                "chat_id":    chat_id,
                                "text":       "✅ <b>Token accepted!</b> SentiStack is starting up…",
                                "parse_mode": "HTML",
                            },
                            timeout=10,
                        )
                    except Exception:
                        pass
                    return new_token
                except Exception as exc:
                    logger.error("Token exchange failed: %s", exc)
                    try:
                        _requests.post(
                            f"{api_base}/sendMessage",
                            json={
                                "chat_id":    chat_id,
                                "text":       f"❌ Token exchange failed: {exc}\nPlease try /token again.",
                                "parse_mode": "HTML",
                            },
                            timeout=10,
                        )
                    except Exception:
                        pass
                    # Keep polling — user will re-send

    def get_token(self) -> Tuple[KiteConnect, str]:
        env_tok = settings.kite.ACCESS_TOKEN
        if env_tok and self._validate_token(env_tok):
            return self._kite, env_tok
        cached = self._load_cached_token()
        if cached and self._validate_token(cached):
            return self._kite, cached
        # Need a fresh token.  Choose flow based on whether we have a terminal.
        headless = not sys.stdin.isatty()
        if headless:
            logger.info("Headless server detected — using Telegram-based token flow.")
            new_tok = self._telegram_oauth_flow()
        else:
            new_tok = self._run_oauth_flow()
        self._save_token(new_tok)
        if not self._validate_token(new_tok):
            raise RuntimeError("Newly generated token failed validation.")
        return self._kite, new_tok


# ===========================================================================
# GRACEFUL SHUTDOWN
# ===========================================================================

class GracefulShutdown:
    def __init__(self) -> None:
        self._tasks: List[asyncio.Task] = []
        self._event = asyncio.Event()

    def register(self, task: asyncio.Task) -> None:
        self._tasks.append(task)

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._trigger)

    def _trigger(self) -> None:
        logger.warning("Shutdown — cancelling %d tasks.", len(self._tasks))
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()


# ===========================================================================
# DYNAMIC WEBSOCKET RESUBSCRIPTION MANAGER
# Handles subscribing/unsubscribing from KiteTicker when universe changes.
# ===========================================================================

class WebSocketSubscriptionManager:
    """
    Manages KiteTicker instrument subscriptions in response to universe changes.
    Debounces rapid changes and batches subscribe/unsubscribe calls.
    """

    def __init__(
        self,
        ticker_wrapper: AsyncKiteTickerWrapper,
        rate_limiter: RateLimiter,
    ) -> None:
        self._ticker     = ticker_wrapper
        self._limiter    = rate_limiter
        self._subscribed: Set[int] = set()
        self._pending_tokens: Optional[List[int]] = None
        self._lock = asyncio.Lock()

    async def on_universe_change(self, new_symbols: List[str]) -> None:
        """
        Callback registered with UniverseEngine.
        Stores pending tokens; actual resubscription happens in the flush loop.
        """
        from universe import UniverseEngine   # avoid circular at module level
        # Tokens are managed by UniverseEngine; we receive symbols here.
        # We store the change request and let flush_pending() handle it.
        logger.info(
            "Universe change notification received (%d symbols). "
            "Queuing WebSocket resubscription.",
            len(new_symbols),
        )
        # Signal the flush loop — actual tokens come from universe engine
        async with self._lock:
            self._pending_tokens = []   # Flag: refresh needed

    async def apply_subscription(self, new_tokens: List[int]) -> None:
        """
        Compare new_tokens with current subscriptions.
        Unsubscribe removed tokens, subscribe added tokens.
        """
        async with self._lock:
            # Check if ticker is ready
            if not self._ticker._ticker or not self._ticker._ticker.is_connected():
                logger.warning("WebSocket: Ticker not connected — skipping subscription update.")
                return

            new_set = set(new_tokens[: settings.universe.MAX_WEBSOCKET_SUBSCRIPTIONS])
            to_add    = new_set - self._subscribed
            to_remove = self._subscribed - new_set

            if not to_add and not to_remove:
                return

            logger.info(
                "WebSocket: +%d / -%d instruments (total → %d)",
                len(to_add), len(to_remove), len(new_set),
            )

            # KiteTicker subscribe/unsubscribe are synchronous fire-and-forget
            loop = asyncio.get_running_loop()
            if to_remove:
                await loop.run_in_executor(
                    None,
                    lambda: self._ticker._ticker.unsubscribe(list(to_remove))
                )
            if to_add:
                await loop.run_in_executor(
                    None,
                    lambda: self._ticker._ticker.subscribe(list(to_add))
                )
                # Set full-depth mode for new subscriptions
                await loop.run_in_executor(
                    None,
                    lambda: self._ticker._ticker.set_mode(
                        self._ticker._ticker.MODE_FULL, list(to_add)
                    )
                )

            self._subscribed = new_set
            self._pending_tokens = None


# ===========================================================================
# MARKET HOURS GUARD
# ===========================================================================

def is_market_open() -> bool:
    """True if NSE is open (09:15–15:29:59 IST, Mon–Fri). India has no DST."""
    from datetime import timedelta as _td
    ist = timezone(_td(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5: return False
    open_time  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    # B-06 FIX: use strict < for close_time — exactly 15:30:00 must be excluded
    return open_time <= now < close_time


def seconds_until_market_open() -> float:
    """Return seconds until 09:15 IST. Returns 0 if market is already open."""
    if is_market_open(): return 0.0
    from datetime import timedelta as _td
    ist = timezone(_td(hours=5, minutes=30))
    now = datetime.now(ist)
    target = now.replace(hour=9, minute=15, second=0, microsecond=0)
    if now >= target: # Already past 9:15, wait for tomorrow
        target += _td(days=1)
        while target.weekday() >= 5: target += _td(days=1)
    return (target - now).total_seconds()


# ===========================================================================
# STRATEGY LOOP (dynamic, frequency-aware)
# ===========================================================================

async def strategy_loop(
    kite: KiteConnect,
    redis_client: aioredis.Redis,
    alt_data: AlternativeDataPipeline,
    geo_monitor: GeopoliticalRiskMonitor,
    universe_engine: UniverseEngine,
    strategy_engine: StrategyEngine,
    executor: OrderExecutor,
    rate_limiter: RateLimiter,
    freq_optimiser: FrequencyOptimiser,
    ws_manager: WebSocketSubscriptionManager,
    telegram: "TelegramNotifier",   # B-01 FIX: was missing from signature
    pos_manager: PositionManager,   # R-12: Added PositionManager
    logbook: Any = None,
    bot_state: Any = None,
    portfolio_risk: Optional["PortfolioRiskMonitor"] = None,  # R-13
) -> None:
    """
    Main evaluation loop with:
      • Dynamic poll_interval computed by FrequencyOptimiser.
      • Full universe evaluation each cycle (batch LTP).
      • WebSocket resubscription applied when universe changes.
      • Macro signals pushed to UniverseEngine for live sector scoring.
      • PositionManager evaluates TSL/SL exits every cycle.
    """
    logger.info("Strategy loop started.")
    # B-17 FIX: use plain local variables instead of function attributes
    _cycle          = 0
    _prev_gri       = 0.0
    _quota_notified = False
    loop = asyncio.get_running_loop()
    last_interval_log = 0.0

    # Position & cooldown guards
    # After placing any order for a symbol, block re-entry for this many seconds.
    # Prevents the same signal from firing on consecutive 1-second cycles.
    SIGNAL_COOLDOWN_S: float = 300.0   # 5 minutes between orders on the same symbol
    _order_cooldowns: Dict[str, float] = {}   # symbol → monotonic ts of last order
    _open_positions:  Dict[str, int]   = {}   # symbol → net MIS qty (+ long, − short)

    # R-10: Simple state object for the forced square-off flag.
    # squareoff_done_today resets each calendar day so the bot squares off
    # exactly once per trading session.
    class _LoopState:
        squareoff_done_today: bool = False
        squareoff_date: Optional[date] = None
        pos_seeded: bool = False   # Flag: seed pos_manager on first successful fetch
        risk_last_mono: float = 0.0      # R-13 throttle
        margin_last_mono: float = 0.0    # R-14 throttle
        margin_seeded: bool = False      # R-14 first-poll flag
    _strategy_loop_state = _LoopState()

    while True:
        cycle_start = time.monotonic()
        _cycle   += 1
        cycle_num = _cycle

        # ---- Market hours guard ----
        _open = is_market_open()
        if cycle_num == 1:
            logger.info("Market hours check: is_open=%s", _open)

        if not _open:
            wait = seconds_until_market_open()
            if wait > 0:
                logger.info("Market closed — sleeping %.0f s until open.", min(wait, 60))
                await asyncio.sleep(min(wait, 60))
            else:
                await asyncio.sleep(60)
            continue

        # ── Task heartbeat — update every cycle so /tasks shows health ───────
        if bot_state is not None:
            _hb = bot_state.task_heartbeats.setdefault("strategy_loop", {"cycles": 0, "errors": 0})
            _hb["last_beat"] = time.time()
            _hb["cycles"]   += 1

        # ---- R-11/R-13: Sync active capital from BotState on every cycle ----
        # Single source of truth = bot_state.paper_capital (misnamed historically —
        # it is the bot's live working capital in both paper and live modes).
        # Any /capital command fans out to RiskManager (position sizing) and
        # PortfolioRiskMonitor (D/W/M loss budgets).
        if bot_state is not None:
            _active_cap = bot_state.paper_capital
            if _active_cap > 0 and abs(_active_cap - risk_manager._capital) > 1.0:
                risk_manager.update_capital(_active_cap)
                if portfolio_risk is not None:
                    portfolio_risk.update_capital(_active_cap)
                # Also update the settings shadow so new components that read
                # TOTAL_CAPITAL at construction time pick up the new value.
                try:
                    settings.strategy.TOTAL_CAPITAL = _active_cap
                except Exception:
                    pass
                logger.warning(
                    "Active capital updated via /capital: ₹%.2f "
                    "(RiskManager + PortfolioRisk synced).", _active_cap,
                )

        # ---- R-14: Auto-sync capital from Zerodha broker balance ----
        # Only in LIVE mode, only if enabled, and throttled by
        # AUTO_SYNC_INTERVAL_S. Safety buffer keeps some margin headroom.
        if (
            not is_paper_trade()
            and getattr(settings.strategy, "AUTO_SYNC_CAPITAL", False)
            and bot_state is not None
        ):
            _msync_interval = float(getattr(settings.strategy, "AUTO_SYNC_INTERVAL_S", 300))
            _now_mono_m     = time.monotonic()
            if (_now_mono_m - _strategy_loop_state.margin_last_mono) >= _msync_interval:
                _strategy_loop_state.margin_last_mono = _now_mono_m
                try:
                    _margins = await executor.get_margins()
                    _equity  = (_margins or {}).get("equity", {}) or {}
                    # Prefer equity.net (available for trading), fall back to
                    # equity.available.live_balance for older API schemas.
                    _raw_bal = float(
                        _equity.get("net")
                        or ((_equity.get("available") or {}).get("live_balance"))
                        or ((_equity.get("available") or {}).get("cash"))
                        or 0.0
                    )
                    if _raw_bal <= 0:
                        logger.warning(
                            "R-14 broker margin poll returned zero balance; "
                            "skipping auto-sync this cycle."
                        )
                    else:
                        _buffer = float(getattr(
                            settings.strategy, "AUTO_SYNC_SAFETY_BUFFER", 0.90
                        ))
                        _ceiling = float(getattr(
                            settings.strategy, "AUTO_SYNC_MAX_CAPITAL", 1e7
                        ))
                        _new_cap = min(_raw_bal * _buffer, _ceiling)
                        _cur_cap = bot_state.paper_capital
                        _min_delta_pct = float(getattr(
                            settings.strategy, "AUTO_SYNC_MIN_DELTA_PCT", 0.01
                        ))
                        _delta = abs(_new_cap - _cur_cap) / max(_cur_cap, 1.0)
                        # On first poll, always apply (seed from broker).
                        if (not _strategy_loop_state.margin_seeded) or _delta >= _min_delta_pct:
                            bot_state.paper_capital = _new_cap
                            _strategy_loop_state.margin_seeded = True
                            logger.warning(
                                "R-14 broker capital auto-sync: "
                                "raw=₹%.2f × buffer=%.2f → active=₹%.2f "
                                "(was ₹%.2f, delta=%.1f%%). "
                                "RiskManager + PortfolioRisk will resync next cycle.",
                                _raw_bal, _buffer, _new_cap, _cur_cap, _delta * 100,
                            )
                            try:
                                await telegram.send_message(
                                    f"💰 <b>Capital auto-synced from Zerodha</b>\n"
                                    f"Broker balance: ₹{_raw_bal:,.2f}\n"
                                    f"Safety buffer: {_buffer:.0%}\n"
                                    f"Active capital: <b>₹{_new_cap:,.2f}</b>\n"
                                    f"(was ₹{_cur_cap:,.2f})"
                                )
                            except Exception:
                                pass
                except Exception as _m_exc:
                    logger.error("R-14 margin auto-sync failed: %s", _m_exc)

        # ---- R-13: Portfolio-level risk budget check (throttled) ----
        if portfolio_risk is not None and bot_state is not None:
            _risk_interval = float(getattr(settings.strategy, "RISK_CHECK_INTERVAL_S", 60))
            _now_mono = time.monotonic()
            if (_now_mono - _strategy_loop_state.risk_last_mono) >= _risk_interval:
                try:
                    _budget, _changed, _reason = portfolio_risk.check(bot_state)
                    _strategy_loop_state.risk_last_mono = _now_mono
                    if _changed and _reason:
                        logger.warning("R-13 RISK HALT TRANSITION: %s", _reason)
                        try:
                            await telegram.send_message(
                                f"🚨 <b>Risk halt transition</b>\n{_reason}"
                            )
                        except Exception as _tg_exc:
                            logger.error("Risk halt Telegram alert failed: %s", _tg_exc)
                except Exception as _risk_exc:
                    logger.exception("PortfolioRiskMonitor.check failed: %s", _risk_exc)

        # ---- Push latest macro + geo signals to universe engine ----
        logger.info("Strategy loop [cycle %d]: updating macro signals...", cycle_num)
        universe_engine.update_macro_signals(
            weather_anomaly=alt_data.weather.get_aggregate_anomaly(),
            sentiment_score=alt_data.sentiment.sentiment_score,
            geo_risk=geo_monitor.current.composite,
        )

        # ---- Get current active universe ----
        active_symbols = universe_engine.get_active_symbols()
        if not active_symbols:
            logger.warning("Empty universe — waiting for initialisation.")
            await asyncio.sleep(5)
            continue

        # ---- Apply pending WebSocket resubscription if universe changed ----
        new_tokens = universe_engine.get_instrument_tokens()
        if new_tokens:
            logger.info("Strategy loop [cycle %d]: applying subscriptions...", cycle_num)
            await ws_manager.apply_subscription(new_tokens)
        
        logger.info("Strategy loop [cycle %d]: fetching batch LTP...", cycle_num)

        # ---- Dynamic frequency ----
        interval = freq_optimiser.compute_interval(len(active_symbols))
        now = time.monotonic()
        if now - last_interval_log > 300:   # Log every 5 min
            logger.info("Frequency: %s", freq_optimiser.describe(len(active_symbols)))
            last_interval_log = now

        # ---- Batch LTP fetch (one API call for all symbols) ----
        exchange_syms = [f"{settings.kite.EXCHANGE}:{s}" for s in active_symbols]
        ltp_map: Dict[str, float] = {}
        try:
            async with rate_limiter.request_slot():
                raw_ltp = await loop.run_in_executor(
                    None, lambda s=exchange_syms: kite.ltp(s)
                )
            for k, v in raw_ltp.items():
                ltp_map[k.split(":")[1]] = float(v["last_price"])

            # Diagnostic: log LTP health every 10 cycles
            priced = sum(1 for p in ltp_map.values() if p > 0)
            if cycle_num % 10 == 1 or priced == 0:
                logger.info(
                    "LTP health: %d/%d symbols priced | sample=%s",
                    priced, len(active_symbols),
                    next(((s, ltp_map[s]) for s in list(active_symbols)[:1]
                          if s in ltp_map), ("none", 0)),
                )
            
            if bot_state:
                bot_state.update(
                    active_symbols=len(active_symbols),
                    top_symbols=list(active_symbols[:3]),
                )

            if priced == 0:
                logger.error("ALL PRICES ZERO — KITE_ACCESS_TOKEN likely expired (resets 6 AM IST). Update .env.sh and restart.")
                await asyncio.sleep(interval)
                continue

        except Exception as exc:
            estr = str(exc).lower()
            if any(x in estr for x in ["token", "403", "401", "invalid",
                                        "incorrect", "access_token"]):
                logger.critical("TOKEN EXPIRED: %s", exc)
                if bot_state:
                    bot_state.token_valid = False
                    bot_state.paused = True
                # Alert via Telegram — instruct user to send /login then /token
                try:
                    login_url = kite.login_url()
                    await telegram.send(
                        f"🔑 <b>Zerodha Token Expired — Action Required</b>\n"
                        f"{'─' * 34}\n"
                        f"The access token has expired. Trading is paused.\n\n"
                        f"<b>To refresh without restarting:</b>\n"
                        f"1. Open: <code>{login_url}</code>\n"
                        f"2. Log in and copy the <code>request_token</code> "
                        f"from the redirect URL\n"
                        f"3. Send: <code>/token &lt;request_token&gt;</code>\n\n"
                        f"Trading resumes automatically once the token is refreshed."
                    )
                except Exception:
                    pass
                # Wait for token to be refreshed via /token command
                if bot_state:
                    logger.info("Waiting for /token via Telegram…")
                    while not bot_state.token_valid:
                        await asyncio.sleep(10)
                    logger.info("Token refreshed — reconnecting WebSocket and resuming.")
                    # Reconnect KiteTicker with the new token (kite.access_token
                    # is updated by TelegramController's /token handler).
                    try:
                        await ticker_wrapper.reconnect_with_token(kite.access_token)
                    except Exception as _tr_exc:
                        logger.error("Ticker reconnect after REST 403 failed: %s", _tr_exc)
                    bot_state.paused = False
            else:
                logger.error("LTP batch failed: %s", exc)
            await asyncio.sleep(interval)
            continue

        # ---- GRI + Sentiment (both cached/non-blocking) ----
        # FIX B-18 (Bug #5): Assign gri HERE — before any continue statements — so
        # bot_state.update() always has fresh GRI data regardless of trading mode.
        # Previously gri was assigned after the paused/paper_monitor continues,
        # which left last_gri=0.000, last_gri_level='UNKNOWN' in /status permanently.
        prev_gri_val = _prev_gri
        gri          = geo_monitor.current
        sentiment    = alt_data.sentiment
        weather_anom = alt_data.weather.get_aggregate_anomaly()

        # FIX B-18 (Bug #2 / #9): In GRI-only mode, ALWAYS recompute synthetic
        # sentiment on EVERY cycle from the latest GRI value.
        # Previously: only ran when sentiment_score == 0.0 (i.e., only on first cycle
        # when Gemini had never run). On subsequent cycles, score was non-zero from
        # the first synthetic pass so the block was never re-entered, meaning GRI
        # changes never propagated to update the sentiment signal throughout the day.
        if bot_state and bot_state.mode.value == "gri_only":
            from alternative_data import SentimentResult as _SR
            gri_score = (gri.composite - 0.30) * -2.5
            gri_score = max(-0.8, min(0.8, gri_score))
            sentiment = _SR(
                sentiment_classification="Fear"      if gri_score < -0.15 else
                                         "Excitement" if gri_score > 0.15  else "Neutral",
                sentiment_score=round(gri_score, 4),
                rationale=f"GRI-synthetic: GRI={gri.composite:.3f} ({gri.level})",
                source_articles=[], key_entities=[], model_latency_ms=0.0,
                risk_context="", volatility_context=gri.level, gpr_context="",
            )

        # FIX B-18 (Bug #5): Update bot_state with live GRI/sentiment BEFORE any
        # continue statements so /status always reflects current data.
        if bot_state:
            bot_state.update(
                last_gri=gri.composite, last_gri_level=gri.level,
                last_vix=gri.india_vix, last_usdinr=gri.usdinr,
                last_sentiment=sentiment.sentiment_score,
                last_sentiment_class=sentiment.sentiment_classification,
                last_alpha_mult=gri.alpha_multiplier,
                last_kelly_mult=gri.kelly_multiplier,
                gemini_working=(bot_state.mode.value != "gri_only" and
                                sentiment.sentiment_score != 0.0),
            )

        # ── Apply trading mode guards ──────────────────────────────────
        if bot_state and bot_state.paused:
            await asyncio.sleep(1)
            continue

        if bot_state and bot_state.mode.value == "paper_monitor":
            await asyncio.sleep(1)
            continue
        if abs(gri.composite - prev_gri_val) > 0.05 and prev_gri_val > 0:
            asyncio.create_task(telegram.notify_gri_spike(
                prev=prev_gri_val, composite=gri.composite,
                level=gri.level, keywords=getattr(gri, "keywords", []),
            ), name="tg_gri_spike")
        _prev_gri = gri.composite

        # ---- Fetch open MIS positions (once per cycle, before signal loop) ----
        # PAPER_TRADE: skip the real Zerodha API call — positions are tracked
        # in-memory from simulated fills.  Calling kite.positions() in paper
        # mode would return real account positions (from manual trades), which
        # could wrongly trigger the 3:15 PM square-off with real orders.
        if not is_paper_trade():
            try:
                async with rate_limiter.request_slot():
                    pos_data = await loop.run_in_executor(None, kite.positions)
                _open_positions = {}
                for pos in pos_data.get("day", []):
                    if pos.get("product") == settings.kite.PRODUCT:
                        sym = pos.get("tradingsymbol", "")
                        qty = int(pos.get("quantity", 0))
                        if sym:
                            _open_positions[sym] = qty
                if _open_positions:
                    logger.debug("Open MIS positions: %s", _open_positions)
                    
                # R-12: Seed PositionManager on startup using first position fetch
                if not _strategy_loop_state.pos_seeded:
                    for sym, qty in _open_positions.items():
                        if qty != 0:
                            # We don't have entry price for pre-existing trades, 
                            # use current LTP as a neutral baseline.
                            _ltp = ltp_map.get(sym, 0.0)
                            if _ltp > 0:
                                pos_manager.on_trade_executed(
                                    sym, 
                                    TradeDirection.BUY if qty > 0 else TradeDirection.SELL,
                                    _ltp, abs(qty)
                                )
                    _strategy_loop_state.pos_seeded = True
            except Exception as exc:
                logger.warning("Positions fetch failed (skipping guard this cycle): %s", exc)
                # Keep _open_positions from the previous cycle as a best-effort fallback

        # ── R-12: Evaluate TSL/SL exits from PositionManager ─────────────────
        if _open_positions:
            exits = pos_manager.update(ltp_map)
            for sym, qty, exit_dir, reason in exits:
                logger.warning("TSL/SL Triggered for %s: %s", sym, reason)
                await telegram.send(
                    f"🛑 <b>RISK EXIT: {sym}</b>\n"
                    f"{'─' * 30}\n"
                    f"<b>Reason:</b> {reason}\n"
                    f"<b>Action:</b> {exit_dir.value} (Close {qty} shares)\n"
                )
                
                # Execute exit order
                # We use a dummy SignalState to trigger a closing order
                from strategy import SignalState
                
                exit_sig = SignalState(
                    symbol=sym, timestamp=datetime.now(timezone.utc),
                    current_price=ltp_map.get(sym, 0.0), vwap=ltp_map.get(sym, 0.0),
                    ofi=0.0, mlofi=0.0, aflow_ratio=0.0,
                    sentiment_score=0.0, sentiment_class="RiskExit",
                    ml_signal=0.0, ml_confidence=1.0, ml_model_version="exit",
                    ml_is_fallback=True, alpha_raw=0.0, alpha=0.0,
                    direction=exit_dir, quantity=qty, position_fraction=0.0,
                    busseti_f=0.0, vol_regime=0.0, geo_risk=0.0,
                    geo_level="RiskExit", geo_alpha_multiplier=1.0,
                    geo_kelly_multiplier=1.0, is_decayed=False,
                    rationale=f"Risk Manager Exit: {reason}",
                    product_type=settings.kite.PRODUCT
                )
                report = await executor.execute(exit_sig)
                if report.success:
                    # Update local position count and clear cooldown
                    _open_positions[sym] = 0
                    if sym in _order_cooldowns:
                        del _order_cooldowns[sym]
                else:
                    # Exit failed! Put it back in _open_positions to retry next cycle
                    logger.error("Risk Exit FAILED for %s. Will retry via position-sync.", sym)

        # ── R-10: Forced square-off at 3:15 PM IST ───────────────────────────
        # Zerodha auto-squares all open MIS positions at ~3:20 PM and charges
        # ₹50 + 18% GST (≈ ₹59) per position. We pre-empt this by closing all
        # open positions ourselves at 3:15 PM using MARKET orders.
        _ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30)))

        # Reset square-off flag at start of each new trading day
        if _strategy_loop_state.squareoff_date != _ist_now.date():
            _strategy_loop_state.squareoff_done_today = False
            _strategy_loop_state.squareoff_date = _ist_now.date()
        _past_squareoff = (
            _ist_now.hour > settings.strategy.SQUARE_OFF_HOUR_IST
            or (
                _ist_now.hour == settings.strategy.SQUARE_OFF_HOUR_IST
                and _ist_now.minute >= settings.strategy.SQUARE_OFF_MINUTE_IST
            )
        )
        if _past_squareoff and not is_paper_trade():
            open_to_close = {s: q for s, q in _open_positions.items() if q != 0}
            if open_to_close and not getattr(_strategy_loop_state, "squareoff_done_today", False):
                logger.warning(
                    "⏰ 3:15 PM IST — force-closing %d open MIS position(s): %s",
                    len(open_to_close), list(open_to_close.keys()),
                )
                await telegram.send(
                    f"⏰ <b>3:15 PM Square-Off</b>\n"
                    f"Closing {len(open_to_close)} open MIS position(s) to avoid "
                    f"Zerodha auto-square charges.\n"
                    + "\n".join(
                        f"  {'SELL' if q > 0 else 'BUY'} {s} qty={abs(q)}"
                        for s, q in open_to_close.items()
                    )
                )
                for sym, qty in open_to_close.items():
                    try:
                        close_direction = (
                            kite.TRANSACTION_TYPE_SELL if qty > 0
                            else kite.TRANSACTION_TYPE_BUY
                        )
                        close_qty = abs(qty)

                        # R-10 FIX: Zerodha MARKET orders require a non-zero price for
                        # the "protection amount" check (NSE circuit-breaker guard).
                        # Sending price=0 (the default when omitted) causes:
                        #   "Market order cannot be placed with protection amount 0"
                        # Solution: use a LIMIT order with a wide ±2% buffer from the
                        # current LTP.  At ±2% the order always fills at market price
                        # while satisfying Zerodha's price validation.
                        _ltp = ltp_map.get(sym, 0.0)
                        if _ltp <= 0:
                            # LTP unavailable — log and skip; alert user to close manually.
                            raise ValueError(
                                f"LTP unavailable for {sym} — cannot compute limit price."
                            )
                        _SQ_BUFFER = 0.02   # 2% — wide enough to always fill
                        if close_direction == kite.TRANSACTION_TYPE_BUY:
                            _sq_price = round(_ltp * (1.0 + _SQ_BUFFER), 1)
                        else:
                            _sq_price = round(_ltp * (1.0 - _SQ_BUFFER), 1)

                        async with rate_limiter.order_slot():
                            sq_id = await loop.run_in_executor(
                                None,
                                lambda s=sym, d=close_direction, q=close_qty, p=_sq_price: kite.place_order(
                                    variety=settings.kite.ORDER_VARIETY,
                                    exchange=settings.kite.EXCHANGE,
                                    tradingsymbol=s,
                                    transaction_type=d,
                                    quantity=q,
                                    product=settings.kite.PRODUCT,
                                    order_type=kite.ORDER_TYPE_LIMIT,  # LIMIT avoids price=0 rejection
                                    price=p,
                                    tag="SS_SQUAREOFF",
                                )
                            )
                        logger.info(
                            "Square-off placed: %s qty=%d @ ₹%.1f (ltp=%.1f) → order_id=%s",
                            sym, close_qty, _sq_price, _ltp, sq_id,
                        )
                    except Exception as sq_exc:
                        logger.error("Square-off FAILED for %s: %s", sym, sq_exc)
                        await telegram.send(
                            f"⚠️ <b>Square-off failed for {sym}</b>: {sq_exc}\n"
                            f"Please close manually in Kite immediately!"
                        )
                _strategy_loop_state.squareoff_done_today = True

        elif _past_squareoff and is_paper_trade():
            # Paper mode: simulate square-off — clear in-memory positions and notify
            open_to_close = {s: q for s, q in _open_positions.items() if q != 0}
            if open_to_close and not getattr(_strategy_loop_state, "squareoff_done_today", False):
                logger.warning(
                    "⏰ [PAPER] 3:15 PM IST — simulating square-off for %d position(s): %s",
                    len(open_to_close), list(open_to_close.keys()),
                )
                await telegram.send(
                    f"⏰ <b>[PAPER] 3:15 PM Square-Off Simulated</b>\n"
                    f"Clearing {len(open_to_close)} simulated MIS position(s).\n"
                    + "\n".join(
                        f"  {'SELL' if q > 0 else 'BUY'} {s} qty={abs(q)}"
                        for s, q in open_to_close.items()
                    )
                )
                _open_positions.clear()
                _strategy_loop_state.squareoff_done_today = True

            # Skip all new signal entries after 3:15 PM — no new positions
            continue
        # ── End forced square-off ────────────────────────────────────────────

        # ---- Per-symbol signal evaluation ----
        actionable_count = 0
        for symbol in active_symbols:
            price = ltp_map.get(symbol, 0.0)
            if price <= 0:
                continue
            try:
                sig = await strategy_engine.evaluate(
                    symbol=symbol,
                    current_price=price,
                    sentiment=sentiment,
                    gri=gri,
                    gpr_normalised=alt_data.gpr_snapshot.gpr_normalised,
                    vol_regime=("HIGH" if gri.composite > 0.50 else
                                "MODERATE" if gri.composite > 0.25 else "LOW"),
                )
                if logbook:
                    await logbook.log_signal(
                        signal          = sig,
                        sentiment_score = sentiment.sentiment_score,
                        sentiment_class = sentiment.sentiment_classification,
                        risk_context    = getattr(sentiment, "risk_context", ""),
                    )
                if cycle_num % 5 == 1 and symbol in list(active_symbols)[:3]:
                    logger.info(
                        "  sig[%s] ml=%.4f dir=%s qty=%d actionable=%s decayed=%s",
                        symbol, sig.ml_signal, sig.direction.value,
                        sig.quantity, sig.is_actionable, sig.is_decayed,
                    )
                if sig.is_actionable:
                    actionable_count += 1

                    # ── Telegram trade guards (/nobuy, /nosell) ────────────
                    if (bot_state and bot_state.no_new_buys
                            and sig.direction == TradeDirection.BUY):
                        logger.info(
                            "TradeGuard[nobuy]: skipping BUY signal for %s", symbol
                        )
                        continue
                    if (bot_state and bot_state.no_new_sells
                            and sig.direction == TradeDirection.SELL):
                        logger.info(
                            "TradeGuard[nosell]: skipping SELL signal for %s", symbol
                        )
                        continue

                    # ── R-13: Symbol blacklist (consecutive-loss guard) ────
                    if bot_state and symbol in getattr(bot_state, "risk_blacklist", set()):
                        logger.info(
                            "R-13 blacklist: skipping %s (consecutive losses today)",
                            symbol,
                        )
                        continue

                    # ── Cooldown guard ─────────────────────────────────────
                    now_mono = time.monotonic()
                    last_ts  = _order_cooldowns.get(symbol, 0.0)
                    remaining = SIGNAL_COOLDOWN_S - (now_mono - last_ts)
                    if remaining > 0:
                        logger.info(
                            "Cooldown: skipping %s %s (%.0f s left before re-entry allowed)",
                            sig.direction.value, symbol, remaining,
                        )
                        continue

                    # ── Position guard ─────────────────────────────────────
                    net_qty = _open_positions.get(symbol, 0)
                    if net_qty > 0 and sig.direction == TradeDirection.BUY:
                        logger.info(
                            "Position guard: already LONG %s (qty=%d) — skipping BUY signal",
                            symbol, net_qty,
                        )
                        continue
                    if net_qty < 0 and sig.direction == TradeDirection.SELL:
                        logger.info(
                            "Position guard: already SHORT %s (qty=%d) — skipping SELL signal",
                            symbol, net_qty,
                        )
                        continue

                    logger.info(
                        "Signal -> %s %s qty=%d alpha=%.5f",
                        sig.direction.value, symbol, sig.quantity, sig.alpha,
                    )
                    asyncio.create_task(telegram.notify_signal_fired(
                        symbol=sig.symbol, direction=sig.direction.value,
                        alpha=sig.alpha, qty=sig.quantity,
                        sentiment_class=sentiment.sentiment_classification,
                        gri_level=gri.level, gri=gri.composite,
                    ), name="tg_signal")
                    report = await executor.execute(sig)

                    # Set cooldown so this symbol is blocked for SIGNAL_COOLDOWN_S
                    if report.success:
                        _order_cooldowns[symbol] = time.monotonic()
                        
                        # R-12: Register new position with manager.
                        # Bug 2.4 fix — forward σ_ann so stops are vol-scaled.
                        pos_manager.on_trade_executed(
                            symbol,
                            sig.direction,
                            report.avg_fill_price or price,
                            report.total_quantity,
                            sigma_ann=float(getattr(sig, "vol_regime", 0.20) or 0.20),
                        )

                        # In paper mode, track simulated positions in-memory so
                        # the position guard works correctly between cycles.
                        # (Real mode tracks via kite.positions() API call above.)
                        if is_paper_trade():
                            delta = report.total_quantity if sig.direction == TradeDirection.BUY else -report.total_quantity
                            _open_positions[symbol] = _open_positions.get(symbol, 0) + delta
                            if _open_positions[symbol] == 0:
                                del _open_positions[symbol]

                    # B-04 FIX: log_trade() was never called — trades CSV was always empty
                    if logbook:
                        trade_mode = "PAPER" if is_paper_trade() else "LIVE"
                        await logbook.log_trade(report, sig, trade_mode)
                    if bot_state:
                        bot_state.trades_today += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Signal error [%s]: %s", symbol, exc, exc_info=True)
                if bot_state is not None:
                    _hb = bot_state.task_heartbeats.setdefault("strategy_loop", {"cycles": 0, "errors": 0, "last_beat": time.time()})
                    _hb["errors"] += 1

        if cycle_num % 10 == 1:
            sample = active_symbols[0] if active_symbols else "N/A"
            logger.info(
                "Pipeline | sent=%.3f (%s) | GRI=%.3f (%s) | "
                "sample=%s @ %.2f | actionable=%d/%d",
                sentiment.sentiment_score, sentiment.sentiment_classification,
                gri.composite, gri.level,
                sample, ltp_map.get(sample, 0.0),
                actionable_count, len(active_symbols),
            )
            if sentiment.sentiment_score == 0.0:
                # FIX B-17: distinguish actual quota exhaustion from other errors.
                # sentiment_score==0.0 fires for ANY failure (network, parse, quota).
                # Only call it a quota problem when the ModelRotator confirms all slots
                # are on cooldown — otherwise log a more accurate message.
                from agent_pipeline import _rotator as _gm_rotator  # noqa: PLC0415
                if _gm_rotator.all_on_cooldown():
                    logger.warning(
                        "Sentiment=0.0 — Gemini quota exhausted (all models on cooldown, "
                        "soonest in %.0fs). Check Google AI Studio dashboard for quota usage.",
                        _gm_rotator.soonest_available_in(),
                    )
                    if not _quota_notified:
                        _quota_notified = True
                        asyncio.create_task(telegram.notify_quota_exhausted(
                            mode=bot_state.mode.value if bot_state else "unknown"
                        ), name="tg_quota")
                else:
                    logger.warning(
                        "Sentiment=0.0 — Gemini call failed (non-quota issue: "
                        "network/parse error or stale result). Bot will retry next cycle."
                    )

        if bot_state:
            bot_state.update(
                last_gri=gri.composite, last_gri_level=gri.level,
                last_vix=gri.india_vix, last_usdinr=gri.usdinr,
                last_sentiment=sentiment.sentiment_score,
                last_sentiment_class=sentiment.sentiment_classification,
                last_alpha_mult=gri.alpha_multiplier,
                last_kelly_mult=gri.kelly_multiplier,
                active_symbols=len(active_symbols),
                top_symbols=list(active_symbols[:3]),
                gemini_working=sentiment.sentiment_score != 0.0,
            )
            bot_state.signals_today += actionable_count

        # ── Dashboard: push live state to Redis every 5 cycles ────────────
        if cycle_num % 5 == 1:
            try:
                await redis_client.hset("bot:state", mapping={
                    "running":        "1",
                    "mode":           bot_state.mode.value if bot_state else "full",
                    "paused":         "1" if (bot_state and bot_state.paused) else "0",
                    "active_symbols": str(len(active_symbols)),
                    "top_symbols":    json.dumps(list(active_symbols[:5])),
                    "gri":            str(gri.composite),
                    "gri_level":      gri.level,
                    "sentiment":      str(sentiment.sentiment_score),
                    "sentiment_class":sentiment.sentiment_classification,
                    "trades_today":   str(bot_state.trades_today if bot_state else 0),
                    "signals_today":  str(bot_state.signals_today if bot_state else 0),
                    "cycle":          str(cycle_num),
                    "last_update":    datetime.now(timezone.utc).isoformat(),
                })
                await redis_client.expire("bot:state", 300)   # 5-min TTL

                # Check for dashboard pause/resume command
                cmd = await redis_client.get("bot:cmd")
                if cmd:
                    cmd_str = cmd.decode() if isinstance(cmd, bytes) else cmd
                    if cmd_str == "pause" and bot_state and not bot_state.paused:
                        bot_state.paused = True
                        logger.info("Dashboard command: PAUSE")
                    elif cmd_str == "resume" and bot_state and bot_state.paused:
                        bot_state.paused = False
                        logger.info("Dashboard command: RESUME")
                    await redis_client.delete("bot:cmd")
            except Exception as _dash_exc:
                logger.debug("Dashboard state push failed: %s", _dash_exc)

        elapsed = time.monotonic() - cycle_start
        await asyncio.sleep(max(0.0, interval - elapsed))


# ===========================================================================
# ASYNC MAIN
# ===========================================================================

async def main(kite: KiteConnect, access_token: str, tg_offset: int = 0) -> None:
    shutdown = GracefulShutdown()
    shutdown.install_signal_handlers()

    print_banner("System Initialisation")

    redis_client = aioredis.Redis(
        host=settings.redis.HOST, port=settings.redis.PORT,
        db=settings.redis.DB, password=settings.redis.PASSWORD or None,
        decode_responses=False, max_connections=30,
    )
    try:
        await redis_client.ping()
        logger.info("Redis  OK  %s:%d", settings.redis.HOST, settings.redis.PORT)
    except Exception as exc:
        logger.critical("Redis unavailable: %s", exc); sys.exit(1)

    rate_limiter   = RateLimiter(
        max_requests_per_second=settings.kite.MAX_REQUESTS_PER_SECOND,
        max_orders_per_minute=settings.kite.MAX_ORDERS_PER_MINUTE,
    )
    telegram       = TelegramNotifier()
    alt_data       = AlternativeDataPipeline(redis_client)
    freq_optimiser = FrequencyOptimiser()
    geo_monitor    = GeopoliticalRiskMonitor(redis_client)
    risk_manager   = RiskManager(settings.strategy.TOTAL_CAPITAL)
    ml_engine      = EnsembleSignalEngine(redis_client)
    agent_pipeline = AgentPipeline()
    logbook        = Logbook()
    bot_state      = BotState()
    tg_controller  = TelegramController(bot_state)
    # If headless startup consumed some Telegram updates (e.g. the /token
    # command), advance the poll offset so those messages are not re-processed.
    if tg_offset > 0:
        tg_controller._offset = tg_offset
        logger.info("TelegramController offset seeded to %d from headless startup.", tg_offset)
    # Register the live kite instance so /login and /token commands can
    # hot-swap the daily access token without restarting the bot.
    tg_controller.set_kite(kite, settings.kite.API_SECRET)
    # R-11: Register logbook so /pnl command can pull today's trade data.
    tg_controller.set_logbook(logbook)
    # Wire ModelRotator so /resetquota command can clear stale Gemini quota state.
    from agent_pipeline import _rotator as _gm_rotator_ref
    tg_controller.set_rotator(_gm_rotator_ref)
    # Wire Telegram ERROR log forwarding — any ERROR/CRITICAL log line is
    # sent to the Telegram chat automatically (rate-limited to 1 per 30 s).
    _tg_log_handler = TelegramLogHandler(
        token=settings.telegram.BOT_TOKEN,
        chat_id=str(settings.telegram.CHAT_ID),
        cooldown=30.0,
    )
    _tg_log_handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(_tg_log_handler)
    logger.info("TelegramLogHandler installed — ERROR logs will be forwarded to Telegram.")
    strategy_engine = StrategyEngine(redis_client, risk_manager, ml_engine)
    executor        = OrderExecutor(kite, rate_limiter, telegram)
    pos_manager     = PositionManager()   # R-12: Position tracking engine
    portfolio_risk_monitor = PortfolioRiskMonitor(
        capital=settings.strategy.TOTAL_CAPITAL
    )   # R-13: Portfolio loss-budget monitor
    tg_controller.set_portfolio_risk(portfolio_risk_monitor)
    # R-14: Wire broker-balance fetcher so /balance and /synccapital work.
    tg_controller.set_margin_fetcher(executor.get_margins)

    print_banner("Geopolitical Risk Feed")
    await geo_monitor.initialise()
    gri_init = geo_monitor.current
    logger.info(
        "Initial GRI: composite=%.3f (%s) | VIX=%.1f | USD/INR=%.2f",
        gri_init.composite, gri_init.level, gri_init.india_vix, gri_init.usdinr,
    )

    universe_engine = UniverseEngine(kite, redis_client, rate_limiter)
    universe_engine.update_macro_signals(
        weather_anomaly=alt_data.weather.get_aggregate_anomaly(),
        sentiment_score=0.0, geo_risk=gri_init.composite,
    )

    print_banner("Universe Selection")
    await universe_engine.initialise()
    active_symbols = universe_engine.get_active_symbols()
    active_tokens  = universe_engine.get_instrument_tokens()
    logger.info("Active universe: %d symbols", len(active_symbols))

    symbol_map = {
        universe_engine.get_metadata(s).instrument_token: s
        for s in active_symbols
        if universe_engine.get_metadata(s) and universe_engine.get_metadata(s).instrument_token > 0
    }
    # ── WebSocket token-expiry callback ──────────────────────────────────────
    # Called from the KiteTicker thread when a 403 is received.
    # Sets the bot to paused state and sends a Telegram alert so the user
    # can supply a fresh token via /token without restarting the service.
    async def _on_ticker_token_error() -> None:
        if not bot_state.token_valid:
            return   # already being handled (e.g. REST also caught it)
        logger.critical("WebSocket 403 — access token expired. Pausing bot and requesting refresh.")
        bot_state.token_valid = False
        bot_state.paused      = True
        try:
            login_url = kite.login_url()
            await telegram.send(
                "🔴 <b>Zerodha Token Expired (WebSocket 403)</b>\n"
                f"{'─' * 34}\n"
                "The daily access token has expired. Trading is paused.\n\n"
                "<b>To refresh without restarting:</b>\n"
                f"1️⃣ Open: <code>{login_url}</code>\n"
                "2️⃣ Complete Zerodha login\n"
                "3️⃣ Copy the <code>request_token</code> from the redirect URL\n"
                "4️⃣ Send: <code>/token &lt;request_token&gt;</code>\n\n"
                "Trading and WebSocket feed resume automatically after /token."
            )
        except Exception as _cb_exc:
            logger.error("Could not send token-expired Telegram alert: %s", _cb_exc)

    ticker_wrapper = AsyncKiteTickerWrapper(
        api_key=settings.kite.API_KEY, access_token=access_token,
        instruments=active_tokens[:settings.universe.MAX_WEBSOCKET_SUBSCRIPTIONS],
        symbol_map=symbol_map, redis_client=redis_client, rate_limiter=rate_limiter,
        token_error_callback=_on_ticker_token_error,
    )
    ws_manager = WebSocketSubscriptionManager(ticker_wrapper, rate_limiter)
    universe_engine.register_change_callback(ws_manager.on_universe_change)
    candle_agg = CandleAggregator(redis_client)

    # Telegram mode selector
    asyncio.create_task(tg_controller.poll_loop(), name="tg_poll")
    
    async def _mode_init_task():
        await tg_controller.send_mode_selector()
        chosen_mode = await tg_controller.wait_for_mode(timeout_s=120)
        bot_state.update(mode=chosen_mode)
        logger.info("Trading mode: %s", chosen_mode.value)
        await telegram.send("MODE SET: " + chosen_mode.value + "\nSend /status for a live update.")

    asyncio.create_task(_mode_init_task(), name="mode_init")

    print_banner("Starting Background Tasks")

    def _is_market_open() -> bool:
        return is_market_open()

    async def sentiment_loop() -> None:
        logger.info("Sentiment loop started.")
        await asyncio.sleep(15)  # Allow NewsFetcher to get initial data
        while True:
            interval = 300
            try:
                if not _is_market_open():
                    await asyncio.sleep(60); continue

                # ── GRI-only mode: synthesise sentiment from GRI; no Gemini ──
                if bot_state and bot_state.mode.value == "gri_only":
                    from alternative_data import SentimentResult as _SR
                    _gri_now   = geo_monitor.current
                    _gri_score = (_gri_now.composite - 0.30) * -2.5
                    _gri_score = max(-0.8, min(0.8, _gri_score))
                    _gri_result = _SR(
                        sentiment_classification=(
                            "Fear"      if _gri_score < -0.15 else
                            "Excitement" if _gri_score >  0.15 else "Neutral"
                        ),
                        sentiment_score=round(_gri_score, 4),
                        rationale=(
                            f"GRI-synthetic: GRI={_gri_now.composite:.3f} "
                            f"({_gri_now.level}) VIX={_gri_now.india_vix:.1f}"
                        ),
                        source_articles=[], key_entities=[], model_latency_ms=0.0,
                        risk_context="GRI-only mode", volatility_context=_gri_now.level,
                        gpr_context="",
                    )
                    alt_data.update_sentiment(_gri_result)
                    logger.info(
                        "GRI-synthetic sentiment: %s score=%.3f | GRI=%.3f (%s)",
                        _gri_result.sentiment_classification,
                        _gri_result.sentiment_score,
                        _gri_now.composite, _gri_now.level,
                    )
                    if bot_state is not None:
                        _shb = bot_state.task_heartbeats.setdefault(
                            "sentiment_loop", {"cycles": 0, "errors": 0}
                        )
                        _shb["last_beat"] = time.time()
                        _shb["cycles"]   += 1
                    await asyncio.sleep(60)
                    continue

                # ── All models exhausted: back off until soonest slot is free ─
                from agent_pipeline import _rotator as _gm_rotator  # noqa: PLC0415
                if _gm_rotator.all_on_cooldown():
                    wait_s = max(60.0, _gm_rotator.soonest_available_in())
                    logger.warning(
                        "Sentiment loop: all Gemini models on cooldown — "
                        "backing off %.0f s (quota exhausted).", wait_s
                    )
                    await asyncio.sleep(wait_s)
                    continue

                headlines = alt_data.news.latest_headlines
                gpr       = alt_data.gpr_snapshot
                logger.info("Sentiment loop: headlines_count=%d", len(headlines))

                gri_comp  = geo_monitor.current.composite
                vix       = geo_monitor.current.vix_score * 35.0
                vol_regime = ("EXTREME" if gri_comp > 0.65 else
                              "HIGH" if gri_comp > 0.50 else
                              "MODERATE" if gri_comp > 0.25 else "LOW")
                interval = agent_pipeline.recommended_interval(gri_comp, vix)

                if headlines:
                    logger.info("Sentiment loop: running analysis cycle...")
                    try:
                        result = await agent_pipeline.run_analysis_cycle(
                            headlines=headlines, gpr=gpr, vol_regime=vol_regime,
                            current_vix=vix, gri_composite=gri_comp,
                        )
                    except Exception as exc:
                        logger.error("AgentPipeline failed — using GRI fallback: %s", exc)
                        # Robust fallback: inverse of GRI (higher risk = more negative sentiment)
                        # GRI 0.3 (base) -> 0.0 score. GRI 0.7 (critical) -> -0.8 score.
                        fallback_score = (gri_comp - 0.30) * -2.5
                        fallback_score = max(-0.8, min(0.8, fallback_score))
                        from alternative_data import SentimentResult as _SR
                        result = _SR(
                            sentiment_classification="Fear" if fallback_score < -0.15 else 
                                                     "Excitement" if fallback_score > 0.15 else "Neutral",
                            sentiment_score=round(fallback_score, 4),
                            rationale=f"Fallback (AI Unavailable): GRI={gri_comp:.3f}",
                            source_articles=[], key_entities=[], model_latency_ms=0.0,
                            risk_context="AI pipeline error", volatility_context=vol_regime,
                            gpr_context="",
                        )
                    
                    alt_data.update_sentiment(result)
                    logger.info(
                        "Sentiment [next in %ds]: %s score=%.3f conf=%.2f | '%s'",
                        interval, result.sentiment_classification,
                        result.sentiment_score, getattr(result, "confidence", 0),
                        result.rationale[:80],
                    )
                    stats = agent_pipeline.model_stats()
                    total_calls = sum(v["calls"] for v in stats.values())
                    if total_calls % 10 == 0:
                        logger.info("Model pool stats: %s", stats)
                # Heartbeat ping (inside try block)
                if bot_state is not None:
                    _shb = bot_state.task_heartbeats.setdefault("sentiment_loop", {"cycles": 0, "errors": 0})
                    _shb["last_beat"] = time.time()
                    _shb["cycles"]   += 1
            except asyncio.CancelledError: raise
            except Exception as exc:
                logger.error("Sentiment loop error: %s", exc, exc_info=True)
                interval = 300
                if bot_state is not None:
                    _shb = bot_state.task_heartbeats.setdefault("sentiment_loop", {"cycles": 0, "errors": 0, "last_beat": time.time()})
                    _shb["errors"] += 1
            await asyncio.sleep(interval)

    # ── Token watch loop ────────────────────────────────────────────────────
    # Handles the case where the WebSocket 403 fires (via _on_ticker_token_error)
    # but the REST API still works, so the strategy_loop exception handler never
    # sees a 403.  This loop detects the token_valid=False signal, waits for the
    # user to send /token, then reconnects the ticker and resumes the bot.
    async def _token_watch_loop() -> None:
        while True:
            await asyncio.sleep(5)
            if bot_state.token_valid:
                continue
            # Token became invalid — wait for refresh
            logger.info("TokenWatcher: token invalid, waiting for /token via Telegram…")
            while not bot_state.token_valid:
                await asyncio.sleep(5)
            # Token is fresh — reconnect WebSocket
            logger.info("TokenWatcher: token refreshed — reconnecting WebSocket ticker.")
            try:
                await ticker_wrapper.reconnect_with_token(kite.access_token)
            except Exception as _tw_exc:
                logger.error("TokenWatcher: ticker reconnect failed: %s", _tw_exc)
            bot_state.paused = False
            logger.info("TokenWatcher: bot resumed after token refresh.")

    candle_task    = asyncio.create_task(candle_agg.run(), name="candle_agg")
    alt_tasks      = await alt_data.start_background_tasks()
    geo_tasks      = await geo_monitor.start_background_tasks()
    universe_tasks = await universe_engine.start_background_tasks()

    strat_task = asyncio.create_task(
        strategy_loop(
            kite=kite, redis_client=redis_client, alt_data=alt_data,
            geo_monitor=geo_monitor, universe_engine=universe_engine,
            strategy_engine=strategy_engine, executor=executor,
            rate_limiter=rate_limiter, freq_optimiser=freq_optimiser,
            ws_manager=ws_manager,
            telegram=telegram,  # B-01 FIX: pass telegram notifier
            pos_manager=pos_manager, # R-12: Risk management
            logbook=logbook, bot_state=bot_state,
            portfolio_risk=portfolio_risk_monitor,  # R-13
        ), name="strategy_loop",
    )
    sentiment_task    = asyncio.create_task(sentiment_loop(), name="sentiment_loop")
    token_watch_task  = asyncio.create_task(_token_watch_loop(), name="token_watcher")
    ml_retrain_task   = asyncio.create_task(
        ml_engine.run_retrain_loop(active_symbols), name="ml_retrain"
    )
    logbook_task   = asyncio.create_task(logbook.run_summary_loop(), name="logbook")
    tg_status_task = asyncio.create_task(tg_controller.status_broadcast_loop(), name="tg_status")

    all_tasks = [candle_task, strat_task, sentiment_task, token_watch_task,
                 ml_retrain_task, logbook_task, tg_status_task] \
                + alt_tasks + geo_tasks + universe_tasks

    for t in all_tasks:
        shutdown.register(t)

    ticker_task = asyncio.create_task(ticker_wrapper.start(), name="ticker")
    shutdown.register(ticker_task)

    await telegram.send(
        "SENTISTACK V2 LIVE\n"
        + datetime.now(timezone(timedelta(hours=5,minutes=30))).strftime("%d %b %Y  %H:%M:%S IST") + "\n\n"
        + "Universe: " + str(len(active_symbols)) + " symbols\n"
        + "Mode:     " + bot_state.mode.value + "\n"
        + "GRI:      " + gri_init.level + " (" + f"{gri_init.composite:.3f})\n"
        + "VIX:      " + f"{gri_init.india_vix:.1f}\n"
        + "USD/INR:  " + f"{gri_init.usdinr:.2f}\n"
        + "Capital:  Rs" + f"{settings.strategy.TOTAL_CAPITAL:,.0f}"
    )

    print_banner(f"Live — {len(active_symbols)} symbols | Ctrl+C to stop")
    # B-18 FIX: also shut down on Telegram /stop (was SystemExit which bypassed cleanup)
    stop_tasks = [
        asyncio.create_task(shutdown.wait(), name="sigterm_wait"),
        asyncio.create_task(tg_controller.wait_for_stop(), name="tg_stop_wait"),
    ]
    done, pending = await asyncio.wait(stop_tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    shutdown._trigger()   # ensure all registered tasks are cancelled regardless of source

    logger.info("Shutting down gracefully...")
    await ticker_wrapper.stop()
    await asyncio.gather(*all_tasks, return_exceptions=True)

    stats = rate_limiter.stats
    logger.info(
        "Stats: req=%d ord=%d req_throttles=%d ord_throttles=%d",
        stats["total_requests"], stats["total_orders"],
        stats["request_throttle_events"], stats["order_throttle_events"],
    )
    await telegram.send(
        "SENTISTACK V2 STOPPED\n"
        + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") + "\n"
        + "Requests: " + str(stats["total_requests"])
        + " | Orders: " + str(stats["total_orders"])
    )
    await redis_client.aclose()
    logger.info("Shutdown complete.")


# ===========================================================================
# HELPERS
# ===========================================================================

def print_banner(text: str) -> None:
    width = 65
    line  = "=" * width
    pad   = " " * max(0, (width - len(text) - 2) // 2)
    print(f"\n+{line}+")
    print(f"|{pad} {text} {pad}|")
    print(f"+{line}+\n")


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    print_banner("SentiStack NSE Trading Bot  v2.0")
    try:
        mgr             = TokenManager(settings.kite.API_KEY, settings.kite.API_SECRET)
        kite_i, acc_tok = mgr.get_token()
    except RuntimeError as exc:
        print(f"\n  Token error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(0)
    print("  Authentication OK — launching engine...\n")
    try:
        asyncio.run(main(kite_i, acc_tok, tg_offset=mgr.tg_offset))
    except KeyboardInterrupt:
        logger.info("Terminated.")
