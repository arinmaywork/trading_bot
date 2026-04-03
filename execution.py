"""
execution.py
============
Module 4 — Order Execution & Telemetry

Responsibilities:
  1. Async order execution wrapper around kite.place_order.
  2. Automatic order slicing: if quantity > FREEZE_LIMIT, splits into child orders.
  3. Robust error handling with exponential backoff:
       - kiteconnect.exceptions.TokenException   → session expired, must re-auth
       - kiteconnect.exceptions.NetworkException → transient, retry with backoff
       - kiteconnect.exceptions.OrderException   → bad order params, do not retry
  4. Telegram Bot API integration: formatted non-blocking telemetry pushes
     after each order fill (or attempt) including slippage analysis.

All order API calls pass through the centralised RateLimiter (order_slot).
All non-order API calls (positions, margins) pass through request_slot.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from kiteconnect import KiteConnect
from kiteconnect import exceptions as KiteExceptions

from config import settings
from logbook import Logbook
from rate_limiter import RateLimiter
from strategy import SignalState, TradeDirection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data structures for order lifecycle
# ---------------------------------------------------------------------------

@dataclass
class OrderSlice:
    """Represents a single child order in a sliced execution."""
    parent_symbol: str
    quantity: int
    order_id: Optional[str] = None
    status: str = "PENDING"           # PENDING | PLACED | FILLED | REJECTED | ERROR
    fill_price: Optional[float] = None
    placed_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ExecutionReport:
    """Aggregated execution outcome for a complete (possibly sliced) order."""
    symbol: str
    direction: str
    total_quantity: int
    slices: List[OrderSlice] = field(default_factory=list)
    avg_fill_price: float = 0.0
    expected_price: float = 0.0        # Price at signal generation time
    slippage_bps: float = 0.0          # (avg_fill - expected) / expected * 10000
    alpha_score: float = 0.0
    alpha_raw: float = 0.0           # Before geo dampening
    sentiment_classification: str = ""
    sentiment_rationale: str = ""
    geo_risk: float = 0.0
    geo_level: str = "LOW"
    geo_alpha_multiplier: float = 1.0
    geo_kelly_multiplier: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = False
    error: Optional[str] = None

    def compute_slippage(self) -> None:
        """Calculate average fill price and slippage after all slices complete."""
        filled = [s for s in self.slices if s.fill_price and s.fill_price > 0]
        if not filled:
            return
        total_qty = sum(s.quantity for s in filled)
        total_value = sum(s.fill_price * s.quantity for s in filled)
        self.avg_fill_price = total_value / total_qty if total_qty else 0.0
        if self.expected_price > 0:
            self.slippage_bps = (
                (self.avg_fill_price - self.expected_price)
                / self.expected_price * 10_000
            )


# ---------------------------------------------------------------------------
# 2. Telegram Notifier
# ---------------------------------------------------------------------------

class TelegramNotifier:
    """
    Non-blocking Telegram Bot API integration.
    Uses aiohttp for async HTTP; failures are logged but never raise.

    Rate-limit handling:
      • Telegram allows ~30 messages/minute per chat.
      • On a 429 response, backs off for the server-specified retry_after seconds
        (capped at 60 s) before the next send is allowed.
      • A per-minute message counter drops any send that would exceed the cap,
        preventing the 421-second ban cascade caused by flooding the API.
    """

    _RATE_CAP_PER_MINUTE = 20   # stay well under Telegram's 30/min limit

    def __init__(self) -> None:
        cfg = settings.telegram
        self._url = f"{cfg.API_BASE}{cfg.BOT_TOKEN}/sendMessage"
        self._chat_id = cfg.CHAT_ID
        self._timeout = aiohttp.ClientTimeout(total=cfg.SEND_TIMEOUT_SECONDS)
        # Rate-limit state
        self._blocked_until: float = 0.0   # monotonic ts; send() waits if now < this
        self._minute_window: float = 0.0   # start of current 60-s window
        self._minute_count:  int   = 0     # messages sent in current window

    async def send(self, text: str) -> None:
        """Fire-and-forget Telegram message. HTML parse mode is used."""
        now = time.monotonic()

        # Enforce per-minute cap — drop silently if over limit
        if now - self._minute_window >= 60.0:
            self._minute_window = now
            self._minute_count  = 0
        if self._minute_count >= self._RATE_CAP_PER_MINUTE:
            logger.debug("Telegram rate cap reached (%d/min) — message dropped.",
                         self._RATE_CAP_PER_MINUTE)
            return
        self._minute_count += 1

        # Wait out any active backoff from a previous 429
        wait = self._blocked_until - time.monotonic()
        if wait > 0:
            logger.debug("Telegram backoff active — waiting %.1f s", wait)
            await asyncio.sleep(wait)

        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.post(self._url, json=payload) as resp:
                    if resp.status == 429:
                        try:
                            body = await resp.json()
                            retry_after = int(
                                body.get("parameters", {}).get("retry_after", 10)
                            )
                        except Exception:
                            retry_after = 10
                        # Cap at 60 s — longer bans mean we just skip messages
                        backoff = min(retry_after, 60)
                        self._blocked_until = time.monotonic() + backoff
                        logger.warning(
                            "Telegram 429 — backing off %d s (server asked %d s)",
                            backoff, retry_after,
                        )
                    elif resp.status != 200:
                        body = await resp.text()
                        logger.warning("Telegram non-200 response %d: %s",
                                       resp.status, body[:200])
        except Exception as exc:
            # Telemetry failure must NEVER disrupt trading
            logger.warning("Telegram send failed (non-critical): %s", exc)


    @staticmethod
    def _ist_now() -> str:
        from datetime import timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        return datetime.now(ist).strftime("%d %b %Y  %H:%M:%S IST")

    async def notify_startup(self, mode: str, symbols: int) -> None:
        msg = ("SENTISTACK V2 — STARTED\n"
               + self._ist_now() + "\n\n"
               + "Mode:    " + mode + "\n"
               + "Symbols: " + str(symbols) + " in universe\n\n"
               + "Commands: /status /mode /pause /resume /stop")
        await self.send(msg)

    async def notify_mode_set(self, mode: str) -> None:
        msg = ("MODE SET: " + mode + "\n"
               + self._ist_now() + "\n"
               + "Bot is now running in this mode.\n"
               + "Send /status for a live update.")
        await self.send(msg)

    async def notify_signal_fired(self, symbol: str, direction: str,
                                   alpha: float, qty: int,
                                   sentiment_class: str,
                                   gri_level: str, gri: float) -> None:
        msg = ("SIGNAL: " + direction + " " + symbol + "\n"
               + self._ist_now() + "\n\n"
               + "Qty:       " + str(qty) + " shares\n"
               + "Alpha:     " + "{:+.5f}".format(alpha) + "\n"
               + "Sentiment: " + sentiment_class + "\n"
               + "GRI:       " + gri_level + " ({:.3f})".format(gri))
        await self.send(msg)

    async def notify_quota_exhausted(self, mode: str) -> None:
        msg = ("WARNING: GEMINI QUOTA EXHAUSTED\n"
               + self._ist_now() + "\n\n"
               + "All Gemini models hit daily limit.\n"
               + "Current mode: " + mode + "\n\n"
               + "Send /mode to switch to GRI-Only\n"
               + "or enable billing to restore sentiment.")
        await self.send(msg)

    async def notify_gri_spike(self, prev: float, composite: float,
                                level: str, keywords: list) -> None:
        kw = ", ".join(keywords[:5]) if keywords else "none"
        msg = ("ALERT: GRI SPIKE\n"
               + self._ist_now() + "\n\n"
               + "GRI: {:.3f} -> {:.3f} ({})\n".format(prev, composite, level)
               + "Keywords: " + kw + "\n"
               + "Alpha/Kelly multipliers adjusted automatically.")
        await self.send(msg)

    async def notify_heartbeat(self, gri: float, gri_level: str,
                                sentiment_score: float, sentiment_class: str,
                                trades_today: int, signals_today: int,
                                top_symbols: list, mode: str,
                                paused: bool) -> None:
        top = ", ".join(top_symbols[:3]) if top_symbols else "loading"
        state = "PAUSED" if paused else "RUNNING"
        msg = ("HEARTBEAT — SentiStack V2\n"
               + self._ist_now() + "\n\n"
               + "Status:   " + state + "\n"
               + "Mode:     " + mode + "\n"
               + "GRI:      " + gri_level + " ({:.3f})\n".format(gri)
               + "Sentiment:" + sentiment_class
               + " ({:+.3f})\n\n".format(sentiment_score)
               + "Top 3:    " + top + "\n"
               + "Signals:  " + str(signals_today) + " today\n"
               + "Trades:   " + str(trades_today) + " today")
        await self.send(msg)

    async def notify_execution(self, report: ExecutionReport) -> None:
        """Format and push an execution report to the Telegram channel."""
        direction_emoji = "🟢 BUY" if report.direction == "BUY" else "🔴 SELL"
        status_emoji = "✅" if report.success else "❌"
        slip_sign = "+" if report.slippage_bps >= 0 else ""

        geo_bar   = "🟢" if report.geo_risk < 0.30 else ("🟡" if report.geo_risk < 0.55 else "🔴")
        alpha_damp = f"{report.geo_alpha_multiplier:.2f}×" if report.geo_alpha_multiplier < 1.0 else "none"
        text = (
            f"{status_emoji} <b>TRADE EXECUTION</b>\n"
            f"{'─' * 30}\n"
            f"🏷️ <b>Symbol:</b>       {report.symbol}\n"
            f"📊 <b>Action:</b>       {direction_emoji}\n"
            f"🔢 <b>Quantity:</b>     {report.total_quantity:,} shares\n"
            f"💰 <b>Avg Fill:</b>     ₹{report.avg_fill_price:,.2f}\n"
            f"🎯 <b>Alpha:</b>        raw={report.alpha_raw:+.5f} adj={report.alpha_score:+.5f}\n"
            f"📉 <b>Slippage:</b>     {slip_sign}{report.slippage_bps:.1f} bps\n"
            f"{'─' * 30}\n"
            f"🌐 <b>Geo Risk:</b>     {geo_bar} {report.geo_level} ({report.geo_risk:.3f})\n"
            f"⚡ <b>α dampening:</b>  {alpha_damp} | K mult: {report.geo_kelly_multiplier:.2f}×\n"
            f"{'─' * 30}\n"
            f"🧠 <b>Sentiment:</b>    {report.sentiment_classification}\n"
            f"📝 <b>Rationale:</b>    <i>{report.sentiment_rationale[:180]}</i>\n"
            f"{'─' * 30}\n"
            f"🕐 <b>Time:</b>         {report.timestamp.strftime('%H:%M:%S UTC')}\n"
            f"📦 <b>Slices:</b>       {len(report.slices)}\n"
        )

        if not report.success and report.error:
            text += f"\n⚠️ <b>Error:</b> <code>{report.error[:150]}</code>\n"

        await self.send(text)

    async def notify_risk_decay(self, symbol: str, reason: str) -> None:
        """Alert when a signal is suppressed by the risk decay logic."""
        text = (
            f"⚠️ <b>RISK DECAY — SIGNAL SUPPRESSED</b>\n"
            f"🏷️ <b>Symbol:</b> {symbol}\n"
            f"📋 <b>Reason:</b> {reason}\n"
            f"🕐 <b>Time:</b> {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )
        await self.send(text)


# ---------------------------------------------------------------------------
# 3. OrderSlicer — splits large orders into freeze-compliant child orders
# ---------------------------------------------------------------------------

class OrderSlicer:
    """
    Splits a total quantity into child orders, each ≤ FREEZE_LIMIT.

    Example:
        total_qty = 12_000, freeze_limit = 5_000
        → slices = [5_000, 5_000, 2_000]
    """

    def __init__(self, freeze_limit: int = settings.kite.ORDER_FREEZE_LIMIT) -> None:
        self._freeze_limit = freeze_limit

    def slice(self, symbol: str, total_qty: int) -> List[OrderSlice]:
        """Return a list of OrderSlice objects summing to total_qty."""
        if total_qty <= 0:
            return []

        slices: List[OrderSlice] = []
        remaining = total_qty

        while remaining > 0:
            qty = min(remaining, self._freeze_limit)
            slices.append(OrderSlice(parent_symbol=symbol, quantity=qty))
            remaining -= qty

        if len(slices) > 1:
            logger.info(
                "OrderSlicer: %s qty=%d → %d slices (%s)",
                symbol, total_qty, len(slices),
                ", ".join(str(s.quantity) for s in slices)
            )

        return slices


# ---------------------------------------------------------------------------
# 4. OrderExecutor — the main execution class
# ---------------------------------------------------------------------------

class OrderExecutor:
    """
    Async execution engine for NSE orders via KiteConnect.

    Key behaviours:
      • All order API calls are gated through rate_limiter.order_slot().
      • TokenException halts the bot and sends a critical Telegram alert.
      • NetworkException triggers exponential backoff (up to MAX_RETRIES).
      • OrderException (bad params) logs the error without retry.
      • After execution, sends a formatted Telegram notification.
    """

    MAX_RETRIES = 5
    BASE_BACKOFF = 1.0   # seconds; actual delay = BASE_BACKOFF ** attempt

    def __init__(
        self,
        kite: KiteConnect,
        rate_limiter: RateLimiter,
        telegram: TelegramNotifier,
    ) -> None:
        self._kite = kite
        self._limiter = rate_limiter
        self._telegram = telegram
        self._slicer = OrderSlicer()
        # B-08 FIX: removed self._loop = asyncio.get_event_loop()
        # get_event_loop() is deprecated in Python 3.10+ and raises RuntimeError in 3.12.
        # Use asyncio.get_running_loop() inside each async method instead.
        # R-08 FIX: track PermissionException state so we send ONE alert and
        # skip all subsequent order attempts without hitting Zerodha or Telegram.
        self._permission_denied: bool = False

    async def _place_single_order(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: int,
        tag: str = "",
        limit_price: float = 0.0,
        product_type: str = "",
    ) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """
        Place one order slice via kite.place_order.

        Returns (order_id, fill_price, None) on success,
                (None, None, error_str) on failure so the caller can surface
                the real exception in Telegram instead of a generic message.
        kite.place_order is synchronous — wrapped in run_in_executor.
        """
        kite_transaction = (
            self._kite.TRANSACTION_TYPE_BUY
            if direction == TradeDirection.BUY
            else self._kite.TRANSACTION_TYPE_SELL
        )
        cfg = settings.kite

        def _sync_place() -> str:
            """Synchronous wrapper called inside executor."""
            # Aggressive LIMIT price: bid slightly above signal price for BUY
            # and slightly below for SELL so the order fills immediately while
            # giving us explicit slippage control. NSE tick = ₹0.05; we round
            # to 1 decimal place (always a valid multiple of 0.05).
            buf = cfg.LIMIT_PRICE_BUFFER_PCT
            if limit_price > 0:
                if direction == TradeDirection.BUY:
                    order_price = round(limit_price * (1.0 + buf), 1)
                else:
                    order_price = round(limit_price * (1.0 - buf), 1)
            else:
                # limit_price=0 means LTP was unavailable; a LIMIT order without a
                # price will always be rejected. Raise immediately so the error
                # message reaches Telegram rather than Zerodha returning a cryptic 400.
                raise ValueError(
                    f"limit_price is 0 for {symbol} — LTP unavailable. "
                    "Order skipped to avoid invalid LIMIT order."
                )

            # R-11: use signal's product_type ("MIS" or "CNC") if provided;
            # fall back to KiteConfig.PRODUCT (default "MIS").
            _product = product_type if product_type in ("MIS", "CNC") else cfg.PRODUCT
            return self._kite.place_order(
                variety=cfg.ORDER_VARIETY,
                exchange=cfg.EXCHANGE,
                tradingsymbol=symbol,
                transaction_type=kite_transaction,
                quantity=quantity,
                product=_product,
                order_type=cfg.ORDER_TYPE,
                price=order_price,
                tag=tag[:20] if tag else "SENTISTACK",
            )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                async with self._limiter.order_slot():
                    # B-08 FIX: use get_running_loop() instead of stored self._loop
                    order_id = await asyncio.get_running_loop().run_in_executor(None, _sync_place)

                logger.info(
                    "Order placed: %s %s %d shares → order_id=%s",
                    direction.value, symbol, quantity, order_id
                )

                # Fetch fill price — use last traded price as proxy
                # (a real implementation would poll order history)
                fill_price = await self._get_ltp(symbol)
                return order_id, fill_price, None

            except KiteExceptions.TokenException as exc:
                # Session token expired — non-retryable, fatal for the session
                logger.critical("TokenException: session expired. %s", exc)
                await self._telegram.send(
                    f"🚨 <b>CRITICAL: TokenException</b>\n"
                    f"Session token expired for {symbol}. Bot halted. "
                    f"Please regenerate the access token and restart."
                )
                raise   # Propagate to main — this is fatal

            except KiteExceptions.NetworkException as exc:
                # Transient network error — use exponential backoff
                backoff = self.BASE_BACKOFF ** attempt
                logger.warning(
                    "NetworkException (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt, self.MAX_RETRIES, exc, backoff
                )
                if attempt == self.MAX_RETRIES:
                    err_msg = f"NetworkException (max retries): {exc}"
                    logger.error("NetworkException: max retries exceeded for %s.", symbol)
                    return None, None, err_msg
                await asyncio.sleep(backoff)

            except KiteExceptions.PermissionException as exc:
                # IP not whitelisted or API permissions issue — non-retryable.
                # Retrying will never succeed; abort immediately.
                # R-08 FIX: Only send ONE Telegram alert per session. The first call
                # sets _permission_denied=True and fires the alert; every subsequent
                # call returns immediately without contacting Telegram or Zerodha,
                # preventing the 421-second ban cascade.
                err_msg = f"PermissionException: {exc}"
                logger.error(
                    "PermissionException (non-retryable) for %s: %s\n"
                    "ACTION REQUIRED: Add your public IP to the Kite developer console → "
                    "https://developers.kite.trade (Settings → Allowed IPs).",
                    symbol, exc
                )
                if not self._permission_denied:
                    self._permission_denied = True
                    await self._telegram.send(
                        f"🚫 <b>PermissionException — ALL orders blocked</b>\n"
                        f"Symbol: {symbol}\n"
                        f"Reason: {exc}\n\n"
                        f"Add your server's public IP / IPv6 at:\n"
                        f"https://developers.kite.trade\n"
                        f"(Settings → Allowed IPs), then restart the bot.\n\n"
                        f"<i>Further order attempts suppressed until restart.</i>"
                    )
                return None, None, err_msg

            except KiteExceptions.OrderException as exc:
                # Bad order parameters — log and abort (do not retry)
                err_msg = f"OrderException: {exc}"
                logger.error("OrderException (non-retryable) for %s: %s", symbol, exc)
                return None, None, err_msg

            except KiteExceptions.InputException as exc:
                # Malformed request (missing field, bad value) — non-retryable.
                # Retrying the same parameters will always fail.
                err_msg = f"InputException: {exc}"
                logger.error("InputException (non-retryable) for %s: %s", symbol, exc)
                return None, None, err_msg

            except Exception as exc:
                # Catch-all for unexpected errors
                err_msg = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "Unexpected order error for %s (attempt %d/%d): %s",
                    symbol, attempt, self.MAX_RETRIES, exc, exc_info=True
                )
                if attempt == self.MAX_RETRIES:
                    return None, None, err_msg
                await asyncio.sleep(self.BASE_BACKOFF ** attempt)

        return None, None, "Max retries exhausted"

    async def _get_ltp(self, symbol: str) -> float:
        """
        Fetch Last Traded Price for slippage calculation.
        Returns 0.0 on failure (slippage metric will be meaningless but non-fatal).
        """
        try:
            async with self._limiter.request_slot():
                loop = asyncio.get_running_loop()
                quote = await loop.run_in_executor(
                    None,
                    lambda: self._kite.ltp([f"{settings.kite.EXCHANGE}:{symbol}"])
                )
            key = f"{settings.kite.EXCHANGE}:{symbol}"
            return float(quote[key]["last_price"])
        except Exception as exc:
            logger.debug("LTP fetch failed for %s: %s", symbol, exc)
            return 0.0

    async def execute(self, signal: SignalState) -> ExecutionReport:
        """
        Execute a complete order for the given SignalState.

        Flow:
          1. Validate signal is actionable.
          2. Slice the total quantity if needed.
          3. Place child orders sequentially (to avoid concurrent race on the
             same instrument; a production system might parallelise across
             different symbols).
          4. Aggregate results into an ExecutionReport.
          5. Push Telegram notification.
        """
        report = ExecutionReport(
            symbol=signal.symbol,
            direction=signal.direction.value,
            total_quantity=signal.quantity,
            expected_price=signal.current_price,
            alpha_score=signal.alpha,
            alpha_raw=signal.alpha_raw,
            sentiment_classification=signal.sentiment_class,
            sentiment_rationale=signal.rationale[:250],
            geo_risk=signal.geo_risk,
            geo_level=signal.geo_level,
            geo_alpha_multiplier=signal.geo_alpha_multiplier,
            geo_kelly_multiplier=signal.geo_kelly_multiplier,
        )

        # ---- Guard: PermissionException already seen this session ----
        # R-08 FIX: If the IP is not whitelisted, every order attempt will fail
        # with a PermissionException. Skip slicing / placing entirely to avoid
        # flooding Zerodha and Telegram. The single alert was already sent on
        # the first failure.
        if self._permission_denied:
            report.error = "Orders blocked: IP not whitelisted (PermissionException). See Telegram alert."
            report.success = False
            logger.warning(
                "execute() skipped for %s — PermissionException already raised this session.",
                signal.symbol,
            )
            return report

        # ---- Guard: only execute actionable signals ----
        if not signal.is_actionable:
            report.error = f"Signal not actionable: {signal.rationale}"
            report.success = False

            if signal.is_decayed:
                await self._telegram.notify_risk_decay(signal.symbol, signal.rationale)

            logger.info("Skipping non-actionable signal for %s: %s",
                        signal.symbol, signal.rationale)
            return report

        # ---- Slice order ----
        slices = self._slicer.slice(signal.symbol, signal.quantity)
        if not slices:
            report.error = "OrderSlicer returned empty slice list."
            report.success = False
            await self._telegram.notify_execution(report)
            return report

        # ---- Execute each slice ----
        errors: List[str] = []

        for idx, order_slice in enumerate(slices):
            order_slice.placed_at = datetime.now(timezone.utc)
            tag = f"SS_{signal.symbol[:6]}_{idx}"

            # ---- Paper trade mode: simulate fill without hitting Zerodha ----
            place_err: Optional[str] = None
            if settings.kite.PAPER_TRADE:
                simulated_id   = f"PAPER-{signal.symbol}-{idx}-{int(time.time())}"
                simulated_fill = signal.current_price   # Assume fill at signal price
                order_id, fill_price = simulated_id, simulated_fill
                logger.info(
                    "[PAPER] Simulated order: %s %s %d shares @ ₹%.2f | id=%s",
                    signal.direction.value, signal.symbol,
                    order_slice.quantity, simulated_fill, simulated_id,
                )
            else:
                order_id, fill_price, place_err = await self._place_single_order(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    quantity=order_slice.quantity,
                    tag=tag,
                    limit_price=signal.current_price,
                    product_type=getattr(signal, "product_type", ""),
                )

            if order_id:
                order_slice.order_id = order_id
                order_slice.fill_price = fill_price or signal.current_price
                order_slice.status = "PLACED"
                order_slice.filled_at = datetime.now(timezone.utc)
            else:
                order_slice.status = "ERROR"
                order_slice.error_message = place_err or "Placement failed."
                # Include the real exception in the error list so Telegram shows it
                errors.append(
                    f"Slice {idx} (qty {order_slice.quantity}): "
                    f"{order_slice.error_message}"
                )

            report.slices.append(order_slice)

            # Inter-slice delay to respect rate limits and reduce market impact
            if idx < len(slices) - 1:
                await asyncio.sleep(0.15)

        # ---- Compute aggregate metrics ----
        report.compute_slippage()
        successful_slices = [s for s in report.slices if s.status == "PLACED"]
        report.success = len(successful_slices) > 0

        if errors:
            report.error = "; ".join(errors)

        if report.success:
            logger.info(
                "Execution complete: %s %s %d shares @ avg ₹%.2f | slippage=%.1f bps",
                signal.direction.value, signal.symbol,
                sum(s.quantity for s in successful_slices),
                report.avg_fill_price, report.slippage_bps
            )
        else:
            logger.error("Execution FAILED for %s: %s", signal.symbol, report.error)

        # ---- Telegram notification (fire-and-forget) ----
        asyncio.create_task(self._telegram.notify_execution(report))

        return report

    async def get_positions(self) -> Dict[str, Any]:
        """
        Fetch current open positions from Kite (day + net).
        Rate-limited via request_slot.
        """
        try:
            async with self._limiter.request_slot():
                loop = asyncio.get_running_loop()
                positions = await loop.run_in_executor(
                    None, self._kite.positions
                )
            return positions
        except Exception as exc:
            logger.error("Failed to fetch positions: %s", exc)
            return {}

    async def get_margins(self) -> Dict[str, Any]:
        """
        Fetch account margins (available cash / collateral).
        Rate-limited via request_slot.
        """
        try:
            async with self._limiter.request_slot():
                loop = asyncio.get_running_loop()
                margins = await loop.run_in_executor(
                    None, self._kite.margins
                )
            return margins
        except Exception as exc:
            logger.error("Failed to fetch margins: %s", exc)
            return {}
