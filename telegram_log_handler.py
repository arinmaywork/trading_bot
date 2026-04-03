"""
telegram_log_handler.py
=======================
A logging.Handler that forwards ERROR and CRITICAL log records to the
Telegram bot chat in real-time, so you can see all bot errors without SSH.

Usage (main.py):
    from telegram_log_handler import TelegramLogHandler
    handler = TelegramLogHandler(bot_token, chat_id)
    logging.getLogger().addHandler(handler)

Features:
  • Rate-limited: at most 1 Telegram message per 10 seconds to avoid flooding
  • Deduplication: identical consecutive error messages collapsed into one
  • Works async-safely: uses a background thread to fire-and-forget HTTP POST
  • Attaches component name, log level, and truncated traceback
"""

import logging
import os
import time
import threading
import urllib.request
import urllib.error
import json
from typing import Optional


class TelegramLogHandler(logging.Handler):
    """
    Logging handler that sends ERROR and CRITICAL records to Telegram.

    Parameters
    ----------
    token    : Telegram bot token
    chat_id  : Telegram chat ID (string)
    min_level: Minimum log level to forward (default: logging.ERROR)
    cooldown : Minimum seconds between successive Telegram messages (default: 30)
    """

    LEVEL_EMOJI = {
        logging.ERROR:    "🔴",
        logging.CRITICAL: "💀",
        logging.WARNING:  "⚠️",
    }

    def __init__(
        self,
        token:    str,
        chat_id:  str,
        min_level: int = logging.ERROR,
        cooldown:  float = 30.0,
    ) -> None:
        super().__init__(min_level)
        self._token    = token
        self._chat_id  = str(chat_id)
        self._cooldown = cooldown
        self._last_sent: float = 0.0
        self._last_msg:  str   = ""
        self._lock = threading.Lock()

    def _send(self, text: str) -> None:
        """Fire-and-forget HTTP POST (runs in a daemon thread)."""
        url  = f"https://api.telegram.org/bot{self._token}/sendMessage"
        body = json.dumps({
            "chat_id":    self._chat_id,
            "text":       text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8):
                pass
        except Exception:
            pass  # Never raise inside a logging handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            now = time.monotonic()
            emoji = self.LEVEL_EMOJI.get(record.levelno, "ℹ️")
            level = record.levelname
            name  = record.name
            msg   = record.getMessage()

            # Truncate very long messages
            if len(msg) > 600:
                msg = msg[:597] + "..."

            # Include short traceback if available
            tb_text = ""
            if record.exc_info:
                import traceback
                tb_lines = traceback.format_exception(*record.exc_info)
                tb_raw   = "".join(tb_lines)[-400:]
                tb_text  = f"\n<pre>{tb_raw}</pre>"

            full = (
                f"{emoji} <b>[{level}] {name}</b>\n"
                f"<code>{msg}</code>"
                f"{tb_text}"
            )

            with self._lock:
                # Rate-limit: don't send faster than self._cooldown
                if (now - self._last_sent) < self._cooldown:
                    return
                # Dedup: skip identical consecutive message
                if full == self._last_msg:
                    return
                self._last_sent = now
                self._last_msg  = full

            t = threading.Thread(target=self._send, args=(full,), daemon=True)
            t.start()

        except Exception:
            self.handleError(record)
