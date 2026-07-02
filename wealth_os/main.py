"""Wealth OS entrypoint: python -m wealth_os.main"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from . import db
from .bot import WealthBot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set (source .env.sh)")
    db.connect().close()  # ensure schema
    asyncio.run(WealthBot(token, int(chat_id)).run())


if __name__ == "__main__":
    main()
