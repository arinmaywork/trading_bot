"""
backtest/data_loader.py
=======================

Loads historical minute-bar OHLCV data from Zerodha Kite Connect and
caches it on disk so repeated backtest runs are fast and cheap (Kite
historical API is rate-limited to 3 req/sec and you burn quota on
every call).

Design notes
------------
- Cache format: pickle (.pkl.gz). Parquet would be nicer but adds a
  pyarrow dependency. Pickle is zero-dep and a compressed DataFrame
  of 6 months of minute bars is ~2 MB, which is fine.
- Cache key:    {symbol}_{interval}_{start}_{end}.pkl.gz
- Cache dir:    backtest/cache/  (git-ignored)
- Zerodha limits a single historical call to 60 days for minute data,
  so we chunk the date range and glue the chunks together.
- Offline mode: if KITE_API_KEY is not set, load_bars() still works on
  already-cached symbols — lets you run the scaffold end-to-end in CI
  or on a dev box without broker credentials.

Public API
----------
    load_bars(symbol, start, end, interval="minute", use_cache=True)
        → pandas.DataFrame with columns [date, open, high, low, close, volume]
          indexed by UTC-aware DatetimeIndex (IST → UTC conversion done here)

    generate_synthetic_bars(symbol, start, end, interval="minute")
        → deterministic synthetic bars for tests (random-walk with seed=hash(symbol))
"""

from __future__ import annotations

import gzip
import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Kite's hard limit on one historical call for minute-interval data
_KITE_MINUTE_CHUNK_DAYS = 60

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(symbol: str, interval: str, start: datetime, end: datetime) -> Path:
    key = f"{symbol}_{interval}_{start.date()}_{end.date()}.pkl.gz"
    return CACHE_DIR / key


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as fh:
            df = pickle.load(fh)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as exc:
        logger.warning("Cache read failed for %s: %s", path.name, exc)
    return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        with gzip.open(path, "wb") as fh:
            pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.warning("Cache write failed for %s: %s", path.name, exc)


def _get_kite_client():
    """
    Late import so the scaffold module loads even on machines without
    kiteconnect installed (e.g. CI). Returns None if creds missing.
    """
    try:
        from kiteconnect import KiteConnect  # type: ignore
    except ImportError:
        logger.warning("kiteconnect not installed — only cached data is available")
        return None

    api_key      = os.getenv("KITE_API_KEY", "")
    access_token = os.getenv("KITE_ACCESS_TOKEN", "")
    if not api_key or not access_token:
        logger.warning("KITE_API_KEY / KITE_ACCESS_TOKEN missing — only cached data is available")
        return None

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _get_instrument_token(kite, symbol: str, exchange: str = "NSE") -> Optional[int]:
    """Look up the numeric instrument token needed by kite.historical_data()."""
    try:
        instruments = kite.instruments(exchange)
    except Exception as exc:
        logger.error("instruments fetch failed: %s", exc)
        return None
    for inst in instruments:
        if inst.get("tradingsymbol") == symbol and inst.get("exchange") == exchange:
            return int(inst["instrument_token"])
    return None


def _fetch_chunk(
    kite,
    instrument_token: int,
    start: datetime,
    end: datetime,
    interval: str,
) -> pd.DataFrame:
    """Single kite.historical_data() call, returned as a DataFrame."""
    raw = kite.historical_data(
        instrument_token=instrument_token,
        from_date=start,
        to_date=end,
        interval=interval,
        continuous=False,
        oi=False,
    )
    if not raw:
        return pd.DataFrame(columns=["date"] + BAR_COLUMNS)
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _fetch_range(
    kite,
    instrument_token: int,
    start: datetime,
    end: datetime,
    interval: str,
) -> pd.DataFrame:
    """
    Chunk a long date range into Kite-sized pieces (60 days for minute data)
    and concatenate the results.
    """
    chunk_days = _KITE_MINUTE_CHUNK_DAYS if "minute" in interval else 365
    all_chunks: list[pd.DataFrame] = []
    cur = start
    while cur < end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        logger.info("Fetching %s %s → %s", interval, cur.date(), chunk_end.date())
        chunk = _fetch_chunk(kite, instrument_token, cur, chunk_end, interval)
        if not chunk.empty:
            all_chunks.append(chunk)
        cur = chunk_end + timedelta(days=1)
    if not all_chunks:
        return pd.DataFrame(columns=["date"] + BAR_COLUMNS)
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def load_bars(
    symbol:    str,
    start:     datetime,
    end:       datetime,
    interval:  str  = "minute",
    use_cache: bool = True,
    exchange:  str  = "NSE",
) -> pd.DataFrame:
    """
    Load historical OHLCV bars for `symbol` between `start` and `end`.

    Returns a DataFrame indexed by UTC-aware timestamp with columns
    `[open, high, low, close, volume]`. Returns an empty DataFrame if
    no data is available (no Kite creds AND no cache hit).
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    cache_file = _cache_path(symbol, interval, start, end)
    if use_cache:
        cached = _read_cache(cache_file)
        if cached is not None:
            logger.info("Cache hit: %s (%d bars)", cache_file.name, len(cached))
            return cached

    kite = _get_kite_client()
    if kite is None:
        logger.warning("No live client and no cache — returning empty frame for %s", symbol)
        return pd.DataFrame(columns=BAR_COLUMNS)

    token = _get_instrument_token(kite, symbol, exchange)
    if token is None:
        logger.error("Instrument token not found for %s on %s", symbol, exchange)
        return pd.DataFrame(columns=BAR_COLUMNS)

    df = _fetch_range(kite, token, start, end, interval)
    if df.empty:
        return df

    df = df.set_index("date")[BAR_COLUMNS]
    _write_cache(cache_file, df)
    return df


# ---------------------------------------------------------------------------
# Synthetic data — lets the scaffold run end-to-end without a broker
# ---------------------------------------------------------------------------
def generate_synthetic_bars(
    symbol:    str,
    start:     datetime,
    end:       datetime,
    interval:  str = "minute",
    base_price: float = 1000.0,
    annual_vol: float = 0.25,
    drift_bps:  float = 0.0,
) -> pd.DataFrame:
    """
    Deterministic random-walk bars for offline testing.

    Seed is a hash of the symbol so two different symbols give different
    price paths but the same symbol always produces the same series
    (reproducible tests).
    """
    import numpy as np

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    # Only generate bars during NSE trading hours: 09:15 IST → 15:30 IST
    # IST = UTC + 5:30, so in UTC that's 03:45 → 10:00 UTC
    freq = "1min" if "minute" in interval else "1D"
    all_ts = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    if "minute" in interval:
        ist = all_ts.tz_convert("Asia/Kolkata")
        mask = (
            (ist.hour > 9) | ((ist.hour == 9) & (ist.minute >= 15))
        ) & (
            (ist.hour < 15) | ((ist.hour == 15) & (ist.minute <= 30))
        )
        mask = mask & (ist.dayofweek < 5)  # weekdays only
        all_ts = all_ts[mask]

    n = len(all_ts)
    if n == 0:
        return pd.DataFrame(columns=BAR_COLUMNS)

    seed = abs(hash(symbol)) % (2 ** 32)
    rng  = np.random.default_rng(seed)

    # Per-bar vol: annual_vol scaled to one-minute bar
    bars_per_year = 252 * 375  # 375 minutes per trading day
    sigma = annual_vol / (bars_per_year ** 0.5)
    mu    = (drift_bps / 1e4) / bars_per_year

    log_returns = rng.normal(loc=mu, scale=sigma, size=n)
    prices = base_price * np.exp(log_returns.cumsum())

    # Build OHLC from close (cheap but deterministic)
    noise = rng.normal(scale=sigma * 0.5, size=n)
    opens  = prices * (1.0 + noise * 0.1)
    highs  = prices * (1.0 + abs(noise))
    lows   = prices * (1.0 - abs(noise))
    volume = rng.integers(1_000, 20_000, size=n)

    df = pd.DataFrame(
        {
            "open":   opens,
            "high":   highs,
            "low":    lows,
            "close":  prices,
            "volume": volume,
        },
        index=all_ts,
    )
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Quick self-test: `python -m backtest.data_loader`
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    start = datetime(2025, 1, 6, tzinfo=timezone.utc)  # Monday
    end   = datetime(2025, 1, 10, tzinfo=timezone.utc)  # Friday
    df = generate_synthetic_bars("RELIANCE", start, end, base_price=2500.0)
    print(f"Synthetic RELIANCE bars: {len(df)} rows")
    print(df.head(3))
    print("...")
    print(df.tail(3))

    # Cache round-trip test
    test_cache = CACHE_DIR / "selftest.pkl.gz"
    _write_cache(test_cache, df)
    round_trip = _read_cache(test_cache)
    assert round_trip is not None and len(round_trip) == len(df)
    test_cache.unlink()
    print("\n[OK] Cache round-trip works.")
