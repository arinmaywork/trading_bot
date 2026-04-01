"""
data_ingestion.py  — V2
=======================
Microstructural Data Ingestion: Multi-Level OFI + Aggressive Flow Tracking.

Enhancement 1 — V2 upgrades:
  • MLOFI: aggregates volume imbalance across all 5 LOB levels with
    exponentially decaying weights [0.40, 0.25, 0.18, 0.10, 0.07].
  • Aggressive Flow Classification: LTP vs best bid/ask classifies each
    trade as aggressive buy ("lifting the offer") or aggressive sell
    ("hitting the bid"), with a rolling 60-second accumulation window.

MLOFI Formula:
    MLOFI = Σ_{i=0}^{4}  w_i × (V_bid_i − V_ask_i) / max(V_bid_i + V_ask_i, 1)

Aggressive Flow:
    LTP >= best_ask  →  aggressive_buy  += LTQ   (buyer initiated)
    LTP <= best_bid  →  aggressive_sell += LTQ   (seller initiated)
    AFD  = aggressive_buy − aggressive_sell   (signed pressure)
    AFR  = AFD / max(total, 1)  ∈ [−1, 1]

Redis keys (V2 additions):
    mlofi:<SYM>  — latest MLOFI float, TTL 60 s
    aflow:<SYM>  — JSON {buy, sell, delta, ratio}, TTL 60 s
    ofi:<SYM>    — V1 level-0 OFI (kept for backward compat)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from kiteconnect import KiteTicker

from config import settings
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

Tick = Dict[str, Any]
LOBLevel = Dict[str, Any]
LOBSnapshot = Dict[str, List[LOBLevel]]

# Exponentially decaying weights for 5 depth levels (sums to 1.00)
MLOFI_WEIGHTS: List[float] = [0.40, 0.25, 0.18, 0.10, 0.07]


# ---------------------------------------------------------------------------
# Aggressive Flow Snapshot
# ---------------------------------------------------------------------------

@dataclass
class AggressiveFlowSnapshot:
    symbol:              str
    aggressive_buy_vol:  float = 0.0
    aggressive_sell_vol: float = 0.0
    delta:               float = 0.0   # buy − sell
    ratio:               float = 0.0   # delta / total ∈ [−1, 1]
    last_ltp:            float = 0.0
    tick_count:          int   = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, float]:
        return {"buy": self.aggressive_buy_vol, "sell": self.aggressive_sell_vol,
                "delta": self.delta, "ratio": self.ratio}


# ---------------------------------------------------------------------------
# V2 Core Calculations
# ---------------------------------------------------------------------------

def calculate_mlofi(tick: Tick) -> float:
    """
    Multi-Level OFI across all 5 depth levels.
    Returns float in [-1.0, 1.0]; 0.0 if depth absent.
    """
    try:
        depth = tick.get("depth")
        if not depth:
            return 0.0
        buy_lvls: List[dict] = depth.get("buy", [])
        sell_lvls: List[dict] = depth.get("sell", [])
        if not buy_lvls or not sell_lvls:
            return 0.0

        mlofi = 0.0
        for i in range(5):
            w     = MLOFI_WEIGHTS[i]
            v_bid = float(buy_lvls[i]["quantity"])  if i < len(buy_lvls)  else 0.0
            v_ask = float(sell_lvls[i]["quantity"]) if i < len(sell_lvls) else 0.0
            total = v_bid + v_ask
            if total > 0:
                mlofi += w * (v_bid - v_ask) / total
        return round(mlofi, 6)
    except Exception as exc:
        logger.debug("MLOFI error: %s", exc)
        return 0.0


def calculate_ofi(tick: Tick) -> float:
    """Level-0 OFI — backward-compatible with V1 strategy."""
    try:
        depth = tick.get("depth")
        if not depth:
            return 0.0
        buy  = depth.get("buy",  [])
        sell = depth.get("sell", [])
        if not buy or not sell:
            return 0.0
        v_bid = float(buy[0].get("quantity", 0))
        v_ask = float(sell[0].get("quantity", 0))
        total = v_bid + v_ask
        return round((v_bid - v_ask) / total, 6) if total > 0 else 0.0
    except Exception:
        return 0.0


def classify_aggressive_flow(tick: Tick, snap: AggressiveFlowSnapshot) -> AggressiveFlowSnapshot:
    """
    Classify each tick's trade as aggressive buy or sell.
    Mutates a *new* snapshot (immutable-ish pattern).
    """
    try:
        depth = tick.get("depth")
        ltp   = float(tick.get("last_price", 0))
        ltq   = float(tick.get("last_traded_quantity", 0))
        if not depth or ltp <= 0 or ltq <= 0:
            return snap

        buy_lvls  = depth.get("buy",  [])
        sell_lvls = depth.get("sell", [])
        if not buy_lvls or not sell_lvls:
            return snap

        best_bid = float(buy_lvls[0].get("price", 0))
        best_ask = float(sell_lvls[0].get("price", float("inf")))

        new_buy  = snap.aggressive_buy_vol
        new_sell = snap.aggressive_sell_vol

        if best_ask > 0 and ltp >= best_ask:
            new_buy  += ltq   # lifting the offer
        elif best_bid > 0 and ltp <= best_bid:
            new_sell += ltq   # hitting the bid

        total   = new_buy + new_sell
        delta   = new_buy - new_sell
        ratio   = delta / max(total, 1.0)

        return AggressiveFlowSnapshot(
            symbol=snap.symbol,
            aggressive_buy_vol=new_buy,
            aggressive_sell_vol=new_sell,
            delta=delta,
            ratio=ratio,
            last_ltp=ltp,
            tick_count=snap.tick_count + 1,
        )
    except Exception as exc:
        logger.debug("Aggressive flow error: %s", exc)
        return snap


def extract_lob_snapshot(tick: Tick) -> LOBSnapshot:
    depth = tick.get("depth", {})
    result: LOBSnapshot = {"buy": [], "sell": []}
    for side in ("buy", "sell"):
        raw: List[dict] = depth.get(side, [])
        out: List[LOBLevel] = []
        for i in range(5):
            if i < len(raw):
                lvl = raw[i]
                out.append({"price": float(lvl.get("price", 0)),
                            "quantity": int(lvl.get("quantity", 0)),
                            "orders":   int(lvl.get("orders", 0))})
            else:
                out.append({"price": 0.0, "quantity": 0, "orders": 0})
        result[side] = out
    return result


# ---------------------------------------------------------------------------
# Redis Stream Writer — V2
# ---------------------------------------------------------------------------

class RedisStreamWriter:
    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis  = redis_client
        self._stream = settings.redis.TICK_STREAM
        self._maxlen = settings.redis.MAX_STREAM_LEN

    async def publish_tick(
        self,
        symbol: str,
        tick: Tick,
        ofi: float,
        mlofi: float,
        aflow: AggressiveFlowSnapshot,
    ) -> None:
        lob = extract_lob_snapshot(tick)
        payload = {
            "symbol":      symbol,
            "ltp":         str(tick.get("last_price", 0)),
            "volume":      str(tick.get("volume_traded", 0)),
            "ltq":         str(tick.get("last_traded_quantity", 0)),
            "ofi":         str(ofi),
            "mlofi":       str(mlofi),
            "aflow_buy":   str(aflow.aggressive_buy_vol),
            "aflow_sell":  str(aflow.aggressive_sell_vol),
            "aflow_delta": str(aflow.delta),
            "aflow_ratio": str(aflow.ratio),
            "lob_buy":     json.dumps(lob["buy"]),
            "lob_sell":    json.dumps(lob["sell"]),
            "ts":          str(tick.get("exchange_timestamp",
                                        datetime.now(timezone.utc).isoformat())),
        }
        await self._redis.xadd(self._stream, payload,
                               maxlen=self._maxlen, approximate=True)


# ---------------------------------------------------------------------------
# Candle Aggregator — V2 (adds mlofi_sum, aflow fields)
# ---------------------------------------------------------------------------

class CandleAggregator:
    def __init__(self, redis_client: aioredis.Redis) -> None:
        self._redis         = redis_client
        self._stream        = settings.redis.TICK_STREAM
        self._last_id       = "0-0"
        self._candle_prefix = settings.redis.CANDLE_HASH
        self._ttl           = 86400   # 24 h

    @staticmethod
    def _bucket_key(symbol: str, ts: datetime) -> str:
        bucket = ts.replace(second=0, microsecond=0)
        return f"candles:1m:{symbol}:{bucket.strftime('%Y%m%d_%H%M')}"

    async def process_batch(self, batch_size: int = 100) -> int:
        # B-15 FIX: block=0 hangs forever on silent TCP drop; use 5000ms timeout instead.
        # If no messages within 5s, returns None — caller's run() loop will retry.
        messages = await self._redis.xread(
            {self._stream: self._last_id}, count=batch_size, block=5000)
        if not messages:
            return 0
        processed = 0
        for _name, entries in messages:
            for msg_id, fields in entries:
                self._last_id = msg_id
                await self._update_candle(fields)
                processed += 1
        return processed

    async def _update_candle(self, fields: Dict[bytes, bytes]) -> None:
        try:
            symbol  = fields[b"symbol"].decode()
            price   = float(fields[b"ltp"])
            volume  = int(fields[b"volume"])
            mlofi   = float(fields.get(b"mlofi", b"0"))
            aflow_d = float(fields.get(b"aflow_delta", b"0"))
            aflow_r = float(fields.get(b"aflow_ratio", b"0"))
            try:
                ts = datetime.fromisoformat(fields[b"ts"].decode())
            except (ValueError, KeyError):
                ts = datetime.now(timezone.utc)

            key  = self._bucket_key(symbol, ts)
            pipe = self._redis.pipeline()
            pipe.hsetnx(key, "open", price)
            pipe.hset(key, "close", price)
            pipe.hincrbyfloat(key, "volume", volume)
            pipe.hincrbyfloat(key, "vwap_sum", price * volume)
            pipe.hincrby(key, "tick_count", 1)
            pipe.hincrbyfloat(key, "mlofi_sum", mlofi)
            pipe.hincrbyfloat(key, "aflow_delta_sum", aflow_d)
            pipe.hset(key, "aflow_ratio", aflow_r)
            pipe.expire(key, self._ttl)
            await pipe.execute()

            await self._redis.eval(
                """
                local h = tonumber(redis.call('HGET', KEYS[1], 'high')) or tonumber(ARGV[1])
                local l = tonumber(redis.call('HGET', KEYS[1], 'low'))  or tonumber(ARGV[1])
                if tonumber(ARGV[1]) > h then redis.call('HSET', KEYS[1], 'high', ARGV[1]) end
                if tonumber(ARGV[1]) < l then redis.call('HSET', KEYS[1], 'low',  ARGV[1]) end
                return 1
                """,
                1, key, str(price)
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Candle update error: %s", exc)

    async def run(self, poll_interval: float = 0.05) -> None:
        logger.info("CandleAggregator V2 started — polling every %.0f ms",
                    poll_interval * 1000)
        while True:
            try:
                count = await self.process_batch()
                if count == 0:
                    await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                logger.info("CandleAggregator shutting down.")
                raise
            except Exception as exc:
                logger.error("CandleAggregator error: %s", exc, exc_info=True)
                await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# Async KiteTicker Wrapper — V2
# ---------------------------------------------------------------------------

class AsyncKiteTickerWrapper:
    """
    V2 WebSocket façade.
    Computes MLOFI + aggressive flow classification on every tick batch.
    Aggressive flow accumulators reset every 60 seconds per symbol (rolling window).
    """

    _AFLOW_WINDOW_S = 60.0

    def __init__(
        self,
        api_key: str,
        access_token: str,
        instruments: List[int],
        symbol_map: Dict[int, str],
        redis_client: aioredis.Redis,
        rate_limiter: RateLimiter,
    ) -> None:
        self._instruments   = instruments
        self._symbol_map    = symbol_map
        self._rate_limiter  = rate_limiter
        self._stream_writer = RedisStreamWriter(redis_client)
        self._redis         = redis_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._aflow:    Dict[str, AggressiveFlowSnapshot] = {}
        self._aflow_ts: Dict[str, float] = {}

        self._ticker = KiteTicker(api_key, access_token)
        self._ticker.on_ticks     = self._on_ticks_sync
        self._ticker.on_connect   = self._on_connect_sync
        self._ticker.on_error     = self._on_error_sync
        self._ticker.on_close     = self._on_close_sync
        self._ticker.on_reconnect = self._on_reconnect_sync

        self._tick_count = 0
        logger.info("AsyncKiteTickerWrapper V2 created (%d instruments).",
                    len(instruments))

    # Sync callbacks --------------------------------------------------------

    def _on_connect_sync(self, ws: Any, _: Any) -> None:
        logger.info("KiteTicker connected. Subscribing %d tokens.", len(self._instruments))
        ws.subscribe(self._instruments)
        ws.set_mode(ws.MODE_FULL, self._instruments)

    def _on_ticks_sync(self, ws: Any, ticks: List[Tick]) -> None:
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._process_ticks_async(ticks), self._loop)

    def _on_error_sync(self, ws: Any, code: int, reason: str) -> None:
        logger.error("KiteTicker error %s: %s", code, reason)

    def _on_close_sync(self, ws: Any, code: int, reason: str) -> None:
        logger.warning("KiteTicker closed %s: %s", code, reason)

    def _on_reconnect_sync(self, ws: Any, attempts: int) -> None:
        logger.info("KiteTicker reconnecting (attempt %d)…", attempts)

    # Async processor -------------------------------------------------------

    async def _process_ticks_async(self, ticks: List[Tick]) -> None:
        now = time.monotonic()
        for tick in ticks:
            try:
                token:  int = tick.get("instrument_token", 0)
                symbol: str = self._symbol_map.get(token, f"TOKEN_{token}")

                ofi   = calculate_ofi(tick)
                mlofi = calculate_mlofi(tick)

                # Rolling aggressive flow window
                if now - self._aflow_ts.get(symbol, 0.0) > self._AFLOW_WINDOW_S:
                    self._aflow[symbol]    = AggressiveFlowSnapshot(symbol=symbol)
                    self._aflow_ts[symbol] = now

                aflow = classify_aggressive_flow(
                    tick,
                    self._aflow.get(symbol, AggressiveFlowSnapshot(symbol=symbol))
                )
                self._aflow[symbol] = aflow

                # Write to Redis atomically
                pipe = self._redis.pipeline()
                pipe.set(f"{settings.redis.OFI_KEY_PREFIX}{symbol}", str(ofi), ex=60)
                pipe.set(f"mlofi:{symbol}", str(mlofi), ex=60)
                pipe.set(f"aflow:{symbol}", json.dumps(aflow.to_dict()), ex=60)
                await pipe.execute()

                await self._stream_writer.publish_tick(symbol, tick, ofi, mlofi, aflow)
                self._tick_count += 1

            except Exception as exc:
                logger.warning("Tick error token=%s: %s",
                               tick.get("instrument_token"), exc)

    # Lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        logger.info("Starting KiteTicker WebSocket feed (V2)…")
        self._ticker.connect(threaded=True)

    async def stop(self) -> None:
        logger.info("Stopping KiteTicker (total ticks: %d).", self._tick_count)
        try:
            self._ticker.close()
        except Exception as exc:
            logger.warning("KiteTicker close error: %s", exc)

    @property
    def tick_count(self) -> int:
        return self._tick_count


# ---------------------------------------------------------------------------
# Public Redis Readers
# ---------------------------------------------------------------------------

async def get_latest_ofi(redis_client: aioredis.Redis, symbol: str) -> float:
    raw = await redis_client.get(f"{settings.redis.OFI_KEY_PREFIX}{symbol}")
    try:
        return float(raw) if raw else 0.0
    except (ValueError, TypeError):
        return 0.0


async def get_latest_mlofi(redis_client: aioredis.Redis, symbol: str) -> float:
    raw = await redis_client.get(f"mlofi:{symbol}")
    try:
        return float(raw) if raw else 0.0
    except (ValueError, TypeError):
        return 0.0


async def get_aggressive_flow(
    redis_client: aioredis.Redis, symbol: str
) -> Dict[str, float]:
    raw = await redis_client.get(f"aflow:{symbol}")
    try:
        return json.loads(raw) if raw else {"buy": 0.0, "sell": 0.0, "delta": 0.0, "ratio": 0.0}
    except Exception:
        return {"buy": 0.0, "sell": 0.0, "delta": 0.0, "ratio": 0.0}


async def get_latest_candle(
    redis_client: aioredis.Redis,
    symbol: str,
    as_of: Optional[datetime] = None,
) -> Optional[Dict[str, float]]:
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    key = CandleAggregator._bucket_key(symbol, as_of)
    raw = await redis_client.hgetall(key)
    if not raw:
        return None
    try:
        volume    = float(raw.get(b"volume", 0))
        vwap_sum  = float(raw.get(b"vwap_sum", 0))
        tick_cnt  = max(int(raw.get(b"tick_count", 1)), 1)
        mlofi_sum = float(raw.get(b"mlofi_sum", 0))
        aflow_d   = float(raw.get(b"aflow_delta_sum", 0))
        vwap      = vwap_sum / volume if volume > 0 else float(raw.get(b"close", 0))
        return {
            "open":        float(raw.get(b"open",  0)),
            "high":        float(raw.get(b"high",  0)),
            "low":         float(raw.get(b"low",   0)),
            "close":       float(raw.get(b"close", 0)),
            "volume":      volume,
            "vwap":        vwap,
            "mlofi_avg":   mlofi_sum / tick_cnt,
            "aflow_delta": aflow_d,
        }
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Candle parse error %s: %s", symbol, exc)
        return None
