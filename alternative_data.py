"""
alternative_data.py  — V2
=========================
Alternative Data Fetching: Enriched Weather + Caldara-Iacoviello GPR Index.

Enhancement 2 — V2 upgrades:
  • Weather: fetches temperature, humidity, wind_speed, and precipitation
    anomalies — all three impact agricultural, power, and consumer stocks.
  • GPR Index: integrates the Caldara & Iacoviello Geopolitical Risk (GPR)
    Index with its GPRT (Threats) and GPRA (Acts) sub-components, fetched
    from the St. Louis FRED API and cached in Redis (15-min TTL).
    - GPRT (Threats): forward-looking fear; impacts option vol + defensives
    - GPRA (Acts): realized conflict; impacts current sector rotation

Enhancement 3 is in agent_pipeline.py — LangGraph multi-agent LLM.
This module still provides SentimentResult for backward compatibility,
but the production path is through AgentPipeline.run_analysis_cycle().
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import yfinance as yf

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WeatherSignal:
    """
    V2 enriched weather signal — temperature, humidity, wind, precipitation.
    All anomalies = measured - seasonal_baseline.
    """
    city:                    str
    temperature_celsius:     float
    feels_like_celsius:      float
    humidity_pct:            float   # 0-100
    wind_speed_ms:           float   # metres per second
    precipitation_mm:        float   # last hour (0 if none)
    description:             str
    anomaly_celsius:         float   # temp - seasonal_baseline
    anomaly_humidity:        float   # humidity - seasonal_baseline_humidity
    anomaly_wind:            float   # wind - seasonal_baseline_wind
    anomaly_precip:          float   # precip - seasonal_baseline_precip
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Sector impact signals ─────────────────────────────────────────────
    @property
    def is_heat_stress(self) -> bool:
        """High temperature + humidity stress → power demand surge."""
        return self.anomaly_celsius > 3.0 and self.humidity_pct > 70

    @property
    def is_cold_stress(self) -> bool:
        return self.anomaly_celsius < -3.0

    @property
    def is_drought_signal(self) -> bool:
        """Low precip + high wind → agricultural stress."""
        return self.anomaly_precip < -5.0 and self.anomaly_wind > 2.0

    @property
    def is_flood_signal(self) -> bool:
        """Extreme precip → logistics/infrastructure disruption."""
        return self.precipitation_mm > 20.0 or self.anomaly_precip > 15.0

    @property
    def composite_weather_stress(self) -> float:
        """
        Scalar 0-1 weather stress index combining all four dimensions.
        Higher = more unusual / impactful weather.
        """
        t_stress  = min(abs(self.anomaly_celsius)     / 8.0,  1.0) * 0.40
        h_stress  = min(abs(self.anomaly_humidity)    / 30.0, 1.0) * 0.20
        w_stress  = min(abs(self.anomaly_wind)        / 5.0,  1.0) * 0.20
        p_stress  = min(abs(self.anomaly_precip)      / 20.0, 1.0) * 0.20
        return round(t_stress + h_stress + w_stress + p_stress, 4)


@dataclass
class GPRSnapshot:
    """
    Caldara & Iacoviello (2022) Geopolitical Risk Index snapshot.
    Published monthly by the Fed; updated near end of month.
    GPRT = Geopolitical Threats (forward-looking newspaper threats)
    GPRA = Geopolitical Acts    (realized conflict events)
    GPR  = overall index        = GPRT + GPRA combined
    All values normalised: 100 = long-run mean (1900-present).
    """
    gpr_index:       float   # Overall GPR (raw value; 100 = mean)
    gprt_threats:    float   # GPR Threats sub-component
    gpra_acts:       float   # GPR Acts sub-component
    gpr_normalised:  float   # [0,1] normalised (0=low, 1=extreme)
    gprt_normalised: float
    gpra_normalised: float
    period:          str     # e.g. "2026-01"
    source:          str     = "FRED"
    timestamp: datetime      = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def threat_dominance(self) -> float:
        """
        Ratio of threats to acts.
        >1 = market pricing in future conflict risk (buy vol).
        <1 = acts dominating = current crisis (de-risk now).
        """
        return self.gprt_threats / max(self.gpra_acts, 1.0)

    @property
    def sector_rotation_signal(self) -> Dict[str, float]:
        """
        Returns per-sector adjustment scores based on GPRT vs GPRA.
        Positive = tailwind during current GPR regime.
        """
        gpr_n = self.gpr_normalised
        gprt  = self.gprt_normalised
        gpra  = self.gpra_normalised
        return {
            "DEFENCE":    +0.80 * gpr_n  + 0.20 * gpra,
            "PHARMA":     +0.30 * gpr_n,
            "CONSUMER":   +0.20 * gpr_n,
            "UTILITIES":  +0.10 * gpr_n,
            "FINANCIALS": -0.80 * gpr_n  - 0.10 * gpra,
            "ENERGY":     -0.50 * gprt   + 0.30 * gpra,  # oil shocks ambiguous
            "REALTY":     -0.60 * gpr_n,
            "IT":         -0.40 * gprt,
            "AUTO":       -0.30 * gpr_n,
        }

    @classmethod
    def neutral(cls) -> "GPRSnapshot":
        return cls(
            gpr_index=100.0, gprt_threats=100.0, gpra_acts=100.0,
            gpr_normalised=0.10, gprt_normalised=0.10, gpra_normalised=0.10,
            period="unknown",
        )


@dataclass
class SentimentResult:
    """V2 sentiment — produced by AgentPipeline; backward compatible with V1."""
    sentiment_classification: str      # "Fear"|"Excitement"|"Neutral"|"Disbelief"
    sentiment_score:          float    # [-1.0, 1.0]
    rationale:                str
    key_entities:             List[str]
    source_articles:          List[str]
    model_latency_ms:         float
    # V2 additions
    risk_context:             str  = ""  # Risk Manager Agent output
    volatility_context:       str  = ""  # vol regime label
    gpr_context:              str  = ""  # GPR sub-component interpretation
    confidence:               float = 0.0 # LLM conviction score [0.0, 1.0]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def neutral(cls) -> "SentimentResult":
        return cls(
            sentiment_classification="Neutral",
            sentiment_score=0.0,
            rationale="No data — defaulting to neutral.",
            key_entities=[], source_articles=[],
            model_latency_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Seasonal baselines (expanded for V2 — temperature, humidity, wind, precip)
# ---------------------------------------------------------------------------

_BASELINE: Dict[str, Dict[str, float]] = {
    #                temp  humidity  wind_ms  precip_mm
    "Mumbai":     {"t": 28.0, "h": 75.0, "w": 3.5, "p": 2.0},
    "Delhi":      {"t": 25.0, "h": 55.0, "w": 2.5, "p": 0.8},
    "Chennai":    {"t": 30.0, "h": 78.0, "w": 4.0, "p": 1.5},
    "Hyderabad":  {"t": 27.0, "h": 60.0, "w": 3.0, "p": 1.0},
    "Ahmedabad":  {"t": 27.0, "h": 50.0, "w": 3.0, "p": 0.5},
    "Pune":       {"t": 26.0, "h": 60.0, "w": 2.8, "p": 1.0},
    "Bengaluru":  {"t": 24.0, "h": 65.0, "w": 2.5, "p": 1.5},
    "Kolkata":    {"t": 28.0, "h": 75.0, "w": 3.0, "p": 1.8},
    "Jaipur":     {"t": 27.0, "h": 45.0, "w": 3.5, "p": 0.4},
    "Surat":      {"t": 28.0, "h": 68.0, "w": 4.0, "p": 1.2},
}

# FRED series IDs for Caldara-Iacoviello GPR index
_GPR_FRED_SERIES  = "GPRC"    # Overall GPR (monthly)
_GPRT_FRED_SERIES = "GPRCT"   # GPR Threats
_GPRA_FRED_SERIES = "GPRCA"   # GPR Acts
_FRED_BASE_URL    = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Historical GPR normalisation bounds (long-run observation)
_GPR_MIN, _GPR_MAX   = 30.0, 600.0
_GPRT_MIN, _GPRT_MAX = 20.0, 500.0
_GPRA_MIN, _GPRA_MAX = 10.0, 400.0


# ---------------------------------------------------------------------------
# Weather Fetcher — V2 (multi-dimensional)
# ---------------------------------------------------------------------------

class WeatherDataFetcher:
    """
    Fetches current weather for all configured cities.
    Computes anomalies for temperature, humidity, wind speed, and precipitation.
    """

    def __init__(self) -> None:
        self._api_key  = settings.weather.API_KEY
        self._base_url = settings.weather.BASE_URL
        self._cities   = settings.weather.CITIES
        self._interval = settings.weather.FETCH_INTERVAL_SECONDS
        self._latest:  Dict[str, WeatherSignal] = {}

    async def _fetch_city(
        self, session: aiohttp.ClientSession, city: str
    ) -> Optional[WeatherSignal]:
        params = {"q": f"{city},IN", "appid": self._api_key, "units": "metric"}
        try:
            async with session.get(
                self._base_url, params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                resp.raise_for_status()
                data: Dict[str, Any] = await resp.json()

            baseline = _BASELINE.get(city, {"t": 27.0, "h": 60.0, "w": 3.0, "p": 1.0})

            temp      = float(data["main"]["temp"])
            humidity  = float(data["main"]["humidity"])
            wind_ms   = float(data.get("wind", {}).get("speed", 0))
            precip_mm = float(data.get("rain", {}).get("1h", 0))

            return WeatherSignal(
                city               = city,
                temperature_celsius= temp,
                feels_like_celsius = float(data["main"]["feels_like"]),
                humidity_pct       = humidity,
                wind_speed_ms      = wind_ms,
                precipitation_mm   = precip_mm,
                description        = data["weather"][0]["description"],
                anomaly_celsius    = round(temp     - baseline["t"], 2),
                anomaly_humidity   = round(humidity - baseline["h"], 2),
                anomaly_wind       = round(wind_ms  - baseline["w"], 2),
                anomaly_precip     = round(precip_mm- baseline["p"], 2),
            )

        except Exception as exc:
            logger.warning("Weather fetch failed %s: %s", city, exc)
            return None

    async def fetch_all(self) -> Dict[str, WeatherSignal]:
        async with aiohttp.ClientSession() as session:
            tasks   = [self._fetch_city(session, c) for c in self._cities]
            # B-10 FIX: return_exceptions=True so one city failure doesn't kill all data
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.warning("Weather city fetch raised exception: %s", r)
                continue
            if r is not None:
                self._latest[r.city] = r
                logger.debug(
                    "Weather %s: %.1f°C (Δtemp=%+.1f Δhum=%+.1f Δwind=%+.1f Δprecip=%+.1f)",
                    r.city, r.temperature_celsius,
                    r.anomaly_celsius, r.anomaly_humidity,
                    r.anomaly_wind, r.anomaly_precip,
                )
        return dict(self._latest)

    async def run(self) -> None:
        logger.info("WeatherDataFetcher V2 started (interval=%ds).", self._interval)
        while True:
            try:
                await self.fetch_all()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("WeatherDataFetcher error: %s", exc, exc_info=True)
            await asyncio.sleep(self._interval)

    def get_aggregate_anomaly(self) -> float:
        """Mean temperature anomaly across cities (V1 compat)."""
        if not self._latest:
            return 0.0
        return sum(s.anomaly_celsius for s in self._latest.values()) / len(self._latest)

    def get_composite_stress(self) -> float:
        """Mean composite weather stress index across all cities [0, 1]."""
        if not self._latest:
            return 0.0
        return sum(s.composite_weather_stress for s in self._latest.values()) / len(self._latest)

    def get_sector_weather_signals(self) -> Dict[str, float]:
        """
        Per-sector weather impact score (averaged across cities).
        Power sector: heat stress dominant.
        Agriculture:  drought/flood signals.
        Consumer:     precipitation disrupts sales.
        """
        if not self._latest:
            return {}
        sigs: Dict[str, float] = {}
        for sig in self._latest.values():
            s = sig.composite_weather_stress
            sigs["UTILITIES"]  = sigs.get("UTILITIES", 0) + (s if sig.is_heat_stress else 0)
            sigs["AGRICULTURE"]= sigs.get("AGRICULTURE", 0) + (s if sig.is_drought_signal else 0)
            sigs["CONSUMER"]   = sigs.get("CONSUMER", 0) - (0.3 * s if sig.is_flood_signal else 0)
        n = max(len(self._latest), 1)
        return {k: round(v / n, 4) for k, v in sigs.items()}

    @property
    def latest(self) -> Dict[str, WeatherSignal]:
        return dict(self._latest)


# ---------------------------------------------------------------------------
# GPR Index Fetcher — Caldara & Iacoviello (2022)
# ---------------------------------------------------------------------------

class GPRIndexFetcher:
    """
    Fetches the Caldara-Iacoviello Geopolitical Risk (GPR) Index and its
    sub-components (Threats/GPRT, Acts/GPRA) from the FRED public data API.

    The index is published monthly.  We cache in Redis (TTL = 6 hours).
    Falls back to a neutral GPR of 100 if the API is unavailable.

    Citation: Caldara & Iacoviello (2022), "Measuring Geopolitical Risk",
    American Economic Review, 112(4), 1194-1225.
    """

    _CACHE_KEY = "gpr:snapshot"
    _CACHE_TTL = 3600 * 6   # 6 hours — monthly data doesn't change intraday

    def __init__(self, redis_client: Any) -> None:
        self._redis   = redis_client
        self._latest: GPRSnapshot = GPRSnapshot.neutral()
        self._fred_key = getattr(settings, "fred", None)
        # FRED API key is optional — CSV endpoint works without key
        self._fred_api_key: Optional[str] = (
            getattr(self._fred_key, "API_KEY", None)
            if self._fred_key else None
        )

    async def _fetch_fred_series(
        self,
        session: aiohttp.ClientSession,
        series_id: str,
    ) -> Optional[float]:
        """
        Fetch the most recent observation from FRED for a given series.
        Uses the CSV endpoint (no API key required for public series).
        """
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                text = await resp.text()
            # CSV format: DATE,VALUE\n2026-01-01,123.4\n...
            lines = [l.strip() for l in text.strip().splitlines()
                     if l.strip() and not l.startswith("DATE")]
            if not lines:
                return None
            last_line = lines[-1]
            parts = last_line.split(",")
            if len(parts) >= 2 and parts[1] not in ("", "."):
                return float(parts[1])
            return None
        except Exception as exc:
            logger.warning("FRED fetch failed for %s: %s", series_id, exc)
            return None

    async def fetch(self) -> GPRSnapshot:
        """Fetch latest GPR, GPRT, GPRA from FRED; cache in Redis."""

        # Check Redis cache first
        try:
            cached = await self._redis.get(self._CACHE_KEY)
            if cached:
                data = json.loads(cached)
                snap = GPRSnapshot(**{k: v for k, v in data.items()
                                     if k != "timestamp"})
                self._latest = snap
                return snap
        except Exception:
            pass

        # Fetch from FRED
        async with aiohttp.ClientSession() as session:
            gpr_val, gprt_val, gpra_val = await asyncio.gather(
                self._fetch_fred_series(session, _GPR_FRED_SERIES),
                self._fetch_fred_series(session, _GPRT_FRED_SERIES),
                self._fetch_fred_series(session, _GPRA_FRED_SERIES),
            )

        if gpr_val is None:
            logger.warning("GPR fetch failed — using neutral baseline.")
            return self._latest

        def normalise(v: float, lo: float, hi: float) -> float:
            return round(max(0.0, min(1.0, (v - lo) / (hi - lo))), 4)

        gpr_v  = gpr_val
        gprt_v = gprt_val or gpr_val
        gpra_v = gpra_val or gpr_val

        snap = GPRSnapshot(
            gpr_index        = gpr_v,
            gprt_threats     = gprt_v,
            gpra_acts        = gpra_v,
            gpr_normalised   = normalise(gpr_v,  _GPR_MIN,  _GPR_MAX),
            gprt_normalised  = normalise(gprt_v, _GPRT_MIN, _GPRT_MAX),
            gpra_normalised  = normalise(gpra_v, _GPRA_MIN, _GPRA_MAX),
            period           = datetime.now(timezone.utc).strftime("%Y-%m"),
            source           = "FRED/Caldara-Iacoviello",
        )

        self._latest = snap

        # Cache in Redis
        try:
            cache_data = {
                "gpr_index": snap.gpr_index,
                "gprt_threats": snap.gprt_threats,
                "gpra_acts": snap.gpra_acts,
                "gpr_normalised": snap.gpr_normalised,
                "gprt_normalised": snap.gprt_normalised,
                "gpra_normalised": snap.gpra_normalised,
                "period": snap.period,
                "source": snap.source,
            }
            await self._redis.set(self._CACHE_KEY, json.dumps(cache_data),
                                   ex=self._CACHE_TTL)
        except Exception as exc:
            logger.warning("GPR Redis cache write failed: %s", exc)

        logger.info(
            "GPR updated: index=%.1f GPRT=%.1f GPRA=%.1f (normalised: %.3f/%.3f/%.3f)",
            snap.gpr_index, snap.gprt_threats, snap.gpra_acts,
            snap.gpr_normalised, snap.gprt_normalised, snap.gpra_normalised,
        )
        return snap

    @property
    def current(self) -> GPRSnapshot:
        return self._latest


# ---------------------------------------------------------------------------
# News Fetcher (unchanged from V1, compatible with LangGraph agent)
# ---------------------------------------------------------------------------

class NewsFetcher:
    """
    V2 NewsFetcher — RSS-based, no paid API required.

    Sources (all free, no key needed):
      • Moneycontrol Markets RSS     — NSE-specific market news
      • Economic Times Markets RSS   — Indian market headlines
      • Business Standard Markets    — macro + corporate news
      • Reuters India RSS            — international + India macro
      • Livemint Markets RSS         — NSE corporate + economy

    Falls back to yfinance if all RSS feeds fail.
    """

    _RSS_FEEDS: List[str] = [
        "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.business-standard.com/rss/markets-106.rss",
        "https://feeds.reuters.com/reuters/INbusinessNews",
        "https://www.livemint.com/rss/markets",
    ]

    def __init__(self) -> None:
        self._interval     = settings.news.FETCH_INTERVAL_SECONDS
        self._max_articles = settings.news.MAX_ARTICLES
        self._watchlist    = settings.strategy.STATIC_WATCHLIST
        self._latest:  List[str] = []

    async def _fetch_rss(
        self, session: aiohttp.ClientSession, url: str
    ) -> List[str]:
        """Fetch and parse a single RSS feed, return list of titles."""
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10),
                headers={"User-Agent": "Mozilla/5.0 SentiStack/2.0"}
            ) as resp:
                resp.raise_for_status()
                text = await resp.text()

            # Parse XML manually — no feedparser dependency required
            import xml.etree.ElementTree as ET
            root = ET.fromstring(text)
            titles: List[str] = []

            # Standard RSS <item><title> and Atom <entry><title>
            for item in root.iter("item"):
                t = item.findtext("title", "").strip()
                if t:
                    titles.append(t)
            for entry in root.iter("entry"):
                t = entry.findtext("{http://www.w3.org/2005/Atom}title", "").strip()
                if t:
                    titles.append(t)

            return titles[:self._max_articles]

        except Exception as exc:
            logger.debug("RSS fetch failed %s: %s", url, exc)
            return []

    async def _fetch_yfinance_fallback(self) -> List[str]:
        """Last-resort fallback using yfinance for watchlist symbols."""
        loop = asyncio.get_running_loop()
        def _sync():
            headlines = []
            for sym in self._watchlist[:3]:   # limit to 3 to avoid rate limits
                try:
                    news = yf.Ticker(f"{sym}.NS").news or []
                    headlines += [n.get("title", "") for n in news[:5] if n.get("title")]
                except Exception:
                    pass
            return headlines
        return await loop.run_in_executor(None, _sync)

    async def fetch_all_headlines(self) -> List[str]:
        all_headlines: List[str] = []

        async with aiohttp.ClientSession() as session:
            tasks   = [self._fetch_rss(session, url) for url in self._RSS_FEEDS]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, list):
                all_headlines.extend(r)

        # Fallback to yfinance if all RSS feeds returned nothing
        if not all_headlines:
            logger.debug("All RSS feeds empty — trying yfinance fallback.")
            all_headlines = await self._fetch_yfinance_fallback()

        # Deduplicate while preserving order
        seen: set = set()
        unique = [h for h in all_headlines if h and h not in seen and not seen.add(h)]
        self._latest = unique
        logger.info("NewsFetcher: %d unique headlines.", len(self._latest))
        return self._latest

    async def run(self) -> None:
        logger.info("NewsFetcher started (interval=%ds).", self._interval)
        while True:
            try:
                await self.fetch_all_headlines()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("NewsFetcher error: %s", exc, exc_info=True)
            await asyncio.sleep(self._interval)

    @property
    def latest_headlines(self) -> List[str]:
        return list(self._latest)


# ---------------------------------------------------------------------------
# AlternativeDataPipeline — V2 aggregator
# ---------------------------------------------------------------------------

class AlternativeDataPipeline:
    """
    Runs all alternative data fetchers as background tasks.
    Exposes unified .sentiment and .weather_stress properties.
    V2: GPR fetcher integrated; LangGraph agent pipeline imported separately.
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis      = redis_client
        self.weather     = WeatherDataFetcher()
        self.news        = NewsFetcher()
        self.gpr         = GPRIndexFetcher(redis_client)
        self._sentiment: SentimentResult = SentimentResult.neutral()
        self._interval                   = settings.news.FETCH_INTERVAL_SECONDS

    def update_sentiment(self, result: SentimentResult) -> None:
        """Called by AgentPipeline after each LangGraph workflow run."""
        self._sentiment = result

    @property
    def sentiment(self) -> SentimentResult:
        return self._sentiment

    @property
    def weather_stress(self) -> float:
        return self.weather.get_composite_stress()

    @property
    def gpr_snapshot(self) -> GPRSnapshot:
        return self.gpr.current

    async def start_background_tasks(self) -> list:
        tasks = [
            asyncio.create_task(self.weather.run(), name="weather_fetcher"),
            asyncio.create_task(self.news.run(),    name="news_fetcher"),
            # B-03 FIX: GPR refresh loop was never launched — GPR stayed neutral forever
            asyncio.create_task(self.run_gpr_refresh_loop(), name="gpr_refresh"),
        ]
        logger.info("AlternativeDataPipeline: %d background tasks started.", len(tasks))
        return tasks

    async def run_gpr_refresh_loop(
        self, interval_seconds: int = 3600
    ) -> None:
        """GPR updates every hour (monthly data — infrequent refresh OK)."""
        while True:
            try:
                await self.gpr.fetch()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("GPR refresh error: %s", exc, exc_info=True)
            await asyncio.sleep(interval_seconds)
