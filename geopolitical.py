"""
geopolitical.py
===============
Real-Time Geopolitical Risk Feed & Composite Index

Data sources (all free, no paid API required):
  1. GDELT Project API  — real-time global event monitoring, Goldstein conflict scale
  2. India VIX          — market-implied fear via Yahoo Finance (^INDIAVIX)
  3. RSS headline scan  — MEA India, PIB, Reuters India, Al Jazeera Asia
  4. USD/INR exchange   — rupee weakness is a leading geopolitical stress indicator

Composite GeopoliticalRiskIndex (GRI) — normalised 0.0 → 1.0:

    GRI = w1 × conflict_score
        + w2 × vix_score
        + w3 × headline_score
        + w4 × fx_stress_score

    Weights: conflict=0.30, vix=0.35, headline=0.20, fx=0.15

Algorithm integration points (4 layers):

  Layer 1 — Alpha Dampening (compute_alpha):
    Soft suppression of the raw alpha signal when geo risk rises.
    Multiplier: piecewise linear, = 1.0 below threshold, → 0.0 at ceiling.

  Layer 2 — Kelly Fraction Scaling (RiskManager):
    Graduated position-size reduction. NOT a binary on/off.
    f_adjusted = f_kelly × max(0, 1 − GRI × GEO_KELLY_SENSITIVITY)

  Layer 3 — Hard Position Decay (RiskManager):
    Existing binary guard: if GRI > GEOPOLITICAL_RISK_THRESHOLD → FLAT.
    Now uses a real GRI instead of the weather proxy.

  Layer 4 — Universe Sector Re-weighting (UniverseEngine):
    High GRI boosts defensive sectors (PHARMA, FMCG/CONSUMER, UTILITIES).
    High GRI penalises vulnerable sectors (FINANCIALS, ENERGY, IT).
    Defence/aerospace stocks get an inverse bonus during conflict spikes.
"""

import asyncio
import collections
import json
import logging
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional, Tuple

import aiohttp

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crisis keyword lexicon — weighted by severity
# ---------------------------------------------------------------------------
# Format: {keyword_lower: severity_weight}
# Severity: 1.0 = full crisis, 0.5 = elevated concern, 0.2 = background noise

CRISIS_KEYWORDS: Dict[str, float] = {
    # Armed conflict
    "war": 1.0, "missile": 1.0, "airstrike": 1.0, "nuclear": 1.0,
    "bomb": 0.9, "attack": 0.7, "military": 0.6, "ceasefire": 0.5,
    "troops": 0.6, "invasion": 1.0, "conflict": 0.6, "artillery": 0.8,
    # India-specific geopolitical
    "pakistan": 0.7, "china border": 0.8, "lac": 0.8, "loc": 0.7,
    "kashmir": 0.7, "doklam": 0.8, "galwan": 0.9, "arunachal": 0.6,
    "india china": 0.7, "india pakistan": 0.8, "quad": 0.3,
    # Economic / sanctions
    "sanctions": 0.8, "embargo": 0.8, "trade war": 0.7, "tariff": 0.4,
    "rupee crash": 0.9, "currency crisis": 0.9, "capital outflow": 0.7,
    "fii sell": 0.5, "fpi outflow": 0.6,
    # Energy / commodity shocks
    "oil shock": 0.7, "opec cut": 0.5, "iran strait": 0.8,
    "red sea": 0.7, "suez": 0.6, "supply disruption": 0.5,
    # Political
    "election violence": 0.6, "coup": 0.9, "political crisis": 0.6,
    "sebi ban": 0.5, "rbi intervention": 0.4, "npa crisis": 0.5,
    "market halt": 1.0, "circuit breaker": 0.8, "trading halt": 0.9,
}

# Sectors and their geopolitical sensitivity:
#   > 0 = risk is bearish for the sector (higher GRI → lower score)
#   < 0 = risk is BULLISH for the sector (inverse: defence, gold, pharma)
GEO_SECTOR_SENSITIVITY: Dict[str, float] = {
    "FINANCIALS":  0.80,   # Banks / NBFCs hit hard by risk-off
    "ENERGY":      0.60,   # Volatile: oil spike cuts margins for importers
    "IT":          0.40,   # Outsourcing contracts, visa risk, USD exposure
    "AUTO":        0.35,   # Demand shock, supply chain
    "REALTY":      0.50,   # Capital flow sensitive
    "METALS":      0.30,   # Trade tariff sensitive
    "INDUSTRIALS": 0.25,
    "TELECOM":     0.20,
    "MEDIA":       0.15,
    "CONSUMER":   -0.20,   # Defensive — benefits from risk-off rotation
    "PHARMA":     -0.30,   # Defensive + export earner during crises
    "UTILITIES":  -0.10,   # Stable cash flows, mild defensive
    "OTHER":       0.20,
}

# RSS feed URLs — all public, no auth required
RSS_FEEDS: List[Tuple[str, float]] = [
    # (url, source_weight)
    ("https://www.mea.gov.in/rss/press-releases.htm", 0.9),           # MEA India
    ("https://feeds.reuters.com/reuters/INbusinessNews", 0.8),        # Reuters India
    ("https://timesofindia.indiatimes.com/rssfeeds/296589292.cms", 0.7),  # TOI World
    ("https://feeds.feedburner.com/ndtvnews-india-news", 0.7),        # NDTV India
    ("https://rss.app/feeds/Xf3OkGZuTLmicD1z.xml", 0.6),             # Al Jazeera Asia
]

# GDELT API endpoint
GDELT_API_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query=India%20conflict%20OR%20India%20military%20OR%20India%20sanctions"
    "&mode=artlist&maxrecords=25&format=json&timespan=4h"
)

# India VIX via Yahoo Finance (^INDIAVIX)
INDIA_VIX_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EINDIAVIX?range=5d&interval=1d"

# USD/INR via Yahoo Finance
USDINR_URL = "https://query1.finance.yahoo.com/v8/finance/chart/USDINR%3DX?range=30d&interval=1d"

# Normal range reference values for normalisation
VIX_NORMAL_LOW  = 10.0   # Calm market
VIX_NORMAL_HIGH = 35.0   # Crisis level
# B-11 FIX: updated USDINR_STABLE from 83.0 → 87.0 to reflect 2025-26 reality.
# At 83.0 the fx_stress_score was permanently ~0.6-0.8, inflating GRI baseline.
# USDINR_STRESS is kept at ~5% above stable for the normalisation spread.
USDINR_STABLE   = 87.0   # Approximate stable value for 2025-26 (update periodically)
USDINR_STRESS   = 92.0   # Stress level (≈5% above stable)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeopoliticalRiskIndex:
    """
    Composite geopolitical risk index for India equity markets.
    All sub-scores and the composite are normalised to [0.0, 1.0].
    """
    composite:        float   # Final weighted composite ∈ [0, 1]
    conflict_score:   float   # GDELT Goldstein-based conflict intensity
    vix_score:        float   # India VIX normalised
    headline_score:   float   # RSS keyword scan intensity
    fx_stress_score:  float   # USD/INR depreciation stress
    top_headlines:    List[str]
    active_keywords:  List[str]   # Which crisis keywords triggered
    india_vix:        float       # Raw India VIX value
    usdinr:           float       # Raw USD/INR rate
    data_age_seconds: float       # Seconds since last successful fetch
    timestamp:        datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def neutral(cls) -> "GeopoliticalRiskIndex":
        """Safe default when all data fetches fail."""
        return cls(
            composite=0.10,       # Small non-zero default — assume some background risk
            conflict_score=0.10,
            vix_score=0.10,
            headline_score=0.05,
            fx_stress_score=0.10,
            top_headlines=[],
            active_keywords=[],
            india_vix=15.0,
            usdinr=USDINR_STABLE,
            data_age_seconds=0.0,
        )

    @property
    def level(self) -> str:
        """Human-readable risk level."""
        if self.composite >= 0.70:
            return "CRITICAL"
        elif self.composite >= 0.50:
            return "HIGH"
        elif self.composite >= 0.30:
            return "ELEVATED"
        elif self.composite >= 0.15:
            return "MODERATE"
        else:
            return "LOW"

    @property
    def alpha_multiplier(self) -> float:
        """
        Graduated alpha dampening factor ∈ [0.0, 1.0].
        Represents how much of the raw alpha signal survives after geo risk.

        Piecewise linear:
            GRI ∈ [0.00, 0.25] → multiplier = 1.0      (full signal)
            GRI ∈ [0.25, 0.65] → multiplier decays linearly to 0.15
            GRI ∈ [0.65, 1.00] → multiplier = 0.0      (signal suppressed)
        """
        if self.composite <= 0.25:
            return 1.0
        elif self.composite <= 0.65:
            return round(1.0 - (self.composite - 0.25) / (0.65 - 0.25) * 0.85, 4)
        else:
            return 0.0

    @property
    def kelly_multiplier(self) -> float:
        """
        Graduated Kelly fraction scalar ∈ [0.0, 1.0].
        More conservative than alpha_multiplier — position sizes shrink faster.

        Piecewise linear:
            GRI ∈ [0.00, 0.20] → multiplier = 1.0
            GRI ∈ [0.20, 0.65] → multiplier decays linearly to 0.0
            GRI > 0.65         → multiplier = 0.0 (hard zero)
        """
        if self.composite <= 0.20:
            return 1.0
        elif self.composite <= 0.65:
            return round(max(0.0, 1.0 - (self.composite - 0.20) / (0.65 - 0.20)), 4)
        else:
            return 0.0

    def sector_impact(self, sector: str) -> float:
        """
        Sector-specific composite impact score adjustment.
        Returns a signed float:
          Positive → GRI reduces this sector's universe score
          Negative → GRI increases this sector's score (defensive rotation)
        """
        sensitivity = GEO_SECTOR_SENSITIVITY.get(sector, 0.20)
        return round(self.composite * sensitivity, 4)

    def max_position_fraction_cap(self, base_cap: float) -> float:
        """
        Tighten the maximum position fraction cap during risk periods.
        At GRI = 0.65: cap halved. At GRI = 1.0: cap = 0.
        """
        return round(base_cap * self.kelly_multiplier, 4)


# ---------------------------------------------------------------------------
# Sub-fetchers
# ---------------------------------------------------------------------------

async def _fetch_gdelt_conflict_score(session: aiohttp.ClientSession) -> Tuple[float, List[str]]:
    """
    Query GDELT for recent India-related conflict events.
    Returns (conflict_score ∈ [0,1], list_of_top_headlines).

    GDELT Goldstein scale: -10 (most conflictual) to +10 (most cooperative).
    We invert and normalise: score = max(0, -goldstein_mean) / 10
    """
    try:
        async with session.get(
            GDELT_API_URL,
            timeout=aiohttp.ClientTimeout(total=12),
            headers={"User-Agent": "SentiStack-Bot/2.0 (research)"},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        articles = data.get("articles", [])
        if not articles:
            return 0.05, []

        headlines = [a.get("title", "") for a in articles[:10] if a.get("title")]

        # Goldstein scale from GDELT tone field: use negative tone as proxy
        # GDELT returns tone as: positive%, negative%, polarity, ...
        tones = []
        for article in articles:
            tone = article.get("tone", "")
            if tone:
                parts = str(tone).split(",")
                if len(parts) >= 2:
                    try:
                        neg_pct = abs(float(parts[1]))
                        tones.append(neg_pct)
                    except ValueError:
                        pass

        if not tones:
            return 0.10, headlines

        # Average negative tone, normalised (>30% neg tone = extreme)
        avg_neg = sum(tones) / len(tones)
        score = min(1.0, avg_neg / 30.0)
        return round(score, 4), headlines

    except asyncio.TimeoutError:
        logger.debug("GDELT fetch timed out.")
        return 0.10, []
    except Exception as exc:
        logger.debug("GDELT fetch error: %s", exc)
        return 0.10, []


async def _fetch_india_vix(session: aiohttp.ClientSession) -> Tuple[float, float]:
    """
    Fetch India VIX via Yahoo Finance.
    Returns (normalised_vix_score ∈ [0,1], raw_vix_value).
    """
    try:
        async with session.get(
            INDIA_VIX_URL,
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "Mozilla/5.0"},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        closes = (
            data.get("chart", {})
                .get("result", [{}])[0]
                .get("indicators", {})
                .get("adjclose", [{}])[0]
                .get("adjclose", [])
        )
        if not closes:
            # Fallback to quote endpoint
            closes = (
                data.get("chart", {})
                    .get("result", [{}])[0]
                    .get("indicators", {})
                    .get("quote", [{}])[0]
                    .get("close", [])
            )

        valid_closes = [c for c in (closes or []) if c is not None]
        if not valid_closes:
            return 0.10, 15.0

        raw_vix = valid_closes[-1]
        # Normalise: score = (vix - low) / (high - low)
        score = (raw_vix - VIX_NORMAL_LOW) / (VIX_NORMAL_HIGH - VIX_NORMAL_LOW)
        score = min(1.0, max(0.0, score))
        return round(score, 4), round(raw_vix, 2)

    except Exception as exc:
        logger.debug("India VIX fetch error: %s", exc)
        return 0.10, 15.0


async def _fetch_usdinr_stress(session: aiohttp.ClientSession) -> Tuple[float, float]:
    """
    Fetch USD/INR rate and compute rupee depreciation stress score.
    Returns (fx_stress_score ∈ [0,1], raw_usdinr_rate).

    Stress = how far current rate is above the 30-day mean.
    At +5% above mean: score = 1.0 (full stress).
    """
    try:
        async with session.get(
            USDINR_URL,
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"User-Agent": "Mozilla/5.0"},
        ) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        closes = (
            data.get("chart", {})
                .get("result", [{}])[0]
                .get("indicators", {})
                .get("quote", [{}])[0]
                .get("close", [])
        )
        valid = [c for c in (closes or []) if c is not None]
        if not valid:
            return 0.10, USDINR_STABLE

        current = valid[-1]
        mean_30d = sum(valid) / len(valid)

        # Stress = deviation above 30d mean, normalised to 5% range
        deviation_pct = (current - mean_30d) / mean_30d
        stress = max(0.0, deviation_pct / 0.05)   # 5% dev = score of 1.0
        return round(min(1.0, stress), 4), round(current, 4)

    except Exception as exc:
        logger.debug("USD/INR fetch error: %s", exc)
        return 0.10, USDINR_STABLE


async def _fetch_rss_headline_score(session: aiohttp.ClientSession) -> Tuple[float, List[str], List[str]]:
    """
    Fetch RSS feeds and scan headlines for crisis keywords.
    Returns (headline_score ∈ [0,1], top_headlines, triggered_keywords).

    Scoring:
        For each article, compute article_risk = max(keyword_weight for matched keywords).
        Final score = weighted average of top-10 article risks, weighted by source trust.
    """
    all_headlines: List[Tuple[str, float]] = []   # (headline, source_weight)

    for url, source_weight in RSS_FEEDS:
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=8),
                headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                if resp.status != 200:
                    continue
                text = await resp.text()

            root = ET.fromstring(text)
            items = root.findall(".//item")
            for item in items[:15]:
                title_el = item.find("title")
                desc_el  = item.find("description")
                title    = title_el.text or "" if title_el is not None else ""
                desc     = desc_el.text  or "" if desc_el  is not None else ""
                combined = (title + " " + desc).lower()
                all_headlines.append((title, source_weight, combined))

        except Exception as exc:
            logger.debug("RSS fetch failed for %s: %s", url, exc)

    if not all_headlines:
        return 0.05, [], []

    # Score each headline
    article_scores: List[float] = []
    top_headlines: List[str] = []
    triggered_set: set = set()

    for title, src_weight, combined in all_headlines:
        article_risk = 0.0
        for keyword, weight in CRISIS_KEYWORDS.items():
            if keyword in combined:
                article_risk = max(article_risk, weight * src_weight)
                triggered_set.add(keyword)

        if article_risk > 0.1:
            article_scores.append(article_risk)
            top_headlines.append(title)

    if not article_scores:
        return 0.05, [], []

    # Use the 90th percentile of article scores (robust to noise)
    sorted_scores = sorted(article_scores, reverse=True)
    top_k = max(1, len(sorted_scores) // 5)
    headline_score = sum(sorted_scores[:top_k]) / top_k

    return (
        round(min(1.0, headline_score), 4),
        top_headlines[:8],
        sorted(triggered_set)[:10],
    )


# ---------------------------------------------------------------------------
# Composite GRI builder
# ---------------------------------------------------------------------------

WEIGHTS = {
    "conflict":  0.30,
    "vix":       0.35,
    "headline":  0.20,
    "fx":        0.15,
}


def _build_composite(
    conflict_score: float,
    vix_score: float,
    headline_score: float,
    fx_score: float,
) -> float:
    """
    Weighted composite GRI. Uses a non-linear amplification for extreme values:
    if any sub-score exceeds 0.80, the composite is nudged upward.
    """
    linear = (
        conflict_score  * WEIGHTS["conflict"] +
        vix_score       * WEIGHTS["vix"]      +
        headline_score  * WEIGHTS["headline"] +
        fx_score        * WEIGHTS["fx"]
    )
    # Non-linear tail boost: if any single component is critical, amplify
    max_component = max(conflict_score, vix_score, headline_score, fx_score)
    if max_component >= 0.80:
        boost = (max_component - 0.80) * 0.25   # up to +5% boost
        linear = min(1.0, linear + boost)

    return round(linear, 4)


# ---------------------------------------------------------------------------
# GeopoliticalRiskMonitor — the main class
# ---------------------------------------------------------------------------

class GeopoliticalRiskMonitor:
    """
    Continuously monitors geopolitical risk indicators and maintains a
    fresh GeopoliticalRiskIndex available to all algorithm components.

    Architecture:
        • _fetch_loop() runs every FETCH_INTERVAL_SECONDS.
        • All 4 sub-fetchers run concurrently via asyncio.gather().
        • On failure, exponential backoff (up to MAX_BACKOFF_SECONDS).
        • Latest index always available via .current property (never blocks).
        • Redis snapshot stored for cross-process sharing.

    Usage:
        monitor = GeopoliticalRiskMonitor(redis_client)
        tasks = await monitor.start_background_tasks()
        ...
        gri = monitor.current
        alpha_adj = raw_alpha * gri.alpha_multiplier
    """

    FETCH_INTERVAL_SECONDS = 300      # Every 5 minutes
    MAX_BACKOFF_SECONDS    = 600      # Max 10 min on repeated failure
    REDIS_KEY              = "geo:risk_index"
    REDIS_TTL              = 900      # 15 min TTL on Redis snapshot

    def __init__(self, redis_client) -> None:
        self._redis      = redis_client
        self._current    = GeopoliticalRiskIndex.neutral()
        self._last_fetch = 0.0
        self._fail_count = 0
        # B-14 FIX: deque(maxlen=12) is O(1) append/pop vs O(n) list.pop(0)
        self._history: Deque[GeopoliticalRiskIndex] = collections.deque(maxlen=12)
        # B-16 FIX: reuse a single TelegramNotifier rather than creating one per alert
        self._telegram_notifier = None   # lazy-init on first escalation alert

    @property
    def current(self) -> GeopoliticalRiskIndex:
        """Non-blocking read of the latest GRI. Always returns a valid object."""
        return self._current

    async def fetch_now(self) -> GeopoliticalRiskIndex:
        """
        Fetch all sub-components concurrently and build the composite index.
        This is the core fetch function; called by the background loop.
        """
        t_start = time.monotonic()
        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                _fetch_gdelt_conflict_score(session),
                _fetch_india_vix(session),
                _fetch_usdinr_stress(session),
                _fetch_rss_headline_score(session),
                return_exceptions=True,
            )

        # Unpack results with safe defaults for any that raised
        conflict_score, conflict_headlines = (
            results[0] if not isinstance(results[0], Exception) else (0.10, [])
        )
        vix_score, raw_vix = (
            results[1] if not isinstance(results[1], Exception) else (0.10, 15.0)
        )
        fx_score, raw_usdinr = (
            results[2] if not isinstance(results[2], Exception) else (0.10, USDINR_STABLE)
        )
        headline_score, rss_headlines, triggered_kw = (
            results[3] if not isinstance(results[3], Exception) else (0.05, [], [])
        )

        composite = _build_composite(conflict_score, vix_score, headline_score, fx_score)

        all_headlines = list(dict.fromkeys(conflict_headlines + rss_headlines))[:10]
        data_age = time.monotonic() - t_start

        gri = GeopoliticalRiskIndex(
            composite=composite,
            conflict_score=conflict_score,
            vix_score=vix_score,
            headline_score=headline_score,
            fx_stress_score=fx_score,
            top_headlines=all_headlines,
            active_keywords=triggered_kw,
            india_vix=raw_vix,
            usdinr=raw_usdinr,
            data_age_seconds=round(data_age, 2),
        )

        self._current    = gri
        self._last_fetch = time.monotonic()
        self._history.append(gri)  # B-14: deque(maxlen=12) auto-evicts oldest

        logger.info(
            "GRI update: composite=%.3f (%s) | vix=%.1f | usdinr=%.2f | "
            "conflict=%.3f | headline=%.3f | fx=%.3f | keywords=[%s]",
            composite, gri.level, raw_vix, raw_usdinr,
            conflict_score, headline_score, fx_score,
            ", ".join(triggered_kw[:5]),
        )

        # Persist to Redis for cross-process reads
        await self._persist_to_redis(gri)

        # Alert Telegram on sharp escalation
        if len(self._history) >= 2:
            prev = self._history[-2].composite
            if gri.composite - prev > 0.15:
                await self._send_escalation_alert(gri, prev)

        return gri

    async def _persist_to_redis(self, gri: GeopoliticalRiskIndex) -> None:
        payload = json.dumps({
            "composite":       gri.composite,
            "conflict_score":  gri.conflict_score,
            "vix_score":       gri.vix_score,
            "headline_score":  gri.headline_score,
            "fx_stress_score": gri.fx_stress_score,
            "india_vix":       gri.india_vix,
            "usdinr":          gri.usdinr,
            "level":           gri.level,
            "alpha_multiplier":gri.alpha_multiplier,
            "kelly_multiplier":gri.kelly_multiplier,
            "active_keywords": gri.active_keywords,
            "top_headlines":   gri.top_headlines[:3],
            "timestamp":       gri.timestamp.isoformat(),
        })
        try:
            await self._redis.set(self.REDIS_KEY, payload, ex=self.REDIS_TTL)
        except Exception as exc:
            logger.debug("GRI Redis persist failed: %s", exc)

    async def _send_escalation_alert(
        self, gri: GeopoliticalRiskIndex, prev_composite: float
    ) -> None:
        """Push a Telegram alert on sharp GRI escalation (non-blocking)."""
        try:
            # B-16 FIX: reuse cached notifier — was creating a new TelegramNotifier()
            # (and a new aiohttp session) on every alert call, leaking connections.
            if self._telegram_notifier is None:
                from execution import TelegramNotifier
                self._telegram_notifier = TelegramNotifier()
            notifier = self._telegram_notifier
            kw_str = ", ".join(gri.active_keywords[:5]) or "N/A"
            headline_str = ("\n  • ".join(gri.top_headlines[:3])
                            if gri.top_headlines else "N/A")
            await notifier.send(
                f"⚠️ <b>GEOPOLITICAL RISK ESCALATION</b>\n"
                f"{'─' * 30}\n"
                f"📈 <b>GRI:</b> {prev_composite:.3f} → <b>{gri.composite:.3f}</b> "
                f"({gri.level})\n"
                f"📊 <b>India VIX:</b> {gri.india_vix:.1f}\n"
                f"💱 <b>USD/INR:</b> {gri.usdinr:.2f}\n"
                f"🔑 <b>Keywords:</b> {kw_str}\n"
                f"📰 <b>Headlines:</b>\n  • {headline_str}\n"
                f"⚡ <b>Alpha mult:</b> {gri.alpha_multiplier:.2f}  "
                f"| <b>Kelly mult:</b> {gri.kelly_multiplier:.2f}\n"
                f"🕐 {gri.timestamp.strftime('%H:%M:%S UTC')}"
            )
        except Exception as exc:
            logger.debug("GRI escalation alert failed: %s", exc)

    async def load_from_redis(self) -> bool:
        """
        Try to restore the latest GRI from Redis on startup.
        Returns True if a recent (< 15 min) snapshot was found.
        """
        try:
            raw = await self._redis.get(self.REDIS_KEY)
            if not raw:
                return False
            data = json.loads(raw)
            ts = datetime.fromisoformat(data.get("timestamp", ""))
            age_s = (datetime.now(timezone.utc) - ts).total_seconds()
            if age_s > self.REDIS_TTL:
                return False
            self._current = GeopoliticalRiskIndex(
                composite       = data["composite"],
                conflict_score  = data["conflict_score"],
                vix_score       = data["vix_score"],
                headline_score  = data["headline_score"],
                fx_stress_score = data["fx_stress_score"],
                top_headlines   = data.get("top_headlines", []),
                active_keywords = data.get("active_keywords", []),
                india_vix       = data["india_vix"],
                usdinr          = data["usdinr"],
                data_age_seconds= age_s,
                timestamp       = ts,
            )
            logger.info(
                "GRI restored from Redis: composite=%.3f (%s), age=%.0fs",
                self._current.composite, self._current.level, age_s,
            )
            return True
        except Exception as exc:
            logger.debug("GRI Redis load failed: %s", exc)
            return False

    async def _fetch_loop(self) -> None:
        """Background loop: fetch GRI every FETCH_INTERVAL_SECONDS."""
        logger.info("GeopoliticalRiskMonitor: background fetch loop started.")
        while True:
            try:
                await self.fetch_now()
                self._fail_count = 0
                await asyncio.sleep(self.FETCH_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                logger.info("GeopoliticalRiskMonitor: shutting down.")
                raise
            except Exception as exc:
                self._fail_count += 1
                backoff = min(
                    self.MAX_BACKOFF_SECONDS,
                    self.FETCH_INTERVAL_SECONDS * (2 ** min(self._fail_count, 4))
                )
                logger.error(
                    "GRI fetch failed (attempt %d): %s — retry in %.0fs",
                    self._fail_count, exc, backoff,
                )
                await asyncio.sleep(backoff)

    async def initialise(self) -> None:
        """On startup: try Redis cache, then do an immediate live fetch."""
        loaded = await self.load_from_redis()
        if not loaded:
            logger.info("GRI: No cache — fetching live data now…")
            try:
                await self.fetch_now()
            except Exception as exc:
                logger.warning("GRI initial fetch failed: %s — using neutral default.", exc)

    async def start_background_tasks(self) -> list:
        tasks = [
            asyncio.create_task(self._fetch_loop(), name="geo_risk_monitor")
        ]
        logger.info("GeopoliticalRiskMonitor: background task started.")
        return tasks

    @property
    def trend(self) -> str:
        """Rising / Falling / Stable based on last 3 readings."""
        if len(self._history) < 3:
            return "STABLE"
        recent   = self._history[-1].composite
        older    = self._history[-3].composite
        delta    = recent - older
        if delta > 0.05:
            return "RISING"
        elif delta < -0.05:
            return "FALLING"
        return "STABLE"
