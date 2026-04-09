"""
agent_pipeline.py  — V2
=======================
Enhancement 3: Multi-Agent LLM Architecture using LangGraph.

Architecture: StateGraph with 3 specialist agent nodes and conditional routing.

    ┌─────────────────────────────────────────────────────────────────┐
    │                     SentiStack Agent Graph                      │
    │                                                                 │
    │   [START] → DataRetrieverAgent → SentimentAnalyzerAgent        │
    │                                         │                       │
    │                              ┌──────────┴──────────┐           │
    │                              ▼ (cond: vol context)  ▼           │
    │                       RiskManagerAgent         [pass-through]   │
    │                              │                      │           │
    │                              └──────────┬───────────┘           │
    │                                         ▼                       │
    │                                      [END]                      │
    └─────────────────────────────────────────────────────────────────┘

Agent Nodes:
  1. DataRetrieverAgent    — Fetches/formats news headlines + GPR snapshot
  2. SentimentAnalyzerAgent — Gemini LLM: produces fear/excitement score + rationale
  3. RiskManagerAgent      — Gemini LLM: contextualizes sentiment vs current vol regime

Conditional Routing:
  After SentimentAnalyzerAgent, route to RiskManagerAgent only when:
    • |sentiment_score| > HIGH_CONVICTION_THRESHOLD (0.4), OR
    • vol_regime label is "HIGH" or "EXTREME"
  Otherwise route directly to END (skip risk manager for neutral signals).

State Schema (TypedDict):
  headlines     : List[str]   — raw news headlines
  gpr_snapshot  : dict        — serialised GPRSnapshot
  vol_regime    : str         — "LOW" | "MODERATE" | "HIGH" | "EXTREME"
  current_vix   : float       — India VIX scalar
  sentiment_raw : dict        — SentimentAnalyzerAgent output (JSON)
  risk_context  : str         — RiskManagerAgent output (plain text)
  final_result  : SentimentResult — assembled final output
  error         : str         — error message if any agent fails
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from google import genai
from google.genai import types as genai_types

from alternative_data import GPRSnapshot, SentimentResult
from config import settings

logger = logging.getLogger(__name__)

# Conviction threshold for conditional routing to RiskManagerAgent
HIGH_CONVICTION_THRESHOLD = 0.40

# B-12 FIX: create genai.Client as a module-level singleton rather than per-call.
# Creating a new client on every API call wastes ~5-10ms and leaks HTTP sessions.
_genai_client: Optional[genai.Client] = None

def _get_genai_client() -> genai.Client:
    """Return the module-level singleton Gemini client, creating it if needed."""
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=settings.gemini.API_KEY)
    return _genai_client



# ---------------------------------------------------------------------------
# Model Rotation & Budget Manager
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone as _tz, timedelta

_IST = _tz(timedelta(hours=5, minutes=30))

@dataclass
class _ModelSlot:
    """Tracks state for one Gemini model."""
    api_name:   str         # exact model string for genai API
    display:    str         # human label for logs
    quality:    int         # 1-5; higher = better sentiment accuracy
    rpm_limit:  int         # requests per minute (free tier)
    rpd_limit:  int         # requests per day (free tier)
    tier:       str         # "flash" | "pro" — determines when it's selected
    supports_system_instruction: bool = True
    # runtime state
    calls_made:      int   = 0
    calls_this_min:  int   = 0    # RPM window counter
    min_window_start: float = 0.0  # start of current 60-second RPM window
    rpd_today:       int   = 0    # calls made today (resets at midnight IST)
    rpd_date:        str   = ""   # IST date string when rpd_today was last reset
    last_call_ts:    float = 0.0
    errors_429:      int   = 0
    cooldown_until:  float = 0.0   # epoch seconds


class ModelRotator:
    """
    Manages a priority-ordered pool of Gemini models.

    Design principles (free-tier optimised):
      • Flash-first: use high-quota Flash models (1500 RPD) for routine calls.
      • Pro-reserved: Pro models (25-50 RPD) only for high-GRI / opening / close.
      • Proactive RPM enforcement: never fire if already at RPM limit for that model.
      • RPD tracking: count daily calls per model; warn when nearing limit.
      • Differentiated cooldown: RPM hit → 65s; RPD exhausted → 24h.
      • RiskManagerAgent gated: only fire the second LLM call when Pro quota is available.

    Pool order = Flash models first (normal use), Pro models last (reserved).

    Call-frequency tiers (returned by `recommended_interval_s`):
      Opening      → 180 s   (09:15–09:45)
      Pre-close    → 180 s   (15:00–15:30)
      EXTREME GRI  → 180 s
      HIGH GRI     → 240 s
      ELEVATED     → 360 s   (default — 62 Flash calls/day)
      MODERATE     → 480 s
      LOW + calm   → 600 s   (conserve budget)
    """

    # Pool in selection priority order.
    # Flash models first — they have 1500 RPD on free tier.
    # Pro models last  — they have only 25-50 RPD; reserved for high-risk moments.
    _POOL: List[_ModelSlot] = [
        _ModelSlot("gemini-2.5-flash", "2.5 Flash", quality=4, rpm_limit=10, rpd_limit=500,  tier="flash"),
        _ModelSlot("gemini-2.0-flash", "2.0 Flash", quality=4, rpm_limit=15, rpd_limit=1500, tier="flash"),
        _ModelSlot("gemini-1.5-flash", "1.5 Flash", quality=3, rpm_limit=15, rpd_limit=1500, tier="flash"),
        _ModelSlot("gemini-2.5-pro",   "2.5 Pro",   quality=5, rpm_limit=5,  rpd_limit=25,   tier="pro"),
        _ModelSlot("gemini-1.5-pro",   "1.5 Pro",   quality=5, rpm_limit=2,  rpd_limit=50,   tier="pro"),
    ]

    # Cooldowns: separate RPM vs RPD exhaustion
    _RPM_COOLDOWN_S  = 65     # just over 1 minute — enough for RPM window to reset
    _RPD_COOLDOWN_S  = 86400  # 24 hours — daily quota exhausted

    # Warn when a model has used this fraction of its daily quota
    _RPD_WARN_FRACTION = 0.80

    # Path for persisting daily quota counters across process restarts.
    # Dot-file keeps it out of the way; excluded from git via .gitignore.
    _QUOTA_STATE_FILE: Path = Path(__file__).parent / ".gemini_quota.json"

    def __init__(self) -> None:
        import copy
        self._slots: List[_ModelSlot] = copy.deepcopy(self._POOL)
        self._load_quota_state()          # restore RPD counters & active cooldowns
        logger.info(
            "ModelRotator initialised — Flash pool: %s | Pro pool: %s",
            [s.display for s in self._slots if s.tier == "flash"],
            [s.display for s in self._slots if s.tier == "pro"],
        )

    # ── Quota state persistence ───────────────────────────────────────────

    def _save_quota_state(self) -> None:
        """
        Persist per-model RPD counters and active cooldown timestamps to disk.
        Called after every mark_success / mark_rate_limited / mark_not_found so
        that process restarts don't reset the daily call count and re-exhaust
        the same quota that was already consumed in this IST day.
        """
        now = time.time()
        state: Dict[str, Any] = {}
        for s in self._slots:
            state[s.api_name] = {
                "rpd_today":     s.rpd_today,
                "rpd_date":      s.rpd_date,
                # Only persist cooldowns that are still active (> 60 s remaining)
                # so short RPM cooldowns are not needlessly carried over restarts.
                "cooldown_until": s.cooldown_until if s.cooldown_until - now > 60 else 0.0,
            }
        try:
            self._QUOTA_STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as exc:
            logger.warning("Could not save Gemini quota state: %s", exc)

    def _load_quota_state(self) -> None:
        """
        Restore RPD counters saved by _save_quota_state.
        Only restores data whose rpd_date matches today (IST) so yesterday's
        counts don't carry over after the Google quota reset (midnight IST).
        """
        if not self._QUOTA_STATE_FILE.exists():
            return
        try:
            data  = json.loads(self._QUOTA_STATE_FILE.read_text())
            today = datetime.now(_IST).strftime("%Y-%m-%d")
            now   = time.time()
            for s in self._slots:
                saved = data.get(s.api_name, {})
                if saved.get("rpd_date") == today:
                    s.rpd_today = int(saved.get("rpd_today", 0))
                    s.rpd_date  = today
                    # Restore still-active cooldowns (e.g. 24-h RPD block)
                    saved_cd = float(saved.get("cooldown_until", 0.0))
                    if saved_cd > now:
                        s.cooldown_until = saved_cd
                        logger.info(
                            "Restored %s: RPD %d/%d  cooldown %.0f s remaining.",
                            s.display, s.rpd_today, s.rpd_limit, saved_cd - now,
                        )
                    else:
                        logger.info(
                            "Restored %s: RPD %d/%d (no active cooldown).",
                            s.display, s.rpd_today, s.rpd_limit,
                        )
        except Exception as exc:
            logger.warning("Could not load Gemini quota state: %s", exc)

    # ── RPD date reset ────────────────────────────────────────────────────

    def _reset_rpd_if_new_day(self, slot: _ModelSlot) -> None:
        today = datetime.now(_IST).strftime("%Y-%m-%d")
        if slot.rpd_date != today:
            slot.rpd_today = 0
            slot.rpd_date  = today

    # ── RPM window management (separated from availability check) ─────────

    def _advance_rpm_window(self, slot: _ModelSlot) -> None:
        """
        Advance the slot's 60-second RPM window if it has expired.
        MUST be called ONLY when we are about to make a call (in pick_model)
        — NOT inside _slot_available — so that merely checking availability
        doesn't corrupt the window counter.
        """
        now = time.time()
        if now - slot.min_window_start >= 60.0:
            slot.min_window_start = now
            slot.calls_this_min   = 0

    # ── Slot availability ─────────────────────────────────────────────────

    def _slot_available(self, slot: _ModelSlot) -> bool:
        """
        Pure read — returns True if slot can accept a call right now.
        Does NOT mutate any state (window reset is in _advance_rpm_window).
        """
        now = time.time()

        # 1. Cooldown block (set after 429 or 404)
        if now < slot.cooldown_until:
            return False

        # 2. Proactive RPM check — read the window WITHOUT resetting it
        elapsed = now - slot.min_window_start
        effective_calls = slot.calls_this_min if elapsed < 60.0 else 0
        if effective_calls >= slot.rpm_limit:
            return False

        # 3. Daily quota check
        today = datetime.now(_IST).strftime("%Y-%m-%d")
        effective_rpd = slot.rpd_today if slot.rpd_date == today else 0
        if effective_rpd >= slot.rpd_limit:
            return False

        return True

    def all_on_cooldown(self) -> bool:
        return not any(self._slot_available(s) for s in self._slots)

    def soonest_available_in(self) -> float:
        """
        Seconds until the first model becomes available.

        Binding constraints:
          • Active cooldown (429/404) → must wait for cooldown_until
          • No cooldown but RPM window full → wait for window to reset

        BUG that was here: used min(cooldown, rpm_reset).  For a model
        on 24h cooldown this returned ~30s (rpm_reset), making the
        sentiment loop wake up every 60s for 24 hours instead of waiting
        the full cooldown duration.
        """
        now   = time.time()
        waits = []
        for s in self._slots:
            if self._slot_available(s):
                return 0.0
            cd = max(0.0, s.cooldown_until - now)
            if cd > 0:
                # Hard cooldown is the binding constraint — RPM window is irrelevant.
                waits.append(cd)
            else:
                # No cooldown; only obstacle is the RPM window being full.
                rpm_reset = max(0.0, 60.0 - (now - s.min_window_start))
                waits.append(rpm_reset)
        return min(waits) if waits else 0.0

    def pick_model(self, gri_composite: float = 0.3, allow_pro: bool = False) -> "_ModelSlot | None":
        """
        Pick the best available model for the current context.

        Strategy:
          - Normal conditions → Flash models only (preserve Pro quota).
          - High GRI / opening / close AND allow_pro=True → try Pro first.
          - If preferred tier unavailable → fall back to any available slot.
        """
        now  = time.time()
        ist  = datetime.fromtimestamp(now, tz=_IST)
        hour = ist.hour + ist.minute / 60.0

        is_opening   = 9.25 <= hour <= 9.75
        is_pre_close = 15.0 <= hour <= 15.5
        is_high_risk = gri_composite >= 0.55

        use_pro = allow_pro and (is_opening or is_pre_close or is_high_risk)

        # Build candidate list
        available_flash = [s for s in self._slots if s.tier == "flash" and self._slot_available(s)]
        available_pro   = [s for s in self._slots if s.tier == "pro"   and self._slot_available(s)]
        available_all   = [s for s in self._slots if self._slot_available(s)]

        if use_pro and available_pro:
            chosen = available_pro[0]   # best available Pro
        elif available_flash:
            chosen = available_flash[0] # best available Flash (default)
        elif available_all:
            chosen = available_all[0]   # last resort — any available
        else:
            soonest = self.soonest_available_in()
            logger.warning(
                "All Gemini models unavailable — skipping call (soonest in %.0fs).",
                soonest,
            )
            return None

        # Advance the RPM window ONLY here (just before a real call is issued),
        # not inside _slot_available, so availability checks don't corrupt the counter.
        self._advance_rpm_window(chosen)
        self._reset_rpd_if_new_day(chosen)

        # Warn if nearing daily limit
        rpd_used_frac = chosen.rpd_today / chosen.rpd_limit if chosen.rpd_limit else 0
        if rpd_used_frac >= self._RPD_WARN_FRACTION:
            logger.warning(
                "⚠️  %s daily quota at %.0f%% (%d/%d calls).",
                chosen.display, rpd_used_frac * 100, chosen.rpd_today, chosen.rpd_limit,
            )

        return chosen

    def mark_success(self, slot: _ModelSlot) -> None:
        now = time.time()
        slot.calls_made      += 1
        slot.last_call_ts     = now
        slot.errors_429       = 0
        slot.calls_this_min  += 1
        self._reset_rpd_if_new_day(slot)
        slot.rpd_today       += 1
        logger.debug(
            "ModelRotator ✓ %s  (RPM window: %d/%d | RPD: %d/%d)",
            slot.display, slot.calls_this_min, slot.rpm_limit,
            slot.rpd_today, slot.rpd_limit,
        )
        self._save_quota_state()   # persist so restarts see today's count

    def mark_rate_limited(self, slot: _ModelSlot, is_daily: bool = False) -> None:
        """
        Apply cooldown after a 429.
        is_daily=True  → daily quota exhausted (24h cooldown).
        is_daily=False → RPM hit (65s + exponential back-off).
        """
        slot.errors_429 += 1
        if is_daily:
            cooldown = self._RPD_COOLDOWN_S
            logger.warning(
                "ModelRotator: %s daily quota exhausted — disabling for 24h.", slot.display
            )
        else:
            # Exponential back-off capped at 10 min for RPM errors
            cooldown = min(self._RPM_COOLDOWN_S * (2 ** min(slot.errors_429 - 1, 3)), 600)
            logger.warning(
                "ModelRotator: %s RPM limit hit — cooldown %.0fs (error #%d).",
                slot.display, cooldown, slot.errors_429,
            )
        slot.cooldown_until = time.time() + cooldown
        self._save_quota_state()   # persist cooldowns (especially 24-h blocks)

    def mark_not_found(self, slot: _ModelSlot) -> None:
        slot.cooldown_until = time.time() + self._RPD_COOLDOWN_S
        logger.error("ModelRotator: %s returned 404 — disabling for today.", slot.display)
        self._save_quota_state()

    # ── Interval recommendation ───────────────────────────────────────────

    def recommended_interval_s(self, gri_composite: float = 0.3, vix: float = 18.0) -> int:
        """
        Recommended sleep between sentiment cycles.
        NSE day = 375 min. At 360s base → ~62 Flash calls/day — well within 1500 RPD.
        Pro models are called at most ~10x/day (only during high-risk windows).
        """
        now  = time.time()
        ist  = datetime.fromtimestamp(now, tz=_IST)
        hour = ist.hour + ist.minute / 60.0

        if 9.25 <= hour <= 9.75:   return 180   # Opening window
        if 15.0 <= hour <= 15.5:   return 180   # Pre-close window
        if gri_composite >= 0.65:  return 180   # EXTREME risk
        if gri_composite >= 0.50:  return 240   # HIGH risk
        if gri_composite >= 0.30:  return 360   # ELEVATED (default)
        if gri_composite >= 0.15:  return 480   # MODERATE
        return 600                              # LOW — conserve budget

    def can_afford_risk_manager(self) -> bool:
        """
        Return True only when a Pro slot has remaining daily quota.
        Prevents RiskManagerAgent from burning Flash quota on a second call.
        """
        return any(
            s.tier == "pro" and self._slot_available(s)
            for s in self._slots
        )

    def daily_stats(self) -> Dict[str, Any]:
        now = time.time()
        return {
            s.display: {
                "calls_total": s.calls_made,
                "rpd_today":   s.rpd_today,
                "rpd_limit":   s.rpd_limit,
                "rpm_window":  s.calls_this_min,
                "rpm_limit":   s.rpm_limit,
                "errors_429":  s.errors_429,
                "available":   self._slot_available(s),
                "cooldown_s":  max(0.0, round(s.cooldown_until - now)),
                "tier":        s.tier,
            }
            for s in self._slots
        }


# ---------------------------------------------------------------------------
# Shared Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    headlines:      List[str]
    gpr_snapshot:   Dict[str, Any]
    vol_regime:     str      # "LOW" | "MODERATE" | "HIGH" | "EXTREME"
    current_vix:    float
    sentiment_raw:  Dict[str, Any]
    risk_context:   str
    final_result:   Optional[Dict[str, Any]]
    error:          str
    iterations:     int


def _initial_state(
    headlines: List[str],
    gpr: GPRSnapshot,
    vol_regime: str,
    current_vix: float,
) -> AgentState:
    return AgentState(
        headlines     = headlines,
        gpr_snapshot  = {
            "gpr_index":       gpr.gpr_index,
            "gprt_threats":    gpr.gprt_threats,
            "gpra_acts":       gpr.gpra_acts,
            "gpr_normalised":  gpr.gpr_normalised,
            "gprt_normalised": gpr.gprt_normalised,
            "gpra_normalised": gpr.gpra_normalised,
            "period":          gpr.period,
            "threat_dominance": gpr.threat_dominance,
        },
        vol_regime    = vol_regime,
        current_vix   = current_vix,
        sentiment_raw = {},
        risk_context  = "",
        final_result  = None,
        error         = "",
        iterations    = 0,
    )


# ---------------------------------------------------------------------------
# Agent Node 1: DataRetrieverAgent
# ---------------------------------------------------------------------------

def data_retriever_agent(state: AgentState) -> AgentState:
    """
    Formats and enriches the raw data context for downstream agents.
    Trims, deduplicates headlines; annotates GPR regime label.

    Pure Python — no LLM call. Acts as a data preparation / filtering node.
    """
    headlines   = list(dict.fromkeys(state["headlines"]))[:20]  # dedup + cap
    gpr         = state["gpr_snapshot"]
    vol_regime  = state["vol_regime"]

    gpr_label = (
        "CRITICAL"  if gpr["gpr_normalised"] > 0.70 else
        "HIGH"      if gpr["gpr_normalised"] > 0.50 else
        "ELEVATED"  if gpr["gpr_normalised"] > 0.30 else
        "MODERATE"  if gpr["gpr_normalised"] > 0.15 else
        "LOW"
    )

    threat_label = (
        "threat-dominant (market pricing future conflict)"
        if gpr["threat_dominance"] > 1.2
        else "act-dominant (realized conflict ongoing)"
        if gpr["threat_dominance"] < 0.8
        else "balanced"
    )

    enriched_context = {
        "headlines":      headlines,
        "gpr_label":      gpr_label,
        "gpr_index":      round(gpr["gpr_index"], 1),
        "gprt":           round(gpr["gprt_threats"], 1),
        "gpra":           round(gpr["gpra_acts"], 1),
        "threat_label":   threat_label,
        "vol_regime":     vol_regime,
        "vix":            state["current_vix"],
        "headline_count": len(headlines),
    }

    logger.debug(
        "DataRetrieverAgent: %d headlines, GPR=%s (%s), vol=%s",
        len(headlines), gpr_label, threat_label, vol_regime
    )

    # Store enriched context in state as updated gpr_snapshot
    updated_gpr = {**state["gpr_snapshot"], **{"label": gpr_label, "threat_label": threat_label}}
    return {**state, "gpr_snapshot": updated_gpr, "headlines": headlines, "iterations": state["iterations"] + 1}


# ---------------------------------------------------------------------------
# Agent Node 2: SentimentAnalyzerAgent
# ---------------------------------------------------------------------------

_SENTIMENT_SYSTEM = """You are a quantitative financial sentiment analyst specialising in
Indian equity markets (NSE/BSE). You receive a set of news headlines and geopolitical context.

Your task: classify market sentiment and output ONLY a valid JSON object.

Output schema (STRICT — Rationale at the end):
{
  "sentiment_classification": "Fear" | "Excitement" | "Neutral" | "Disbelief",
  "sentiment_score": <float in [-1.0, 1.0]>,
  "confidence": <float in [0.0, 1.0]>,
  "key_entities": ["<NSE ticker or company name>", ...],
  "rationale": "<one concise sentence explaining the dominant signal>"
}

Rules:
- Fear:       bearish macro, conflict, recession signals → score < 0
- Excitement: strong earnings, deals, policy tailwinds  → score > 0
- Disbelief:  contradictory or extreme signals          → score near 0 but high confidence
- Neutral:    no clear signal                            → score ≈ 0, confidence < 0.5
"""

_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "parts": ["Headlines: [\"RBI holds rates, signals easing bias\", \"TCS beats Q3 estimates\"]\nGPR: LOW | VIX: 12.3 | Vol regime: LOW"],
        "response": '{"sentiment_classification":"Excitement","sentiment_score":0.62,"confidence":0.78,"key_entities":["TCS"],"rationale":"Policy easing bias combined with large-cap earnings beat signals risk-on environment."}'
    },
    {
        "role": "user",
        "parts": ["Headlines: [\"India-Pakistan border tensions escalate\", \"FII sold ₹4,200 Cr\"]\nGPR: HIGH (threat-dominant) | VIX: 28.5 | Vol regime: HIGH"],
        "response": '{"sentiment_classification":"Fear","sentiment_score":-0.79,"confidence":0.91,"key_entities":["NIFTY","BANKNIFTY"],"rationale":"Border escalation coupled with heavy institutional selling triggers defensive positioning."}'
    },
]


# Module-level rotator shared by both agent functions
_rotator = ModelRotator()


async def _call_gemini_with_rotation(
    prompt: str,
    system: str,
    gri_composite: float,
    generation_config: Any,
    history: Optional[List[Dict]] = None,
    response_mime_type: str = "text/plain",
    allow_pro: bool = False,
) -> tuple:
    """
    Calls Gemini with automatic model rotation on 429 / 404 errors.
    Returns (response_text, model_display_name).

    allow_pro=True: passed for RiskManagerAgent calls so the rotator may
    select a Pro-tier model when quota allows (Pro reserved for high-risk
    contextualisation; Flash used for all routine sentiment calls).
    """
    slots_tried = 0
    last_exc = None

    while slots_tried < len(_rotator._slots):
        slot = _rotator.pick_model(gri_composite, allow_pro=allow_pro)
        if slot is None:
            # All models on cooldown — raise immediately so callers use fallback
            raise RuntimeError("All Gemini models on cooldown — quota exhausted.")
        slots_tried += 1

        try:
            client = _get_genai_client()  # B-12 FIX: use singleton, not per-call creation
            loop   = asyncio.get_running_loop()

            temp    = getattr(generation_config, 'temperature', 0.15)
            max_tok = getattr(generation_config, 'max_output_tokens', 1024)

            def _generate():
                config = genai_types.GenerateContentConfig(
                    system_instruction=system if slot.supports_system_instruction else None,
                    temperature=temp,
                    max_output_tokens=max_tok,
                    response_mime_type=response_mime_type,
                )
                prompt_to_send = prompt
                if not slot.supports_system_instruction:
                    prompt_to_send = f"{system}\n\n---\n{prompt}"
                
                return client.models.generate_content(
                    model=slot.api_name, contents=prompt_to_send, config=config
                )

            response = await loop.run_in_executor(None, _generate)
            _rotator.mark_success(slot)
            return response.text.strip(), slot.display

        except Exception as exc:
            last_exc = exc
            err_msg  = str(exc)

            # Quota/Rate Limit handling
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                # Distinguish daily (RPD) exhaustion from per-minute (RPM) rate limit.
                #
                # Google's standard phrases:
                #   RPM → "generate_requests_per_minute"  (contains "per_minute")
                #   RPD → "generate_requests_per_day"     (contains "per_day")
                #
                # DO NOT use "quota" alone — both RPM and RPD errors contain it.
                # Treating RPM hits as daily would give Flash models a 24h ban
                # from a simple per-minute rate limit, permanently disabling them.
                #
                # Fallback heuristics for short-form error messages that only say
                # "RESOURCE_EXHAUSTED" without the quota-metric detail:
                #   1. calls_made == 0 → we just started; can't have hit RPM yet
                #      (RPM requires prior successful calls to fill the window)
                #      → must be daily exhaustion from a previous run.
                #   2. errors_429 >= 3 → three consecutive 429s even after RPM
                #      cooldowns → window keeps exhausting → treat as daily.
                err_lower = err_msg.lower()
                is_daily = (
                    "per_day"       in err_lower    # Google standard RPD phrase
                    or "daily"      in err_lower    # generic / older SDK phrasing
                    or "limit: 0"   in err_msg      # free-tier: zero quota allocated
                    or "limit: 0.0" in err_msg
                    # Heuristics for opaque "RESOURCE_EXHAUSTED" messages:
                    or slot.calls_made == 0         # no successes → can't be RPM
                    or slot.errors_429 >= 3         # repeated 429s → escalate
                )
                _rotator.mark_rate_limited(slot, is_daily=is_daily)
                logger.warning("⟳ Model %s rate-limited%s — rotating.",
                               slot.display, " (daily → 24h)" if is_daily else " (RPM → 65s)")
                await asyncio.sleep(5)
                continue

            # Not Found handling
            if "404" in err_msg or "NOT_FOUND" in err_msg:
                _rotator.mark_not_found(slot)
                logger.warning("⟳ Model %s not found — disabling.", slot.display)
                continue

            # Other errors — treat as temporary RPM-style cooldown (is_daily=False)
            logger.warning("⟳ Model %s error: %s — trying next.", slot.display, exc)
            _rotator.mark_rate_limited(slot, is_daily=False)
            await asyncio.sleep(5)
            continue

    raise RuntimeError(f"All Gemini models exhausted. Last error: {last_exc}")


async def sentiment_analyzer_agent(state: AgentState) -> AgentState:
    """
    Node 2: Sends enriched headlines + GPR context to Gemini for sentiment scoring.
    """
    t0 = time.perf_counter()

    headlines = state["headlines"]
    gpr       = state["gpr_snapshot"]
    vol       = state["vol_regime"]
    vix       = state["current_vix"]
    gri       = float(gpr.get("gpr_normalised", 0.3))

    headline_block = "\n".join(f"• {h}" for h in headlines[:20]) or "No headlines available."
    prompt = (
        f"Context:\nHeadlines:\n{headline_block}\n\n"
        f"GPR Index: {gpr.get('gpr_index','?')} ({gpr.get('label','?')}) | "
        f"VIX: {vix:.1f} | Vol regime: {vol}"
    )

    gen_cfg = genai_types.GenerateContentConfig(temperature=0.15, max_output_tokens=1024)

    try:
        raw_text, model_used = await _call_gemini_with_rotation(
            prompt=prompt, system=_SENTIMENT_SYSTEM,
            gri_composite=gri, generation_config=gen_cfg,
            response_mime_type="application/json",
        )
        
        # Clean up markdown if model ignored JSON mode instructions
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        # Robust Truncation Repair
        if raw_text and not raw_text.endswith("}"):
            logger.debug("SentimentAnalyzerAgent: Attempting JSON repair on truncated response.")
            # 1. Close open string if odd number of quotes
            if raw_text.count('"') % 2 != 0:
                raw_text += '"'
            
            # 2. Backtrack to last comma to discard partial key/value
            last_comma = raw_text.rfind(",")
            if last_comma > 0:
                raw_text = raw_text[:last_comma]
            
            # 3. Close braces
            open_braces = raw_text.count("{") - raw_text.count("}")
            if open_braces > 0:
                raw_text += "}" * open_braces
            logger.debug("Repaired JSON: %s", raw_text)

        parsed = json.loads(raw_text)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "SentimentAnalyzerAgent [%s]: class=%s score=%.3f conf=%.2f (%.0f ms)",
            model_used,
            parsed.get("sentiment_classification"),
            parsed.get("sentiment_score", 0),
            parsed.get("confidence", 0),
            latency_ms,
        )

        parsed["latency_ms"]  = latency_ms
        parsed["model_used"]  = model_used
        return {**state, "sentiment_raw": parsed, "iterations": state["iterations"] + 1}

    except Exception as exc:
        logger.warning("SentimentAnalyzerAgent parse/logic error: %s", exc)
        return {**state, "sentiment_raw": {
                "sentiment_score": 0.0,
                "sentiment_classification": "Neutral",
                "rationale": f"Error: {str(exc)[:50]}", 
                "key_entities": [],
                "confidence": 0.0, "latency_ms": 0.0},
                "error": str(exc), "iterations": state["iterations"] + 1}


# ---------------------------------------------------------------------------
# Agent Node 3: RiskManagerAgent (conditional)
# ---------------------------------------------------------------------------

_RISK_SYSTEM = """You are a quantitative risk manager at a Mumbai-based prop trading desk.
You receive a preliminary sentiment score and must contextualise it against the current
volatility regime, VIX level, and geopolitical risk profile.

Output a single paragraph (2-3 sentences) describing:
1. Whether the sentiment score should be AMPLIFIED, ATTENUATED, or MAINTAINED given vol context
2. The specific risk or opportunity this creates for NSE equity positioning
3. A recommended action bias: INCREASE_CONVICTION | REDUCE_CONVICTION | MAINTAIN | AVOID

Keep it concise. Do not output JSON.
"""


async def risk_manager_agent(state: AgentState) -> AgentState:
    """
    Node 3 (conditional): Contextualises sentiment against vol regime.
    Only invoked when sentiment conviction is high OR vol regime is stressed.
    """
    t0  = time.perf_counter()
    raw = state["sentiment_raw"]
    gpr = state["gpr_snapshot"]

    prompt = (
        f"Sentiment: {raw.get('sentiment_classification')} "
        f"(score={raw.get('sentiment_score', 0):.3f}, "
        f"confidence={raw.get('confidence', 0):.2f})\n"
        f"Rationale: {raw.get('rationale', '')}\n\n"
        f"Market Context:\n"
        f"  Vol regime:   {state['vol_regime']}\n"
        f"  India VIX:    {state['current_vix']:.1f}\n"
        f"  GPR label:    {gpr.get('label', '?')}\n"
        f"  GPRT (threats): {gpr.get('gprt', '?')} | "
        f"GPRA (acts): {gpr.get('gpra', '?')}\n"
        f"  Threat bias:  {gpr.get('threat_label', '?')}"
    )

    gri = float(state.get("gpr_snapshot", {}).get("gpr_normalised", 0.3))
    gen_cfg = genai_types.GenerateContentConfig(temperature=0.25, max_output_tokens=512)
    try:
        context, model_used = await _call_gemini_with_rotation(
            prompt=prompt, system=_RISK_SYSTEM,
            gri_composite=gri, generation_config=gen_cfg,
            allow_pro=True,   # RiskManager may use Pro model if quota allows
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("RiskManagerAgent [%s]: completed in %.0f ms", model_used, latency_ms)
        return {**state, "risk_context": context, "iterations": state["iterations"] + 1}

    except Exception as exc:
        logger.warning("RiskManagerAgent error (non-fatal): %s", exc)
        return {**state, "risk_context": "Risk context unavailable.",
                "iterations": state["iterations"] + 1}


# ---------------------------------------------------------------------------
# Conditional Router
# ---------------------------------------------------------------------------

def _route_after_sentiment(state: AgentState) -> str:
    """
    Conditional edge: decides whether to invoke RiskManagerAgent.
    Routes to 'risk_manager' if conviction is high OR vol is stressed.
    Routes to 'assemble' otherwise (skip risk manager for neutral signals).
    """
    raw    = state.get("sentiment_raw", {})
    score  = abs(float(raw.get("sentiment_score", 0.0)))
    conf   = float(raw.get("confidence", 0.0))
    vol    = state.get("vol_regime", "LOW")
    stress = state.get("current_vix", 15.0)
    error  = state.get("error", "")

    if error:
        return "assemble"   # Don't call risk manager if sentiment failed

    high_conviction = score >= HIGH_CONVICTION_THRESHOLD and conf >= 0.55
    high_vol        = vol in ("HIGH", "EXTREME") or stress > 22.0

    if high_conviction or high_vol:
        if _rotator.can_afford_risk_manager():
            logger.debug(
                "Router → risk_manager (score=%.3f vol=%s vix=%.1f)",
                score, vol, stress,
            )
            return "risk_manager"
        logger.info(
            "Router → assemble: conviction/vol warrants risk_manager but "
            "no Pro quota remaining — skipping to conserve Flash budget."
        )
        return "assemble"

    logger.debug("Router → assemble (low conviction / low vol — skipping risk manager)")
    return "assemble"


# ---------------------------------------------------------------------------
# Final Assembler Node
# ---------------------------------------------------------------------------

def assemble_result(state: AgentState) -> AgentState:
    """
    Collects outputs from all agent nodes and builds the final SentimentResult.
    Applies risk context to adjust the final sentiment score if available.
    """
    raw          = state.get("sentiment_raw", {})
    risk_context = state.get("risk_context", "")

    score = float(raw.get("sentiment_score", 0.0))
    cls   = raw.get("sentiment_classification", "Neutral")
    conf  = float(raw.get("confidence", 0.0))

    # Apply risk context attenuation: if risk manager flags REDUCE_CONVICTION
    # or AVOID, dampen the final score by 40%
    if risk_context:
        if "REDUCE_CONVICTION" in risk_context:
            score *= 0.60
            logger.debug("RiskManager attenuated score by 40%%.")
        elif "AVOID" in risk_context:
            score *= 0.20
            logger.debug("RiskManager strongly attenuated score (AVOID).")
        elif "INCREASE_CONVICTION" in risk_context:
            score = max(-1.0, min(1.0, score * 1.20))
            logger.debug("RiskManager amplified score by 20%%.")

    final = {
        "sentiment_classification": cls,
        "sentiment_score":          round(score, 4),
        "rationale":                raw.get("rationale", ""),
        "key_entities":             raw.get("key_entities", []),
        "confidence":               conf,
        "risk_context":             risk_context,
        "vol_regime":               state.get("vol_regime", ""),
        "gpr_label":                state["gpr_snapshot"].get("label", ""),
    }
    return {**state, "final_result": final}


# ---------------------------------------------------------------------------
# LangGraph StateGraph Construction
# ---------------------------------------------------------------------------

def _build_graph():
    """
    Builds and compiles the LangGraph StateGraph.
    Nodes: data_retriever → sentiment_analyzer → (conditional) → assemble
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.error(
            "LangGraph not installed. Run: pip install langgraph\n"
            "Falling back to linear (non-graph) execution mode."
        )
        return None

    graph = StateGraph(AgentState)

    graph.add_node("data_retriever",   data_retriever_agent)
    graph.add_node("sentiment_analyzer", sentiment_analyzer_agent)
    graph.add_node("risk_manager",     risk_manager_agent)
    graph.add_node("assemble",         assemble_result)

    graph.set_entry_point("data_retriever")
    graph.add_edge("data_retriever", "sentiment_analyzer")

    # Conditional routing after sentiment analysis
    graph.add_conditional_edges(
        "sentiment_analyzer",
        _route_after_sentiment,
        {
            "risk_manager": "risk_manager",
            "assemble":     "assemble",
        }
    )

    graph.add_edge("risk_manager", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()


# Lazily compiled graph (None if langgraph not installed)
_COMPILED_GRAPH = None


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()
    return _COMPILED_GRAPH


# ---------------------------------------------------------------------------
# AgentPipeline — public interface
# ---------------------------------------------------------------------------

class AgentPipeline:
    """
    Main entry point for the V2 multi-agent LLM architecture.
    Accepts a snapshot of inputs and returns a SentimentResult.

    If LangGraph is not installed, falls back to sequential node execution
    (same logic, no graph runtime overhead).
    """

    def __init__(self) -> None:
        self._graph   = _get_graph()
        self._rotator = _rotator
        if self._graph is None:
            logger.warning(
                "AgentPipeline running in FALLBACK mode (LangGraph not installed)."
            )

    def recommended_interval(self, gri_composite: float = 0.3, vix: float = 18.0) -> int:
        """Delegate to rotator — used by sentiment_loop in main.py."""
        return self._rotator.recommended_interval_s(gri_composite, vix)

    def model_stats(self) -> Dict[str, Any]:
        """Return per-model usage stats for logging."""
        return self._rotator.daily_stats()

    async def run_analysis_cycle(
        self,
        headlines:      List[str],
        gpr:            GPRSnapshot,
        vol_regime:     str   = "MODERATE",
        current_vix:    float = 15.0,
        gri_composite:  float = 0.3,
    ) -> SentimentResult:
        """
        Execute the full agent workflow and return a SentimentResult.

        Args:
            headlines:   Latest news headlines
            gpr:         Current GPR snapshot
            vol_regime:  Volatility regime label
            current_vix: India VIX scalar

        Returns:
            SentimentResult with sentiment score, rationale, risk context
        """
        state = _initial_state(headlines, gpr, vol_regime, current_vix)

        try:
            if self._graph is not None:
                # Use ainvoke — required for async node functions in LangGraph
                try:
                    final_state = await self._graph.ainvoke(state)
                except (AttributeError, TypeError):
                    logger.warning("LangGraph ainvoke unavailable — using sequential fallback.")
                    final_state = None
                if final_state is None:
                    final_state = data_retriever_agent(state)
                    final_state = await sentiment_analyzer_agent(final_state)
                    route       = _route_after_sentiment(final_state)
                    if route == "risk_manager":
                        final_state = await risk_manager_agent(final_state)
                    final_state = assemble_result(final_state)
            else:
                # No LangGraph — sequential execution
                final_state = data_retriever_agent(state)
                final_state = await sentiment_analyzer_agent(final_state)
                route       = _route_after_sentiment(final_state)
                if route == "risk_manager":
                    final_state = await risk_manager_agent(final_state)
                final_state = assemble_result(final_state)

            result_dict  = final_state.get("final_result") or {}
            raw          = final_state.get("sentiment_raw", {})

            return SentimentResult(
                sentiment_classification = result_dict.get("sentiment_classification", "Neutral"),
                sentiment_score          = float(result_dict.get("sentiment_score", 0.0)),
                rationale                = result_dict.get("rationale", ""),
                key_entities             = result_dict.get("key_entities", []),
                source_articles          = headlines[:5],
                model_latency_ms         = float(raw.get("latency_ms", 0.0)),
                risk_context             = result_dict.get("risk_context", ""),
                volatility_context       = vol_regime,
                gpr_context              = result_dict.get("gpr_label", ""),
                confidence               = float(result_dict.get("confidence", 0.0)),
            )

        except Exception as exc:
            logger.error("AgentPipeline error: %s", exc, exc_info=True)
            return SentimentResult.neutral()
