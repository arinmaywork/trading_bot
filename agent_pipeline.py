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
    supports_system_instruction: bool = True   # False for Gemma models
    # runtime state
    calls_made:      int   = 0
    last_call_ts:    float = 0.0
    errors_429:      int   = 0
    cooldown_until:  float = 0.0   # epoch seconds


class ModelRotator:
    """
    Manages a priority-ordered pool of Gemini models.

    Strategy:
      • Quality-first: always tries highest-quality model first.
      • Rate-aware: respects per-model RPM; backs off on 429.
      • Budget-aware: distributes call budget across the NSE trading day
        (09:15–15:30 IST, 375 min) so the pool never runs dry.
      • GRI-adaptive: uses best models during high-risk / opening / close
        windows; lighter models during calm mid-session.

    Call-frequency tiers (returned by `recommended_interval_s`):
      EXTREME GRI  → 120 s   (max responsiveness, opens all models)
      HIGH GRI     → 180 s
      ELEVATED     → 300 s   (default)
      MODERATE     → 420 s
      LOW + calm   → 600 s   (conserve budget)
      Opening      → 120 s   (09:15–09:45, always)
      Pre-close    → 150 s   (15:00–15:30, always)
    """

    # Models in priority order (best quality first).
    # API names verified against Google AI Studio free-tier model list.
    _POOL: List[_ModelSlot] = [
        _ModelSlot("gemini-1.5-pro",             "1.5 Pro",         5,  2),
        _ModelSlot("gemini-2.5-pro",             "2.5 Pro",         5,  2),
        _ModelSlot("gemini-2.5-flash",           "2.5 Flash",       4,  5),
        _ModelSlot("gemini-2.0-flash",           "2.0 Flash",       4, 10),
        _ModelSlot("gemini-1.5-flash",           "1.5 Flash",       3, 15),
    ]

    # Seconds to cool down after a 429 (doubles each consecutive error)
    _BASE_COOLDOWN_S = 300  # Increase base cooldown to 5 min

    def __init__(self) -> None:
        # Deep-copy slots so instance state is independent
        import copy
        self._slots: List[_ModelSlot] = copy.deepcopy(self._POOL)
        self._call_log: List[float] = []   # epoch timestamps of all calls
        logger.info(
            "ModelRotator initialised with %d models: %s",
            len(self._slots),
            [s.display for s in self._slots],
        )

    # ── Slot selection ────────────────────────────────────────────────────

    def _slot_available(self, slot: _ModelSlot) -> bool:
        """Return True if slot is not in cooldown and within RPM."""
        now = time.time()
        if now < slot.cooldown_until:
            return False
        return True

    def all_on_cooldown(self) -> bool:
        """Return True when every model slot is still rate-limited."""
        return not any(self._slot_available(s) for s in self._slots)

    def soonest_available_in(self) -> float:
        """Seconds until the first slot comes off cooldown (0 if one is free)."""
        now = time.time()
        waits = [max(0.0, s.cooldown_until - now) for s in self._slots]
        return min(waits)

    def pick_model(self, gri_composite: float = 0.3) -> _ModelSlot:
        """
        Pick the best available model for the current context.
        """
        now   = time.time()
        ist   = datetime.fromtimestamp(now, tz=_IST)
        hour  = ist.hour + ist.minute / 60.0

        is_opening   = 9.25 <= hour <= 9.75    # 09:15 – 09:45
        is_pre_close = 15.0 <= hour <= 15.5    # 15:00 – 15:30
        is_high_risk = gri_composite >= 0.50
        use_best     = is_opening or is_pre_close or is_high_risk

        available = [s for s in self._slots if self._slot_available(s)]
        if not available:
            # All slots still on cooldown — do NOT force a doomed API call.
            soonest = self.soonest_available_in()
            logger.warning(
                "All Gemini models on cooldown — skipping call (soonest available in %.0fs).",
                soonest,
            )
            return None  # type: ignore[return-value]

        if use_best or len(available) == 1:
            chosen = available[0]
        else:
            # Use second-tier model to preserve premium quota for high-risk moments
            chosen = available[min(1, len(available) - 1)]

        return chosen

    def mark_success(self, slot: _ModelSlot) -> None:
        slot.calls_made   += 1
        slot.last_call_ts  = time.time()
        slot.errors_429    = 0
        self._call_log.append(time.time())
        logger.debug("ModelRotator ✓ %s  (total calls: %d)", slot.display, slot.calls_made)

    def mark_rate_limited(self, slot: _ModelSlot, is_permanent: bool = False) -> None:
        """Back off exponentially on 429 or limit:0."""
        slot.errors_429 += 1
        if is_permanent:
            cooldown = 86400 # 24h
        else:
            cooldown = self._BASE_COOLDOWN_S * (2 ** min(slot.errors_429 - 1, 4))
        slot.cooldown_until = time.time() + cooldown
        logger.warning(
            "ModelRotator 429/Limit on %s — cooldown %.0fs  (error #%d)",
            slot.display, cooldown, slot.errors_429,
        )

    def mark_not_found(self, slot: _ModelSlot) -> None:
        """Permanently disable a slot (404 / model not available in this project)."""
        slot.cooldown_until = time.time() + 86400  # 24h effectively disabled
        logger.error(
            "ModelRotator: %s returned 404 — disabling for today.", slot.display
        )

    # ── Interval recommendation ───────────────────────────────────────────

    def recommended_interval_s(
        self,
        gri_composite: float = 0.3,
        vix:           float = 18.0,
    ) -> int:
        """
        Return the recommended sleep interval between sentiment calls.

        Balances responsiveness vs API budget conservation.
        NSE trading day = 375 minutes. At 300s base → 75 calls/day.
        """
        now  = time.time()
        ist  = datetime.fromtimestamp(now, tz=_IST)
        hour = ist.hour + ist.minute / 60.0

        # Market window priority overrides
        if 9.25 <= hour <= 9.75:    return 120   # Opening 30 min
        if 15.0 <= hour <= 15.5:    return 120   # Pre-close 30 min

        # GRI-based
        if gri_composite >= 0.65:   return 120   # EXTREME
        if gri_composite >= 0.50:   return 180   # HIGH
        if gri_composite >= 0.30:   return 300   # ELEVATED (default)
        if gri_composite >= 0.15:   return 420   # MODERATE
        return 600                               # LOW — conserve budget

    def daily_stats(self) -> Dict[str, Any]:
        return {
            s.display: {
                "calls": s.calls_made,
                "errors_429": s.errors_429,
                "available": self._slot_available(s),
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
) -> tuple:
    """
    Calls Gemini with automatic model rotation on 429 / 404 errors.
    Returns (response_text, model_display_name).
    """
    slots_tried = 0
    last_exc = None

    while slots_tried < len(_rotator._slots):
        slot = _rotator.pick_model(gri_composite)
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
                # Check for "limit: 0" which means permanent lack of quota for this model
                is_perm = "limit: 0" in err_msg or "limit: 0.0" in err_msg
                _rotator.mark_rate_limited(slot, is_permanent=is_perm)
                logger.warning("⟳ Model %s rate-limited%s — rotating.", 
                               slot.display, " (PERMANENT)" if is_perm else "")
                await asyncio.sleep(2)
                continue

            # Not Found handling
            if "404" in err_msg or "NOT_FOUND" in err_msg:
                _rotator.mark_not_found(slot)
                logger.warning("⟳ Model %s not found — disabling.", slot.display)
                continue

            # Other errors
            logger.warning("⟳ Model %s error: %s — trying next.", slot.display, exc)
            _rotator.mark_rate_limited(slot) # treat as temporary
            await asyncio.sleep(2)
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
        logger.debug("Router → risk_manager (score=%.3f vol=%s vix=%.1f)",
                     score, vol, stress)
        return "risk_manager"

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
