"""
Phase 0 signal diagnostics and filter funnel tracking.

Standalone module (no codebase imports except logging) for tracking signal flow
through the strategy pipeline, trade attribution, and signal distribution analysis.
Designed to be wired into strategy.py and telegram_controller.py for debugging
signal quality and filter effectiveness.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, stdev
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class FunnelSnapshot:
    """Counts at each filter stage for a single evaluation cycle."""

    timestamp: float = 0.0
    total_symbols_evaluated: int = 0
    signals_generated: int = 0  # ML model produced a non-FLAT signal
    passed_alpha_gate: int = 0  # |signal| > alpha threshold
    passed_rsi_filter: int = 0  # Not vetoed by RSI
    passed_trend_alignment: int = 0  # Price vs VWAP direction match
    passed_cost_hurdle: int = 0  # Expected P&L > hurdle * cost
    passed_sector_cap: int = 0  # Sector exposure still under cap
    passed_cooldown: int = 0  # Symbol not in cooldown
    passed_risk_halt: int = 0  # No daily/weekly/monthly halt
    # Block stages (signal rejected at a specific gate)
    blocked_nobuy_guard: int = 0     # Buy blocked by no-buy guard
    blocked_nosell_guard: int = 0    # Sell blocked by no-sell guard
    blocked_risk_blacklist: int = 0  # Symbol on risk blacklist
    blocked_news_blackout: int = 0   # Blocked by news blackout
    blocked_cooldown: int = 0        # Symbol still in order cooldown
    blocked_position_guard_long: int = 0   # Already long, can't buy more
    blocked_position_guard_short: int = 0  # Already short, can't sell more
    blocked_sector_cap: int = 0      # Sector exposure cap breached
    blocked_max_positions: int = 0   # Tune-3: Max concurrent positions limit reached
    passed_all_gates: int = 0  # Cleared every filter — about to execute
    orders_executed: int = 0  # Actually placed an order


class FilterFunnel:
    """
    Tracks how many signals pass through each stage of the filter pipeline per day.

    Provides insight into filter effectiveness and signal quality by recording
    pass-through counts at each stage of the strategy pipeline.
    """

    def __init__(self):
        """Initialize funnel with empty counts and no session date."""
        self._daily_totals = FunnelSnapshot()
        self._session_date: str = ""

    def record(self, stage: str, count: int = 1) -> None:
        """
        Increment a specific stage counter.

        Args:
            stage: Field name in FunnelSnapshot (e.g., "signals_generated", "passed_alpha_gate")
            count: Number to increment (default 1)

        Raises:
            ValueError: If stage is not a valid FunnelSnapshot field
        """
        if not hasattr(self._daily_totals, stage):
            # Graceful fallback: log a warning but do NOT raise.
            # Diagnostics must never crash the trading loop.  Unknown stages
            # are stored in the overflow dict so they still appear in /funnel.
            if not hasattr(self, '_overflow'):
                self._overflow: Dict[str, int] = {}
            self._overflow[stage] = self._overflow.get(stage, 0) + count
            logger.warning("FilterFunnel: unknown stage '%s' (overflow +%d)", stage, count)
            return

        current = getattr(self._daily_totals, stage)
        setattr(self._daily_totals, stage, current + count)
        logger.debug(f"Recorded {count} to stage '{stage}' (now {current + count})")

    def reset_if_new_day(self, date_str: str) -> None:
        """
        Reset counters at the start of a new trading day.

        Args:
            date_str: Date string in YYYY-MM-DD format
        """
        if date_str != self._session_date:
            logger.info(
                f"FilterFunnel: rolling over from {self._session_date} to {date_str}"
            )
            self._session_date = date_str
            self._daily_totals = FunnelSnapshot(
                timestamp=datetime.now().timestamp()
            )

    def snapshot(self) -> FunnelSnapshot:
        """
        Return current day's funnel counts.

        Returns:
            FunnelSnapshot with current stage counts
        """
        return FunnelSnapshot(
            timestamp=self._daily_totals.timestamp,
            total_symbols_evaluated=self._daily_totals.total_symbols_evaluated,
            signals_generated=self._daily_totals.signals_generated,
            passed_alpha_gate=self._daily_totals.passed_alpha_gate,
            passed_rsi_filter=self._daily_totals.passed_rsi_filter,
            passed_trend_alignment=self._daily_totals.passed_trend_alignment,
            passed_cost_hurdle=self._daily_totals.passed_cost_hurdle,
            passed_sector_cap=self._daily_totals.passed_sector_cap,
            passed_cooldown=self._daily_totals.passed_cooldown,
            passed_risk_halt=self._daily_totals.passed_risk_halt,
            orders_executed=self._daily_totals.orders_executed,
        )

    def format_funnel(self) -> str:
        """
        Format funnel snapshot as HTML for Telegram display.

        Shows each stage with absolute count and pass-through rate (percentage
        of previous stage that passed to current stage).

        Returns:
            HTML-formatted string suitable for Telegram
        """
        snap = self.snapshot()

        lines = [
            "<code>",
            f"📊 <b>Filter Funnel ({self._session_date})</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        # Symbols evaluated
        lines.append(f"Symbols evaluated:  {snap.total_symbols_evaluated}")

        # Signals generated
        pct = (
            100.0 * snap.signals_generated / snap.total_symbols_evaluated
            if snap.total_symbols_evaluated > 0
            else 0.0
        )
        lines.append(f"Signals generated:  {snap.signals_generated:3d}  ({pct:5.1f}%)")

        # Alpha gate
        pct = (
            100.0 * snap.passed_alpha_gate / snap.signals_generated
            if snap.signals_generated > 0
            else 0.0
        )
        lines.append(f"→ Alpha gate:       {snap.passed_alpha_gate:3d}  ({pct:5.1f}%)")

        # RSI filter
        pct = (
            100.0 * snap.passed_rsi_filter / snap.passed_alpha_gate
            if snap.passed_alpha_gate > 0
            else 0.0
        )
        lines.append(f"→ RSI filter:       {snap.passed_rsi_filter:3d}  ({pct:5.1f}%)")

        # Trend alignment
        pct = (
            100.0 * snap.passed_trend_alignment / snap.passed_rsi_filter
            if snap.passed_rsi_filter > 0
            else 0.0
        )
        lines.append(
            f"→ Trend alignment:  {snap.passed_trend_alignment:3d}  ({pct:5.1f}%)"
        )

        # Cost hurdle
        pct = (
            100.0 * snap.passed_cost_hurdle / snap.passed_trend_alignment
            if snap.passed_trend_alignment > 0
            else 0.0
        )
        lines.append(f"→ Cost hurdle:      {snap.passed_cost_hurdle:3d}  ({pct:5.1f}%)")

        # Sector cap
        pct = (
            100.0 * snap.passed_sector_cap / snap.passed_cost_hurdle
            if snap.passed_cost_hurdle > 0
            else 0.0
        )
        lines.append(f"→ Sector cap:       {snap.passed_sector_cap:3d}  ({pct:5.1f}%)")

        # Cooldown
        pct = (
            100.0 * snap.passed_cooldown / snap.passed_sector_cap
            if snap.passed_sector_cap > 0
            else 0.0
        )
        lines.append(f"→ Cooldown:         {snap.passed_cooldown:3d}  ({pct:5.1f}%)")

        # Risk halt
        pct = (
            100.0 * snap.passed_risk_halt / snap.passed_cooldown
            if snap.passed_cooldown > 0
            else 0.0
        )
        lines.append(f"→ Risk halt:        {snap.passed_risk_halt:3d}  ({pct:5.1f}%)")

        # Orders executed
        lines.append(f"✅ Orders executed: {snap.orders_executed:3d}")

        lines.append("</code>")
        return "\n".join(lines)


@dataclass
class TradeRecord:
    """Record of a single trade entry with attribution data and eventual exit data."""

    timestamp: float
    symbol: str
    direction: str  # BUY or SELL
    ml_signal: float
    ml_confidence: float
    tier: str  # TIER1_MICRO, TIER2_MEANREV, TIER3_SENTIMENT
    regime: str  # TRENDING, MEAN_REVERTING, VOLATILE
    top_features: list = field(default_factory=list)  # [(name, importance), ...]
    predicted_return: float = 0.0
    actual_return: float = 0.0
    gross_pnl: float = 0.0
    cost: float = 0.0


class TradeAttribution:
    """
    Logs per-trade feature attribution and computes summary statistics.

    Tracks entry and exit data for each trade, including ML signal, predicted
    return, actual return, and top contributing features. Provides attribution
    analysis to understand which features drive wins/losses and whether
    predictions are well-calibrated.
    """

    def __init__(self, max_records: int = 500):
        """
        Initialize trade attribution logger.

        Args:
            max_records: Maximum number of trades to keep in memory (FIFO)
        """
        self._records: list[TradeRecord] = []
        self._max = max_records
        logger.info(f"TradeAttribution initialized with max_records={max_records}")

    def record_entry(
        self,
        symbol: str,
        direction: str,
        ml_signal: float,
        ml_confidence: float,
        tier: str,
        regime: str,
        top_features: list,
        predicted_return: float,
    ) -> int:
        """
        Record a trade entry.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            direction: "BUY" or "SELL"
            ml_signal: Model signal magnitude and sign
            ml_confidence: Confidence score [0, 1]
            tier: Tier classification (TIER1_MICRO, TIER2_MEANREV, TIER3_SENTIMENT)
            regime: Market regime (TRENDING, MEAN_REVERTING, VOLATILE)
            top_features: List of (feature_name, importance) tuples
            predicted_return: Expected return from model

        Returns:
            Index of recorded trade (for later update via record_exit)
        """
        record = TradeRecord(
            timestamp=datetime.now().timestamp(),
            symbol=symbol,
            direction=direction,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            tier=tier,
            regime=regime,
            top_features=top_features,
            predicted_return=predicted_return,
        )

        self._records.append(record)

        # Enforce max_records FIFO
        if len(self._records) > self._max:
            self._records.pop(0)

        idx = len(self._records) - 1
        logger.debug(f"Recorded trade entry at index {idx}: {symbol} {direction}")
        return idx

    def record_exit(
        self, index: int, actual_return: float, gross_pnl: float, cost: float
    ) -> None:
        """
        Update trade record with exit data.

        Args:
            index: Record index from record_entry
            actual_return: Actual realized return
            gross_pnl: Gross profit/loss in currency
            cost: Transaction cost/slippage
        """
        if 0 <= index < len(self._records):
            self._records[index].actual_return = actual_return
            self._records[index].gross_pnl = gross_pnl
            self._records[index].cost = cost
            logger.debug(
                f"Recorded exit for index {index}: "
                f"actual_return={actual_return:.4f}, pnl={gross_pnl:.2f}"
            )
        else:
            logger.warning(f"Invalid record index {index} in record_exit")

    def summary(self, last_n: int = 50) -> dict:
        """
        Compute attribution summary statistics for recent trades.

        Args:
            last_n: Number of most recent trades to analyze

        Returns:
            Dictionary with:
            - win_rate: Fraction of trades with actual_return > 0
            - avg_predicted: Mean predicted return across trades
            - avg_actual: Mean actual return across trades
            - calibration_ratio: avg_actual / avg_predicted (1.0 = perfect)
            - correct_direction: Fraction where direction of actual matched prediction sign
            - pnl_gross: Total gross P&L
            - pnl_net: Gross P&L minus costs
            - pnl_by_tier: {tier -> net_pnl}
            - pnl_by_regime: {regime -> net_pnl}
            - top_features_in_winners: Most common features in profitable trades
            - top_features_in_losers: Most common features in losing trades
        """
        subset = self._records[-last_n:] if last_n > 0 else self._records

        if not subset:
            return {
                "count": 0,
                "win_rate": 0.0,
                "avg_predicted": 0.0,
                "avg_actual": 0.0,
                "calibration_ratio": 0.0,
                "correct_direction": 0.0,
                "pnl_gross": 0.0,
                "pnl_net": 0.0,
                "pnl_by_tier": {},
                "pnl_by_regime": {},
                "top_features_in_winners": [],
                "top_features_in_losers": [],
            }

        # Basic stats
        count = len(subset)
        wins = sum(1 for r in subset if r.actual_return > 0)
        win_rate = wins / count if count > 0 else 0.0

        predicted_returns = [r.predicted_return for r in subset]
        actual_returns = [r.actual_return for r in subset]

        avg_predicted = mean(predicted_returns) if predicted_returns else 0.0
        avg_actual = mean(actual_returns) if actual_returns else 0.0

        calibration_ratio = (
            avg_actual / avg_predicted if avg_predicted != 0 else 0.0
        )

        # Direction correctness
        correct_dir = sum(
            1
            for r in subset
            if (r.direction == "BUY" and r.actual_return > 0)
            or (r.direction == "SELL" and r.actual_return < 0)
        )
        correct_direction = correct_dir / count if count > 0 else 0.0

        # P&L aggregation
        pnl_gross = sum(r.gross_pnl for r in subset)
        pnl_net = sum(r.gross_pnl - r.cost for r in subset)

        # P&L by tier and regime
        pnl_by_tier = {}
        pnl_by_regime = {}

        for r in subset:
            if r.tier not in pnl_by_tier:
                pnl_by_tier[r.tier] = 0.0
            pnl_by_tier[r.tier] += r.gross_pnl - r.cost

            if r.regime not in pnl_by_regime:
                pnl_by_regime[r.regime] = 0.0
            pnl_by_regime[r.regime] += r.gross_pnl - r.cost

        # Top features in winners vs losers
        winner_features = {}
        loser_features = {}

        for r in subset:
            is_winner = r.actual_return > 0
            feature_dict = winner_features if is_winner else loser_features

            for feature_name, importance in r.top_features:
                if feature_name not in feature_dict:
                    feature_dict[feature_name] = 0.0
                feature_dict[feature_name] += importance

        # Sort by importance and return top 5
        top_winners = (
            sorted(winner_features.items(), key=lambda x: x[1], reverse=True)[:5]
            if winner_features
            else []
        )
        top_losers = (
            sorted(loser_features.items(), key=lambda x: x[1], reverse=True)[:5]
            if loser_features
            else []
        )

        return {
            "count": count,
            "win_rate": win_rate,
            "avg_predicted": avg_predicted,
            "avg_actual": avg_actual,
            "calibration_ratio": calibration_ratio,
            "correct_direction": correct_direction,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "pnl_by_tier": pnl_by_tier,
            "pnl_by_regime": pnl_by_regime,
            "top_features_in_winners": top_winners,
            "top_features_in_losers": top_losers,
        }

    def format_attribution(self, last_n: int = 20) -> str:
        """
        Format recent trade attribution as HTML for Telegram display.

        Shows summary statistics, P&L breakdown, and feature attribution
        for the most recent trades.

        Args:
            last_n: Number of recent trades to include

        Returns:
            HTML-formatted string suitable for Telegram
        """
        stats = self.summary(last_n)

        if stats["count"] == 0:
            return "<code>📈 Trade Attribution: No trades recorded yet</code>"

        lines = [
            "<code>",
            f"📈 <b>Trade Attribution (last {stats['count']} trades)</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Win rate:           {stats['win_rate']*100:5.1f}%",
            f"Direction correct:  {stats['correct_direction']*100:5.1f}%",
            f"Calibration ratio:  {stats['calibration_ratio']:6.3f}x",
            f"Avg predicted ret:  {stats['avg_predicted']:7.3f}%",
            f"Avg actual ret:     {stats['avg_actual']:7.3f}%",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Gross P&L:          ${stats['pnl_gross']:10.2f}",
            f"Net P&L:            ${stats['pnl_net']:10.2f}",
        ]

        # P&L by tier
        if stats["pnl_by_tier"]:
            lines.append("P&L by Tier:")
            for tier, pnl in sorted(stats["pnl_by_tier"].items()):
                lines.append(f"  {tier:20s} ${pnl:10.2f}")

        # P&L by regime
        if stats["pnl_by_regime"]:
            lines.append("P&L by Regime:")
            for regime, pnl in sorted(stats["pnl_by_regime"].items()):
                lines.append(f"  {regime:20s} ${pnl:10.2f}")

        # Top features
        if stats["top_features_in_winners"]:
            lines.append("Top features (winners):")
            for feat, importance in stats["top_features_in_winners"][:3]:
                lines.append(f"  {feat:20s} {importance:8.4f}")

        if stats["top_features_in_losers"]:
            lines.append("Top features (losers):")
            for feat, importance in stats["top_features_in_losers"][:3]:
                lines.append(f"  {feat:20s} {importance:8.4f}")

        lines.append("</code>")
        return "\n".join(lines)


class SignalDistribution:
    """
    Tracks the distribution of absolute ML signal values for adaptive gating.

    Maintains a rolling window of signal magnitudes and computes percentiles
    to inform dynamic alpha thresholds and signal quality analysis.
    """

    def __init__(self, window: int = 1000):
        """
        Initialize signal distribution tracker.

        Args:
            window: Maximum number of signal values to keep in history
        """
        self._values: list[float] = []
        self._window = window
        logger.info(f"SignalDistribution initialized with window={window}")

    def add(self, abs_signal: float) -> None:
        """
        Add a signal observation to the distribution.

        Args:
            abs_signal: Absolute value of ML signal
        """
        if abs_signal < 0:
            logger.warning(f"SignalDistribution.add received negative value {abs_signal}")
            abs_signal = abs(abs_signal)

        self._values.append(abs_signal)

        # Enforce rolling window (FIFO)
        if len(self._values) > self._window:
            self._values.pop(0)

        logger.debug(f"Added signal {abs_signal:.4f} (now {len(self._values)} in buffer)")

    def percentile(self, pct: float) -> float:
        """
        Return the given percentile of stored signal values.

        Args:
            pct: Percentile value in [0, 1] (e.g., 0.5 for median)

        Returns:
            Percentile value, or 0.0 if no data
        """
        if not self._values:
            return 0.0

        if not (0 <= pct <= 1):
            raise ValueError(f"Percentile must be in [0, 1], got {pct}")

        sorted_vals = sorted(self._values)
        idx = int(pct * (len(sorted_vals) - 1))
        return float(sorted_vals[idx])

    def stats(self) -> dict:
        """
        Compute summary statistics for the signal distribution.

        Returns:
            Dictionary with keys: count, mean, std, p25, p50, p75, p90, p95, p99
        """
        if not self._values:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        count = len(self._values)
        mean_val = mean(self._values)
        std_val = stdev(self._values) if count > 1 else 0.0

        return {
            "count": count,
            "mean": mean_val,
            "std": std_val,
            "p25": self.percentile(0.25),
            "p50": self.percentile(0.50),
            "p75": self.percentile(0.75),
            "p90": self.percentile(0.90),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
        }

    def format_distribution(self) -> str:
        """
        Format signal distribution statistics as HTML for Telegram display.

        Shows percentile breakdown and mean/std to assess signal quality
        and help inform alpha threshold settings.

        Returns:
            HTML-formatted string suitable for Telegram
        """
        stats = self.stats()

        if stats["count"] == 0:
            return "<code>📉 Signal Distribution: No signals recorded yet</code>"

        lines = [
            "<code>",
            f"📉 <b>Signal Distribution ({stats['count']} signals)</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Mean:               {stats['mean']:8.4f}",
            f"Std dev:            {stats['std']:8.4f}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"P25:                {stats['p25']:8.4f}",
            f"P50 (median):       {stats['p50']:8.4f}",
            f"P75:                {stats['p75']:8.4f}",
            f"P90:                {stats['p90']:8.4f}",
            f"P95:                {stats['p95']:8.4f}",
            f"P99:                {stats['p99']:8.4f}",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Range:              [{self.percentile(0.0):.4f}, {self.percentile(1.0):.4f}]",
            "</code>",
        ]

        return "\n".join(lines)
