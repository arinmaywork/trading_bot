"""
Model Health Monitoring Module

Tracks ML model calibration and predictive quality. Provides:
- CalibrationTracker: Monitors predicted vs actual returns alignment
- FeatureImportanceTracker: Tracks feature contribution over time
- Helper functions for statistical computations (numpy-free)

All computations use stdlib math only for maximum portability.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CalibrationSnapshot:
    """Represents model calibration metrics at a point in time."""
    symbol: str
    timestamp: float
    calibration_slope: float
    information_coefficient: float
    sharpe_ratio: float
    n_samples: int
    status: str  # "HEALTHY", "DEGRADED", "BROKEN"

    def __repr__(self) -> str:
        return (
            f"CalibrationSnapshot(symbol={self.symbol}, "
            f"status={self.status}, ic={self.information_coefficient:.4f}, "
            f"slope={self.calibration_slope:.4f}, n={self.n_samples})"
        )


@dataclass
class FeatureImportanceRecord:
    """Records feature importance from a model training/retraining cycle."""
    timestamp: float
    symbol: str
    importances: dict[str, float]


# =============================================================================
# Helper Functions (numpy-free)
# =============================================================================


def _mean(values: list[float]) -> float:
    """Compute arithmetic mean. Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float], mean_val: Optional[float] = None) -> float:
    """
    Compute population variance.

    Args:
        values: List of numeric values
        mean_val: Pre-computed mean (computed if None)

    Returns:
        Population variance. Returns 0.0 for empty or single-element lists.
    """
    if len(values) <= 1:
        return 0.0

    if mean_val is None:
        mean_val = _mean(values)

    sum_sq_diff = sum((x - mean_val) ** 2 for x in values)
    return sum_sq_diff / len(values)


def _covariance(x: list[float], y: list[float]) -> float:
    """
    Compute covariance between two equal-length sequences.

    Args:
        x: First sequence
        y: Second sequence

    Returns:
        Covariance. Returns 0.0 if sequences empty or length mismatch.
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    mean_x = _mean(x)
    mean_y = _mean(y)

    sum_product = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    return sum_product / len(x)


def _rank(values: list[float]) -> list[float]:
    """
    Assign ranks to values, handling ties with average rank.

    Args:
        values: List of numeric values (duplicates allowed)

    Returns:
        List of ranks (1-indexed). Values are left in original order;
        ties receive their average rank.

    Example:
        >>> _rank([10.0, 20.0, 10.0])
        [1.5, 3.0, 1.5]
    """
    if not values:
        return []

    # Create (value, original_index) pairs
    indexed = [(v, i) for i, v in enumerate(values)]

    # Sort by value
    indexed.sort(key=lambda x: x[0])

    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        # Find range of equal values
        j = i
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1

        # Assign average rank to ties
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank

        i = j

    return ranks


def _linear_regression_slope(x: list[float], y: list[float]) -> float:
    """
    Compute slope of linear regression y = slope * x + intercept.

    Uses the formula: slope = cov(x, y) / var(x)

    Args:
        x: Predictor sequence
        y: Response sequence (must equal length to x)

    Returns:
        Regression slope. Returns 0.0 for invalid inputs:
        - Length mismatch
        - Empty sequences
        - Zero variance in x (constant predictor)
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0

    var_x = _variance(x)
    if var_x == 0.0:
        return 0.0

    cov_xy = _covariance(x, y)
    return cov_xy / var_x


def _spearman_rank_correlation(x: list[float], y: list[float]) -> float:
    """
    Compute Spearman rank correlation coefficient (Information Coefficient).

    Uses the formula: IC = 1 - 6*sum(d_i^2) / (n*(n^2-1))
    where d_i is the rank difference for observation i.

    Args:
        x: First sequence
        y: Second sequence (must equal length to x)

    Returns:
        Spearman correlation in range [-1, 1]. Returns 0.0 for:
        - Length mismatch
        - Empty sequences
        - Single-element sequences (undefined for rank correlation)
    """
    if len(x) != len(y) or len(x) <= 1:
        return 0.0

    n = len(x)
    ranks_x = _rank(x)
    ranks_y = _rank(y)

    # Compute sum of squared rank differences
    sum_sq_diff = sum((rx - ry) ** 2 for rx, ry in zip(ranks_x, ranks_y))

    denominator = n * (n * n - 1)
    if denominator == 0:
        return 0.0

    return 1.0 - (6.0 * sum_sq_diff) / denominator


def _sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio from returns.

    Sharpe = (mean_return - risk_free_rate) / std_return

    Args:
        returns: Sequence of returns
        risk_free_rate: Risk-free rate (default 0.0)

    Returns:
        Sharpe ratio. Returns 0.0 for:
        - Empty or single-element sequences
        - Zero standard deviation (constant returns)
    """
    if len(returns) <= 1:
        return 0.0

    mean_ret = _mean(returns)
    var_ret = _variance(returns, mean_ret)

    if var_ret == 0.0:
        return 0.0

    std_ret = math.sqrt(var_ret)
    return (mean_ret - risk_free_rate) / std_ret


# =============================================================================
# CalibrationTracker
# =============================================================================


class CalibrationTracker:
    """
    Tracks model calibration by comparing predicted vs actual returns.

    Maintains health status (HEALTHY/DEGRADED/BROKEN) based on:
    - Information Coefficient (Spearman rank correlation)
    - Calibration slope (regression slope of predicted vs actual)
    - Minimum sample size requirement

    Used to dynamically adjust position sizing based on model reliability.
    """

    def __init__(
        self,
        degraded_ic_threshold: float = 0.01,
        broken_ic_threshold: float = 0.005,
        degraded_slope_threshold: float = 0.3,
        broken_slope_threshold: float = 0.1,
        min_samples: int = 20,
    ):
        """
        Initialize calibration tracker with thresholds.

        Args:
            degraded_ic_threshold: IC level below which status is DEGRADED
            broken_ic_threshold: IC level below which status is BROKEN
            degraded_slope_threshold: Slope below which status is DEGRADED
            broken_slope_threshold: Slope below which status is BROKEN
            min_samples: Minimum samples required for calibration snapshot
        """
        self.degraded_ic_threshold = degraded_ic_threshold
        self.broken_ic_threshold = broken_ic_threshold
        self.degraded_slope_threshold = degraded_slope_threshold
        self.broken_slope_threshold = broken_slope_threshold
        self.min_samples = min_samples

        self._snapshots: dict[str, CalibrationSnapshot] = {}

        logger.info(
            f"CalibrationTracker initialized: "
            f"degraded_ic={degraded_ic_threshold}, "
            f"broken_ic={broken_ic_threshold}, "
            f"degraded_slope={degraded_slope_threshold}, "
            f"broken_slope={broken_slope_threshold}"
        )

    def update(
        self,
        symbol: str,
        predicted: list[float],
        actual: list[float],
    ) -> CalibrationSnapshot:
        """
        Update calibration snapshot for a symbol.

        Computes calibration metrics from holdout predictions and actual returns,
        then determines health status.

        Args:
            symbol: Asset symbol identifier
            predicted: List of predicted returns
            actual: List of actual returns (must match predicted length)

        Returns:
            CalibrationSnapshot with computed metrics and health status

        Raises:
            ValueError: If predicted and actual have different lengths
        """
        if len(predicted) != len(actual):
            raise ValueError(
                f"Length mismatch: predicted={len(predicted)}, "
                f"actual={len(actual)}"
            )

        n_samples = len(predicted)
        timestamp = datetime.utcnow().timestamp()

        # Handle minimum sample requirement
        if n_samples < self.min_samples:
            logger.warning(
                f"{symbol}: Insufficient samples ({n_samples} < {self.min_samples})"
            )
            snapshot = CalibrationSnapshot(
                symbol=symbol,
                timestamp=timestamp,
                calibration_slope=0.0,
                information_coefficient=0.0,
                sharpe_ratio=0.0,
                n_samples=n_samples,
                status="UNKNOWN",
            )
            self._snapshots[symbol] = snapshot
            return snapshot

        # Compute metrics
        slope = _linear_regression_slope(predicted, actual)
        ic = _spearman_rank_correlation(predicted, actual)
        sharpe = _sharpe_ratio(actual)

        # Determine health status
        if ic > self.degraded_ic_threshold and slope > self.degraded_slope_threshold:
            status = "HEALTHY"
        elif ic > self.broken_ic_threshold and slope > self.broken_slope_threshold:
            status = "DEGRADED"
        else:
            status = "BROKEN"

        snapshot = CalibrationSnapshot(
            symbol=symbol,
            timestamp=timestamp,
            calibration_slope=slope,
            information_coefficient=ic,
            sharpe_ratio=sharpe,
            n_samples=n_samples,
            status=status,
        )

        self._snapshots[symbol] = snapshot

        logger.info(
            f"{symbol}: {status} "
            f"(IC={ic:.6f}, slope={slope:.4f}, sharpe={sharpe:.4f}, n={n_samples})"
        )

        return snapshot

    def get_status(self, symbol: str) -> str:
        """
        Get health status for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            Status string: "HEALTHY", "DEGRADED", "BROKEN", or "UNKNOWN"
        """
        if symbol not in self._snapshots:
            return "UNKNOWN"
        return self._snapshots[symbol].status

    def get_sizing_multiplier(self, symbol: str) -> float:
        """
        Get position sizing multiplier based on model health.

        Multiplier controls how aggressively to trade based on model confidence:
        - HEALTHY (1.0): Full position size
        - DEGRADED (0.5): Half position size
        - BROKEN (0.0): No trading
        - UNKNOWN (0.7): Cautious default

        Args:
            symbol: Asset symbol

        Returns:
            Multiplier in range [0.0, 1.0]
        """
        status = self.get_status(symbol)

        multipliers = {
            "HEALTHY": 1.0,
            "DEGRADED": 0.5,
            "BROKEN": 0.0,
            "UNKNOWN": 0.7,
        }

        return multipliers.get(status, 0.7)

    def all_snapshots(self) -> dict[str, CalibrationSnapshot]:
        """
        Get all latest calibration snapshots.

        Returns:
            Dictionary mapping symbol -> CalibrationSnapshot
        """
        return self._snapshots.copy()

    def format_health(self, top_n: int = 10) -> str:
        """
        Format health status as HTML for Telegram display.

        Shows top N symbols by Information Coefficient.
        Uses emoji status indicators and displays key metrics.

        Args:
            top_n: Number of top symbols to display

        Returns:
            HTML-formatted string ready for Telegram
        """
        if not self._snapshots:
            return "<b>📊 Model Health</b><pre>No data</pre>"

        # Sort by IC descending
        sorted_snapshots = sorted(
            self._snapshots.values(),
            key=lambda s: s.information_coefficient,
            reverse=True,
        )[:top_n]

        status_emoji = {
            "HEALTHY": "🟢",
            "DEGRADED": "🟡",
            "BROKEN": "🔴",
            "UNKNOWN": "⚪",
        }

        lines = ["<b>📊 Model Health</b><pre>"]
        lines.append(f"{'Symbol':<8} {'Status':<12} {'IC':>8} {'Slope':>8} {'N':>6}")
        lines.append("-" * 48)

        for snap in sorted_snapshots:
            emoji = status_emoji.get(snap.status, "⚪")
            status_str = f"{emoji} {snap.status}"

            lines.append(
                f"{snap.symbol:<8} {status_str:<12} "
                f"{snap.information_coefficient:>8.4f} "
                f"{snap.calibration_slope:>8.4f} "
                f"{snap.n_samples:>6}"
            )

        lines.append("</pre>")

        return "\n".join(lines)


# =============================================================================
# FeatureImportanceTracker
# =============================================================================


class FeatureImportanceTracker:
    """
    Tracks feature importance over time across model retraining cycles.

    Maintains historical records of feature contributions and computes
    rolling averages to smooth out noise from individual training runs.

    Useful for identifying which features drive predictions and detecting
    when feature relevance shifts over time.
    """

    def __init__(self, max_records_per_symbol: int = 100):
        """
        Initialize feature importance tracker.

        Args:
            max_records_per_symbol: Maximum records to keep per symbol
                (older records are discarded when exceeded)
        """
        self.max_records_per_symbol = max_records_per_symbol
        self._records: dict[str, list[FeatureImportanceRecord]] = {}

        logger.info(
            f"FeatureImportanceTracker initialized: "
            f"max_records_per_symbol={max_records_per_symbol}"
        )

    def record(self, symbol: str, importances: dict[str, float]) -> None:
        """
        Record feature importance from a model training cycle.

        Args:
            symbol: Asset symbol
            importances: Dictionary mapping feature_name -> importance_value

        Note:
            If number of records exceeds max_records_per_symbol, oldest
            records are automatically discarded.
        """
        if symbol not in self._records:
            self._records[symbol] = []

        timestamp = datetime.utcnow().timestamp()
        record = FeatureImportanceRecord(
            timestamp=timestamp,
            symbol=symbol,
            importances=importances.copy(),
        )

        self._records[symbol].append(record)

        # Trim old records
        if len(self._records[symbol]) > self.max_records_per_symbol:
            self._records[symbol] = self._records[symbol][-self.max_records_per_symbol:]

        logger.debug(
            f"{symbol}: Recorded {len(importances)} features, "
            f"total records={len(self._records[symbol])}"
        )

    def avg_importance(
        self,
        symbol: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Compute average feature importance across recent records.

        Args:
            symbol: Specific symbol to average (None = all symbols)

        Returns:
            Dictionary mapping feature_name -> average_importance,
            sorted by importance (highest first). Empty dict if no records.
        """
        # Collect all relevant records
        all_records = []

        if symbol is None:
            # Aggregate across all symbols
            for records in self._records.values():
                all_records.extend(records)
        else:
            # Single symbol
            all_records = self._records.get(symbol, [])

        if not all_records:
            return {}

        # Aggregate importances by feature name
        feature_totals: dict[str, float] = {}
        feature_counts: dict[str, int] = {}

        for record in all_records:
            for feature_name, importance in record.importances.items():
                if feature_name not in feature_totals:
                    feature_totals[feature_name] = 0.0
                    feature_counts[feature_name] = 0

                feature_totals[feature_name] += importance
                feature_counts[feature_name] += 1

        # Compute averages
        avg_importances = {
            feature: feature_totals[feature] / feature_counts[feature]
            for feature in feature_totals
        }

        # Sort by importance (descending)
        sorted_importances = dict(
            sorted(
                avg_importances.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        return sorted_importances

    def format_importance(
        self,
        symbol: Optional[str] = None,
        top_n: int = 8,
    ) -> str:
        """
        Format feature importance as HTML text-based bar chart for Telegram.

        Args:
            symbol: Specific symbol (None = aggregate all)
            top_n: Number of top features to display

        Returns:
            HTML-formatted string with feature importance visualization
        """
        importances = self.avg_importance(symbol)

        if not importances:
            scope = f"({symbol})" if symbol else "(all)"
            return (
                f"<b>🔬 Feature Importance {scope}</b><pre>No data</pre>"
            )

        # Get top N features
        top_features = list(importances.items())[:top_n]

        # Find max importance for scaling bar length
        max_importance = max(imp for _, imp in top_features) if top_features else 1.0
        if max_importance == 0:
            max_importance = 1.0

        scope = f"({symbol})" if symbol else "(all)"
        lines = [
            f"<b>🔬 Feature Importance {scope}</b><pre>",
        ]

        # Add each feature with bar
        for feature_name, importance in top_features:
            # Scale bar to 20 characters max
            bar_length = int((importance / max_importance) * 20)
            bar = "█" * bar_length

            lines.append(
                f"{feature_name:<20} {bar:<20} {importance:>7.4f}"
            )

        lines.append("</pre>")

        return "\n".join(lines)


# =============================================================================
# Module initialization
# =============================================================================


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test CalibrationTracker
    ct = CalibrationTracker()

    predicted = [0.01, 0.02, -0.01, 0.03, 0.015, 0.025, -0.005, 0.02,
                 0.01, 0.03, 0.02, 0.015, 0.025, 0.01, 0.02, -0.01,
                 0.03, 0.015, 0.025, -0.005, 0.02, 0.01, 0.03, 0.02]
    actual = [0.009, 0.018, -0.011, 0.028, 0.012, 0.022, -0.008, 0.019,
              0.011, 0.031, 0.021, 0.013, 0.026, 0.009, 0.021, -0.012,
              0.032, 0.014, 0.024, -0.007, 0.021, 0.011, 0.029, 0.019]

    snap = ct.update("BTC", predicted, actual)
    print(f"BTC: {snap}")
    print(f"Multiplier: {ct.get_sizing_multiplier('BTC')}")
    print(ct.format_health())

    # Test FeatureImportanceTracker
    fit = FeatureImportanceTracker()

    fit.record("BTC", {
        "mlofi": 0.231,
        "aflow_ratio": 0.178,
        "vwap_z": 0.134,
        "spread_z": 0.112,
        "rsi": 0.085,
    })

    fit.record("BTC", {
        "mlofi": 0.225,
        "aflow_ratio": 0.182,
        "vwap_z": 0.140,
        "spread_z": 0.108,
        "rsi": 0.088,
    })

    print(fit.format_importance("BTC", top_n=5))
