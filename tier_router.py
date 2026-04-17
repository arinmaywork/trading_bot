from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from regime_detector import MarketRegime

logger = logging.getLogger(__name__)


class StrategyTier(Enum):
    """Enum for trading strategy tiers."""
    TIER1_MICRO = "TIER1_MICRO"           # Microstructure scalper
    TIER2_MEANREV = "TIER2_MEANREV"       # Mean-reversion swinger
    TIER3_SENTIMENT = "TIER3_SENTIMENT"   # Sentiment swing (CNC)
    NO_TRADE = "NO_TRADE"                 # Don't trade


@dataclass
class TierConfig:
    """Configuration for strategy tier routing and capital allocation."""
    tier1_enabled: bool = True
    tier2_enabled: bool = True
    tier3_enabled: bool = True
    # Universe sizes per tier
    tier1_universe_size: int = 8
    tier2_universe_size: int = 20
    tier3_universe_size: int = 50
    # Capital allocation per tier (must sum to <= 1.0)
    tier1_capital_pct: float = 0.30
    tier2_capital_pct: float = 0.50
    tier3_capital_pct: float = 0.20


class TierRouter:
    """
    Routes trading signals to appropriate strategy tier based on market regime and time.

    Routing logic:
    - TRENDING: Tier 1 (micro) and Tier 3 (sentiment)
    - MEAN_REVERTING: Tier 1 and Tier 2 (mean-reversion)
    - VOLATILE: Tier 3 only (contrarian, reduced sizing)
    - UNKNOWN: Tier 1 and Tier 2

    Time-based overrides:
    - Lunch (12:00-13:00): Tier 1 & 2 OFF, Tier 3 allowed
    - First 5 min (9:15-9:20): Tier 1 & 2 OFF, Tier 3 allowed
    - Last 15 min (15:15-15:30): Tier 1 only (momentum into close)
    """

    def __init__(self, config: TierConfig = None):
        """
        Initialize tier router.

        Args:
            config: TierConfig instance. Uses defaults if None.
        """
        self._config = config or TierConfig()

        # Validate capital allocation
        total_pct = (
            self._config.tier1_capital_pct +
            self._config.tier2_capital_pct +
            self._config.tier3_capital_pct
        )
        if total_pct > 1.0:
            logger.warning(
                f"Total capital allocation {total_pct:.2%} exceeds 100%. "
                f"Will normalize proportionally."
            )

        logger.info(
            f"TierRouter initialized: Tier1={self._config.tier1_enabled}, "
            f"Tier2={self._config.tier2_enabled}, Tier3={self._config.tier3_enabled}"
        )

    def route(
        self,
        regime: MarketRegime,
        hour: int,
        minute: int,
        is_no_trade_zone: bool = False
    ) -> list[StrategyTier]:
        """
        Determine active trading tiers for current market conditions.

        Routing table by regime:
                         | TRENDING | MEAN_REVERTING | VOLATILE | UNKNOWN |
        Tier 1 (micro)   |   ON     |      ON        |   OFF    |   ON    |
        Tier 2 (MR)      |   OFF    |      ON        |   OFF    |   ON    |
        Tier 3 (sent)    |   ON     |     OFF        |   ON*    |   OFF   |

        * Tier 3 in VOLATILE = contrarian, reduced sizing

        Time-based overrides:
        - Lunch (12:00-12:59): Tier 1 & 2 OFF, Tier 3 allowed
        - First 5 min (9:15-9:19): Tier 1 & 2 OFF, Tier 3 allowed
        - Last 15 min (15:15-15:29): Tier 1 only (momentum into close)

        Args:
            regime: Current market regime from RegimeDetector
            hour: Current hour (0-23)
            minute: Current minute (0-59)
            is_no_trade_zone: Override to disable all trading

        Returns:
            List of active StrategyTier enums
        """
        if is_no_trade_zone:
            logger.debug("In no-trade zone, returning empty tier list")
            return [StrategyTier.NO_TRADE]

        # Import here to avoid circular dependency
        from regime_detector import MarketRegime

        # Check for time-based overrides
        in_lunch = 12 <= hour < 13
        in_open_5min = hour == 9 and 15 <= minute < 20
        in_close_15min = hour == 15 and 15 <= minute < 30

        logger.debug(
            f"Time check: lunch={in_lunch}, open_5min={in_open_5min}, "
            f"close_15min={in_close_15min}"
        )

        # Last 15 minutes: only Tier 1 (momentum into close)
        if in_close_15min:
            active = []
            if self._config.tier1_enabled:
                active.append(StrategyTier.TIER1_MICRO)
            logger.info(f"In last 15 min: routing to {active}")
            return active if active else [StrategyTier.NO_TRADE]

        # Lunch and first 5 min: Tier 1 & 2 OFF, Tier 3 allowed
        if in_lunch or in_open_5min:
            active = []
            if self._config.tier3_enabled:
                active.append(StrategyTier.TIER3_SENTIMENT)
            logger.debug(
                f"In {'lunch' if in_lunch else 'open 5min'}: routing to {active}"
            )
            return active if active else [StrategyTier.NO_TRADE]

        # Regime-based routing
        active = []

        if regime == MarketRegime.TRENDING:
            if self._config.tier1_enabled:
                active.append(StrategyTier.TIER1_MICRO)
            if self._config.tier3_enabled:
                active.append(StrategyTier.TIER3_SENTIMENT)

        elif regime == MarketRegime.MEAN_REVERTING:
            if self._config.tier1_enabled:
                active.append(StrategyTier.TIER1_MICRO)
            if self._config.tier2_enabled:
                active.append(StrategyTier.TIER2_MEANREV)

        elif regime == MarketRegime.VOLATILE:
            # Tier 3 in VOLATILE = contrarian, reduced sizing
            if self._config.tier3_enabled:
                active.append(StrategyTier.TIER3_SENTIMENT)

        elif regime == MarketRegime.UNKNOWN:
            if self._config.tier1_enabled:
                active.append(StrategyTier.TIER1_MICRO)
            if self._config.tier2_enabled:
                active.append(StrategyTier.TIER2_MEANREV)

        logger.info(
            f"Routed {regime.value}: active tiers = {[t.value for t in active]}"
        )

        return active if active else [StrategyTier.NO_TRADE]

    def get_capital_allocation(
        self,
        active_tiers: list[StrategyTier],
        total_capital: float
    ) -> dict[StrategyTier, float]:
        """
        Calculate capital allocation across active tiers.

        If some tiers are inactive, their capital share is redistributed
        proportionally among active tiers.

        Args:
            active_tiers: List of active StrategyTier enums
            total_capital: Total capital to allocate (in currency units)

        Returns:
            Dictionary mapping StrategyTier to allocated capital
        """
        allocation = {}

        # Base allocations
        tier_allocations = {
            StrategyTier.TIER1_MICRO: self._config.tier1_capital_pct,
            StrategyTier.TIER2_MEANREV: self._config.tier2_capital_pct,
            StrategyTier.TIER3_SENTIMENT: self._config.tier3_capital_pct,
        }

        # Filter to only active tiers
        active_allocations = {
            tier: pct
            for tier, pct in tier_allocations.items()
            if tier in active_tiers
        }

        if not active_allocations:
            logger.warning("No active tiers for capital allocation")
            return allocation

        # Calculate total allocation for active tiers
        total_active_pct = sum(active_allocations.values())

        # Normalize to 100% of active tiers
        if total_active_pct > 0:
            for tier in active_allocations:
                normalized_pct = active_allocations[tier] / total_active_pct
                allocation[tier] = total_capital * normalized_pct
                logger.debug(
                    f"{tier.value}: {normalized_pct:.1%} = "
                    f"{allocation[tier]:.2f} ({total_capital:.2f} total)"
                )
        else:
            logger.error("Total active allocation is zero")

        return allocation

    def get_universe_size(self, tier: StrategyTier) -> int:
        """
        Return target universe size for a given strategy tier.

        Args:
            tier: StrategyTier enum

        Returns:
            Target number of symbols/positions for this tier
        """
        if tier == StrategyTier.TIER1_MICRO:
            return self._config.tier1_universe_size
        elif tier == StrategyTier.TIER2_MEANREV:
            return self._config.tier2_universe_size
        elif tier == StrategyTier.TIER3_SENTIMENT:
            return self._config.tier3_universe_size
        else:
            logger.warning(f"Unknown tier {tier}, returning 0")
            return 0
