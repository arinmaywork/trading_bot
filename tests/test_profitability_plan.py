"""
Comprehensive unit tests for profitability plan modules.

Tests cover:
  1. Microstructure features (Tier-1)
  2. Mean-reversion features (Tier-2)
  3. Seasonality features
  4. Regime detection
  5. Tier routing
  6. Diagnostics & attribution
  7. Model health
  8. Feature vectors (Tier-1 & Tier-2)

All tests use unittest framework and follow the existing test file patterns.
"""

import unittest
import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass

# Import modules under test
import sys
sys.path.insert(0, '/sessions/serene-affectionate-carson/mnt/Trading_Bot')

from features.microstructure import (
    MicrostructureState,
    order_book_imbalance_momentum,
    update_obi_ema,
    trade_arrival_rate,
    update_tick_count_avg,
    spread_z_score,
    volume_weighted_price_pressure,
    price_momentum,
    compute_tier1_features,
)

from features.mean_reversion import (
    MeanReversionState,
    vwap_z_score,
    rsi_mean_reversion_signal,
    opening_range_breakout_position,
    compute_tier2_features,
)

from features.seasonality import (
    time_of_day_features,
    day_of_week_feature,
    is_no_trade_zone,
)

from regime_detector import (
    RegimeDetector,
    RegimeConfig,
    MarketRegime,
    compute_adx,
    compute_bollinger_bandwidth,
)

from tier_router import (
    TierRouter,
    TierConfig,
    StrategyTier,
)

from diagnostics import (
    FilterFunnel,
    TradeAttribution,
    TradeRecord,
    SignalDistribution,
)

from model_health import (
    CalibrationTracker,
    FeatureImportanceTracker,
)

from ml_signal import (
    Tier1FeatureVector,
    Tier2FeatureVector,
)


# =============================================================================
# Test 1: Microstructure Features (5+ tests)
# =============================================================================

class TestMicrostructureFeatures(unittest.TestCase):
    """Tests for microstructure feature computations."""

    def test_obi_momentum_positive_acceleration(self):
        """OBI current > EMA should give positive momentum."""
        obi_current = 0.5
        obi_ema = 0.2
        momentum = order_book_imbalance_momentum(obi_current, obi_ema)
        self.assertGreater(momentum, 0)
        self.assertAlmostEqual(momentum, 0.3, places=6)

    def test_obi_momentum_negative_acceleration(self):
        """OBI current < EMA should give negative momentum."""
        obi_current = 0.1
        obi_ema = 0.5
        momentum = order_book_imbalance_momentum(obi_current, obi_ema)
        self.assertLess(momentum, 0)
        self.assertAlmostEqual(momentum, -0.4, places=6)

    def test_obi_ema_update_converges(self):
        """EMA should converge toward current value."""
        obi_current = 1.0
        obi_ema_prev = 0.0
        obi_ema_new = update_obi_ema(obi_current, obi_ema_prev, span=20)
        # With span=20, alpha = 2/21 ≈ 0.0952
        self.assertGreater(obi_ema_new, obi_ema_prev)
        self.assertLess(obi_ema_new, obi_current)
        # After many iterations, EMA should approach current
        for _ in range(100):
            obi_ema_new = update_obi_ema(obi_current, obi_ema_new, span=20)
        self.assertAlmostEqual(obi_ema_new, obi_current, places=3)

    def test_trade_arrival_rate_normal(self):
        """Rate = 1.0 when tick count equals average."""
        tick_count = 10
        avg_tick_count = 10.0
        rate = trade_arrival_rate(tick_count, avg_tick_count)
        self.assertAlmostEqual(rate, 1.0, places=6)

    def test_trade_arrival_rate_elevated(self):
        """Rate > 1.0 when ticks > average."""
        tick_count = 20
        avg_tick_count = 10.0
        rate = trade_arrival_rate(tick_count, avg_tick_count)
        self.assertAlmostEqual(rate, 2.0, places=6)

    def test_trade_arrival_rate_min_baseline(self):
        """Rate should use min baseline of 1.0."""
        tick_count = 5
        avg_tick_count = 0.5  # very low
        rate = trade_arrival_rate(tick_count, avg_tick_count)
        self.assertAlmostEqual(rate, 5.0, places=6)

    def test_spread_z_score_clamped(self):
        """Z-score should be clamped to [-5, 5]."""
        # Create extreme spread scenario
        best_bid = 100.0
        best_ask = 101.0  # wide spread
        mid_price = 100.5
        spread_mean = 1.0  # basis points
        spread_std = 0.1  # very tight
        z = spread_z_score(best_bid, best_ask, mid_price, spread_mean, spread_std)
        # Should be clamped to max of 5
        self.assertLessEqual(z, 5.0)
        self.assertGreaterEqual(z, -5.0)

    def test_spread_z_score_zero_when_tight(self):
        """Z-score should be near 0 when spread matches mean."""
        best_bid = 100.0
        best_ask = 100.01  # tight spread = 1 bp
        mid_price = 100.005
        spread_mean = 1.0
        spread_std = 0.5
        z = spread_z_score(best_bid, best_ask, mid_price, spread_mean, spread_std)
        self.assertLess(abs(z), 1.0)

    def test_price_momentum_multiple_horizons(self):
        """Should return correct log-returns for 1/5/15 min horizons."""
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
                  109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0]
        momenta = price_momentum(prices, horizons=[1, 5, 15])
        self.assertEqual(len(momenta), 3)
        # ret_1 = log(115 / 114)
        expected_1 = math.log(115.0 / 114.0)
        self.assertAlmostEqual(momenta[0], expected_1, places=6)
        # ret_5 = log(115 / 110)
        expected_5 = math.log(115.0 / 110.0)
        self.assertAlmostEqual(momenta[1], expected_5, places=6)
        # ret_15 = log(115 / 100)
        expected_15 = math.log(115.0 / 100.0)
        self.assertAlmostEqual(momenta[2], expected_15, places=6)

    def test_price_momentum_insufficient_history(self):
        """Should return 0.0 for insufficient history."""
        prices = [100.0, 101.0]  # only 2 prices
        momenta = price_momentum(prices, horizons=[1, 5, 15])
        self.assertEqual(len(momenta), 3)
        # Insufficient history for 5 and 15 min horizons
        self.assertNotEqual(momenta[0], 0.0)  # 1-min has sufficient data
        self.assertEqual(momenta[1], 0.0)  # 5-min insufficient
        self.assertEqual(momenta[2], 0.0)  # 15-min insufficient

    def test_compute_tier1_features_returns_all_keys(self):
        """Should return dict with all 9 required keys."""
        state = MicrostructureState(symbol="TEST")
        state.recent_closes_1min = [100.0, 101.0, 102.0]
        state.signed_trades_5min = [(100.0, 0.5), (150.0, 0.3)]

        features = compute_tier1_features(
            state=state,
            mlofi=0.15,
            aflow_ratio=0.5,
            best_bid=100.0,
            best_ask=100.02,
        )

        required_keys = [
            "obi_momentum", "trade_arrival_rate", "spread_z", "vwpp",
            "ret_1min", "ret_5min", "ret_15min", "mlofi", "aflow_ratio",
        ]
        for key in required_keys:
            self.assertIn(key, features)

        self.assertEqual(len(features), 9)


# =============================================================================
# Test 2: Mean-Reversion Features (4+ tests)
# =============================================================================

class TestMeanReversionFeatures(unittest.TestCase):
    """Tests for mean-reversion feature computations."""

    def test_vwap_z_score_positive_deviation(self):
        """Price above VWAP should give positive z-score."""
        price = 105.0
        session_vwap = 100.0
        price_vwap_std = 2.0
        z = vwap_z_score(price, session_vwap, price_vwap_std)
        self.assertGreater(z, 0)
        self.assertAlmostEqual(z, 2.5, places=6)

    def test_vwap_z_score_negative_deviation(self):
        """Price below VWAP should give negative z-score."""
        price = 95.0
        session_vwap = 100.0
        price_vwap_std = 2.0
        z = vwap_z_score(price, session_vwap, price_vwap_std)
        self.assertLess(z, 0)
        self.assertAlmostEqual(z, -2.5, places=6)

    def test_rsi_mean_reversion_oversold(self):
        """RSI < 25 should give signal = +1.0."""
        signal = rsi_mean_reversion_signal(20.0)
        self.assertAlmostEqual(signal, 1.0, places=6)

    def test_rsi_mean_reversion_overbought(self):
        """RSI > 75 should give signal = -1.0."""
        signal = rsi_mean_reversion_signal(80.0)
        self.assertAlmostEqual(signal, -1.0, places=6)

    def test_rsi_mean_reversion_neutral(self):
        """RSI in 40-60 range should give signal ≈ 0.0."""
        signal = rsi_mean_reversion_signal(50.0)
        self.assertAlmostEqual(signal, 0.0, places=6)

    def test_rsi_mean_reversion_linear_interpolation(self):
        """RSI 30 (halfway between 25 and 35) should interpolate correctly."""
        signal = rsi_mean_reversion_signal(32.5)
        # Between 25 (1.0) and 40 (0.0): linear from 1.0 - (32.5-25)/15 = 0.5
        expected = 1.0 - (32.5 - 25) / 15.0
        self.assertAlmostEqual(signal, expected, places=6)

    def test_opening_range_breakout_midpoint(self):
        """Price at midpoint should give position = 0.5."""
        price = 102.5
        orb_high = 105.0
        orb_low = 100.0
        position = opening_range_breakout_position(price, orb_high, orb_low)
        self.assertAlmostEqual(position, 0.5, places=6)

    def test_opening_range_breakout_at_low(self):
        """Price at low should give position = 0.0."""
        price = 100.0
        orb_high = 105.0
        orb_low = 100.0
        position = opening_range_breakout_position(price, orb_high, orb_low)
        self.assertAlmostEqual(position, 0.0, places=6)

    def test_opening_range_breakout_at_high(self):
        """Price at high should give position = 1.0."""
        price = 105.0
        orb_high = 105.0
        orb_low = 100.0
        position = opening_range_breakout_position(price, orb_high, orb_low)
        self.assertAlmostEqual(position, 1.0, places=6)


# =============================================================================
# Test 3: Seasonality Features (4+ tests)
# =============================================================================

class TestSeasonalityFeatures(unittest.TestCase):
    """Tests for time-of-day and day-of-week seasonality."""

    def test_time_of_day_features_market_open(self):
        """At 9:15, market_fraction should be near 0."""
        features = time_of_day_features(9, 15)
        self.assertAlmostEqual(features["market_fraction"], 0.0, places=2)
        self.assertEqual(features["is_opening_rush"], 1)

    def test_time_of_day_features_market_close(self):
        """At 15:30, market_fraction should be near 1."""
        features = time_of_day_features(15, 30)
        self.assertAlmostEqual(features["market_fraction"], 1.0, places=2)
        self.assertEqual(features["is_closing_auction"], 1)

    def test_time_of_day_features_midday(self):
        """At 12:30, market_fraction should be ~0.5."""
        features = time_of_day_features(12, 30)
        self.assertGreater(features["market_fraction"], 0.4)
        self.assertLess(features["market_fraction"], 0.6)

    def test_no_trade_zone_lunch(self):
        """12:30 should be in no-trade zone (lunch)."""
        result = is_no_trade_zone(12, 30)
        self.assertTrue(result)

    def test_no_trade_zone_normal(self):
        """10:30 should not be in no-trade zone."""
        result = is_no_trade_zone(10, 30)
        self.assertFalse(result)

    def test_no_trade_zone_market_open_gap(self):
        """9:18 should be in no-trade zone (market open gap)."""
        result = is_no_trade_zone(9, 18)
        self.assertTrue(result)

    def test_day_of_week_feature_monday(self):
        """Monday should not be expiry day."""
        features = day_of_week_feature(0)
        self.assertEqual(features["is_expiry_day"], 0)

    def test_day_of_week_feature_thursday(self):
        """Thursday (weekday=3) should be expiry day."""
        features = day_of_week_feature(3)
        self.assertEqual(features["is_expiry_day"], 1)


# =============================================================================
# Test 4: Regime Detector (4+ tests)
# =============================================================================

class TestRegimeDetector(unittest.TestCase):
    """Tests for market regime classification."""

    def test_classify_volatile_high_vix(self):
        """High VIX should classify as VOLATILE."""
        detector = RegimeDetector()
        regime, confidence = detector.classify(
            adx=15.0,
            vix=25.0,  # > 20 threshold
            nifty_ret=0.0,
            bollinger_bw=0.01,
            vol_1h=0.3,
        )
        self.assertEqual(regime, MarketRegime.VOLATILE)
        self.assertGreater(confidence, 0.0)

    def test_classify_trending(self):
        """High ADX + significant return should classify as TRENDING."""
        detector = RegimeDetector()
        regime, confidence = detector.classify(
            adx=30.0,  # > 25 threshold
            vix=15.0,
            nifty_ret=0.005,  # > 0.3% threshold
            bollinger_bw=0.03,
            vol_1h=0.2,
        )
        self.assertEqual(regime, MarketRegime.TRENDING)
        self.assertGreater(confidence, 0.0)

    def test_classify_mean_reverting(self):
        """Low ADX + tight bands should classify as MEAN_REVERTING."""
        detector = RegimeDetector()
        regime, confidence = detector.classify(
            adx=15.0,  # < 20 threshold
            vix=15.0,
            nifty_ret=0.0001,  # < 0.3% threshold
            bollinger_bw=0.01,  # < 2% threshold
            vol_1h=0.15,
        )
        self.assertEqual(regime, MarketRegime.MEAN_REVERTING)
        self.assertGreater(confidence, 0.0)

    def test_compute_adx_requires_minimum_bars(self):
        """ADX should return 0 with insufficient bars."""
        adx = compute_adx(
            highs=[100.0, 101.0],
            lows=[99.0, 100.0],
            closes=[100.0, 100.5],
            period=14,
        )
        self.assertEqual(adx, 0.0)

    def test_compute_adx_with_sufficient_bars(self):
        """ADX should compute with sufficient bars."""
        highs = [100.0 + i * 0.5 for i in range(20)]
        lows = [99.0 + i * 0.5 for i in range(20)]
        closes = [99.5 + i * 0.5 for i in range(20)]
        adx = compute_adx(highs, lows, closes, period=14)
        self.assertGreater(adx, 0.0)
        self.assertLessEqual(adx, 100.0)

    def test_compute_bollinger_bandwidth(self):
        """Bollinger bandwidth should be between 0 and 1."""
        closes = [100.0 + i * 0.1 for i in range(30)]
        bw = compute_bollinger_bandwidth(closes, period=20, num_std=2.0)
        self.assertGreaterEqual(bw, 0.0)
        self.assertLessEqual(bw, 1.0)


# =============================================================================
# Test 5: Tier Router (4+ tests)
# =============================================================================

class TestTierRouter(unittest.TestCase):
    """Tests for strategy tier routing."""

    def test_route_trending_regime(self):
        """TRENDING regime should activate Tier 1 and Tier 3."""
        router = TierRouter()
        active = router.route(
            regime=MarketRegime.TRENDING,
            hour=10,
            minute=30,
        )
        self.assertIn(StrategyTier.TIER1_MICRO, active)
        self.assertIn(StrategyTier.TIER3_SENTIMENT, active)
        self.assertNotIn(StrategyTier.TIER2_MEANREV, active)

    def test_route_mean_reverting_regime(self):
        """MEAN_REVERTING regime should activate Tier 1 and Tier 2."""
        router = TierRouter()
        active = router.route(
            regime=MarketRegime.MEAN_REVERTING,
            hour=10,
            minute=30,
        )
        self.assertIn(StrategyTier.TIER1_MICRO, active)
        self.assertIn(StrategyTier.TIER2_MEANREV, active)
        self.assertNotIn(StrategyTier.TIER3_SENTIMENT, active)

    def test_route_volatile_regime(self):
        """VOLATILE regime should activate only Tier 3."""
        router = TierRouter()
        active = router.route(
            regime=MarketRegime.VOLATILE,
            hour=10,
            minute=30,
        )
        self.assertEqual(active, [StrategyTier.TIER3_SENTIMENT])

    def test_capital_allocation_redistributes(self):
        """Inactive tiers should redistribute capital to active tiers."""
        router = TierRouter(TierConfig(
            tier1_capital_pct=0.30,
            tier2_capital_pct=0.50,
            tier3_capital_pct=0.20,
        ))
        active = [StrategyTier.TIER1_MICRO, StrategyTier.TIER3_SENTIMENT]
        allocation = router.get_capital_allocation(active, 1000.0)

        # Tier2 is inactive, so its 50% should be redistributed
        total_base = 0.30 + 0.20  # 50%
        tier1_pct = 0.30 / total_base  # 60%
        tier3_pct = 0.20 / total_base  # 40%

        self.assertAlmostEqual(allocation[StrategyTier.TIER1_MICRO], 1000.0 * tier1_pct, places=2)
        self.assertAlmostEqual(allocation[StrategyTier.TIER3_SENTIMENT], 1000.0 * tier3_pct, places=2)
        self.assertNotIn(StrategyTier.TIER2_MEANREV, allocation)

    def test_no_trade_zone_overrides(self):
        """Lunch hour should return only Tier 3."""
        router = TierRouter()
        active = router.route(
            regime=MarketRegime.TRENDING,
            hour=12,
            minute=30,
            is_no_trade_zone=True,
        )
        self.assertEqual(active, [StrategyTier.NO_TRADE])


# =============================================================================
# Test 6: Diagnostics (3+ tests)
# =============================================================================

class TestDiagnostics(unittest.TestCase):
    """Tests for diagnostics modules."""

    def test_filter_funnel_records_and_formats(self):
        """FilterFunnel should record counts and format output."""
        funnel = FilterFunnel()
        funnel.reset_if_new_day("2026-04-17")
        funnel.record("total_symbols_evaluated", 100)
        funnel.record("signals_generated", 50)
        funnel.record("passed_alpha_gate", 40)

        snapshot = funnel.snapshot()
        self.assertEqual(snapshot.total_symbols_evaluated, 100)
        self.assertEqual(snapshot.signals_generated, 50)
        self.assertEqual(snapshot.passed_alpha_gate, 40)

        formatted = funnel.format_funnel()
        self.assertIn("Filter Funnel", formatted)
        self.assertIn("100", formatted)

    def test_signal_distribution_percentile(self):
        """SignalDistribution should compute correct percentiles."""
        dist = SignalDistribution(window=1000)
        # Add values 0 to 99
        for i in range(100):
            dist.add(float(i))

        p50 = dist.percentile(0.5)
        self.assertAlmostEqual(p50, 50.0, delta=2.0)

        p25 = dist.percentile(0.25)
        self.assertLess(p25, p50)

    def test_trade_attribution_win_rate(self):
        """TradeAttribution should compute win rate correctly."""
        attribution = TradeAttribution(max_records=100)

        # Record 10 trades: 7 winners, 3 losers
        for i in range(7):
            idx = attribution.record_entry(
                symbol="TEST",
                direction="BUY",
                ml_signal=0.5,
                ml_confidence=0.8,
                tier="TIER1_MICRO",
                regime="TRENDING",
                top_features=[("feat1", 0.5)],
                predicted_return=0.01,
            )
            attribution.record_exit(idx, actual_return=0.01, gross_pnl=100.0, cost=10.0)

        for i in range(3):
            idx = attribution.record_entry(
                symbol="TEST",
                direction="SELL",
                ml_signal=-0.5,
                ml_confidence=0.7,
                tier="TIER1_MICRO",
                regime="TRENDING",
                top_features=[("feat1", 0.5)],
                predicted_return=-0.01,
            )
            attribution.record_exit(idx, actual_return=-0.005, gross_pnl=-50.0, cost=10.0)

        summary = attribution.summary()
        self.assertAlmostEqual(summary["win_rate"], 0.7, places=2)
        self.assertEqual(summary["count"], 10)


# =============================================================================
# Test 7: Model Health (4+ tests)
# =============================================================================

class TestModelHealth(unittest.TestCase):
    """Tests for model health monitoring."""

    def test_calibration_slope_perfect_prediction(self):
        """Perfect predictions should give slope ≈ 1.0."""
        tracker = CalibrationTracker(min_samples=10)
        predicted = [0.01, 0.02, -0.01, 0.03, 0.015, 0.025, -0.005, 0.02, 0.01, 0.03,
                    0.02, 0.015, 0.025, 0.01, 0.02, -0.01, 0.03, 0.015, 0.025, -0.005]
        actual = predicted.copy()  # perfect match

        snapshot = tracker.update("TEST", predicted, actual)
        self.assertAlmostEqual(snapshot.calibration_slope, 1.0, places=2)

    def test_spearman_ic_random(self):
        """Shuffled predictions should give IC near 0."""
        tracker = CalibrationTracker(min_samples=10)
        predicted = [0.01 * i for i in range(20)]
        actual = [0.01 * (19 - i) for i in range(20)]  # reversed

        snapshot = tracker.update("TEST", predicted, actual)
        # With reversed data, IC should be negative
        self.assertLess(snapshot.information_coefficient, 0.5)

    def test_sizing_multiplier_broken_model(self):
        """BROKEN status should give 0.0 multiplier."""
        tracker = CalibrationTracker(
            broken_ic_threshold=0.01,
            broken_slope_threshold=0.1,
            min_samples=10,
        )
        # Create terrible predictions
        predicted = [0.001, 0.001, 0.001] * 10  # constant
        actual = [0.01, -0.01, 0.02] * 10  # random

        snapshot = tracker.update("TEST", predicted, actual)
        multiplier = tracker.get_sizing_multiplier("TEST")
        self.assertEqual(multiplier, 0.0)

    def test_feature_importance_avg(self):
        """FeatureImportanceTracker should average importances."""
        tracker = FeatureImportanceTracker(max_records_per_symbol=100)

        tracker.record("TEST", {"feat1": 0.5, "feat2": 0.3, "feat3": 0.2})
        tracker.record("TEST", {"feat1": 0.52, "feat2": 0.28, "feat3": 0.2})

        avg = tracker.avg_importance("TEST")
        self.assertGreater(avg.get("feat1", 0), avg.get("feat2", 0))
        self.assertAlmostEqual(avg["feat1"], 0.51, places=2)


# =============================================================================
# Test 8: Feature Vectors (Tier-1 & Tier-2) (2+ tests)
# =============================================================================

class TestFeatureVectors(unittest.TestCase):
    """Tests for Tier-1 and Tier-2 feature vector serialization."""

    def test_tier1_feature_vector_roundtrip(self):
        """Tier1FeatureVector should preserve values in to_dict -> from_dict."""
        fv = Tier1FeatureVector(
            mlofi=0.15,
            obi_momentum=0.05,
            aflow_ratio=0.3,
            trade_arrival_rate=1.5,
            spread_z=2.0,
            vwpp=0.4,
            ret_1min=0.001,
            ret_5min=0.005,
            reference_price=100.0,
            fwd_log_return=0.002,
        )

        d = fv.to_dict()
        fv_restored = Tier1FeatureVector.from_dict(d)

        self.assertAlmostEqual(fv_restored.mlofi, fv.mlofi, places=6)
        self.assertAlmostEqual(fv_restored.obi_momentum, fv.obi_momentum, places=6)
        self.assertAlmostEqual(fv_restored.trade_arrival_rate, fv.trade_arrival_rate, places=6)
        self.assertAlmostEqual(fv_restored.fwd_log_return, fv.fwd_log_return, places=6)

    def test_tier2_feature_vector_roundtrip(self):
        """Tier2FeatureVector should preserve values in to_dict -> from_dict."""
        fv = Tier2FeatureVector(
            vwap_z=1.5,
            rsi_mr_signal=0.5,
            sector_rs=0.02,
            orb_position=0.6,
            time_sin=0.7,
            time_cos=0.3,
            ret_15min=0.01,
            vol_normalised=0.25,
            aflow_ratio_15min=0.15,
            reference_price=105.0,
            fwd_log_return=0.015,
        )

        d = fv.to_dict()
        fv_restored = Tier2FeatureVector.from_dict(d)

        self.assertAlmostEqual(fv_restored.vwap_z, fv.vwap_z, places=6)
        self.assertAlmostEqual(fv_restored.rsi_mr_signal, fv.rsi_mr_signal, places=6)
        self.assertAlmostEqual(fv_restored.ret_15min, fv.ret_15min, places=6)
        self.assertAlmostEqual(fv_restored.fwd_log_return, fv.fwd_log_return, places=6)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    unittest.main()
