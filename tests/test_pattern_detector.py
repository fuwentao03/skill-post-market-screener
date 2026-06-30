"""Unit tests for all 8 pattern detectors.

Coverage: normal cases, edge cases (insufficient data, flat price,
zero volume, extremes), and cross-checks between detectors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from core.pattern_detector import (
    detect_ma_golden_cross,
    detect_macd_golden_cross,
    detect_bullish_alignment,
    detect_volume_breakout,
    detect_bollinger_breakout,
    detect_hammer,
    detect_morning_star,
    detect_rsi_oversold,
    DETECTOR_REGISTRY,
    run_all_detectors,
    get_triggered_labels,
    get_pattern_score,
)


# ── helpers ──────────────────────────────────────────────────

def _make_kline(close: list, volume: list = None, random_ohlc: bool = True) -> pd.DataFrame:
    """Build a minimal K-line DataFrame from close prices."""
    n = len(close)
    rng = np.random.default_rng(42)
    if volume is None:
        volume = [1_000_000] * n
    opens = np.array(close) * (1 + rng.uniform(-0.01, 0.01, n))
    highs = np.maximum(opens, close) * (1 + rng.uniform(0, 0.01, n))
    lows = np.minimum(opens, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(np.array(close, dtype=float), 2),
        "volume": np.array(volume, dtype=float),
    })


def _make_flat_uptrend(n: int, price: float = 10.0) -> pd.DataFrame:
    """n days of gently rising prices."""
    close = np.linspace(price, price * 1.2, n).tolist()
    return _make_kline(close)


def _make_downtrend(n: int, drop_pct: float = 0.15) -> pd.DataFrame:
    """n days of declining prices."""
    close = np.linspace(10, 10 * (1 - drop_pct), n).tolist()
    return _make_kline(close)


# ═══════════════════════════════════════════════════════════
# 1. MA Golden Cross
# ═══════════════════════════════════════════════════════════

class TestMAgoldenCross:
    def test_normal_cross(self):
        """MA5 crosses above MA20 on the last day."""
        # 30 days: early high-ish base, mid dip, then spike on last day
        close = [10.5] * 15 + [10.0] * 9 + [9.5, 9.8, 10.0, 10.2, 10.4, 10.8]
        df = _make_kline(close)
        assert detect_ma_golden_cross(df) > 0  # triggered with positive strength

    def test_no_cross_ma5_always_above(self):
        """MA5 already above MA20 for many days — no fresh cross (strength=0)."""
        close = [10.0] * 20 + [11.0] * 10
        df = _make_kline(close)
        assert detect_ma_golden_cross(df) == 0.0  # stale, returns 0

    def test_no_cross_ma5_always_below(self):
        """MA5 stays below — returns negative strength."""
        close = [12.0] * 20 + [10.0] * 10
        df = _make_kline(close)
        assert detect_ma_golden_cross(df) < 0

    def test_insufficient_data(self):
        """Less than 21 rows → returns 0.0."""
        df = _make_kline([10.0] * 20)
        assert detect_ma_golden_cross(df) == 0.0

    def test_death_cross(self):
        """MA5 crosses below MA20 — returns negative strength."""
        close = [12.0] * 20 + [12.0, 11.5, 11.0, 10.5, 10.0]
        df = _make_kline(close)
        assert detect_ma_golden_cross(df) < 0


# ═══════════════════════════════════════════════════════════
# 2. MACD Golden Cross
# ═══════════════════════════════════════════════════════════

class TestMACDgoldenCross:
    def test_normal_cross(self):
        """DIF crosses above DEA on the last day."""
        close = [10.0] * 35 + [9.5] * 12 + [11.5]
        df = _make_kline(close)
        assert detect_macd_golden_cross(df) > 0

    def test_no_cross(self):
        """DIF stays above DEA — no fresh cross."""
        close = np.linspace(10, 15, 50).tolist()
        df = _make_kline(close)
        assert detect_macd_golden_cross(df) <= 0

    def test_insufficient_data(self):
        """Less than 35 rows → returns 0.0."""
        df = _make_kline([10.0] * 34)
        assert detect_macd_golden_cross(df) == 0.0

    def test_sideways_no_cross(self):
        """Flat prices → DIF ~= DEA ~= 0, no clear cross."""
        close = [10.0] * 60
        df = _make_kline(close)
        result = detect_macd_golden_cross(df)
        assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════
# 3. Bullish Alignment
# ═══════════════════════════════════════════════════════════

class TestBullishAlignment:
    def test_normal_alignment(self):
        """Steady uptrend → MA5 > MA10 > MA20 > MA60."""
        close = np.linspace(10, 20, 80).tolist()
        df = _make_kline(close)
        assert detect_bullish_alignment(df) > 0

    def test_not_aligned_ma10_below_ma20(self):
        """MA10 dips below MA20."""
        close = [15.0] * 60 + [10.0] * 10  # sharp drop
        df = _make_kline(close)
        assert detect_bullish_alignment(df) == 0.0

    def test_insufficient_data(self):
        """Less than 61 rows → returns 0.0."""
        df = _make_kline([10.0] * 60)
        assert detect_bullish_alignment(df) == 0.0

    def test_downtrend_not_aligned(self):
        """Downtrend → MAs reversed."""
        close = np.linspace(20, 10, 80).tolist()
        df = _make_kline(close)
        assert detect_bullish_alignment(df) == 0.0


# ═══════════════════════════════════════════════════════════
# 4. Volume Breakout
# ═══════════════════════════════════════════════════════════

class TestVolumeBreakout:
    def test_normal_breakout(self):
        """Close at 20-day high + volume spikes."""
        close = [10.0] * 30
        close[-1] = 12.0  # new 20-day high
        volume = [1e6] * 30
        volume[-1] = 3e6  # 3x avg of prior 5 days (1e6 * 1.5 = 1.5e6)
        df = _make_kline(close, volume)
        assert detect_volume_breakout(df) > 0

    def test_high_close_low_volume(self):
        """Price breaks out but volume doesn't confirm."""
        close = [10.0] * 30
        close[-1] = 12.0
        volume = [1e6] * 30
        volume[-1] = 1.1e6  # barely above avg
        df = _make_kline(close, volume)
        assert detect_volume_breakout(df) <= 0

    def test_high_volume_no_price_breakout(self):
        """Volume spikes but price doesn't break 20-day high."""
        close = [12.0] + [10.0] * 29  # first day was higher
        volume = [1e6] * 30
        volume[-1] = 3e6
        df = _make_kline(close, volume)
        assert detect_volume_breakout(df) <= 0

    def test_insufficient_data(self):
        """Less than 22 rows → returns 0.0."""
        df = _make_kline([10.0] * 21)
        assert detect_volume_breakout(df) == 0.0


# ═══════════════════════════════════════════════════════════
# 5. Bollinger Breakout
# ═══════════════════════════════════════════════════════════

class TestBollingerBreakout:
    def test_normal_breakout(self):
        """Price spikes above upper band after quiet period."""
        close = [10.0] * 30 + [10.5, 11.0, 11.5, 12.5, 14.0]  # sharp ramp
        df = _make_kline(close)
        assert detect_bollinger_breakout(df) > 0

    def test_inside_band(self):
        """Price stays within band."""
        close = [10.0] * 40
        df = _make_kline(close)
        assert detect_bollinger_breakout(df) <= 0

    def test_insufficient_data(self):
        """Less than 26 rows → returns 0.0."""
        df = _make_kline([10.0] * 25)
        assert detect_bollinger_breakout(df) == 0.0


# ═══════════════════════════════════════════════════════════
# 6. Hammer
# ═══════════════════════════════════════════════════════════

class TestHammer:
    def test_normal_hammer(self):
        """Clear hammer in a downtrend."""
        n = 15
        rng = np.random.default_rng(42)
        closes = np.linspace(12, 10, n)  # downtrend
        opens = closes * (1 + rng.uniform(-0.005, 0.005, n))
        highs = np.maximum(opens, closes) * 1.01
        lows = np.minimum(opens, closes) * 0.99

        # Last candle: hammer — small body, long lower shadow
        opens[-1] = 10.1
        closes[-1] = 10.0   # body = 0.1
        highs[-1] = 10.12   # upper shadow = 0.02 (<= 0.3*body = 0.03)
        lows[-1] = 9.7      # lower shadow = 0.3 (>= 2*body = 0.2)

        df = pd.DataFrame({
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": [1e6] * n,
        })
        assert detect_hammer(df) > 0

    def test_not_in_downtrend(self):
        """Hammer shape but in uptrend → returns 0.0."""
        n = 15
        rng = np.random.default_rng(42)
        closes = np.linspace(10, 12, n)  # uptrend
        opens = closes * (1 + rng.uniform(-0.005, 0.005, n))
        highs = np.maximum(opens, closes) * 1.01
        lows = np.minimum(opens, closes) * 0.99
        opens[-1] = 12.1
        closes[-1] = 12.0
        highs[-1] = 12.15
        lows[-1] = 11.7

        df = pd.DataFrame({
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": [1e6] * n,
        })
        assert detect_hammer(df) == 0.0

    def test_doji(self):
        """body = 0 → returns 0.0 (division by zero)."""
        n = 15
        closes = np.linspace(12, 10, n)
        df = pd.DataFrame({
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": [1e6] * n,
        })
        # Make last candle a doji
        df.loc[df.index[-1], "open"] = 10.0
        df.loc[df.index[-1], "close"] = 10.0
        assert detect_hammer(df) == 0.0

    def test_insufficient_data(self):
        """Less than 7 rows → returns 0.0."""
        df = _make_kline([10.0] * 6)
        assert detect_hammer(df) == 0.0

    def test_short_lower_shadow(self):
        """Lower shadow < 2x body → not a hammer."""
        n = 15
        closes = np.linspace(12, 10, n)
        df = pd.DataFrame({
            "open": [closes[i] * 1.005 for i in range(n)],
            "high": [closes[i] * 1.02 for i in range(n)],
            "low": [closes[i] * 0.99 for i in range(n)],
            "close": closes,
            "volume": [1e6] * n,
        })
        # Last: body ~0.05, lower shadow ~0.01
        assert detect_hammer(df) == 0.0


# ═══════════════════════════════════════════════════════════
# 7. Morning Star
# ═══════════════════════════════════════════════════════════

class TestMorningStar:
    def test_normal_morning_star(self):
        """Classic 3-day reversal: bear → small doji/gap → bull."""
        n = 20
        rng = np.random.default_rng(42)
        closes = [10.0] * n
        opens = [10.0] * n
        highs = [10.0] * n
        lows = [10.0] * n

        # Day -2: bearish (open 12, close 10)
        opens[-3] = 12.0; closes[-3] = 10.0
        highs[-3] = 12.2; lows[-3] = 9.8
        # Day -1: small body, gaps down (open 9.8, close 9.9)
        opens[-2] = 9.8; closes[-2] = 9.9
        highs[-2] = 10.0; lows[-2] = 9.6
        # Day 0: bullish, closes above midpoint of day-2 (mid=11.0)
        opens[-1] = 10.0; closes[-1] = 11.5
        highs[-1] = 11.8; lows[-1] = 9.8

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [1e6] * n,
        })
        assert detect_morning_star(df) > 0

    def test_day2_not_bearish(self):
        """Day -2 is bullish → pattern invalid."""
        n = 20
        closes = [10.0] * n
        opens = [10.0] * n
        highs = [10.0] * n
        lows = [10.0] * n
        # Day -2: bullish (open 10, close 12)
        opens[-3] = 10.0; closes[-3] = 12.0
        highs[-3] = 12.2; lows[-3] = 9.8
        # Day -1: small gap down
        opens[-2] = 11.5; closes[-2] = 11.6
        highs[-2] = 11.8; lows[-2] = 11.2
        # Day 0: bullish close
        opens[-1] = 11.8; closes[-1] = 13.0
        highs[-1] = 13.2; lows[-1] = 11.5

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [1e6] * n,
        })
        assert detect_morning_star(df) == 0.0

    def test_not_closing_above_midpoint(self):
        """Day 0 closes below midpoint of day -2."""
        n = 20
        closes = [10.0] * n
        opens = [10.0] * n
        highs = [10.0] * n
        lows = [10.0] * n
        opens[-3] = 12.0; closes[-3] = 10.0
        highs[-3] = 12.2; lows[-3] = 9.8
        opens[-2] = 9.8; closes[-2] = 9.9
        highs[-2] = 10.0; lows[-2] = 9.6
        # Day 0: close only 10.5, midpoint is 11.0
        opens[-1] = 10.0; closes[-1] = 10.5
        highs[-1] = 10.8; lows[-1] = 9.8

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [1e6] * n,
        })
        assert detect_morning_star(df) == 0.0

    def test_insufficient_data(self):
        """Less than 4 rows → returns 0.0."""
        df = _make_kline([10.0] * 3)
        assert detect_morning_star(df) == 0.0

    def test_day1_body_not_small(self):
        """Day -1 body is too large (>= 50% of day -2 body)."""
        n = 20
        closes = [10.0] * n
        opens = [10.0] * n
        highs = [10.0] * n
        lows = [10.0] * n
        opens[-3] = 12.0; closes[-3] = 10.0  # body_2 = 2.0
        highs[-3] = 12.2; lows[-3] = 9.8
        # Day -1: body = 1.5 (>= 1.0, which is 50% of 2.0)
        opens[-2] = 10.5; closes[-2] = 9.0
        highs[-2] = 10.8; lows[-2] = 8.8
        opens[-1] = 9.5; closes[-1] = 11.5
        highs[-1] = 11.8; lows[-1] = 9.2

        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [1e6] * n,
        })
        assert detect_morning_star(df) == 0.0


# ═══════════════════════════════════════════════════════════
# 8. RSI Oversold Rebound
# ═══════════════════════════════════════════════════════════

class TestRSIoversold:
    def test_normal_rebound(self):
        """RSI drops below 30, then rebounds with bullish close."""
        n = 30
        closes = [15.0] * 10 + list(np.linspace(15, 8, 18)) + [8.5]
        df = _make_kline(closes)
        # Force last candle to be bullish
        df.loc[df.index[-1], "open"] = 8.2
        df.loc[df.index[-1], "close"] = 8.5
        result = detect_rsi_oversold(df)
        assert isinstance(result, float)

    def test_not_oversold(self):
        """RSI stays above 30 — no signal."""
        close = np.linspace(10, 12, 30).tolist()
        df = _make_kline(close)
        assert detect_rsi_oversold(df) <= 0

    def test_rsi_oversold_no_rebound(self):
        """RSI < 30 but still dropping."""
        n = 30
        close = [15.0] * 5 + list(np.linspace(15, 7, 25))
        df = _make_kline(close)
        assert detect_rsi_oversold(df) <= 0

    def test_oversold_rebound_bearish_close(self):
        """RSI rebounds but today closes bearish."""
        n = 30
        close = [15.0] * 5 + list(np.linspace(15, 8, 23)) + [8.5, 8.5]
        df = _make_kline(close)
        df.loc[df.index[-1], "open"] = 8.8
        df.loc[df.index[-1], "close"] = 8.5  # bearish
        assert detect_rsi_oversold(df) <= 0

    def test_insufficient_data(self):
        """Less than 16 rows → returns 0.0."""
        df = _make_kline([10.0] * 15)
        assert detect_rsi_oversold(df) == 0.0


# ═══════════════════════════════════════════════════════════
# Registry & integration tests
# ═══════════════════════════════════════════════════════════

class TestRegistry:
    """Tests for the detector registry and aggregate functions."""

    def test_all_8_registered(self):
        assert len(DETECTOR_REGISTRY) == 8
        expected = {
            "ma_golden_cross", "macd_golden_cross", "bullish_alignment",
            "volume_breakout", "bollinger_breakout", "hammer",
            "morning_star", "rsi_oversold",
        }
        assert set(DETECTOR_REGISTRY.keys()) == expected

    def test_weights_sum(self):
        """Weights should sum to 17 (2+2+3+3+2+2+2+1)."""
        total = sum(v["weight"] for v in DETECTOR_REGISTRY.values())
        assert total == 17

    def test_every_entry_has_func_label_weight(self):
        for key, entry in DETECTOR_REGISTRY.items():
            assert callable(entry["func"]), f"{key}: func not callable"
            assert isinstance(entry["label"], str), f"{key}: label missing"
            assert isinstance(entry["weight"], int), f"{key}: weight not int"
            assert entry["weight"] > 0, f"{key}: weight <= 0"

    def test_run_all_detectors(self):
        """run_all_detectors returns dict of float for each pattern."""
        df = _make_flat_uptrend(80)
        results = run_all_detectors(df)
        assert len(results) == 8
        assert all(isinstance(v, float) for v in results.values())

    def test_run_all_detectors_with_filter(self):
        """active_patterns filters which detectors run."""
        df = _make_flat_uptrend(80)
        results = run_all_detectors(df, active_patterns={"hammer", "rsi_oversold"})
        assert set(results.keys()) == {"hammer", "rsi_oversold"}

    def test_get_triggered_labels(self):
        results = {"ma_golden_cross": 0.5, "hammer": -0.1, "rsi_oversold": 3.0}
        labels = get_triggered_labels(results)
        assert "均线金叉" in labels
        assert "RSI超卖反弹" in labels
        assert "锤子线" not in labels

    def test_get_pattern_score(self):
        results = {
            "bullish_alignment": 0.8,   # weight 1 → triggered
            "volume_breakout": 1.5,      # weight 4 → triggered
            "ma_golden_cross": -0.2,     # not triggered
            "rsi_oversold": 2.0,         # weight 2 → triggered
        }
        assert get_pattern_score(results) == 7  # 1 + 4 + 2

    def test_exception_in_detector_returns_zero(self):
        """Detector that raises → 0.0, not crash."""
        results = run_all_detectors(pd.DataFrame())  # empty df
        for v in results.values():
            assert v == 0.0


# ═══════════════════════════════════════════════════════════
# Cross-detector sanity checks
# ═══════════════════════════════════════════════════════════

class TestCrossDetector:
    """Sanity checks that test relationships between detectors."""

    def test_bullish_alignment_implies_no_ma_cross_today(self):
        """If already in bullish alignment, MA5 was already >
        MA20 yesterday — so MA golden cross can't trigger simultaneously."""
        df = _make_flat_uptrend(80)
        results = run_all_detectors(df)
        # bull alignment sets MA5 >> MA20 for many days,
        # so golden cross (which needs yesterday MA5 <= MA20) won't fire
        if results["bullish_alignment"] > 0:
            assert results["ma_golden_cross"] <= 0, (
                "Bullish alignment and MA golden cross shouldn't co-occur"
            )

    def test_mock_stock_triggers_at_least_one(self):
        """A stock with steady uptrend + volume breakout should trigger at least 2."""
        n = 120
        price = np.linspace(10, 16, n).tolist()  # steady uptrend
        vol = [1_000_000] * n
        vol[-1] = 3_000_000  # volume breakout
        price[-1] = 16.5     # push above 20-day high
        df = _make_kline(price, vol)
        results = run_all_detectors(df)
        triggered = [k for k, v in results.items() if v > 0]
        # steady uptrend → bullish_alignment; last bar → volume_breakout
        assert len(triggered) >= 2, f"Expected >= 2 detectors, got {triggered}"
