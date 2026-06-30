"""8 technical pattern detectors for A-share daily K-line data.

Each detector takes a pandas DataFrame with columns:
    open, high, low, close, volume
sorted by date ascending (oldest first), and returns a float signal strength.

Strength > 0  → pattern triggered (larger = stronger signal).
Strength <= 0 → pattern not triggered (more negative = further from trigger).

This continuous output enables proper Spearman rank IC calculation in the
backtest (avoids point-biserial degeneration from 0/1 binary signals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_ma_golden_cross(df: pd.DataFrame) -> float:
    """MA5 crosses above MA20 today.

    Strength = (MA5 - MA20) / close * 100  (percentage gap).
    Positive when crossed today; zero when already above (stale);
    negative when MA5 is below MA20.
    """
    if len(df) < 21:
        return 0.0
    close = df["close"].values
    ma5 = pd.Series(close).rolling(5).mean().values
    ma20 = pd.Series(close).rolling(20).mean().values
    gap_pct = (ma5[-1] - ma20[-1]) / close[-1] * 100
    crossed = ma5[-1] > ma20[-1] and ma5[-2] <= ma20[-2]
    if crossed:
        return round(gap_pct, 6)
    elif ma5[-1] > ma20[-1]:
        return 0.0  # already above, stale — no fresh signal
    else:
        return round(gap_pct, 6)  # negative


def detect_macd_golden_cross(df: pd.DataFrame) -> float:
    """DIF crosses above DEA today (MACD golden cross).

    Strength = (DIF - DEA) / close  (normalized gap).
    Positive when crossed today; zero when already above; negative when below.
    """
    if len(df) < 35:
        return 0.0
    close = df["close"].values
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    gap = (dif[-1] - dea[-1]) / close[-1]
    crossed = dif[-1] > dea[-1] and dif[-2] <= dea[-2]
    if crossed:
        return round(gap, 6)
    elif dif[-1] > dea[-1]:
        return 0.0
    else:
        return round(gap, 6)


def detect_bullish_alignment(df: pd.DataFrame) -> float:
    """MA5 > MA10 > MA20 > MA60 (all at today's values).

    Strength = minimum inter-MA gap as % of close.
    Positive when aligned; zero when not.
    """
    if len(df) < 61:
        return 0.0
    close = df["close"].values
    ma5 = pd.Series(close).rolling(5).mean().values[-1]
    ma10 = pd.Series(close).rolling(10).mean().values[-1]
    ma20 = pd.Series(close).rolling(20).mean().values[-1]
    ma60 = pd.Series(close).rolling(60).mean().values[-1]
    aligned = ma5 > ma10 > ma20 > ma60
    if aligned:
        gaps = [ma5 - ma10, ma10 - ma20, ma20 - ma60]
        min_gap_pct = min(gaps) / close[-1] * 100
        return round(min_gap_pct, 6)
    return 0.0


def detect_volume_breakout(df: pd.DataFrame) -> float:
    """Close at 20-day high AND volume > 1.5x 5-day average volume.

    Strength = average of price breakout % and volume excess ratio.
    Positive when both conditions met; negative or zero when not.
    """
    if len(df) < 22:
        return 0.0
    close = df["close"].values
    volume = df["volume"].values
    close_20_high = np.max(close[-21:-1])
    avg_vol_5 = np.mean(volume[-6:-1])
    if avg_vol_5 == 0:
        return 0.0
    price_break = (close[-1] - close_20_high) / close_20_high * 100  # %
    vol_ratio = volume[-1] / (1.5 * avg_vol_5) - 1.0  # > 0 means exceeds threshold
    triggered = close[-1] > close_20_high and volume[-1] > 1.5 * avg_vol_5
    if triggered:
        strength = (price_break + vol_ratio * 100) / 2  # avg of both dimensions
        return round(strength, 6)
    else:
        # Negative: how far from meeting both conditions
        return round(min(price_break / 100, vol_ratio), 6)


def detect_bollinger_breakout(df: pd.DataFrame) -> float:
    """Price breaks upper band AND bandwidth expanding vs 5 days ago.

    Strength = (close - upper) / std  (number of std beyond upper band).
    Positive when breakout; negative when inside band.
    """
    if len(df) < 26:
        return 0.0
    close = df["close"].values
    ma20 = pd.Series(close).rolling(20).mean().values
    std20 = pd.Series(close).rolling(20).std().values
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    bandwidth = (upper - lower) / ma20
    breakout = close[-1] > upper[-1] and bandwidth[-1] > bandwidth[-6]
    if std20[-1] == 0:
        return 0.0
    strength = (close[-1] - upper[-1]) / std20[-1]
    if breakout:
        return round(strength, 6)
    else:
        return round(min(0.0, strength), 6)


def detect_hammer(df: pd.DataFrame) -> float:
    """Lower shadow >= 2x body, small upper shadow, in short-term downtrend.

    Strength = composite of shadow/body ratio excess and trend depth.
    Positive when hammer pattern is valid; zero when not.
    """
    if len(df) < 7:
        return 0.0
    o, h, l, c = df["open"].values[-1], df["high"].values[-1], df["low"].values[-1], df["close"].values[-1]
    body = abs(c - o)
    if body == 0:
        return 0.0
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    shadow_ratio = lower_shadow / body  # ≥ 2 for valid hammer
    upper_ratio = upper_shadow / body   # ≤ 0.3 for valid hammer
    close_6 = df["close"].values[-6]
    close_1 = df["close"].values[-2]
    trend_5 = (close_1 - close_6) / close_6 if close_6 != 0 else 0

    valid = (
        lower_shadow >= 2 * body
        and upper_shadow <= 0.3 * body
        and trend_5 < -0.03
    )
    if valid:
        # Strength: how much shadow exceeds 2x threshold + how deep the downtrend
        shadow_excess = shadow_ratio - 2.0
        trend_depth = abs(trend_5) - 0.03
        # Scale trend_depth to comparable range with shadow_excess.
        # trend_depth is typically 0.00–0.10 (0–10% decline over 5 days),
        # shadow_excess is 0.0–1.0+ (how much shadow exceeds 2× body).
        # Factor 30 maps a 10% downtrend ≈ 3.0 contribution, balancing
        # shadow quality against trend confirmation strength.
        strength = shadow_excess + trend_depth * 30
        return round(strength, 6)
    return 0.0


def detect_morning_star(df: pd.DataFrame) -> float:
    """Three-day reversal: bear → small body → bull, closing above midpoint of day-2.

    Strength = close above midpoint as fraction of day-2 body.
    Positive when pattern is valid; zero when not.
    """
    if len(df) < 4:
        return 0.0
    o = df["open"].values
    c = df["close"].values
    o3, c3 = o[-3], c[-3]
    o2, c2 = o[-2], c[-2]
    o1, c1 = o[-1], c[-1]

    body3 = abs(c3 - o3)
    body2 = abs(c2 - o2)
    if body3 == 0:
        return 0.0

    cond1 = c3 < o3
    cond2 = body2 < body3 * 0.5 and max(o2, c2) < min(o3, c3)
    cond3 = c1 > o1 and c1 > (o3 + c3) / 2

    if cond1 and cond2 and cond3:
        midpoint = (o3 + c3) / 2
        # How far above midpoint, scaled by day-2 body
        strength = (c1 - midpoint) / body3
        return round(strength, 6)
    return 0.0


def detect_rsi_oversold(df: pd.DataFrame, period: int = 14) -> float:
    """RSI(14) was < 30 yesterday, turning up today, and today closes bullish.

    Strength = RSI change (today - yesterday).
    Positive when valid oversold rebound; zero or negative when not.
    """
    if len(df) < period + 2:
        return 0.0
    close = df["close"].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period).mean().values
    avg_loss = pd.Series(loss).rolling(period).mean().values

    rsi = np.full_like(close, np.nan)
    for i in range(period, len(close)):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain[i] / avg_loss[i]))

    valid = (
        not np.isnan(rsi[-2])
        and rsi[-2] < 30
        and rsi[-1] > rsi[-2]
        and close[-1] > df["open"].values[-1]
    )
    if valid:
        return round(rsi[-1] - rsi[-2], 6)
    elif not np.isnan(rsi[-2]) and not np.isnan(rsi[-1]):
        # Signal exists but conditions not met: return negative or zero delta
        delta_rsi = rsi[-1] - rsi[-2]
        return round(min(0.0, delta_rsi), 6)
    return 0.0


# --- Registry ---

DEFAULT_WEIGHTS = {
    # Calibrated via backtest on 5192 A-share stocks (2025-11 ~ 2026-06).
    # See scripts/analyze_weights.py for IC analysis methodology.
    # Re-calibrate quarterly and update config.json → detector_weights.
    "ma_golden_cross":    2,
    "macd_golden_cross":  1,
    "bullish_alignment":  1,
    "volume_breakout":    3,   # backtest: IC(5d)=-0.011, hit=44.5% → reduced from 4
    "bollinger_breakout": 3,
    "hammer":             1,
    "morning_star":       3,
    "rsi_oversold":       1,   # backtest: IC(5d)=-0.007, hit=43.2% → reduced from 2
}

# Weights calibrated via backtest on ~5000 A-share stocks
# See scripts/analyze_weights.py for the full IC analysis
DETECTOR_REGISTRY = {
    "ma_golden_cross":    {"func": detect_ma_golden_cross,    "weight": DEFAULT_WEIGHTS["ma_golden_cross"],    "label": "均线金叉"},
    "macd_golden_cross":  {"func": detect_macd_golden_cross,  "weight": DEFAULT_WEIGHTS["macd_golden_cross"],  "label": "MACD金叉"},
    "bullish_alignment":  {"func": detect_bullish_alignment,  "weight": DEFAULT_WEIGHTS["bullish_alignment"],  "label": "多头排列"},
    "volume_breakout":    {"func": detect_volume_breakout,    "weight": DEFAULT_WEIGHTS["volume_breakout"],    "label": "放量突破"},
    "bollinger_breakout": {"func": detect_bollinger_breakout, "weight": DEFAULT_WEIGHTS["bollinger_breakout"], "label": "布林带突破"},
    "hammer":             {"func": detect_hammer,              "weight": DEFAULT_WEIGHTS["hammer"],              "label": "锤子线"},
    "morning_star":       {"func": detect_morning_star,        "weight": DEFAULT_WEIGHTS["morning_star"],        "label": "启明星"},
    "rsi_oversold":       {"func": detect_rsi_oversold,        "weight": DEFAULT_WEIGHTS["rsi_oversold"],        "label": "RSI超卖反弹"},
}


def run_all_detectors(
    kline_df: pd.DataFrame, active_patterns: Optional[set[str]] = None
) -> dict[str, float]:
    """Run all active detectors against a single stock's K-line DataFrame.

    Args:
        kline_df: K-line data for one stock, sorted by date ascending.
        active_patterns: Set of pattern keys to run. None = run all.

    Returns:
        {pattern_key: signal_strength (float)}
        Strength > 0  → triggered; Strength <= 0 → not triggered.
    """
    if active_patterns is None:
        active_patterns = set(DETECTOR_REGISTRY.keys())

    results: dict[str, float] = {}
    for key in active_patterns:
        entry = DETECTOR_REGISTRY.get(key)
        if entry is None:
            continue
        try:
            results[key] = entry["func"](kline_df)
        except Exception:
            results[key] = 0.0
    return results


def get_triggered_labels(results: dict[str, float]) -> list[str]:
    """Return Chinese labels for triggered patterns (strength > 0)."""
    labels: list[str] = []
    for key, strength in results.items():
        if strength > 0:
            entry = DETECTOR_REGISTRY.get(key)
            if entry:
                labels.append(entry["label"])
    return labels


def get_pattern_score(results: dict[str, float]) -> float:
    """Sum weights of triggered patterns (strength > 0).

    Each triggered pattern contributes its base weight.
    The continuous strength value is used by the backtest for IC,
    not directly in scoring.
    """
    score = 0.0
    for key, strength in results.items():
        if strength > 0:
            entry = DETECTOR_REGISTRY.get(key)
            if entry:
                score += entry["weight"]
    return score


def apply_weights_from_config(config: dict) -> dict[str, dict[str, float]]:
    """Apply optimized weights from config to DETECTOR_REGISTRY.

    Reads ``config.detector_weights`` if present, otherwise uses defaults.
    Returns a dict of {key: {"old": old_weight, "new": new_weight}} for logging.

    Args:
        config: Full runtime config dict (from config.json).
    """
    weight_overrides: dict[str, int] = config.get("detector_weights", {})
    changes: dict[str, dict[str, float]] = {}
    for key in DETECTOR_REGISTRY:
        old = DETECTOR_REGISTRY[key]["weight"]
        new = weight_overrides.get(key, old)
        DETECTOR_REGISTRY[key]["weight"] = new
        changes[key] = {"old": old, "new": new}
    return changes


def get_pattern_details(kline_df: pd.DataFrame, triggered_keys: set[str]) -> dict[str, str]:
    """Build per-pattern context strings for LLM prompt enrichment.

    Args:
        kline_df: K-line DataFrame for one stock, sorted by date ascending.
        triggered_keys: Set of detector keys that triggered.

    Returns:
        {pattern_key: human_readable_context_string}
    """
    details: dict[str, str] = {}
    if kline_df.empty or len(kline_df) < 7:
        return details

    close = kline_df["close"].values
    o = kline_df["open"].values
    h = kline_df["high"].values
    l = kline_df["low"].values
    vol = kline_df["volume"].values

    # ── MA Golden Cross ──
    if "ma_golden_cross" in triggered_keys and len(close) >= 21:
        ma5 = pd.Series(close).rolling(5).mean().values
        ma20 = pd.Series(close).rolling(20).mean().values
        details["ma_golden_cross"] = (
            f"MA5({ma5[-1]:.2f})今日上穿MA20({ma20[-1]:.2f})，"
            f"此前MA5({ma5[-2]:.2f})≤MA20({ma20[-2]:.2f})，"
            f"收盘价{close[-1]:.2f}"
        )

    # ── MACD Golden Cross ──
    if "macd_golden_cross" in triggered_keys and len(close) >= 35:
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
        macd_hist = 2 * (dif - dea)
        details["macd_golden_cross"] = (
            f"DIF({dif[-1]:.3f})上穿DEA({dea[-1]:.3f})，"
            f"MACD柱转正({macd_hist[-1]:.3f})，"
            f"此前DIF({dif[-2]:.3f})≤DEA({dea[-2]:.3f})"
        )

    # ── Bullish Alignment ──
    if "bullish_alignment" in triggered_keys and len(close) >= 61:
        ma5 = pd.Series(close).rolling(5).mean().values[-1]
        ma10 = pd.Series(close).rolling(10).mean().values[-1]
        ma20 = pd.Series(close).rolling(20).mean().values[-1]
        ma60 = pd.Series(close).rolling(60).mean().values[-1]
        details["bullish_alignment"] = (
            f"MA5({ma5:.2f})>MA10({ma10:.2f})>MA20({ma20:.2f})>MA60({ma60:.2f})，"
            f"多头排列完整，收盘价{close[-1]:.2f}"
        )

    # ── Volume Breakout ──
    if "volume_breakout" in triggered_keys and len(close) >= 22:
        close_20_high = np.max(close[-21:-1])
        avg_vol_5 = np.mean(vol[-6:-1])
        vol_ratio = vol[-1] / avg_vol_5 if avg_vol_5 > 0 else 0
        details["volume_breakout"] = (
            f"收盘{close[-1]:.2f}突破20日最高{close_20_high:.2f}，"
            f"量比{vol_ratio:.1f}倍（今日量{vol[-1]:.0f}/5日均{avg_vol_5:.0f}）"
        )

    # ── Bollinger Breakout ──
    if "bollinger_breakout" in triggered_keys and len(close) >= 26:
        ma20 = pd.Series(close).rolling(20).mean().values
        std20 = pd.Series(close).rolling(20).std().values
        upper = ma20 + 2 * std20
        bandwidth = (upper - (ma20 - 2 * std20)) / ma20
        bw_now = bandwidth[-1]
        bw_prev = bandwidth[-6]
        details["bollinger_breakout"] = (
            f"收盘{close[-1]:.2f}突破布林上轨{upper[-1]:.2f}，"
            f"带宽从{bw_prev:.1%}扩至{bw_now:.1%}，波动扩大中"
        )

    # ── Hammer ──
    if "hammer" in triggered_keys:
        body = abs(close[-1] - o[-1])
        lower_shadow = min(o[-1], close[-1]) - l[-1]
        upper_shadow = h[-1] - max(o[-1], close[-1])
        close_6 = close[-6]
        close_1 = close[-2]
        trend_5 = (close_1 - close_6) / close_6 * 100 if close_6 != 0 else 0
        ratio = lower_shadow / body if body > 0 else 0
        details["hammer"] = (
            f"实体{body:.2f}，下影线{lower_shadow:.2f}({ratio:.1f}倍实体)，"
            f"上影线{upper_shadow:.2f}，前5日跌幅{trend_5:.1f}%"
        )

    # ── Morning Star ──
    if "morning_star" in triggered_keys and len(close) >= 4:
        o3, c3 = o[-3], close[-3]
        o2, c2 = o[-2], close[-2]
        o1, c1 = o[-1], close[-1]
        body3 = abs(c3 - o3)
        body2 = abs(c2 - o2)
        details["morning_star"] = (
            f"前日阴线(开{o3:.2f}收{c3:.2f}，实体{body3:.2f})→"
            f"昨日小实体(开{o2:.2f}收{c2:.2f}，实体{body2:.2f}，仅{body2/body3*100:.0f}%前日)→"
            f"今日阳线(开{o1:.2f}收{c1:.2f})，收盘超越前日中点{(o3+c3)/2:.2f}"
        )

    # ── RSI Oversold ──
    if "rsi_oversold" in triggered_keys and len(close) >= 16:
        period = 14
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(period).mean().values
        avg_loss = pd.Series(loss).rolling(period).mean().values
        if avg_loss[-2] > 0:
            rsi_yesterday = 100.0 - 100.0 / (1.0 + avg_gain[-2] / avg_loss[-2])
        else:
            rsi_yesterday = 100.0
        if avg_loss[-1] > 0:
            rsi_today = 100.0 - 100.0 / (1.0 + avg_gain[-1] / avg_loss[-1])
        else:
            rsi_today = 100.0
        details["rsi_oversold"] = (
            f"RSI从昨日{rsi_yesterday:.0f}反弹至今日{rsi_today:.0f}，"
            f"今日收阳({close[-1]:.2f}>{o[-1]:.2f})，前14日累计跌幅约"
            f"{(close[-15]-close[-1])/close[-15]*100:.1f}%"
        )

    return details
