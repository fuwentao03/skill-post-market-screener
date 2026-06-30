# Pattern Detector Formulas

All 8 detectors run against each stock's daily K-line data. Each detector returns a **continuous float strength value**: `> 0` = pattern triggered (higher = stronger signal), `≤ 0` = not triggered. The strength magnitude is used for backtest IC calculation; the scoring system treats `> 0` as binary (triggered = adds weight, not triggered = 0).

## 1. MA Golden Cross (均线金叉)

**Weight: 2**

```
MA5  = SMA(close, 5)
MA20 = SMA(close, 20)

Condition:
  MA5[0] > MA20[0]           # today MA5 above MA20
  AND MA5[-1] <= MA20[-1]    # yesterday MA5 at or below MA20 (cross just happened)
```

## 2. MACD Golden Cross (MACD 金叉)

**Weight: 1**

```
EMA12 = EMA(close, 12)
EMA26 = EMA(close, 26)
DIF   = EMA12 - EMA26
DEA   = EMA(DIF, 9)

Condition:
  DIF[0] > DEA[0]            # today DIF above DEA
  AND DIF[-1] <= DEA[-1]     # yesterday DIF at or below DEA (cross just happened)
```

## 3. Bullish Alignment (多头排列)

**Weight: 1**

```
MA5  = SMA(close, 5)
MA10 = SMA(close, 10)
MA20 = SMA(close, 20)
MA60 = SMA(close, 60)

Condition:
  MA5 > MA10 > MA20 > MA60   # all at today's values
```

## 4. Volume Breakout (放量突破)

**Weight: 4**

```
close_20_high = MAX(close[-1], 20)     # highest close in prior 20 days (excluding today)
avg_vol_5     = AVG(volume[-1], 5)     # average volume over prior 5 days (excluding today)

Condition:
  close[0] > close_20_high             # today's close breaks 20-day high
  AND volume[0] > 1.5 * avg_vol_5      # volume > 1.5x 5-day average
```

## 5. Bollinger Breakout (布林带突破)

**Weight: 3**

```
MA20    = SMA(close, 20)
std_20  = STD(close, 20)
upper   = MA20 + 2 * std_20
bandwidth = (upper - lower) / MA20

Condition:
  close[0] > upper[0]                          # price breaks above upper band
  AND bandwidth[0] > bandwidth[-5]             # bandwidth expanding vs 5 days ago
```

## 6. Hammer (锤子线)

**Weight: 1**

```
body        = ABS(close[0] - open[0])
lower_shadow = MIN(open[0], close[0]) - low[0]
upper_shadow = high[0] - MAX(open[0], close[0])
trend_5     = (close[-5] - close[-1]) / close[-5]   # 5-day return (positive = downtrend recovery check)

Condition:
  lower_shadow >= 2 * body                  # long lower shadow
  AND upper_shadow <= 0.3 * body            # small or no upper shadow
  AND trend_5 < -0.03                       # in a short-term downtrend (>3% drop over 5 days)
  AND body > 0                              # non-doji
```

## 7. Morning Star (启明星)

**Weight: 3**

Three-day pattern:

```
Day -2 (bear candle):
  close[-2] < open[-2]                      # bearish day
  body_2 = ABS(close[-2] - open[-2])

Day -1 (small body):
  body_1 = ABS(close[-1] - open[-1])
  body_1 < body_2 * 0.5                     # small body, less than half of day -2
  AND MIN(open[-1], close[-1]) < MIN(open[-2], close[-2])  # gaps down

Day 0 (bull candle):
  close[0] > open[0]                        # bullish day
  AND close[0] > (open[-2] + close[-2]) / 2 # closes above midpoint of day -2
```

## 8. RSI Oversold Rebound (RSI 超卖反弹)

**Weight: 2**

```
RSI(14) = 100 - (100 / (1 + RS))
  where RS = AVG(gain, 14) / AVG(loss, 14)

Condition:
  RSI[-1] < 30                              # yesterday RSI oversold
  AND RSI[0] > RSI[-1]                      # RSI turning up
  AND close[0] > open[0]                    # today closes bullish
```

## Scoring Translation

Each triggered pattern contributes its weight to `pattern_score`:

```
pattern_score = SUM(weight_i for each triggered detector_i)
```

Max possible pattern_score = 2+1+1+4+3+1+3+2 = 17.

## Edge Cases

| Case | Handling |
| --- | --- |
| Insufficient history (< min_data_rows, default 30) | Skip stock entirely; configurable via `scan.min_data_rows` in config.json |
| Flat price (body = 0 for hammer) | Skip hammer (returns 0.0) |
| Gap-down morning star where day -1 isn't lower | Relax gap requirement to "day -1 low < day -2 low" |
| RSI when all gains or all losses in window | RSI = 100 or 0 respectively; still valid |
| Detector exception (e.g. NumPy error) | Returns 0.0, does not halt the scan |
| Stale signal (cross already happened) | Returns 0.0 — only fresh crosses on the latest day trigger |
