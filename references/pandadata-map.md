# Pandadata Map — Post-Market Screener

Use this map to plan the daily screener. The actual implementation uses the `panda_data` Python SDK (v0.0.9) which communicates with a Pandadata Java backend service via HTTP. All methods are accessed via `panda_data.<method>()` after calling `panda_data.init_token()`.

> **注意：** 当前 SDK 版本无需调用 `panda_data.init()`，仅需 `panda_data.init_token(username, password, base_url)`。

## Core Date And Universe

| Need | Preferred method | Notes |
| --- | --- | --- |
| Latest trading day | `get_last_trade_date` | Use when the user says 今天扫描 or does not specify a date. |
| Trading calendar | `get_trade_cal` | Confirm whether the target date is open. |
| Tradable A-share universe | `get_trade_list` | Use as the stock list for batch K-line and fund-flow queries. Market-cap filtering is applied per-stock during the scan phase. |

## Data Groups

| Data group | Preferred methods | Key fields | Notes |
| --- | --- | --- | --- |
| **K-line (120 days)** | `get_stock_daily` | `open, high, low, close, volume, amount` | Need 120 trading days for MA60/MA20/MA10/MA5, MACD, Bollinger, and RSI calculation windows. |
| **Fund flow** | AKShare `stock_individual_fund_flow_rank` | `main_net_inflow, super_large_net_inflow, large_net_inflow, main_inflow_rate` | Per-stock daily flow from East Money via AKShare. Pandadata `get_stock_rt_daily` 不可用（权限不足），资金流向由 AKShare 替代。Main inflow rate = main_net_inflow / turnover. |
| **Stock basic info** | `get_stock_detail` | `name, market_cap, industry, list_status, exchange` | Used for industry grouping, market-cap quality bonus, and filtering out suspended/ST stocks. |
| **Stock industry** | `get_stock_detail` | `sector_code_name` (用作 industry) | 行业分类从 `get_stock_detail` 的 `sector_code_name` 字段获取。`get_stock_industry` 不可用（权限不足）。 |

## Pattern Calculation Fields (from K-line)

| Pattern | Required fields | Lookback window |
| --- | --- | --- |
| MA Golden Cross | `close` | MA5, MA20 (today and yesterday) |
| MACD Golden Cross | `close` | EMA12, EMA26, DIF, DEA (today and yesterday) |
| Bullish Alignment | `close` | MA5, MA10, MA20, MA60 (today) |
| Volume Breakout | `close, volume` | 20-day high of close, 5-day avg volume |
| Bollinger Breakout | `close` | MA20, upper band (20-day, 2 std), bandwidth |
| Hammer | `open, high, low, close` | Today's candle + 5-day trend check |
| Morning Star | `open, high, low, close` | Today + yesterday + day-before-yesterday |
| RSI Oversold | `close` | RSI(14) today and yesterday |

## API Resilience

All Pandadata API calls (`get_trade_list`, `get_stock_daily`, `get_stock_detail`,
`get_share_float`) are wrapped with `_retry_api_call()`: 3 attempts, exponential
backoff (1s → 2s → 4s). Transient failures on a single chunk do not lose data.
AKShare fund flow uses `FlowFetcher` with its own 3-attempt retry.

K-line chunks (200 symbols each) are fetched in parallel via `ThreadPoolExecutor`
(max 4 workers). Fund flow is fetched at the pipeline layer (not inside
`DataFetcher`) for clean separation of concerns.

## Degradation

If a data interface is unavailable or too slow:

1. Skip the affected stock, continue scanning others (single-stock exceptions
   are caught and logged; they do not halt the full scan).
2. Keep all stocks that have complete pattern + flow data.
3. Add the skipped stock count and failed interface names under "数据说明" in the report.
4. If the fund-flow interface is entirely down: output a "资金数据不可用，仅做形态扫描"
   note and skip the flow filter. In mock mode, synthetic flow data is used with a
   visible degradation note ("AKShare资金流向不可用，使用模拟资金流数据").
5. If K-line data is unavailable for all stocks: abort with "K线数据不可用，今日扫描终止".
6. Corrupt cache files (parquet read errors) are handled gracefully — the affected
   DataFrame is returned empty, triggering a fresh fetch.
