"""Mock data generator for testing the screener pipeline without Pandadata."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Realistic A-share stock names and industries for mock generation
STOCK_POOL = [
    ("600519.SH", "贵州茅台", "白酒", 25000),
    ("000858.SZ", "五粮液", "白酒", 8000),
    ("300750.SZ", "宁德时代", "电池", 12000),
    ("002594.SZ", "比亚迪", "汽车", 9000),
    ("601318.SH", "中国平安", "保险", 10000),
    ("600036.SH", "招商银行", "银行", 9500),
    ("000333.SZ", "美的集团", "家电", 5500),
    ("600276.SH", "恒瑞医药", "医药", 3500),
    ("002415.SZ", "海康威视", "安防", 3800),
    ("300059.SZ", "东方财富", "证券", 4200),
    ("600900.SH", "长江电力", "电力", 6000),
    ("601012.SH", "隆基绿能", "光伏", 2200),
    ("002714.SZ", "牧原股份", "养殖", 2800),
    ("688981.SH", "中芯国际", "半导体", 5000),
    ("000651.SZ", "格力电器", "家电", 2500),
    ("300124.SZ", "汇川技术", "工控", 2000),
    ("601166.SH", "兴业银行", "银行", 4800),
    ("002230.SZ", "科大讯飞", "AI", 1800),
    ("600030.SH", "中信证券", "证券", 4000),
    ("601888.SH", "中国中免", "旅游", 1500),
    ("002475.SZ", "立讯精密", "电子", 2800),
    ("300274.SZ", "阳光电源", "光伏", 1900),
    ("688111.SH", "金山办公", "软件", 1600),
    ("601899.SH", "紫金矿业", "矿业", 4500),
    ("300015.SZ", "爱尔眼科", "医疗", 2000),
    ("603259.SH", "药明康德", "医药", 2200),
    ("000625.SZ", "长安汽车", "汽车", 1700),
    ("002352.SZ", "顺丰控股", "物流", 2400),
    ("300498.SZ", "温氏股份", "养殖", 1500),
    ("601390.SH", "中国中铁", "基建", 1800),
    ("000002.SZ", "万科A", "地产", 1400),
    ("002142.SZ", "宁波银行", "银行", 2200),
    ("300014.SZ", "亿纬锂能", "电池", 1200),
    ("601668.SH", "中国建筑", "建筑", 2200),
    ("002460.SZ", "赣锋锂业", "锂矿", 1100),
    ("300347.SZ", "泰格医药", "医药", 900),
    ("000568.SZ", "泸州老窖", "白酒", 3200),
    ("600809.SH", "山西汾酒", "白酒", 3600),
    ("002304.SZ", "洋河股份", "白酒", 2000),
    ("300760.SZ", "迈瑞医疗", "医疗", 4500),
    ("601088.SH", "中国神华", "煤炭", 5500),
    ("000831.SZ", "中国稀土", "稀土", 800),
    ("002466.SZ", "天齐锂业", "锂矿", 700),
    ("300450.SZ", "先导智能", "锂电设备", 600),
    ("600104.SH", "上汽集团", "汽车", 2800),
    ("002074.SZ", "国轩高科", "电池", 500),
    ("300207.SZ", "欣旺达", "电池", 400),
    ("688005.SH", "容百科技", "正极材料", 300),
    ("300896.SZ", "爱美客", "医美", 1100),
    ("600745.SH", "闻泰科技", "半导体", 800),
]

# Pattern injection: which patterns to force for certain stocks
# index, (pattern_key, description_of_what_to_inject)
PATTERN_INJECTIONS = [
    (0, "ma_golden_cross", "MA5 crosses MA20"),
    (1, "macd_golden_cross", "DIF crosses DEA"),
    (2, "bullish_alignment", "MA5>MA10>MA20>MA60"),
    (3, "volume_breakout", "20-day high close + high volume"),
    (4, "bollinger_breakout", "Close above upper band"),
    (5, "hammer", "Long lower shadow in downtrend"),
    (6, "morning_star", "Bear→small→bull 3-day reversal"),
    (7, "rsi_oversold", "RSI<30 yesterday, up today"),
    (2, "ma_golden_cross", "also MA cross"),
    (3, "bollinger_breakout", "also bollinger"),
    (8, "volume_breakout", "breakout"),
    (9, "ma_golden_cross", "golden cross"),
    (10, "bullish_alignment", "bullish alignment"),
    (11, "macd_golden_cross", "macd cross"),
    (12, "hammer", "hammer"),
    (14, "ma_golden_cross", "golden cross"),
    (16, "volume_breakout", "breakout"),
    (18, "bollinger_breakout", "bollinger"),
    (20, "morning_star", "morning star"),
    (22, "rsi_oversold", "rsi oversold"),
    (24, "bullish_alignment", "bullish alignment"),
    (26, "ma_golden_cross", "golden cross"),
    (28, "macd_golden_cross", "macd cross"),
    (30, "volume_breakout", "breakout"),
    (32, "bollinger_breakout", "bollinger"),
    (34, "hammer", "hammer"),
    (36, "ma_golden_cross", "golden cross"),
    (38, "morning_star", "morning star"),
    (7, "volume_breakout", "extra breakout"),
    (15, "ma_golden_cross", "golden cross"),
    (25, "bollinger_breakout", "bollinger"),
    (0, "bullish_alignment", "also alignment"),
    (5, "morning_star", "also morning star"),
]


def _generate_base_prices(n_days: int, trend: float = 0.0, volatility: float = 0.02) -> np.ndarray:
    """Generate a random price series with geometric Brownian motion."""
    rng = np.random.default_rng()
    returns = rng.normal(trend / 252, volatility, n_days)
    return 10.0 * np.cumprod(1 + returns)


def _generate_kline_for_stock(
    n_days: int,
    injections: list[dict],
    name: str,
    code: str,
) -> pd.DataFrame:
    """Generate K-line data for a single stock, with injected patterns."""
    rng = np.random.default_rng()

    # Base price series with slight uptrend
    prices = _generate_base_prices(n_days, trend=0.05, volatility=0.025)
    volume_base = rng.integers(1_000_000, 10_000_000, n_days).astype(float)

    # Apply pattern injections (modify the last few candles)
    for inj_map in injections:
        if "ma_golden_cross" in inj_map.get("type", ""):
            # Force MA5 to cross above MA20 on the last day
            prices[-25:-6] = prices[-25:-6] * 0.95  # dip before
            prices[-5:] = prices[-5:] * 1.08  # rally to cross
            volume_base[-1] *= 1.3

        if "macd_golden_cross" in inj_map.get("type", ""):
            prices[-40:-6] = prices[-40:-6] * 0.93
            prices[-5:] = prices[-5:] * 1.12
            volume_base[-1] *= 1.5

        if "bullish_alignment" in inj_map.get("type", ""):
            # Steady uptrend to create MA5>MA10>MA20>MA60
            ramp = np.linspace(0.85, 1.15, n_days)
            prices = prices * ramp
            volume_base[-1] *= 1.2

        if "volume_breakout" in inj_map.get("type", ""):
            prices[-1] = np.max(prices[:-1]) * 1.02  # 20-day high
            volume_base[-1] = np.mean(volume_base[-6:-1]) * 2.0  # 2x avg vol

        if "bollinger_breakout" in inj_map.get("type", ""):
            # Make price spike above the rolling upper band
            prices[-7:] = prices[-7:] * np.linspace(1.0, 1.18, 7)
            volume_base[-1] *= 1.6

        if "hammer" in inj_map.get("type", ""):
            # Long lower shadow in downtrend
            prices[-6:] = prices[-6:] * 0.93  # downtrend
            # Today: open near high, big dip, close back near open
            open_price = prices[-1] * 0.99
            low_price = prices[-1] * 0.90
            close_price = prices[-1] * 0.98
            high_price = prices[-1] * 0.995
            # We'll handle OHLC separately

        if "morning_star" in inj_map.get("type", ""):
            prices[-3] = prices[-4] * 0.92  # bear day
            prices[-2] = prices[-3] * 0.98  # small body
            prices[-1] = prices[-3] * 1.05  # bull day

        if "rsi_oversold" in inj_map.get("type", ""):
            prices[-16:-2] = prices[-16:-2] * np.linspace(1.0, 0.75, 14)  # steep drop
            prices[-1] = prices[-2] * 1.04  # rebound

    # Build OHLC from close prices
    body_ratio = 0.01
    shadow_ratio = 0.015
    opens = prices * (1 + rng.uniform(-body_ratio, body_ratio, n_days))
    highs = np.maximum(opens, prices) * (1 + rng.uniform(0, shadow_ratio, n_days))
    lows = np.minimum(opens, prices) * (1 - rng.uniform(0, shadow_ratio, n_days))
    closes = prices

    # Apply hammer-specific OHLC if needed
    for inj_map in injections:
        if "hammer" in inj_map.get("type", ""):
            # Last candle is a hammer
            opens[-1] = closes[-2] * 0.92
            lows[-1] = opens[-1] * 0.85  # very long lower shadow
            closes[-1] = opens[-1] * 1.005  # close near open
            highs[-1] = closes[-1] * 1.01
            volume_base[-1] *= 1.5

    pre_closes = np.roll(closes, 1)
    pre_closes[0] = closes[0] * 0.99

    return pd.DataFrame({
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(closes, 2),
        "volume": volume_base.astype(int),
        "pre_close": np.round(pre_closes, 2),
        "amount": (volume_base * closes).astype(float),
    })


def _generate_synthetic_pool(n: int) -> list[tuple]:
    """Generate synthetic stock entries beyond the fixed STOCK_POOL."""
    rng = np.random.default_rng(123)
    industries = ["银行", "白酒", "电池", "汽车", "保险", "家电", "医药", "安防",
                  "证券", "电力", "光伏", "养殖", "半导体", "工控", "AI", "电子",
                  "软件", "矿业", "医疗", "物流", "基建", "地产", "锂矿", "煤炭",
                  "稀土", "医美", "正极材料", "锂电设备"]
    extra = []
    for i in range(len(STOCK_POOL), n):
        if i % 3 == 0:
            code = f"6{10000 + i:05d}.SH"
        elif i % 3 == 1:
            code = f"0{10000 + i:05d}.SZ"
        else:
            code = f"3{10000 + i:05d}.SZ"
        name = f"合成股票{i:04d}"
        industry = rng.choice(industries)
        mcap = round(rng.uniform(100, 20000), 0)
        extra.append((code, name, industry, mcap))
    return extra


def generate_mock_data(
    trade_date: str = "20260629",
    n_stocks: int = 50,
    n_days: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate mock K-line, fund flow, and stock info data.

    Returns:
        (kline_df, flow_df, info_df) — same format as DataFetcher.fetch_all_data()
    """
    rng = np.random.default_rng(42)
    all_klines: list[pd.DataFrame] = []
    all_flows: list[dict] = []
    all_infos: list[dict] = []

    # Build pool: fixed entries + synthetic if needed
    pool = list(STOCK_POOL)
    if n_stocks > len(pool):
        pool += _generate_synthetic_pool(n_stocks)
    pool = pool[:n_stocks]

    # Map stock pool index to injections
    inj_map: dict[int, list[str]] = {}
    for pool_idx, pat_key, _desc in PATTERN_INJECTIONS:
        if pool_idx < n_stocks:
            if pool_idx not in inj_map:
                inj_map[pool_idx] = []
            inj_map[pool_idx].append(pat_key)

    # Add pattern injections for synthetic stocks beyond STOCK_POOL
    if n_stocks > len(STOCK_POOL):
        pat_keys = list(set(k for _, k, _ in PATTERN_INJECTIONS))
        extra_rng = np.random.default_rng(99)
        for i in range(len(STOCK_POOL), n_stocks):
            if extra_rng.random() < 0.4:  # 40% of extra stocks get a pattern
                if i not in inj_map:
                    inj_map[i] = []
                inj_map[i].append(extra_rng.choice(pat_keys))

    for i, (code, name, industry, mcap) in enumerate(pool):
        patterns_for_stock = inj_map.get(i, [])

        # Generate injection descriptors for this stock
        injections = [{"type": p} for p in patterns_for_stock]

        df = _generate_kline_for_stock(n_days, injections, name, code)
        df["symbol"] = code
        df["date_idx"] = range(len(df))

        all_klines.append(df)

        # Mock fund flow data
        has_patterns = len(patterns_for_stock) > 0
        inflow_rate = rng.uniform(6, 20) if has_patterns else rng.uniform(-5, 8)
        turnover = rng.uniform(5000, 500000) if has_patterns else rng.uniform(500, 80000)
        super_large = rng.uniform(100, 50000) if has_patterns and inflow_rate > 5 else rng.uniform(-1000, 5000)

        all_flows.append({
            "symbol": code,
            "main_net_inflow": round(turnover * inflow_rate / 100, 0),
            "super_large_net_inflow": round(super_large, 0),
            "large_net_inflow": round(turnover * inflow_rate / 100 * 0.4, 0),
            "turnover": round(turnover, 0),
            "main_inflow_rate": round(inflow_rate, 2),
        })

        # Mock stock info
        all_infos.append({
            "symbol": code,
            "name": name,
            "industry": industry,
            "market_cap": mcap,
            "list_status": "正常",
        })

    kline_df = pd.concat(all_klines, ignore_index=True)
    flow_df = pd.DataFrame(all_flows)
    info_df = pd.DataFrame(all_infos)

    return kline_df, flow_df, info_df
