#!/usr/bin/env python3
"""Fetch real A-share K-line data via AKShare and save to cache format.

Usage:
    python scripts/fetch_real_cache.py                # 50 stocks, 120 days
    python scripts/fetch_real_cache.py --n 100 --days 250
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("fetch_cache")

# 50 representative A-share stocks across industries
DEFAULT_POOL = [
    "600519", "000858", "300750", "002594", "601318",  # 茅台 五粮液 宁德 比亚迪 平安
    "600036", "000333", "600276", "002415", "300059",  # 招行 美的 恒瑞 海康 东财
    "600900", "601012", "002714", "688981", "000651",  # 长电 隆基 牧原 中芯 格力
    "300124", "601166", "002230", "600030", "601888",  # 汇川 兴业 讯飞 中信 中免
    "002475", "300274", "688111", "601899", "300015",  # 立讯 阳光 金山 紫金 爱尔
    "603259", "000625", "002352", "300498", "601390",  # 药明 长安 顺丰 温氏 中铁
    "000002", "002142", "300014", "601668", "002460",  # 万科 宁波 亿纬 建筑 赣锋
    "300347", "000568", "600809", "002304", "300760",  # 泰格 老窖 汾酒 洋河 迈瑞
    "601088", "000831", "002466", "300450", "600104",  # 神华 稀土 天齐 先导 上汽
    "002074", "300207", "688005", "300896", "600745",  # 国轩 欣旺达 容百 爱美客 闻泰
]

AK_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
}


def fetch_one_stock(code: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily K-line for one stock."""
    try:
        raw = ak.stock_zh_a_hist(
            symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq"
        )
        if raw.empty:
            return None
        df = raw.rename(columns=AK_COLUMN_MAP)
        df["symbol"] = code
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Add pre_close (previous day's close)
        df = df.sort_values("date")
        df["pre_close"] = df["close"].shift(1)
        # Drop first row (NaN pre_close)
        df = df.dropna(subset=["pre_close"])
        keep = ["symbol", "date", "open", "high", "low", "close", "volume", "amount", "pre_close"]
        return df[[c for c in keep if c in df.columns]]
    except Exception as e:
        logger.warning("Fetch failed for %s: %s", code, e)
        return None


def main():
    p = argparse.ArgumentParser(description="Fetch real A-share data for cache")
    p.add_argument("--n", type=int, default=50, help="Number of stocks")
    p.add_argument("--days", type=int, default=120, help="Lookback days")
    args = p.parse_args()

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=args.days * 2)).strftime("%Y%m%d")
    today = datetime.now().strftime("%Y%m%d")

    pool = DEFAULT_POOL[: args.n]
    logger.info("Fetching %d stocks from %s to %s", len(pool), start_date, end_date)

    all_frames: list[pd.DataFrame] = []
    success = 0
    for i, code in enumerate(pool):
        df = fetch_one_stock(code, start_date, end_date)
        if df is not None and not df.empty:
            all_frames.append(df)
            success += 1
            logger.info("[%d/%d] %s: %d days", i + 1, len(pool), code, len(df))
        else:
            logger.warning("[%d/%d] %s: FAILED", i + 1, len(pool), code)
        if i < len(pool) - 1:
            time.sleep(0.3)  # be polite to the API

    if not all_frames:
        logger.error("No data fetched.")
        return 1

    kline_df = pd.concat(all_frames, ignore_index=True)
    logger.info("Total: %d stocks, %d rows", success, len(kline_df))

    # Save to cache
    cache_dir = SKILL_ROOT / "cache" / today
    cache_dir.mkdir(parents=True, exist_ok=True)
    kline_df.to_parquet(cache_dir / "kline.parquet", index=False)

    # Also save dummy flow and info so CacheManager.load() works
    symbols = sorted(kline_df["symbol"].unique())
    flow_df = pd.DataFrame({"symbol": symbols, "main_inflow_rate": [0.0] * len(symbols)})
    info_df = pd.DataFrame({"symbol": symbols, "name": symbols, "industry": ["未知"] * len(symbols)})
    flow_df.to_parquet(cache_dir / "flow.parquet", index=False)
    info_df.to_parquet(cache_dir / "info.parquet", index=False)

    from core.cache import CacheManager
    mgr = CacheManager(cache_root=str(SKILL_ROOT / "cache"))
    logger.info("Cache saved: %s (%d stocks, %d rows)", cache_dir, success, len(kline_df))
    logger.info("Cached dates: %s", mgr.cached_dates)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
