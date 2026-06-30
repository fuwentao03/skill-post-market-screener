#!/usr/bin/env python3
"""Post-Market Screener — main entry point.

Data sources:
  - K-line & stock info: Pandadata (real API)
  - Fund flow: 同花顺 (Tonghuashun/10jqka) primary, East Money fallback
  - LLM analysis: Claude / DeepSeek API (real AI-generated)

Usage:
    python run.py                          # Scan latest trading day
    python run.py --date 20260629          # Scan specific date
    python run.py --no-flow                # Skip fund flow filter (纯形态扫描)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Ensure the skill root is on sys.path
SKILL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_ROOT))

from core.data_fetcher import _load_config
from core.pipeline import ScreenerPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-Market Screener — 双因子收盘扫描")
    p.add_argument("--date", help="Target trade date (YYYYMMDD). Default: latest.")
    p.add_argument("--config", default=str(SKILL_ROOT / "config.json"), help="Config path")
    p.add_argument("--no-flow", action="store_true", help="Skip fund flow filter (纯形态扫描)")
    p.add_argument("--top-n", type=int, default=None, help="Number of stocks in report")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config()
    scan_cfg = config.get("scan", {})
    top_n = args.top_n or scan_cfg.get("top_n", 20)

    pipeline = ScreenerPipeline(config)

    # Check trading day before running
    if args.date:
        from core.data_fetcher import DataFetcher
        fetcher = DataFetcher(config)
        try:
            fetcher.init_api()
        except RuntimeError as e:
            logger.error("API init failed: %s", e)
            return 1
        if not fetcher.is_trading_day(args.date):
            logger.info("%s is not a trading day. 今日休市.", args.date)
            print(f"{args.date} 休市，无需扫描。")
            return 0

    # Run pipeline — always real data + real LLM
    result = pipeline.run(
        trade_date=args.date,
        use_mock=False,
        no_flow=args.no_flow,
        dry_run=False,
        top_n=top_n,
    )

    if not result.ranked_stocks:
        if result.total_stocks == 0:
            logger.error("K-line data unavailable. Real-data pipeline failed.")
            return 1
        print(f"{result.trade_date} 无股票通过筛选。")
        return 0

    # Console summary
    ranked = result.ranked_stocks
    print(f"\n{'='*60}")
    print(f"收盘扫描完成 [完整版 双因子] — {result.trade_date}")
    print(f"全市场: {result.total_stocks} 只 | 入选: {len(ranked)} 只")
    for s in ranked[:5]:
        patterns = "+".join(s.get("triggered_patterns", [])[:2])
        print(f"  {s['rank']:>2}. {s['name']} ({s['code']}) 得分 {s['score']:.1f} | {patterns}")
    if len(ranked) > 5:
        print(f"  ... (共 {len(ranked)} 只)")
    print(f"\n报告已保存:")
    print(f"  Markdown: {result.md_path}")
    print(f"  JSON:     {result.json_path}")
    print(f"{'='*60}")

    # Validator — always strict for real-data runs
    validator = SKILL_ROOT / "scripts" / "validate_screener.py"
    if validator.exists():
        cmd = ["python", str(validator), result.md_path, result.json_path]
        if not args.no_flow and not result.flow_degraded_note:
            cmd.append("--strict")
        val_result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info("Validator: %s", val_result.stdout.strip() or val_result.stderr.strip())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
