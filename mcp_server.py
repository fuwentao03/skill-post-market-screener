#!/usr/bin/env python3
"""MCP server exposing the post-market screener as callable tools.

Start with:
    python mcp_server.py
Or via mcp CLI:
    mcp dev mcp_server.py

LLMs can then call:
    - run_screener: Full daily scan with dual-factor screening
    - get_latest_report: Read the most recent report
    - check_trading_day: Verify if a date is an A-share trading day
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

SKILL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_ROOT))

from core.data_fetcher import DataFetcher, _load_config
from core.pipeline import ScreenerPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_server")

mcp = FastMCP("post-market-screener")


def _format_result(result) -> dict:
    """Format PipelineResult as MCP-friendly dict."""
    if not result.ranked_stocks:
        return {
            "status": "ok" if result.total_stocks > 0 else "error",
            "message": (
                f"{result.trade_date} 无股票通过筛选。"
                if result.total_stocks > 0
                else "K线数据不可用，今日扫描终止。"
            ),
            "trade_date": result.trade_date,
            "total_stocks": result.total_stocks,
        }

    top3 = []
    for s in result.ranked_stocks[:3]:
        top3.append({
            "name": s.get("name", "?"),
            "code": s.get("code", "?"),
            "score": s.get("score", 0),
            "pct_change": s.get("pct_change", 0),
        })

    return {
        "status": "ok",
        "trade_date": result.trade_date,
        "total_stocks": result.total_stocks,
        "flow_passed": result.flow_passed_count,
        "cross_validated": result.cross_validated,
        "selected_count": len(result.ranked_stocks),
        "top3": top3,
        "all_stocks": [
            {
                "rank": s.get("rank"),
                "name": s.get("name"),
                "code": s.get("code"),
                "score": s.get("score"),
                "pct_change": s.get("pct_change"),
                "triggered_patterns": s.get("triggered_patterns", []),
                "inflow_rate": s.get("inflow_rate"),
                "main_inflow": s.get("main_inflow"),
                "llm_analysis": s.get("llm_analysis", ""),
            }
            for s in result.ranked_stocks
        ],
        "md_path": result.md_path,
        "json_path": result.json_path,
    }


# ═══════════════════════════════════════════════════════════
# MCP Tools
# ═══════════════════════════════════════════════════════════

@mcp.tool(
    name="run_screener",
    annotations={
        "title": "运行收盘扫描",
        "readOnlyHint": False,
        "destructiveHint": False,
    },
)
def run_screener(
    date: str | None = None,
    no_flow: bool = False,
    top_n: int = 20,
) -> str:
    """Run the A-share post-market dual-factor stock screener.

    Scans all ~5189 A-share stocks using 8 technical pattern detectors and
    main-capital flow filtering, then returns top-ranked stocks with LLM analysis.

    All data is from real sources: Pandadata (K-line + stock info), 同花顺 (fund flow),
    and Claude/DeepSeek API (LLM analysis).

    Args:
        date: Target trading date in YYYYMMDD format. Defaults to latest trading day.
        no_flow: If True, skip fund flow filter (pattern-only scan).
        top_n: Number of top-ranked stocks to return (default 20).
    """
    logger.info("MCP tool: run_screener(date=%s, no_flow=%s, top_n=%s)",
                date, no_flow, top_n)

    config = _load_config()
    pipeline = ScreenerPipeline(config)
    result = pipeline.run(
        trade_date=date, use_mock=False, no_flow=no_flow,
        dry_run=False, top_n=top_n,
    )
    return json.dumps(_format_result(result), ensure_ascii=False, indent=2)


@mcp.tool(
    name="get_latest_report",
    annotations={
        "title": "获取最新扫描报告",
        "readOnlyHint": True,
        "destructiveHint": False,
    },
)
def get_latest_report() -> str:
    """Read and return the most recent daily screener Markdown report."""
    output_dir = SKILL_ROOT / "output"
    if not output_dir.exists():
        return json.dumps({"status": "error", "message": "报告目录不存在，请先运行扫描。"}, ensure_ascii=False)

    md_files = sorted(output_dir.glob("*/daily_screener_*.md"), reverse=True)
    if not md_files:
        return json.dumps({"status": "error", "message": "暂无报告，请先运行扫描。"}, ensure_ascii=False)

    latest = md_files[0]
    content = latest.read_text(encoding="utf-8")
    return json.dumps({
        "status": "ok",
        "file": str(latest),
        "trade_date": latest.stem.replace("daily_screener_", ""),
        "content_preview": content[:2000],
        "total_chars": len(content),
    }, ensure_ascii=False, indent=2)


@mcp.tool(
    name="check_trading_day",
    annotations={
        "title": "检查交易日",
        "readOnlyHint": True,
        "destructiveHint": False,
    },
)
def check_trading_day(date: str | None = None) -> str:
    """Check if a given date is an A-share trading day.

    Args:
        date: Date in YYYYMMDD format. Defaults to latest trading day.
    """
    config = _load_config()
    fetcher = DataFetcher(config)
    try:
        fetcher.init_api()
    except RuntimeError as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)

    resolved = date or fetcher.get_last_trade_date()
    is_trade = fetcher.is_trading_day(resolved)
    return json.dumps({
        "status": "ok",
        "date": resolved,
        "is_trading_day": is_trade,
        "message": f"{resolved} 是交易日" if is_trade else f"{resolved} 休市",
    }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
