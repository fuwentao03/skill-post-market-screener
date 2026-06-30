#!/usr/bin/env python3
"""Validate post-market screener output files (Markdown report + JSON data)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# --- Markdown validation ---

MD_REQUIRED_SECTIONS = [
    ("title", r"^#\s+收盘扫描日报", "一级标题需要为「收盘扫描日报」"),
    ("scan_config", r"^##\s+扫描配置", "缺少扫描配置章节"),
    ("picks", r"^##\s+今日精选", "缺少今日精选章节"),
    ("pattern_stats", r"^##\s+形态触发统计", "缺少形态触发统计章节"),
    ("industry_dist", r"^##\s+行业分布", "缺少行业分布章节"),
    ("data_notes", r"^##\s+数据说明", "缺少数据说明章节"),
]


def validate_md(text: str) -> list[str]:
    issues: list[str] = []

    if len(text.strip()) < 800:
        issues.append("报告内容过短，可能不是完整扫描")

    for _key, pattern, message in MD_REQUIRED_SECTIONS:
        if not re.search(pattern, text, flags=re.MULTILINE):
            issues.append(message)

    # Check for per-stock analysis: 触发形态 + 资金信号
    if not re.search(r"触发形态", text):
        issues.append("缺少个股形态标注")
    if not re.search(r"资金信号", text):
        issues.append("缺少个股资金信号")

    # Check data source and date
    if not re.search(r"(数据来源|使用接口|Pandadata)", text):
        issues.append("缺少数据来源或使用接口说明")
    if not re.search(r"(数据截止|截止时间)", text):
        issues.append("缺少数据截止时间说明")

    # Check scoring reference
    if not re.search(r"(得分|score)", text, re.IGNORECASE):
        issues.append("缺少个股权重得分")

    # Disclaimer
    if not re.search(r"(不构成.*投资建议|不提供操作建议|仅[供作].*参考)", text):
        issues.append("缺少非投资建议声明")

    return issues


# --- JSON validation ---

JSON_REQUIRED_TOP_KEYS = [
    "trade_date",
    "scan_time",
    "total_stocks",
    "selected_count",
    "stocks",
    "pattern_stats",
    "industry_distribution",
]

JSON_REQUIRED_STOCK_KEYS = [
    "rank",
    "code",
    "name",
    "industry",
    "market_cap",
    "close",
    "pct_change",
    "score",
    "pattern_score",
    "flow_score",
    "quality_bonus",
    "triggered_patterns",
    "main_inflow",
    "inflow_rate",
    "super_large_inflow",
    "turnover",
    "llm_analysis",
]


def validate_json(data: dict, strict_flow: bool = True) -> list[str]:
    issues: list[str] = []

    for key in JSON_REQUIRED_TOP_KEYS:
        if key not in data:
            issues.append(f"JSON 缺少顶层字段: {key}")

    stocks = data.get("stocks", [])
    if not isinstance(stocks, list):
        issues.append("JSON 'stocks' 应为数组")
        return issues

    if len(stocks) == 0:
        issues.append("JSON 'stocks' 为空，今日可能无入选股票")
        return issues

    for i, stock in enumerate(stocks):
        for key in JSON_REQUIRED_STOCK_KEYS:
            if key not in stock:
                issues.append(f"stocks[{i}] 缺少字段: {key}")

        # Validate score = pattern_score + flow_score + quality_bonus
        score = stock.get("score", 0)
        pattern_s = stock.get("pattern_score", 0)
        flow_s = stock.get("flow_score", 0)
        quality_b = stock.get("quality_bonus", 0)
        if abs(score - (pattern_s + flow_s + quality_b)) > 0.01:
            issues.append(
                f"stocks[{i}] ({stock.get('code', '?')}) 得分不一致: "
                f"score={score} != pattern({pattern_s}) + flow({flow_s}) + quality({quality_b})"
            )

        # Flow threshold checks (only in strict dual-factor mode)
        if strict_flow:
            inflow_rate = stock.get("inflow_rate", 0)
            turnover = stock.get("turnover", 0)
            super_large = stock.get("super_large_inflow", 0)
            if inflow_rate <= 5:
                issues.append(
                    f"stocks[{i}] ({stock.get('code', '?')}) 主力流入率 {inflow_rate}% <= 5% 阈值"
                )
            if turnover <= 5000:
                issues.append(
                    f"stocks[{i}] ({stock.get('code', '?')}) 成交额 {turnover}万 <= 5000万 阈值"
                )
            if super_large <= 0:
                issues.append(
                    f"stocks[{i}] ({stock.get('code', '?')}) 超大单净流入 {super_large}万 <= 0"
                )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("md_path", type=Path, help="Path to the Markdown report")
    parser.add_argument("json_path", type=Path, help="Path to the JSON data file")
    parser.add_argument("--strict", action="store_true", help="Enable strict flow threshold checking (dual-factor mode)")
    args = parser.parse_args()

    all_issues: list[str] = []
    exit_code = 0

    # Validate Markdown
    try:
        md_text = args.md_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        all_issues.append(f"Markdown 文件未找到: {args.md_path}")
        exit_code = 2
        md_text = ""

    if md_text:
        md_issues = validate_md(md_text)
        all_issues.extend(md_issues)

    # Validate JSON
    try:
        json_text = args.json_path.read_text(encoding="utf-8-sig")
        json_data = json.loads(json_text)
    except FileNotFoundError:
        all_issues.append(f"JSON 文件未找到: {args.json_path}")
        exit_code = 2
        json_data = {}
    except json.JSONDecodeError as e:
        all_issues.append(f"JSON 解析失败: {e}")
        exit_code = 2
        json_data = {}

    if json_data:
        json_issues = validate_json(json_data, strict_flow=args.strict)
        all_issues.extend(json_issues)

    if all_issues:
        print("FAIL")
        for issue in all_issues:
            print(f"- {issue}")
        return exit_code or 1

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
