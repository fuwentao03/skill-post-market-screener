"""Report generator: produces Markdown and JSON output."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """# 收盘扫描日报 — {trade_date}

{data_banner}
> 扫描时间：{scan_time} | 全市场 {total_stocks} 只 | 入选 {selected_count} 只

## 扫描配置

| 参数 | 值 |
|---|---|
| 最小市值 | {min_market_cap}亿 |
| 最小成交额 | {min_turnover}万 |
| 形态检测器 | {active_detectors} |
| 主力净流入率阈值 | {main_inflow_rate_min}% |

## 今日精选

{stock_entries}

## 形态触发统计

{pattern_stats_table}

## 行业分布

{industry_table}

## 数据来源

{data_provenance_table}

## 数据说明

- 数据截止时间：{data_cutoff}
- 缺失或降级数据：{missing_data_note}
- 评分公式：pattern_score（行业中性化）+ flow_score + quality_bonus（详见 JSON 输出）
- 统计口径：涨跌停不含 ST、不含一字板

## 免责声明

> **风险提示：** 本报告由自动化扫描系统生成，所有数据仅供参考，不构成任何投资建议。股市有风险，投资需谨慎。
"""

STOCK_ENTRY_TEMPLATE = """### {rank}. {name}（{code}） 得分 {score} | 收盘 {close} | {pct_change}

{llm_badge}> {llm_analysis}

#### 触发形态：{triggered_patterns}
#### 资金信号：主力净流入 {main_inflow}万 | 流入率 {inflow_rate}% | 超大单 {super_large}万

---
"""

# Rendered when one or more data sources are not real
DATA_WARNING_BANNER = """> **⚠️ 数据真实性警告：** 本次扫描使用了非真实数据源。报告结果仅供测试参考，**不构成任何投资建议**。正常运行时，系统使用 Pandadata + 同花顺 实盘数据。
"""


def _safe_format(value, fmt: str = ".2f") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def _render_data_provenance(prov: dict | None) -> str:
    """Render the data provenance table and overall attestation."""
    if not prov:
        return "（数据来源未记录）"

    rows = []
    status_icons = {True: "✅ 真实", False: "❌ 非真实"}

    for key, label in [
        ("kline", "K线数据"),
        ("flow", "资金流向"),
        ("stock_info", "股票信息"),
        ("llm", "LLM分析"),
    ]:
        entry = prov.get(key)
        if entry:
            icon = status_icons.get(entry["is_real"], "⚠️ 未知")
            rows.append(f"| {label} | {entry['source']} | {icon} |")

    table = "| 数据类别 | 来源 | 状态 |\n|---|---|---|\n" + "\n".join(rows)

    if prov.get("all_real", False):
        attestation = "\n> ✅ **数据真实性确认：** 本次扫描所有数据均来自真实数据源，大模型分析结果由 API 实时生成。"
    else:
        attestation = "\n> ❌ **数据真实性警告：** 部分数据源非真实数据，报告结果仅供参考。"

    return table + "\n" + attestation


def _render_llm_badge(stock: dict) -> str:
    """Return a small badge indicating the LLM analysis source."""
    source = stock.get("llm_source", "")
    if source == "real":
        return ""
    elif source == "dry_run":
        return "> 🟡 分析来源：Dry-run（跳过LLM）\n\n"
    elif source == "fallback":
        return "> 🟠 分析来源：模板回退（LLM API不可用）\n\n"
    elif source == "error":
        return "> 🔴 分析来源：生成失败\n\n"
    return ""


def generate_markdown(
    stocks: list[dict],
    trade_date: str,
    total_stocks: int,
    config: dict,
    pattern_stats: Optional[dict] = None,
    industry_dist: Optional[dict] = None,
    missing_note: str = "无",
    flow_degraded_note: Optional[str] = None,
    data_provenance: Optional[dict] = None,
) -> str:
    """Generate Markdown daily report."""
    scan_cfg = config.get("scan", {})
    flow_cfg = config.get("flow", {})
    pattern_cfg = config.get("patterns", {})

    active_detectors = ", ".join(k for k, v in pattern_cfg.items() if v)
    if not active_detectors:
        active_detectors = "全部"

    # Data warning banner
    all_real = data_provenance.get("all_real", True) if data_provenance else True
    data_banner = "" if all_real else DATA_WARNING_BANNER

    # Stock entries
    stock_entries = ""
    for s in stocks:
        triggered = s.get("triggered_patterns", [])
        if isinstance(triggered, list):
            triggered = " + ".join(triggered)
        pct = s.get("pct_change", 0) or 0
        pct_str = f"+{pct:.2f}%" if pct >= 0 else f"{pct:.2f}%"
        llm_badge = _render_llm_badge(s)
        entry = STOCK_ENTRY_TEMPLATE.format(
            rank=s.get("rank", "?"),
            name=s.get("name", "?"),
            code=s.get("code", "?"),
            score=_safe_format(s.get("score"), ".1f"),
            close=_safe_format(s.get("close"), ".2f"),
            pct_change=pct_str,
            llm_badge=llm_badge,
            llm_analysis=s.get("llm_analysis", "（分析待生成）"),
            triggered_patterns=triggered,
            main_inflow=_safe_format(s.get("main_inflow"), ".0f"),
            inflow_rate=_safe_format(s.get("inflow_rate"), ".1f"),
            super_large=_safe_format(s.get("super_large_inflow"), ".0f"),
        )
        stock_entries += entry

    # Pattern stats table
    if pattern_stats:
        rows = ""
        for name, count in pattern_stats.items():
            pct_val = (count / max(total_stocks, 1)) * 100
            rows += f"| {name} | {count} | {pct_val:.1f}% |\n"
    else:
        rows = "| N/A | 0 | 0% |\n"
    pattern_stats_table = f"| 形态 | 触发次数 | 占比 |\n|---|---|---|\n{rows}"

    # Industry table
    if industry_dist:
        rows = ""
        for ind, count in industry_dist.items():
            rows += f"| {ind} | {count} |\n"
    else:
        rows = "| N/A | 0 |\n"
    industry_table = f"| 行业 | 入选数量 |\n|---|---|\n{rows}"

    report = REPORT_TEMPLATE.format(
        trade_date=trade_date,
        data_banner=data_banner,
        scan_time=datetime.now().strftime("%H:%M:%S"),
        total_stocks=total_stocks,
        selected_count=len(stocks),
        min_market_cap=scan_cfg.get("min_market_cap", 30),
        min_turnover=scan_cfg.get("min_turnover", 5000),
        active_detectors=active_detectors,
        main_inflow_rate_min=int(flow_cfg.get("main_inflow_rate_min", 0.05) * 100),
        stock_entries=stock_entries,
        pattern_stats_table=pattern_stats_table,
        industry_table=industry_table,
        data_provenance_table=_render_data_provenance(data_provenance),
        data_cutoff=f"{trade_date} 收盘",
        missing_data_note=(
            "; ".join(filter(None, [missing_note, flow_degraded_note]))
            if flow_degraded_note
            else missing_note
        ),
    )
    return report


def generate_json(
    stocks: list[dict],
    trade_date: str,
    total_stocks: int,
    pattern_stats: Optional[dict] = None,
    industry_dist: Optional[dict] = None,
    data_provenance: Optional[dict] = None,
) -> dict:
    """Generate structured JSON output."""
    return {
        "trade_date": trade_date,
        "scan_time": datetime.now().isoformat(),
        "total_stocks": total_stocks,
        "selected_count": len(stocks),
        "stocks": stocks,
        "pattern_stats": pattern_stats or {},
        "industry_distribution": industry_dist or {},
        "data_provenance": data_provenance or {},
    }


def save_report(
    stocks: list[dict],
    trade_date: str,
    total_stocks: int,
    config: dict,
    pattern_stats: Optional[dict] = None,
    industry_dist: Optional[dict] = None,
    output_dir: Optional[str] = None,
    flow_degraded_note: Optional[str] = None,
    data_provenance: Optional[dict] = None,
) -> tuple[str, str]:
    """Save Markdown and JSON reports to output_dir.

    Returns:
        (md_path, json_path)
    """
    base_dir = Path(output_dir or config.get("output", {}).get("dir", "output"))
    # Convert YYYYMMDD → YYYY-MM-DD for the date folder
    date_folder = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}"
    out_dir = base_dir / date_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    missing_note = "无"
    if not pattern_stats:
        missing_note = "形态统计数据不可用"

    md_content = generate_markdown(
        stocks, trade_date, total_stocks, config, pattern_stats, industry_dist,
        missing_note, flow_degraded_note, data_provenance=data_provenance,
    )
    json_content = generate_json(
        stocks, trade_date, total_stocks, pattern_stats, industry_dist,
        data_provenance=data_provenance,
    )

    md_path = out_dir / f"daily_screener_{trade_date}.md"
    json_path = out_dir / f"daily_screener_{trade_date}.json"

    md_path.write_text(md_content, encoding="utf-8")
    json_path.write_text(json.dumps(json_content, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Report saved: %s, %s", md_path, json_path)
    return str(md_path), str(json_path)
