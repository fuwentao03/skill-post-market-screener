"""Unit tests for report generation (Markdown + JSON)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reporter import (
    _safe_format,
    generate_json,
    generate_markdown,
    save_report,
)


# ── fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_stocks():
    return [
        {
            "rank": 1,
            "name": "贵州茅台",
            "code": "600519.SH",
            "industry": "白酒",
            "score": 22.0,
            "pct_change": 2.35,
            "triggered_patterns": ["均线金叉", "放量突破"],
            "pattern_details": {},
            "main_inflow": 25000.0,
            "inflow_rate": 8.5,
            "super_large_inflow": 15000.0,
            "llm_analysis": "贵州茅台触发均线金叉和放量突破，值得关注。",
        },
        {
            "rank": 2,
            "name": "宁德时代",
            "code": "300750.SZ",
            "industry": "电池",
            "score": 18.5,
            "pct_change": -1.2,
            "triggered_patterns": ["MACD金叉"],
            "pattern_details": {},
            "main_inflow": 8000.0,
            "inflow_rate": 3.5,
            "super_large_inflow": 2000.0,
            "llm_analysis": "MACD金叉值得跟踪。",
        },
    ]


@pytest.fixture
def sample_config():
    return {
        "scan": {
            "min_market_cap": 30,
            "min_turnover": 5000,
        },
        "flow": {
            "main_inflow_rate_min": 0.05,
        },
        "patterns": {
            "ma_golden_cross": True,
            "macd_golden_cross": True,
            "bullish_alignment": True,
            "volume_breakout": True,
            "bollinger_breakout": True,
            "hammer": True,
            "morning_star": True,
            "rsi_oversold": True,
        },
        "output": {"dir": "output"},
    }


# ═══════════════════════════════════════════════════════════
# _safe_format tests
# ═══════════════════════════════════════════════════════════

class TestSafeFormat:
    def test_normal_float(self):
        assert _safe_format(3.14159, ".2f") == "3.14"

    def test_none_value(self):
        assert _safe_format(None, ".2f") == "N/A"

    def test_string_value(self):
        assert _safe_format("hello") == "hello"

    def test_bad_format_string(self):
        assert _safe_format(42, "bad") == "42"

    def test_zero(self):
        assert _safe_format(0, ".0f") == "0"


# ═══════════════════════════════════════════════════════════
# generate_markdown tests
# ═══════════════════════════════════════════════════════════

class TestGenerateMarkdown:
    def test_basic_report(self, sample_stocks, sample_config):
        md = generate_markdown(
            sample_stocks, "20260629", 5189, sample_config,
            pattern_stats={"均线金叉": 42, "放量突破": 18},
            industry_dist={"白酒": 3, "电池": 5},
        )
        assert "收盘扫描日报 — 20260629" in md
        assert "全市场 5189 只" in md
        assert "入选 2 只" in md
        assert "贵州茅台" in md
        assert "600519.SH" in md
        assert "宁德时代" in md
        assert "免责声明" in md

    def test_includes_triggered_patterns(self, sample_stocks, sample_config):
        md = generate_markdown(sample_stocks, "20260629", 100, sample_config)
        assert "均线金叉" in md
        assert "MACD金叉" in md

    def test_includes_fund_flow(self, sample_stocks, sample_config):
        md = generate_markdown(sample_stocks, "20260629", 100, sample_config)
        assert "25000万" in md
        assert "8.5%" in md

    def test_llm_analysis_included(self, sample_stocks, sample_config):
        md = generate_markdown(sample_stocks, "20260629", 100, sample_config)
        assert "值得关注" in md

    def test_negative_pct_change(self, sample_stocks, sample_config):
        md = generate_markdown(sample_stocks, "20260629", 100, sample_config)
        assert "-1.20%" in md

    def test_empty_stocks_list(self, sample_config):
        md = generate_markdown([], "20260629", 100, sample_config)
        assert "入选 0 只" in md
        assert "收盘扫描日报" in md

    def test_empty_pattern_stats(self, sample_stocks, sample_config):
        md = generate_markdown(
            sample_stocks, "20260629", 100, sample_config,
            pattern_stats={},
        )
        assert "形态触发统计" in md
        assert "N/A" in md

    def test_empty_industry_dist(self, sample_stocks, sample_config):
        md = generate_markdown(
            sample_stocks, "20260629", 100, sample_config,
            industry_dist={},
        )
        assert "行业分布" in md
        assert "N/A" in md

    def test_flow_degraded_note(self, sample_stocks, sample_config):
        md = generate_markdown(
            sample_stocks, "20260629", 100, sample_config,
            flow_degraded_note="AKShare资金流向不可用，使用模拟数据",
        )
        assert "AKShare资金流向不可用，使用模拟数据" in md

    def test_all_patterns_disabled(self, sample_stocks):
        config = {
            "scan": {"min_market_cap": 30, "min_turnover": 5000},
            "flow": {"main_inflow_rate_min": 0.05},
            "patterns": {},
        }
        md = generate_markdown(sample_stocks, "20260629", 100, config)
        assert "全部" in md  # fallback label

    def test_none_values_in_stock(self, sample_config):
        stock = {
            "rank": 1,
            "name": "测试",
            "code": "000001.SZ",
            "score": None,
            "pct_change": None,
            "triggered_patterns": [],
            "main_inflow": None,
            "inflow_rate": None,
            "super_large_inflow": None,
            "llm_analysis": None,
        }
        md = generate_markdown([stock], "20260629", 100, sample_config)
        assert "000001.SZ" in md
        assert "N/A" in md  # safe_format uses N/A for None


# ═══════════════════════════════════════════════════════════
# generate_json tests
# ═══════════════════════════════════════════════════════════

class TestGenerateJson:
    def test_basic_structure(self, sample_stocks):
        result = generate_json(sample_stocks, "20260629", 5189)
        assert result["trade_date"] == "20260629"
        assert result["total_stocks"] == 5189
        assert result["selected_count"] == 2
        assert len(result["stocks"]) == 2

    def test_stock_data_preserved(self, sample_stocks):
        result = generate_json(sample_stocks, "20260629", 100)
        assert result["stocks"][0]["name"] == "贵州茅台"
        assert result["stocks"][0]["score"] == 22.0
        assert result["stocks"][1]["code"] == "300750.SZ"

    def test_pattern_stats_passed_through(self, sample_stocks):
        result = generate_json(
            sample_stocks, "20260629", 100,
            pattern_stats={"均线金叉": 42},
            industry_dist={"白酒": 3},
        )
        assert result["pattern_stats"]["均线金叉"] == 42
        assert result["industry_distribution"]["白酒"] == 3

    def test_empty_stocks(self):
        result = generate_json([], "20260629", 0)
        assert result["selected_count"] == 0
        assert result["stocks"] == []

    def test_none_stats_become_empty_dicts(self):
        result = generate_json([], "20260629", 0)
        assert result["pattern_stats"] == {}
        assert result["industry_distribution"] == {}


# ═══════════════════════════════════════════════════════════
# save_report tests
# ═══════════════════════════════════════════════════════════

class TestSaveReport:
    def test_saves_both_files(self, sample_stocks, sample_config, tmp_path):
        md_path, json_path = save_report(
            sample_stocks, "20260629", 5189, sample_config,
            output_dir=str(tmp_path),
        )
        assert Path(md_path).exists()
        assert Path(json_path).exists()
        assert md_path.endswith(".md")
        assert json_path.endswith(".json")

    def test_date_folder_format(self, sample_stocks, sample_config, tmp_path):
        md_path, _ = save_report(
            sample_stocks, "20260629", 5189, sample_config,
            output_dir=str(tmp_path),
        )
        assert "2026-06-29" in md_path

    def test_md_content_is_valid_markdown(self, sample_stocks, sample_config, tmp_path):
        md_path, _ = save_report(
            sample_stocks, "20260629", 5189, sample_config,
            output_dir=str(tmp_path),
        )
        content = Path(md_path).read_text(encoding="utf-8")
        assert "# " in content
        assert "贵州茅台" in content

    def test_json_content_is_valid_json(self, sample_stocks, sample_config, tmp_path):
        _, json_path = save_report(
            sample_stocks, "20260629", 5189, sample_config,
            output_dir=str(tmp_path),
        )
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert data["trade_date"] == "20260629"
        assert len(data["stocks"]) == 2

    def test_flow_degraded_note_in_files(self, sample_stocks, sample_config, tmp_path):
        md_path, _ = save_report(
            sample_stocks, "20260629", 5189, sample_config,
            output_dir=str(tmp_path),
            flow_degraded_note="模拟数据",
        )
        content = Path(md_path).read_text(encoding="utf-8")
        assert "模拟数据" in content

    def test_empty_stocks_saves_files(self, sample_config, tmp_path):
        md_path, json_path = save_report(
            [], "20260629", 0, sample_config, output_dir=str(tmp_path),
        )
        assert Path(md_path).exists()
        assert Path(json_path).exists()

    def test_default_output_dir_from_config(self, sample_stocks, sample_config, tmp_path):
        sample_config["output"]["dir"] = str(tmp_path / "custom_out")
        md_path, _ = save_report(
            sample_stocks, "20260629", 5189, sample_config,
        )
        assert Path(md_path).exists()
        assert "custom_out" in md_path
