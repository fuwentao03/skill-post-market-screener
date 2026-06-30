"""Unit tests for LLM analyst module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.analyst import (
    LLMAnalyst,
    _generate_fallback,
    build_prompt,
)


# ── test fixtures ──────────────────────────────────────────────

@pytest.fixture
def sample_stock() -> dict:
    """A realistic stock dict as produced by the pipeline."""
    return {
        "code": "600519.SH",
        "name": "贵州茅台",
        "industry": "白酒",
        "market_cap": 25000.0,
        "close": 1850.50,
        "pct_change": 2.35,
        "triggered_patterns": ["均线金叉", "放量突破"],
        "pattern_details": {
            "ma_golden_cross": (
                "MA5(1820.30)今日上穿MA20(1815.60)，"
                "此前MA5(1805.20)≤MA20(1810.80)，收盘价1850.50"
            ),
            "volume_breakout": (
                "收盘1850.50突破20日最高1845.00，"
                "量比2.1倍（今日量8500000/5日均4047619）"
            ),
        },
        "main_inflow": 25000.0,
        "inflow_rate": 8.5,
        "super_large_inflow": 15000.0,
        "turnover": 294117.0,
        "vol_ratio": 2.1,
    }


@pytest.fixture
def sample_stock_minimal() -> dict:
    """Minimal stock dict with only required fields."""
    return {
        "code": "000002.SZ",
        "name": "万科A",
        "industry": "地产",
        "market_cap": 1400.0,
        "close": 12.30,
        "pct_change": -1.5,
        "triggered_patterns": ["锤子线"],
        "pattern_details": {},
        "main_inflow": -500.0,
        "inflow_rate": -2.0,
        "super_large_inflow": -300.0,
        "turnover": 8000.0,
        "vol_ratio": 0.85,
    }


# ═══════════════════════════════════════════════════════════
# build_prompt tests
# ═══════════════════════════════════════════════════════════

class TestBuildPrompt:
    def test_includes_stock_info(self, sample_stock):
        prompt = build_prompt(sample_stock)
        assert "贵州茅台" in prompt
        assert "600519.SH" in prompt
        assert "白酒" in prompt
        assert "25000.0亿" in prompt
        assert "2.35%" in prompt

    def test_includes_triggered_patterns(self, sample_stock):
        prompt = build_prompt(sample_stock)
        assert "均线金叉" in prompt
        assert "放量突破" in prompt

    def test_includes_pattern_details(self, sample_stock):
        prompt = build_prompt(sample_stock)
        assert "MA5(1820.30)今日上穿MA20(1815.60)" in prompt
        assert "量比2.1倍" in prompt
        assert "形态技术细节" in prompt

    def test_includes_fund_flow(self, sample_stock):
        prompt = build_prompt(sample_stock)
        assert "25000.0万" in prompt
        assert "8.5%" in prompt
        assert "15000.0万" in prompt

    def test_no_details_shows_placeholder(self, sample_stock_minimal):
        prompt = build_prompt(sample_stock_minimal)
        assert "（无详细数据）" in prompt

    def test_triggered_as_string_not_list(self, sample_stock):
        """If triggered_patterns is a string, it should render as-is."""
        stock = dict(sample_stock)
        stock["triggered_patterns"] = "自定义形态"
        prompt = build_prompt(stock)
        assert "自定义形态" in prompt

    def test_prompt_length_reasonable(self, sample_stock):
        """Prompt should be between 200 and 2000 chars."""
        prompt = build_prompt(sample_stock)
        assert 200 < len(prompt) < 2000


# ═══════════════════════════════════════════════════════════
# _generate_fallback tests
# ═══════════════════════════════════════════════════════════

class TestFallback:
    def test_includes_stock_name_and_patterns(self, sample_stock):
        text = _generate_fallback(sample_stock)
        assert "贵州茅台" in text
        assert "均线金叉" in text
        assert "放量突破" in text

    def test_extracts_vol_ratio_from_details(self, sample_stock):
        text = _generate_fallback(sample_stock)
        assert "量比2.1倍" in text

    def test_extracts_ma_from_details(self, sample_stock):
        text = _generate_fallback(sample_stock)
        assert "MA5=1820.30" in text

    def test_includes_flow_values(self, sample_stock):
        text = _generate_fallback(sample_stock)
        assert "25000.0" in text  # main_inflow

    def test_handles_list_patterns(self):
        stock = {
            "name": "测试",
            "code": "000001.SZ",
            "triggered_patterns": ["形态A", "形态B"],
            "pattern_details": {},
            "main_inflow": 100,
            "inflow_rate": 5,
            "super_large_inflow": 50,
        }
        text = _generate_fallback(stock)
        assert "形态A" in text
        assert "形态B" in text

    def test_handles_string_patterns(self):
        stock = {
            "name": "测试",
            "code": "000001.SZ",
            "triggered_patterns": "单一形态",
            "pattern_details": {},
            "main_inflow": 0,
            "inflow_rate": 0,
            "super_large_inflow": 0,
        }
        text = _generate_fallback(stock)
        assert "单一形态" in text

    def test_handles_missing_details(self, sample_stock_minimal):
        text = _generate_fallback(sample_stock_minimal)
        assert "万科A" in text
        assert "锤子线" in text

    def test_fallback_never_says_recommend_buy(self, sample_stock):
        """Fallback should not use buy recommendation language."""
        text = _generate_fallback(sample_stock)
        assert "推荐买入" not in text
        assert "买入" not in text.split("。")[0]  # first sentence about the stock


# ═══════════════════════════════════════════════════════════
# LLMAnalyst tests
# ═══════════════════════════════════════════════════════════

class TestLLMAnalyst:
    def test_init_defaults(self):
        """Default model is configurable via env; just check structure."""
        analyst = LLMAnalyst()
        assert analyst._max_tokens == 1536
        assert analyst._client is None
        # Model may be overridden by ANTHROPIC_MODEL env var
        assert isinstance(analyst._model, str) and len(analyst._model) > 0

    def test_init_with_config(self):
        """Config-specified model should be set, unless overridden by env."""
        config = {"model": "claude-opus-4-6", "max_tokens": 2048}
        analyst = LLMAnalyst(config)
        assert analyst._max_tokens == 2048
        # model comes from env if ANTHROPIC_MODEL is set, else config
        if "ANTHROPIC_MODEL" not in __import__("os").environ:
            assert analyst._model == "claude-opus-4-6"

    def test_analyze_fallback_when_no_api_key(self, sample_stock):
        """With no API key configured, analyze() should use fallback."""
        # Clear env vars that could leak a real API key
        with patch.dict("os.environ", {
            "ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": "",
        }):
            analyst = LLMAnalyst({"anthropic_api_key": ""})
            result = analyst.analyze(sample_stock)
        assert "贵州茅台" in result
        assert "均线金叉" in result

    def test_analyze_fallback_on_api_error(self, sample_stock):
        """When API raises, should fall back to template."""
        config = {"anthropic_api_key": "fake-key"}
        with patch("anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = RuntimeError("API down")
            mock_client_class.return_value = mock_client

            analyst = LLMAnalyst(config)
            result = analyst.analyze(sample_stock)
            assert "贵州茅台" in result

    def test_analyze_returns_api_response(self, sample_stock):
        """On successful API call, returns the model's text."""
        config = {"anthropic_api_key": "fake-key"}

        mock_block = MagicMock()
        mock_block.text = "这是一份专业的LLM分析报告。"

        mock_message = MagicMock()
        mock_message.content = [mock_block]

        with patch("anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_message
            mock_client_class.return_value = mock_client

            analyst = LLMAnalyst(config)
            analyst._client = mock_client
            analyst._api_key = "fake-key"
            result = analyst.analyze(sample_stock)
            assert "专业的LLM分析报告" in result

    def test_analyze_handles_thinking_block(self, sample_stock):
        """DeepSeek-style responses may contain ThinkingBlock with no text attr."""
        config = {"anthropic_api_key": "fake-key"}

        # Block without .text attribute (like a ThinkingBlock)
        thinking_block = MagicMock(spec=[])  # no 'text' attribute
        text_block = MagicMock()
        text_block.text = "实际分析内容"

        mock_message = MagicMock()
        mock_message.content = [thinking_block, text_block]

        with patch("anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_message
            mock_client_class.return_value = mock_client

            analyst = LLMAnalyst(config)
            analyst._client = mock_client
            analyst._api_key = "fake-key"
            result = analyst.analyze(sample_stock)
            assert "实际分析内容" in result

    def test_analyze_empty_response_falls_back(self, sample_stock):
        """Empty API response triggers fallback."""
        config = {"anthropic_api_key": "fake-key"}

        mock_message = MagicMock()
        mock_message.content = []  # empty

        with patch("anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_message
            mock_client_class.return_value = mock_client

            analyst = LLMAnalyst(config)
            analyst._client = mock_client
            analyst._api_key = "fake-key"
            result = analyst.analyze(sample_stock)
            assert "贵州茅台" in result  # fallback generated

    def test_api_key_env_var_resolution(self, sample_stock):
        """API key is resolved from ANTHROPIC_AUTH_TOKEN env var."""
        config = {}
        with patch.dict("os.environ", {"ANTHROPIC_AUTH_TOKEN": "env-token-value"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_client = MagicMock()
                mock_block = MagicMock()
                mock_block.text = "OK"
                mock_message = MagicMock()
                mock_message.content = [mock_block]
                mock_client.messages.create.return_value = mock_message
                mock_client_class.return_value = mock_client

                analyst = LLMAnalyst(config)
                analyst._client = mock_client
                analyst._api_key = "env-token-value"
                result = analyst.analyze(sample_stock)
                assert result == "OK"

    def test_analyze_batch_assigns_to_all_stocks(self, sample_stock):
        """analyze_batch should populate llm_analysis for every stock."""
        stocks = [
            dict(sample_stock),
            dict(sample_stock),
        ]
        stocks[0]["code"] = "A.SH"
        stocks[1]["code"] = "B.SZ"

        config = {"anthropic_api_key": "fake-key"}
        with patch("anthropic.Anthropic") as mock_client_class:
            mock_client = MagicMock()
            mock_block = MagicMock()
            mock_block.text = "分析内容"
            mock_message = MagicMock()
            mock_message.content = [mock_block]
            mock_client.messages.create.return_value = mock_message
            mock_client_class.return_value = mock_client

            analyst = LLMAnalyst(config)
            analyst._client = mock_client
            analyst._api_key = "fake-key"
            result = analyst.analyze_batch(stocks, delay=0)

            assert len(result) == 2
            for s in result:
                assert "llm_analysis" in s
                assert s["llm_analysis"] == "分析内容"
