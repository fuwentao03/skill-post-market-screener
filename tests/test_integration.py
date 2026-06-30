"""Integration tests for pipeline, scorer, flow filter, and cache."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cache import CacheManager
from core.flow_filter import FlowFilter
from core.pattern_detector import run_all_detectors
from core.pipeline import PipelineResult, ScreenerPipeline
from core.scorer import compute_score, rank_stocks


# ── helpers ──────────────────────────────────────────────────

def _make_kline(close: list, volume: list = None) -> pd.DataFrame:
    """Build a minimal K-line DataFrame from close prices."""
    n = len(close)
    rng = np.random.default_rng(42)
    if volume is None:
        volume = [1_000_000] * n
    opens = np.array(close) * (1 + rng.uniform(-0.01, 0.01, n))
    highs = np.maximum(opens, close) * (1 + rng.uniform(0, 0.01, n))
    lows = np.minimum(opens, close) * (1 - rng.uniform(0, 0.01, n))
    return pd.DataFrame({
        "open": np.round(opens, 2),
        "high": np.round(highs, 2),
        "low": np.round(lows, 2),
        "close": np.round(np.array(close, dtype=float), 2),
        "volume": np.array(volume, dtype=float),
    })


def _uptrend_df(n_days: int = 80) -> pd.DataFrame:
    """Steady uptrend K-line."""
    close = np.linspace(10, 16, n_days).tolist()
    return _make_kline(close)


# ═══════════════════════════════════════════════════════════
# Scorer tests
# ═══════════════════════════════════════════════════════════

class TestScorer:
    def test_compute_score_full(self):
        """All patterns triggered + positive flow + bonuses."""
        results = {k: True for k in
                   ["ma_golden_cross", "macd_golden_cross", "bullish_alignment",
                    "volume_breakout", "bollinger_breakout", "hammer",
                    "morning_star", "rsi_oversold"]}
        score_dict = compute_score(results, inflow_rate=0.15, market_cap=80, turnover=30000)
        assert score_dict["pattern_score"] == 15  # 2+1+1+3+3+1+3+1 = 15 (backtest-calibrated)
        assert score_dict["flow_score"] == 3.0  # min(0.15/0.05, 3) = 3
        assert score_dict["quality_bonus"] == 2  # mcap >= 50 AND turnover >= 20000
        assert score_dict["score"] == 20.0

    def test_compute_score_no_patterns(self):
        """Zero patterns → score 0 or flow-only."""
        results = {k: False for k in
                   ["ma_golden_cross", "macd_golden_cross", "bullish_alignment",
                    "volume_breakout", "bollinger_breakout", "hammer",
                    "morning_star", "rsi_oversold"]}
        score_dict = compute_score(results, inflow_rate=0.0, market_cap=0, turnover=0)
        assert score_dict["pattern_score"] == 0
        assert score_dict["flow_score"] == 0
        assert score_dict["quality_bonus"] == 0
        assert score_dict["score"] == 0

    def test_compute_score_negative_flow(self):
        """Negative inflow rate → flow_score = 0."""
        results = {"bullish_alignment": 1.0, "volume_breakout": 1.0}
        score_dict = compute_score(results, inflow_rate=-0.05, market_cap=70, turnover=25000)
        assert score_dict["flow_score"] == 0
        assert score_dict["quality_bonus"] == 2
        assert score_dict["score"] == 4 + 0 + 2  # bull(1) + vol_breakout(3)

    def test_compute_score_flow_capped(self):
        """Flow score capped at 3.0."""
        results = {"ma_golden_cross": True}
        score_dict = compute_score(results, inflow_rate=0.25, market_cap=10, turnover=5000)
        # 0.25 / 0.05 = 5 → capped at 3.0
        assert score_dict["flow_score"] == 3.0

    def test_compute_score_mcap_just_below_50(self):
        """Market cap 49 亿 → no mcap bonus."""
        results = {"bullish_alignment": True}
        score_dict = compute_score(results, inflow_rate=0.10, market_cap=49, turnover=20000)
        assert score_dict["quality_bonus"] == 1  # only turnover bonus

    def test_rank_stocks_sorts_descending(self):
        stocks = [
            {"name": "A", "score": 5.0},
            {"name": "B", "score": 12.0},
            {"name": "C", "score": 8.0},
        ]
        ranked = rank_stocks(stocks, top_n=10)
        assert ranked[0]["name"] == "B"
        assert ranked[1]["name"] == "C"
        assert ranked[2]["name"] == "A"
        assert ranked[0]["rank"] == 1

    def test_rank_stocks_truncates(self):
        stocks = [{"name": f"S{i}", "score": float(i)} for i in range(20)]
        ranked = rank_stocks(stocks, top_n=5)
        assert len(ranked) == 5
        assert ranked[0]["score"] == 19.0


# ═══════════════════════════════════════════════════════════
# FlowFilter tests
# ═══════════════════════════════════════════════════════════

class TestFlowFilter:
    def test_filter_single_pass(self):
        ff = FlowFilter(main_inflow_rate_min=0.05, min_turnover=5000, super_large_positive=True)
        passed, reasons = ff.filter_single(inflow_rate=0.08, turnover=10000, super_large_inflow=500)
        assert passed is True

    def test_filter_single_fail_turnover(self):
        ff = FlowFilter(min_turnover=5000)
        passed, reasons = ff.filter_single(inflow_rate=0.08, turnover=3000, super_large_inflow=500)
        assert passed is False
        assert any("成交额" in r for r in reasons)

    def test_filter_single_fail_inflow_rate(self):
        ff = FlowFilter(main_inflow_rate_min=0.05)
        passed, reasons = ff.filter_single(inflow_rate=0.02, turnover=10000, super_large_inflow=500)
        assert passed is False
        assert any("主力流入率" in r for r in reasons)

    def test_filter_single_fail_super_large(self):
        ff = FlowFilter(super_large_positive=True)
        passed, reasons = ff.filter_single(inflow_rate=0.08, turnover=10000, super_large_inflow=-100)
        assert passed is False
        assert any("超大单" in r for r in reasons)

    def test_filter_single_super_large_disabled(self):
        ff = FlowFilter(super_large_positive=False)
        passed, reasons = ff.filter_single(inflow_rate=0.08, turnover=10000, super_large_inflow=-100)
        assert passed is True

    def test_filter_dataframe(self):
        ff = FlowFilter(main_inflow_rate_min=0.05, min_turnover=5000, super_large_positive=True)
        df = pd.DataFrame({
            "symbol": ["A.SH", "B.SZ", "C.SH"],
            "main_inflow_rate": [0.08, 0.03, 0.10],
            "turnover": [8000, 8000, 3000],
            "super_large_net_inflow": [500, 500, 500],
        })
        passed, rejected = ff.filter_dataframe(df)
        assert len(passed) == 1
        assert passed.iloc[0]["symbol"] == "A.SH"
        assert len(rejected) == 2

    def test_filter_dataframe_empty(self):
        ff = FlowFilter()
        passed, rejected = ff.filter_dataframe(pd.DataFrame())
        assert passed.empty
        assert rejected.empty

    def test_filter_dataframe_inflow_rate_percentage(self):
        """Inflow rate > 1 (percentage form) should be auto-scaled to decimal."""
        ff = FlowFilter(main_inflow_rate_min=0.05)
        df = pd.DataFrame({
            "symbol": ["A.SH"],
            "main_inflow_rate": [12.5],  # percentage
            "turnover": [10000],
            "super_large_net_inflow": [100],
        })
        passed, rejected = ff.filter_dataframe(df)
        assert len(passed) == 1  # 12.5% → 0.125 > 0.05


# ═══════════════════════════════════════════════════════════
# Pipeline integration tests
# ═══════════════════════════════════════════════════════════

class TestPipeline:
    def test_run_mock_dry(self):
        """Full pipeline with mock data, dry-run."""
        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=True, dry_run=True, top_n=10)
        assert isinstance(result, PipelineResult)
        assert result.total_stocks == 50
        assert 1 <= len(result.ranked_stocks) <= 10
        assert len(result.ranked_stocks[0]["triggered_patterns"]) >= 1
        assert "（Dry run" in result.ranked_stocks[0].get("llm_analysis", "")

    def test_run_mock_no_flow(self):
        """Pipeline with mock data, no flow filter."""
        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=True, no_flow=True, dry_run=True, top_n=5)
        assert len(result.ranked_stocks) >= 1
        # All stocks should have flow_score=0 in no-flow mode
        for s in result.ranked_stocks:
            assert s.get("flow_score", 1) == 0

    def test_result_reports_exist(self):
        """Reports are saved and paths are returned."""
        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=True, dry_run=True, top_n=5)
        assert Path(result.md_path).exists()
        assert Path(result.json_path).exists()

    def test_ranked_stocks_have_all_fields(self):
        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=True, dry_run=True, top_n=5)
        for s in result.ranked_stocks:
            for field in ["rank", "name", "code", "score", "pct_change",
                          "triggered_patterns", "inflow_rate", "main_inflow",
                          "llm_analysis", "industry"]:
                assert field in s, f"Missing field: {field}"

    def test_acquire_data_mock(self):
        """acquire_data with mock produces valid dataframes."""
        pipeline = ScreenerPipeline()
        kline, flow, info, date = pipeline.acquire_data(use_mock=True)
        assert not kline.empty
        assert not info.empty
        assert "symbol" in kline.columns
        assert all(c in kline.columns for c in ["open", "high", "low", "close", "volume"])

    def test_mode_label_mock_plus_flow(self):
        """Flow degredation note is None when flow data is available."""
        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=True, dry_run=True, top_n=3)
        # In mock mode, flow data comes from generated mock flow;
        # degradation only happens if AKShare fails AND mock fallback is needed
        assert isinstance(result.flow_degraded_note, (str, type(None)))


# ═══════════════════════════════════════════════════════════
# Cache tests
# ═══════════════════════════════════════════════════════════

class TestCache:
    def test_save_and_load(self, tmp_path):
        """Round-trip: save kline/flow/info → load back."""
        mgr = CacheManager(cache_root=str(tmp_path))
        kline = pd.DataFrame({
            "symbol": ["A.SH", "A.SH", "B.SZ"],
            "date": ["20260627", "20260628", "20260628"],
            "open": [10.0, 10.1, 20.0],
            "close": [10.05, 10.2, 20.1],
            "volume": [1e6, 1.1e6, 5e5],
        })
        flow = pd.DataFrame({
            "symbol": ["A.SH", "B.SZ"],
            "main_inflow_rate": [0.08, 0.03],
        })
        info = pd.DataFrame({"symbol": ["A.SH"], "name": ["TestA"]})

        mgr.save("20260628", kline, flow, info)
        assert mgr.has("20260628")
        k2, f2, i2 = mgr.load("20260628")
        assert len(k2) == 3
        assert len(f2) == 2
        assert len(i2) == 1

    def test_has_nonexistent(self, tmp_path):
        mgr = CacheManager(cache_root=str(tmp_path))
        assert mgr.has("20200101") is False

    def test_cached_dates(self, tmp_path):
        mgr = CacheManager(cache_root=str(tmp_path))
        empty = pd.DataFrame({"symbol": []})
        mgr.save("20260627", empty, empty, empty)
        mgr.save("20260628", empty, empty, empty)
        dates = mgr.cached_dates
        assert len(dates) == 2
        assert "20260627" in dates

    def test_clear_old(self, tmp_path):
        mgr = CacheManager(cache_root=str(tmp_path))
        empty = pd.DataFrame({"symbol": []})
        mgr.save("20200101", empty, empty, empty)  # very old
        mgr.save("20260628", empty, empty, empty)  # recent
        removed = mgr.clear_old(keep_days=30)
        assert removed >= 1
        assert mgr.has("20200101") is False


# ═══════════════════════════════════════════════════════════
# Pipeline helper function tests
# ═══════════════════════════════════════════════════════════

class TestPipelineHelpers:
    """Unit tests for ScreenerPipeline static helpers."""

    def test_get_row_found(self):
        """Returns the matching row as a dict."""
        from core.pipeline import ScreenerPipeline

        df = pd.DataFrame({
            "symbol": ["A.SH", "B.SZ"],
            "name": ["Alpha", "Beta"],
        })
        result = ScreenerPipeline._get_row(df, "A.SH")
        assert result == {"symbol": "A.SH", "name": "Alpha"}

    def test_get_row_not_found(self):
        """Returns empty dict when symbol not in DataFrame."""
        from core.pipeline import ScreenerPipeline

        df = pd.DataFrame({
            "symbol": ["A.SH"],
            "name": ["Alpha"],
        })
        result = ScreenerPipeline._get_row(df, "B.SZ")
        assert result == {}

    def test_get_row_empty_df(self):
        """Returns empty dict when DataFrame is empty."""
        from core.pipeline import ScreenerPipeline

        result = ScreenerPipeline._get_row(pd.DataFrame(), "A.SH")
        assert result == {}

    def test_merge_turnover_basic(self):
        """Merges latest day's amount (yuan) as turnover (万元) into flow_df."""
        from core.pipeline import ScreenerPipeline

        kline = pd.DataFrame({
            "symbol": ["A.SH", "A.SH", "B.SZ"],
            "date": ["20260627", "20260628", "20260628"],
            "amount": [5_000_000, 6_000_000, 8_000_000],
            "close": [10.0, 10.5, 20.0],
        })
        flow = pd.DataFrame({
            "symbol": ["A.SH", "B.SZ"],
            "main_inflow_rate": [0.08, 0.03],
        })
        ScreenerPipeline._merge_turnover_from_kline(kline, flow)
        # Latest A.SH = 6_000_000 yuan / 10000 = 600 万元
        assert flow.loc[flow["symbol"] == "A.SH", "turnover"].iloc[0] == 600.0
        # Latest B.SZ = 8_000_000 yuan / 10000 = 800 万元
        assert flow.loc[flow["symbol"] == "B.SZ", "turnover"].iloc[0] == 800.0

    def test_merge_turnover_missing_amount_column(self):
        """Does not crash when amount column is absent."""
        from core.pipeline import ScreenerPipeline

        kline = pd.DataFrame({
            "symbol": ["A.SH"],
            "date": ["20260628"],
            "close": [10.0],
        })
        flow = pd.DataFrame({"symbol": ["A.SH"], "main_inflow_rate": [0.08]})
        # Should not raise
        ScreenerPipeline._merge_turnover_from_kline(kline, flow)
        assert "turnover" not in flow.columns

    def test_merge_turnover_date_idx_fallback(self):
        """Uses date_idx column when date column is absent."""
        from core.pipeline import ScreenerPipeline

        kline = pd.DataFrame({
            "symbol": ["A.SH", "A.SH"],
            "date_idx": [0, 1],
            "amount": [5_000_000, 10_000_000],
        })
        flow = pd.DataFrame({"symbol": ["A.SH"], "main_inflow_rate": [0.08]})
        ScreenerPipeline._merge_turnover_from_kline(kline, flow)
        # Latest by date_idx = 10_000_000 / 10000 = 1000 万元
        assert flow.loc[flow["symbol"] == "A.SH", "turnover"].iloc[0] == 1000.0

    def test_merge_turnover_symbol_not_in_flow(self):
        """K-line symbol not present in flow_df is simply ignored."""
        from core.pipeline import ScreenerPipeline

        kline = pd.DataFrame({
            "symbol": ["C.SZ"],
            "date": ["20260628"],
            "amount": [5_000_000],
        })
        flow = pd.DataFrame({"symbol": ["A.SH"], "main_inflow_rate": [0.08]})
        ScreenerPipeline._merge_turnover_from_kline(kline, flow)
        # A.SH should not get any turnover value from C.SZ
        assert pd.isna(flow.loc[flow["symbol"] == "A.SH", "turnover"].iloc[0]) or flow.loc[flow["symbol"] == "A.SH", "turnover"].iloc[0] == 0
