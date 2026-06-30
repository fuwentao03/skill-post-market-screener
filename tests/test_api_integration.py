"""Integration tests with mocked Pandadata and AKShare API responses.

These tests verify that the data fetching and processing layers correctly
handle realistic API response shapes without requiring actual network
access or credentials.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.data_fetcher import DataFetcher, _load_config
from core.flow_fetcher import FlowFetcher, _add_exchange_suffix


# ── Mock data factories ────────────────────────────────────────

def _make_mock_universe(n: int = 100) -> pd.DataFrame:
    """Realistic stock universe DataFrame mimicking pdd.get_trade_list."""
    rng = np.random.default_rng(42)
    codes = []
    for i in range(n):
        if i % 3 == 0:
            codes.append(f"60{i:04d}.SH")
        elif i % 3 == 1:
            codes.append(f"00{i:04d}.SZ")
        else:
            codes.append(f"30{i:04d}.SZ")
    names = [f"股票{i:04d}" for i in range(n)]
    industries = rng.choice(
        ["银行", "白酒", "半导体", "医药", "电力", "汽车", "地产", "证券"],
        n,
    )
    return pd.DataFrame({
        "symbol": codes,
        "name": names,
        "sector_code_name": industries,
        "list_status": ["正常"] * n,
        "status": ["1"] * n,
    })


def _make_mock_kline(**kwargs):
    """Realistic daily K-line DataFrame — matches pdd.get_stock_daily kwargs."""
    rng = np.random.default_rng(42)
    syms = kwargs.get("symbol", kwargs.get("symbols", []))
    if isinstance(syms, str):
        syms = [syms]
    dates = pd.date_range(start="2026-01-01", end="2026-06-29", freq="B")[-120:]
    date_strs = [d.strftime("%Y%m%d") for d in dates]

    rows = []
    for sym in syms[:30]:
        for ds in date_strs:
            base = 10 + rng.random() * 90
            close = base * (1 + rng.normal(0.0005, 0.02))
            open_p = close * (1 + rng.uniform(-0.01, 0.01))
            high = max(open_p, close) * (1 + rng.uniform(0, 0.015))
            low = min(open_p, close) * (1 - rng.uniform(0, 0.015))
            volume = rng.integers(500_000, 20_000_000)
            rows.append({
                "symbol": sym,
                "date": ds,
                "open": round(open_p, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
                "amount": round(volume * close, 2),
                "pre_close": round(close * 0.99, 2),
            })
    return pd.DataFrame(rows)


def _make_mock_stock_detail(**kwargs):
    """Realistic stock detail DataFrame — matches pdd.get_stock_detail kwargs."""
    syms = kwargs.get("symbol", kwargs.get("symbols", []))
    if isinstance(syms, str):
        syms = [syms]
    rows = []
    for sym in syms:
        rows.append({
            "symbol": sym,
            "name": f"股票{sym[:6]}",
            "sector_code_name": "半导体",
            "status": "正常",
        })
    return pd.DataFrame(rows)


def _make_mock_share_float(**kwargs):
    """Realistic share float DataFrame — matches pdd.get_share_float kwargs."""
    rng = np.random.default_rng(42)
    syms = kwargs.get("symbol", kwargs.get("symbols", []))
    if isinstance(syms, str):
        syms = [syms]
    rows = []
    end_date = kwargs.get("end_date", "20260629")
    for sym in syms:
        rows.append({
            "symbol": sym,
            "date": end_date,
            "total_a": int(rng.integers(1_0000_0000, 50_0000_0000)),
        })
    return pd.DataFrame(rows)


def _make_mock_fund_flow_raw(indicator=None):
    """Realistic AKShare fund flow DataFrame."""
    rng = np.random.default_rng(42)
    n = 200
    rows = []
    for i in range(n):
        code = f"{600000 + i:06d}" if i % 2 == 0 else f"{300000 + i:06d}"
        inflow = rng.uniform(-5000, 50000)
        rows.append({
            "代码": code,
            "名称": f"股票{code}",
            "今日主力净流入-净额": inflow,
            "今日主力净流入-净占比": rng.uniform(-8, 15),
            "今日超大单净流入-净额": inflow * rng.uniform(0.3, 0.7),
            "今日大单净流入-净额": inflow * 0.4,
        })
    return pd.DataFrame(rows)


def _make_mock_tonghuashun_flow() -> pd.DataFrame:
    """Realistic Tonghuashun (同花顺) fund flow DataFrame."""
    rng = np.random.default_rng(42)
    n = 200
    rows = []
    for i in range(n):
        code = f"{600000 + i:06d}" if i % 2 == 0 else f"{300000 + i:06d}"
        net_wan = rng.uniform(-5000, 50000)
        turnover_wan = rng.uniform(10000, 200000)
        # Mixed 亿/万 format as returned by the API
        if abs(net_wan) >= 10000:
            net_str = f"{net_wan / 10000:.2f}亿"
        else:
            net_str = f"{net_wan:.2f}万"
        if turnover_wan >= 10000:
            turnover_str = f"{turnover_wan / 10000:.2f}亿"
        else:
            turnover_str = f"{turnover_wan:.2f}万"
        rows.append({
            "序号": i + 1,
            "股票代码": int(code),
            "股票简称": f"股票{code}",
            "最新价": rng.uniform(5, 200),
            "涨跌幅": f"{rng.uniform(-5, 5):.2f}%",
            "换手率": f"{rng.uniform(0.5, 15):.2f}%",
            "流入资金": f"{rng.uniform(5000, 100000):.2f}万",
            "流出资金": f"{rng.uniform(5000, 100000):.2f}万",
            "净额": net_str,
            "成交额": turnover_str,
        })
    return pd.DataFrame(rows)


# ── Mocked API fixtures ────────────────────────────────────────

@pytest.fixture
def mock_pandadata():
    """Mock all panda_data SDK functions used by DataFetcher."""
    with patch("core.data_fetcher.pdd") as mock_pdd:
        mock_pdd.get_last_trade_date.return_value = "20260629"
        mock_pdd.get_trade_cal.return_value = pd.DataFrame({
            "is_trade": ["1"],
        })
        mock_pdd.get_trade_list.return_value = _make_mock_universe(100)
        mock_pdd.get_stock_daily.side_effect = _make_mock_kline
        mock_pdd.get_stock_detail.side_effect = _make_mock_stock_detail
        mock_pdd.get_share_float.side_effect = _make_mock_share_float
        mock_pdd.init_token.return_value = None
        yield mock_pdd


@pytest.fixture
def mock_akshare():
    """Mock akshare fund flow function — imported lazily inside FlowFetcher."""
    with patch("akshare.stock_individual_fund_flow_rank") as mock_func:
        mock_func.return_value = _make_mock_fund_flow_raw()
        yield mock_func


# ═══════════════════════════════════════════════════════════════
# DataFetcher tests (mocked Pandadata API)
# ═══════════════════════════════════════════════════════════════

class TestDataFetcherMocked:
    """Test DataFetcher with mocked Pandadata SDK."""

    def test_get_last_trade_date(self, mock_pandadata):
        fetcher = DataFetcher()
        fetcher.init_api()
        result = fetcher.get_last_trade_date()
        assert result == "20260629"
        mock_pandadata.get_last_trade_date.assert_called_once()

    def test_is_trading_day_true(self, mock_pandadata):
        fetcher = DataFetcher()
        fetcher.init_api()
        assert fetcher.is_trading_day("20260629") is True

    def test_is_trading_day_empty_calendar(self, mock_pandadata):
        mock_pandadata.get_trade_cal.return_value = pd.DataFrame()
        fetcher = DataFetcher()
        fetcher.init_api()
        assert fetcher.is_trading_day("20260629") is False

    def test_get_stock_universe(self, mock_pandadata):
        fetcher = DataFetcher()
        fetcher.init_api()
        universe = fetcher.get_stock_universe("20260629")
        assert len(universe) == 100
        assert "symbol" in universe.columns
        mock_pandadata.get_trade_list.assert_called_with(date="20260629")

    def test_fetch_kline_batch(self, mock_pandadata):
        fetcher = DataFetcher()
        fetcher.init_api()
        symbols = ["600000.SH", "000001.SZ", "300001.SZ"]
        df = fetcher.fetch_kline_batch(symbols, "20260101", "20260629")
        assert not df.empty
        assert "symbol" in df.columns
        assert "close" in df.columns
        assert len(df["symbol"].unique()) >= 1

    def test_fetch_kline_batch_empty_symbols(self, mock_pandadata):
        fetcher = DataFetcher()
        fetcher.init_api()
        df = fetcher.fetch_kline_batch([], "20260101", "20260629")
        assert df.empty

    def test_fetch_kline_batch_api_error(self, mock_pandadata):
        mock_pandadata.get_stock_daily.side_effect = RuntimeError("API error")
        fetcher = DataFetcher()
        fetcher.init_api()
        df = fetcher.fetch_kline_batch(["600000.SH"], "20260101", "20260629")
        assert df.empty  # graceful degradation

    def test_fetch_stock_details_batch(self, mock_pandadata):
        fetcher = DataFetcher()
        fetcher.init_api()
        symbols = ["600000.SH", "000001.SZ"]
        df = fetcher.fetch_stock_details_batch(symbols)
        assert not df.empty
        assert "name" in df.columns
        assert "industry" in df.columns

    def test_fetch_stock_details_empty(self, mock_pandadata):
        mock_pandadata.get_stock_detail.side_effect = RuntimeError("down")
        fetcher = DataFetcher()
        fetcher.init_api()
        df = fetcher.fetch_stock_details_batch(["600000.SH"])
        assert df.empty

    def test_fetch_market_cap_batch(self, mock_pandadata):
        fetcher = DataFetcher()
        fetcher.init_api()
        symbols = ["600000.SH", "000001.SZ"]
        result = fetcher.fetch_market_cap_batch(symbols, "20260629")
        assert len(result) > 0
        for v in result.values():
            assert v > 0  # positive shares

    def test_fetch_all_data_full(self, mock_pandadata, mock_akshare):
        """Full fetch_all_data generates valid DataFrames (kline + info only)."""
        fetcher = DataFetcher()
        fetcher.init_api()
        kline, info = fetcher.fetch_all_data("20260629", lookback_days=60)
        assert not kline.empty
        assert not info.empty
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in kline.columns
        for col in ["symbol", "name", "industry"]:
            assert col in info.columns


# ═══════════════════════════════════════════════════════════════
# FlowFetcher tests (mocked AKShare API)
# ═══════════════════════════════════════════════════════════════

class TestFlowFetcherMocked:
    """Test FlowFetcher with mocked AKShare responses."""

    def test_fetch_normal(self, mock_akshare, monkeypatch):
        """Tonghuashun primary path returns valid flow data."""
        # Mock Tonghuashun as primary source
        ths_df = pd.DataFrame([{
            "symbol": "600000.SH", "main_net_inflow": 1000.0,
            "main_inflow_rate": 8.5, "super_large_net_inflow": 500.0,
            "large_net_inflow": 500.0, "turnover": 50000.0,
        }, {
            "symbol": "600001.SH", "main_net_inflow": 2000.0,
            "main_inflow_rate": 6.0, "super_large_net_inflow": 1000.0,
            "large_net_inflow": 1000.0, "turnover": 80000.0,
        }])
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun",
            lambda _self, _symbols: ths_df,
        )

        ff = FlowFetcher()
        symbols = ["600000.SH", "600001.SH", "300000.SZ"]
        df = ff.fetch(symbols)
        assert not df.empty
        for col in ["symbol", "main_net_inflow", "main_inflow_rate",
                     "super_large_net_inflow"]:
            assert col in df.columns
        assert all(s.endswith((".SH", ".SZ")) for s in df["symbol"])
        assert df.attrs.get("flow_source") == "tonghuashun"

    def test_fetch_empty_symbols(self, mock_akshare):
        ff = FlowFetcher()
        df = ff.fetch([])
        assert df.empty

    def test_fetch_retry_then_succeed(self, mock_akshare, monkeypatch):
        """AKShare succeeds on retry after transient failures (Tonghuashun fails first)."""
        # Tonghuashun fails → falls through to AKShare
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun",
            lambda _self, _symbols: pd.DataFrame(),
        )

        call_count = [0]

        def flaky_fetch(indicator=None):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Transient error")
            return _make_mock_fund_flow_raw()

        mock_akshare.side_effect = flaky_fetch

        ff = FlowFetcher(max_retries=3, base_delay=0.01, backoff_factor=1.0)
        df = ff.fetch(["600000.SH"])
        assert not df.empty
        assert call_count[0] == 3  # 2 failures + 1 success
        assert df.attrs.get("flow_source") == "akshare"

    def test_fetch_all_retries_exhausted(self, mock_akshare, monkeypatch):
        """Returns empty DataFrame when ALL paths (AKShare + East Money + Tonghuashun) fail."""
        mock_akshare.side_effect = ConnectionError("Down")

        def mock_fail(_self, _symbols):
            return pd.DataFrame()
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_eastmoney", mock_fail,
        )
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun", mock_fail,
        )

        ff = FlowFetcher(max_retries=3, base_delay=0.01, backoff_factor=1.0)
        df = ff.fetch(["600000.SH"])
        assert df.empty

    def test_fetch_empty_raw_data_falls_back(self, mock_akshare, monkeypatch):
        """When Tonghuashun + AKShare return empty, East Money fallback is attempted."""
        # Tonghuashun fails
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun",
            lambda _self, _symbols: pd.DataFrame(),
        )
        # AKShare returns empty
        mock_akshare.return_value = pd.DataFrame()

        em_called = []

        def mock_em(_self, _symbols):
            em_called.append(True)
            # Return a minimal valid flow row
            return pd.DataFrame([{
                "main_net_inflow": 1000.0, "main_inflow_rate": 8.5,
                "super_large_net_inflow": 500.0, "large_net_inflow": 500.0,
                "symbol": "600000.SH", "turnover": 50000.0,
            }])

        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_eastmoney", mock_em,
        )
        ff = FlowFetcher()
        df = ff.fetch(["600000.SH"])
        assert not df.empty, "East Money fallback should provide data"
        assert len(em_called) == 1, "East Money fallback must be called"
        assert df.attrs.get("flow_source") == "eastmoney"

    def test_fetch_tonghuashun_primary(self, mock_akshare, monkeypatch):
        """Tonghuashun is the primary fund flow source (used first)."""
        ths_called = []

        def mock_ths(_self, _symbols):
            ths_called.append(True)
            return pd.DataFrame([{
                "symbol": "600000.SH", "main_net_inflow": 2000.0,
                "main_inflow_rate": 5.0, "super_large_net_inflow": 1000.0,
                "large_net_inflow": 1000.0, "turnover": 40000.0,
            }])

        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun", mock_ths,
        )
        ff = FlowFetcher()
        df = ff.fetch(["600000.SH"])
        assert not df.empty, "Tonghuashun primary should provide data"
        assert len(ths_called) == 1, "Tonghuashun must be called first"
        assert df.attrs.get("flow_source") == "tonghuashun"

    def test_tonghuashun_unit_parsing(self, monkeypatch):
        """Tonghuashun correctly parses 亿/万 string units to numeric 万元."""
        import akshare as ak

        # Mock akshare.stock_fund_flow_individual
        mock_data = pd.DataFrame([{
            "序号": 1, "股票代码": 600519, "股票简称": "贵州茅台",
            "最新价": 1680.0, "涨跌幅": "1.23%", "换手率": "0.45%",
            "流入资金": "5.50亿", "流出资金": "3.20亿",
            "净额": "2.30亿",    # = 23000万
            "成交额": "8.70亿",  # = 87000万
        }, {
            "序号": 2, "股票代码": 300750, "股票简称": "宁德时代",
            "最新价": 210.0, "涨跌幅": "-0.50%", "换手率": "2.10%",
            "流入资金": "8000.00万", "流出资金": "9500.00万",
            "净额": "-1500.00万",  # = -1500万
            "成交额": "17500.00万",  # = 17500万
        }, {
            "序号": 3, "股票代码": 600000, "股票简称": "浦发银行",
            "最新价": 8.5, "涨跌幅": "0.00%", "换手率": "0.30%",
            "流入资金": "0", "流出资金": "0",
            "净额": "-",        # zero
            "成交额": "1.20亿",  # = 12000万
        }])

        def mock_ths():
            return mock_data
        monkeypatch.setattr(ak, "stock_fund_flow_individual", mock_ths)

        ff = FlowFetcher()
        df = ff.fetch(["600519.SH", "300750.SZ", "600000.SH"])

        assert len(df) == 3
        # 贵州茅台: 2.30亿 = 23000万
        row_600519 = df[df["symbol"] == "600519.SH"].iloc[0]
        assert abs(row_600519["main_net_inflow"] - 23000.0) < 0.01
        assert abs(row_600519["turnover"] - 87000.0) < 0.01
        assert abs(row_600519["main_inflow_rate"] - (23000 / 87000 * 100)) < 0.01
        assert abs(row_600519["super_large_net_inflow"] - 11500.0) < 0.01  # net/2

        # 宁德时代: -1500万
        row_300750 = df[df["symbol"] == "300750.SZ"].iloc[0]
        assert abs(row_300750["main_net_inflow"] - (-1500.0)) < 0.01
        assert abs(row_300750["turnover"] - 17500.0) < 0.01

        # 浦发银行: net is "-" → 0
        row_600000 = df[df["symbol"] == "600000.SH"].iloc[0]
        assert row_600000["main_net_inflow"] == 0.0
        assert abs(row_600000["turnover"] - 12000.0) < 0.01

    def test_add_exchange_suffix_sh(self):
        """Codes starting with 6 → .SH."""
        assert _add_exchange_suffix("600519") == "600519.SH"
        assert _add_exchange_suffix("688981") == "688981.SH"

    def test_add_exchange_suffix_sz(self):
        """Other codes → .SZ."""
        assert _add_exchange_suffix("000858") == "000858.SZ"
        assert _add_exchange_suffix("300750") == "300750.SZ"
        assert _add_exchange_suffix("002594") == "002594.SZ"

    def test_add_exchange_suffix_pads_zeros(self):
        """Bare codes are zero-padded to 6 digits."""
        assert _add_exchange_suffix("1") == "000001.SZ"


# ═══════════════════════════════════════════════════════════════
# Pipeline with mocked APIs
# ═══════════════════════════════════════════════════════════════

class TestPipelineMockedAPIs:
    """End-to-end pipeline tests using mocked Pandadata and AKShare."""

    def test_pipeline_with_mocked_apis(self, mock_pandadata, mock_akshare, monkeypatch):
        """Full pipeline run with all external APIs mocked."""
        from core.pipeline import ScreenerPipeline
        from core.cache import CacheManager

        # Force cache miss to use fresh mock data (avoid stale real cache)
        monkeypatch.setattr(CacheManager, "has", lambda _self, _date: False)

        # Tonghuashun fails → falls through to mocked AKShare
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun",
            lambda _self, _symbols: pd.DataFrame(),
        )

        # Make mock flow data generous so flow filter doesn't exclude pattern-triggered stocks
        import numpy as np
        flow_rows = []
        for i in range(100):
            code = f"{600000 + i:06d}" if i % 2 == 0 else f"{300000 + i:06d}"
            flow_rows.append({
                "代码": code, "名称": f"股票{code}",
                "今日主力净流入-净额": 5000.0,
                "今日主力净流入-净占比": 10.0,
                "今日超大单净流入-净额": 3000.0,
                "今日大单净流入-净额": 2000.0,
            })
        mock_akshare.return_value = pd.DataFrame(flow_rows)

        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=False, dry_run=True, top_n=10)

        assert result.total_stocks >= 1
        assert len(result.ranked_stocks) >= 1, (
            f"Expected ranked stocks but got none. flow_passed={result.flow_passed_count}"
        )
        for s in result.ranked_stocks:
            assert "rank" in s
            assert "score" in s
            assert "triggered_patterns" in s

    def test_pipeline_dry_run_output(self, mock_pandadata, mock_akshare, monkeypatch):
        """Pipeline produces dry-run analysis in every stock."""
        from core.pipeline import ScreenerPipeline

        # Tonghuashun fails → falls through to mocked AKShare
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun",
            lambda _self, _symbols: pd.DataFrame(),
        )

        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=False, dry_run=True, top_n=5)

        for s in result.ranked_stocks:
            assert "（Dry run" in s.get("llm_analysis", "")

    def test_pipeline_no_flow_fallback(self, mock_pandadata, mock_akshare, monkeypatch, tmp_path):
        """When ALL flow paths (AKShare + East Money + Tonghuashun + cache) fail, pipeline degrades gracefully."""
        from core.pipeline import ScreenerPipeline
        from core.cache import CacheManager

        mock_akshare.return_value = pd.DataFrame()

        def mock_fail(_self, _symbols):
            return pd.DataFrame()
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_eastmoney", mock_fail,
        )
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun", mock_fail,
        )

        # Force cache miss by using a non-existent temp cache root
        def mock_cache_has(_self, _date):
            return False
        monkeypatch.setattr(CacheManager, "has", mock_cache_has)

        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=False, dry_run=True, top_n=5)

        assert len(result.ranked_stocks) >= 1
        assert result.flow_degraded_note is not None, "Should report flow degradation"
        for s in result.ranked_stocks:
            assert s.get("flow_score", 1) == 0

    def test_pipeline_score_components(self, mock_pandadata, mock_akshare, monkeypatch):
        """Each ranked stock has pattern_score, flow_score, quality_bonus."""
        from core.pipeline import ScreenerPipeline

        # Tonghuashun fails → falls through to mocked AKShare
        monkeypatch.setattr(
            "core.flow_fetcher.FlowFetcher._fetch_via_tonghuashun",
            lambda _self, _symbols: pd.DataFrame(),
        )

        pipeline = ScreenerPipeline()
        result = pipeline.run(use_mock=False, dry_run=True, top_n=5)

        for s in result.ranked_stocks:
            for field in ["pattern_score", "flow_score", "quality_bonus"]:
                assert field in s, f"Missing {field}"
            total = s["pattern_score"] + s["flow_score"] + s["quality_bonus"]
            assert abs(s["score"] - total) < 0.01, (
                f"Score mismatch: {s['score']} != {total}"
            )
