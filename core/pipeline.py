"""Shared screener pipeline — used by both run.py and mcp_server.py.

Extracts the common data-acquisition → pattern-detection → flow-filter →
scoring → LLM-analysis workflow so the two entry points only differ in
how they present results (console vs MCP JSON).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .cache import CacheManager
from .data_fetcher import DataFetcher, _load_config
from .flow_filter import FlowFilter, normalize_inflow_rate
from .pattern_detector import (
    apply_weights_from_config,
    get_pattern_details,
    get_triggered_labels,
    run_all_detectors,
)
from .reporter import save_report
from .scorer import compute_score, rank_stocks

logger = logging.getLogger("pipeline")


@dataclass
class PipelineResult:
    """Result of a full screener run."""

    trade_date: str
    total_stocks: int
    ranked_stocks: list[dict]
    pattern_stats: dict[str, int] = field(default_factory=dict)
    industry_dist: dict[str, int] = field(default_factory=dict)
    flow_degraded_note: str | None = None
    flow_passed_count: int = 0
    cross_validated: int = 0
    md_path: str = ""
    json_path: str = ""
    stage_times: dict[str, float] = field(default_factory=dict)


class ScreenerPipeline:
    """Encapsulates the full dual-factor screener workflow.

    Both ``run.py`` and ``mcp_server.py`` delegate to this class so that
    the scanning logic lives in one place.

    Production usage always uses real data + real LLM::

        pipeline = ScreenerPipeline()
        result = pipeline.run()
        for s in result.ranked_stocks:
            print(s["name"], s["score"])
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or _load_config()
        self._data_fetcher: DataFetcher | None = None

    # ── data acquisition ──────────────────────────────────────

    def acquire_data(
        self,
        trade_date: str | None = None,
        use_mock: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
        """Fetch K-line, fund-flow, and stock-info data.

        Production path: Pandadata API + 同花顺 fund flow.
        Mock path (use_mock=True): synthetic data — internal fallback only.

        Returns:
            (kline_df, flow_df, info_df, resolved_trade_date)
        """
        scan_cfg = self.config.get("scan", {})

        if use_mock:
            from .mock_data import generate_mock_data

            resolved_date = trade_date or "20260629"
            kline_df, _mock_flow_df, info_df = generate_mock_data(trade_date=resolved_date)

            # Try AKShare for real fund flow, fall back to mock
            logger.info("Fetching real fund flow from AKShare ...")
            from .flow_fetcher import FlowFetcher
            symbols_for_flow = kline_df["symbol"].dropna().unique().tolist()
            flow_fetcher = FlowFetcher()
            flow_df = flow_fetcher.fetch(symbols_for_flow)
            if flow_df.empty:
                logger.warning("AKShare fund flow unavailable, using mock flow data")
                flow_df = _mock_flow_df
                self._mock_flow_fallback = True
            else:
                self._mock_flow_fallback = False
                self._merge_turnover_from_kline(kline_df, flow_df)
        else:
            fetcher = DataFetcher(self.config)
            fetcher.init_api()
            self._data_fetcher = fetcher
            resolved_date = trade_date or fetcher.get_last_trade_date()

            cache = CacheManager()
            if cache.has(resolved_date):
                logger.info("Loading data from cache ...")
                kline_df, flow_df, info_df = cache.load(resolved_date)
                # Restore flow source from cache meta (parquet doesn't preserve attrs)
                if not flow_df.empty:
                    meta = cache.load_meta(resolved_date)
                    if meta.get("flow_source"):
                        flow_df.attrs["flow_source"] = meta["flow_source"]
                # Re-fetch flow if cached flow is empty (e.g. previous AKShare failure)
                if flow_df.empty and not kline_df.empty:
                    logger.info("Cached flow is empty — re-fetching fund flow ...")
                    from .flow_fetcher import FlowFetcher
                    symbols_for_flow = kline_df["symbol"].dropna().unique().tolist()
                    flow_fetcher = FlowFetcher()
                    flow_df = flow_fetcher.fetch(symbols_for_flow)
                    if not flow_df.empty:
                        self._merge_turnover_from_kline(kline_df, flow_df)
                        # Update cache with fresh flow data
                        cache.save(resolved_date, kline_df, flow_df, info_df)
            else:
                kline_df, info_df = fetcher.fetch_all_data(resolved_date)
                if not kline_df.empty:
                    # Fetch fund flow separately (via AKShare/FlowFetcher)
                    from .flow_fetcher import FlowFetcher
                    symbols_for_flow = kline_df["symbol"].dropna().unique().tolist()
                    flow_fetcher = FlowFetcher()
                    flow_df = flow_fetcher.fetch(symbols_for_flow)
                    if not flow_df.empty:
                        self._merge_turnover_from_kline(kline_df, flow_df)
                    cache.save(resolved_date, kline_df, flow_df, info_df)
                else:
                    flow_df = pd.DataFrame()

        return kline_df, flow_df, info_df, resolved_date

    # ── flow filter ───────────────────────────────────────────

    def apply_flow_filter(
        self,
        flow_df: pd.DataFrame,
        symbols: pd.Index,
        flow_enabled: bool,
    ) -> tuple[set[str], bool, str | None]:
        """Apply main-capital flow filter.

        Returns:
            (passed_symbols, flow_enabled, flow_degraded_note)
        """
        flow_degraded_note: str | None = None

        if not flow_enabled:
            return set(symbols), False, None

        if flow_df.empty:
            logger.warning("Fund flow data unavailable — falling back to pattern-only scan")
            return set(symbols), False, (
                "资金流向数据不可用（AKShare 返回空），本次仅做形态扫描，未经过资金过滤"
            )

        flow_cfg = self.config.get("flow", {})
        scan_cfg = self.config.get("scan", {})
        flow_filter = FlowFilter(
            main_inflow_rate_min=flow_cfg.get("main_inflow_rate_min", 0.05),
            min_turnover=scan_cfg.get("min_turnover", 5000),
            super_large_positive=flow_cfg.get("super_large_positive", True),
        )
        passed_flow_df, _ = flow_filter.filter_dataframe(flow_df)
        passed = set()
        if not passed_flow_df.empty and "symbol" in passed_flow_df.columns:
            passed = set(passed_flow_df["symbol"].dropna().unique())
        logger.info("Flow filter: %d passed", len(passed))
        return passed, True, None

    # ── cross-validate & score ────────────────────────────────

    def _process_symbol(
        self,
        sym: str,
        kline_df: pd.DataFrame,
        flow_df: pd.DataFrame,
        info_df: pd.DataFrame,
        flow_enabled: bool,
        flow_passed_symbols: set[str],
        active_patterns: set[str] | None,
        date_col: str,
        min_mcap: float,
    ) -> dict | None:
        """Run detectors + scoring for a single stock. Returns stock dict or None."""
        stock_kline = kline_df[kline_df["symbol"] == sym].sort_values(date_col)
        min_rows = int(self.config.get("scan", {}).get("min_data_rows", 30))
        if len(stock_kline) < min_rows:
            return None

        det_results = run_all_detectors(stock_kline, active_patterns)
        triggered = get_triggered_labels(det_results)
        if not triggered:
            return None

        if flow_enabled and sym not in flow_passed_symbols:
            return None

        stock_info = self._get_row(info_df, sym)
        flow_data = self._get_row(flow_df, sym)

        # Latest K-line metrics
        latest_row = stock_kline.iloc[-1]
        prev_close = float(latest_row.get("pre_close", 0) or 0)
        close = float(latest_row.get("close", 0))
        pct_change = ((close - prev_close) / prev_close * 100) if prev_close else 0

        market_cap = float(stock_info.get("market_cap", 0) or 0)
        if min_mcap > 0 and market_cap < min_mcap:
            return None

        volume = float(latest_row.get("volume", 0))
        avg_vol_5 = (
            float(stock_kline["volume"].iloc[-6:-1].mean())
            if len(stock_kline) >= 6
            else volume
        )
        vol_ratio = round(volume / avg_vol_5, 2) if avg_vol_5 else 1.0

        # Flow values
        inflow_rate = normalize_inflow_rate(float(flow_data.get("main_inflow_rate", 0) or 0))
        turnover = float(flow_data.get("turnover", 0) or 0)
        main_inflow = float(flow_data.get("main_net_inflow", 0) or 0)
        super_large = float(flow_data.get("super_large_net_inflow", 0) or 0)

        # Score — always use compute_score; pass 0 inflow when flow disabled
        effective_inflow = inflow_rate if (flow_enabled and not flow_df.empty) else 0.0
        score_dict = compute_score(det_results, effective_inflow, market_cap, turnover)

        industry = stock_info.get("industry", "未知")

        return {
            "code": sym,
            "name": stock_info.get("name", sym),
            "industry": industry,
            "market_cap": market_cap,
            "close": close,
            "pct_change": round(pct_change, 2),
            "triggered_patterns": triggered,
            "pattern_details": get_pattern_details(
                stock_kline,
                {k for k, v in det_results.items() if v > 0},
            ),
            "main_inflow": main_inflow,
            "inflow_rate": round(inflow_rate * 100 if inflow_rate < 1 else inflow_rate, 2),
            "super_large_inflow": super_large,
            "turnover": turnover,
            "vol_ratio": vol_ratio,
            "_det_results": det_results,
            **score_dict,
        }

    @staticmethod
    def _apply_industry_neutralization(results: list[dict]) -> list[dict]:
        """Apply z-score industry neutralization to reduce sector bias.

        Standardizes each stock's pattern_score within its industry:
          z = (raw − μ_industry) / σ_industry

        This accounts for both the mean AND the dispersion of each industry.
        Stocks in systematically high-trigger industries (e.g., volatile tech)
        don't get an unfair boost; stocks in quiet industries aren't penalized.
        The result is floor-bounded at 0 so no pattern_score goes negative.
        """
        if len(results) < 5:
            return results

        # Compute per-industry mean and std of pattern_score
        ind_scores: dict[str, list[float]] = {}
        for r in results:
            ind = r.get("industry", "未知")
            ind_scores.setdefault(ind, []).append(r["pattern_score"])

        ind_stats: dict[str, tuple[float, float]] = {}
        for ind, scores in ind_scores.items():
            mean = sum(scores) / len(scores)
            if len(scores) > 1:
                variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
                std = variance ** 0.5
            else:
                std = 1.0
            # Only use std normalization if there's meaningful dispersion
            if std < 0.01:
                std = 1.0
            ind_stats[ind] = (mean, std)

        global_mean = sum(r["pattern_score"] for r in results) / len(results)

        for r in results:
            ind = r.get("industry", "未知")
            ind_mean, ind_std = ind_stats.get(ind, (global_mean, 1.0))
            raw_ps = r["pattern_score"]
            # z-score: subtract mean, divide by std
            z_score = (raw_ps - ind_mean) / ind_std
            # Floor at 0 so no negative pattern_scores
            neutralized = max(0.0, z_score)
            # Recompute total score
            r["pattern_score_raw"] = raw_ps
            r["pattern_score"] = round(neutralized, 2)
            r["industry_adj"] = round(ind_mean, 2)
            r["industry_std"] = round(ind_std, 4)
            r["score"] = round(neutralized + r["flow_score"] + r["quality_bonus"], 2)

        return results

    def scan_stocks(
        self,
        kline_df: pd.DataFrame,
        flow_df: pd.DataFrame,
        info_df: pd.DataFrame,
        flow_enabled: bool,
        flow_passed_symbols: set[str],
        active_patterns: set[str] | None = None,
        top_n: int = 20,
    ) -> tuple[list[dict], dict[str, int], dict[str, int]]:
        """Run pattern detection + scoring on every stock.

        Uses ThreadPoolExecutor for parallel per-stock processing.
        Applies industry neutralization to reduce sector bias.

        Returns:
            (ranked_stocks, pattern_stats, industry_dist)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        symbols = kline_df["symbol"].dropna().unique()
        date_col = "date" if "date" in kline_df.columns else "date_idx"
        scan_cfg = self.config.get("scan", {})
        min_mcap = float(scan_cfg.get("min_market_cap", 0) or 0)

        results: list[dict] = []
        pattern_stats: dict[str, int] = {}
        industry_dist: dict[str, int] = {}

        # Process stocks in parallel
        max_workers = min(8, len(symbols))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_symbol,
                    sym, kline_df, flow_df, info_df, flow_enabled,
                    flow_passed_symbols, active_patterns, date_col, min_mcap,
                ): sym
                for sym in symbols
            }

            for future in as_completed(futures):
                sym = futures[future]
                try:
                    stock_dict = future.result()
                except Exception:
                    logger.exception("Stock processing failed for %s", sym)
                    continue
                if stock_dict is None:
                    continue

                # Accumulate pattern stats
                for label in stock_dict["triggered_patterns"]:
                    pattern_stats[label] = pattern_stats.get(label, 0) + 1

                # Accumulate industry distribution
                ind = stock_dict["industry"]
                industry_dist[ind] = industry_dist.get(ind, 0) + 1

                # Clean up internal field
                del stock_dict["_det_results"]
                results.append(stock_dict)

        # Apply industry neutralization before ranking
        results = self._apply_industry_neutralization(results)

        ranked = rank_stocks(results, top_n=top_n)
        return ranked, pattern_stats, industry_dist

    # ── LLM analysis ─────────────────────────────────────────

    def analyze(self, ranked: list[dict], dry_run: bool = False) -> list[dict]:
        """Attach LLM analysis to each ranked stock (mutates list in-place)."""
        if dry_run:
            for s in ranked:
                s["llm_analysis"] = "（Dry run — 跳过 LLM 分析）"
                s["llm_source"] = "dry_run"
            return ranked

        try:
            from llm.analyst import LLMAnalyst

            analyst = LLMAnalyst(self.config.get("llm", {}))
            return analyst.analyze_batch(ranked)
        except Exception as e:
            logger.warning("LLM analysis failed: %s — continuing without", e)
            for s in ranked:
                s["llm_analysis"] = f"（LLM 分析不可用: {e}）"
                s["llm_source"] = "error"
            return ranked

    # ── end-to-end ────────────────────────────────────────────

    def run(
        self,
        trade_date: str | None = None,
        use_mock: bool = False,
        no_flow: bool = False,
        dry_run: bool = False,
        top_n: int = 20,
    ) -> PipelineResult:
        """Run the complete screener pipeline and return structured results.

        Production path (default): Pandadata K-line + 同花顺 fund flow + real LLM.

        Args:
            trade_date: Target trading date (YYYYMMDD). None = latest.
            use_mock: INTERNAL — use synthetic mock data when Pandadata is unavailable.
            no_flow: Skip fund flow filter (pattern-only scan).
            dry_run: INTERNAL — skip LLM analysis (testing only).
            top_n: Number of top-ranked stocks to return.

        Returns:
            PipelineResult with ranked stocks, stats, and report paths.
        """
        t_start = time.time()
        stage_times: dict[str, float] = {}

        # 1. Acquire data
        t0 = time.time()
        logger.info("=== Step 1: Data acquisition ===")
        kline_df, flow_df, info_df, resolved_date = self.acquire_data(
            trade_date=trade_date, use_mock=use_mock
        )
        stage_times["1_data_acquisition"] = round(time.time() - t0, 1)

        if kline_df.empty:
            logger.error("K-line data unavailable.")
            return PipelineResult(trade_date=resolved_date or "", total_stocks=0,
                                  ranked_stocks=[])

        symbols = kline_df["symbol"].dropna().unique()
        total_stocks = len(symbols)
        logger.info("Total stocks with K-line data: %d", total_stocks)

        # 2. Load optimized detector weights if available
        weight_changes = apply_weights_from_config(self.config)
        if weight_changes:
            changed = {k: v for k, v in weight_changes.items() if v["old"] != v["new"]}
            if changed:
                logger.info("Backtest-optimized weights applied: %s",
                            ", ".join(f"{k}={v['new']}" for k, v in changed.items()))

        # 3. Pattern detection config
        pattern_cfg = self.config.get("patterns", {})
        active_patterns = {k for k, v in pattern_cfg.items() if v} if pattern_cfg else None

        # 4. Flow filter
        t0 = time.time()
        flow_enabled = not no_flow
        logger.info("=== Step 2: Flow filter ===" if flow_enabled else "=== Step 2: Skipped ===")
        flow_passed, flow_enabled, flow_degraded_note = self.apply_flow_filter(
            flow_df, symbols, flow_enabled
        )
        stage_times["2_flow_filter"] = round(time.time() - t0, 1)

        # When mock mode falls back to synthetic flow data, flag it
        if use_mock and getattr(self, "_mock_flow_fallback", False):
            mock_note = "AKShare资金流向不可用，使用模拟资金流数据（仅供参考）"
            if flow_degraded_note:
                flow_degraded_note = f"{flow_degraded_note}；{mock_note}"
            else:
                flow_degraded_note = mock_note

        # 4. Scan & score
        t0 = time.time()
        logger.info("=== Step 3: Pattern scanning & scoring ===")
        ranked, pattern_stats, industry_dist = self.scan_stocks(
            kline_df, flow_df, info_df, flow_enabled, flow_passed,
            active_patterns=active_patterns, top_n=top_n,
        )
        stage_times["3_scan_score"] = round(time.time() - t0, 1)

        if not ranked:
            logger.info("No stocks passed all filters.")
            return PipelineResult(
                trade_date=resolved_date, total_stocks=total_stocks,
                ranked_stocks=[], pattern_stats=pattern_stats,
                industry_dist=industry_dist,
                flow_degraded_note=flow_degraded_note,
                flow_passed_count=len(flow_passed),
                stage_times=stage_times,
            )

        # 5. LLM analysis
        t0 = time.time()
        logger.info("=== Step 4: LLM analysis ===")
        ranked = self.analyze(ranked, dry_run=dry_run)
        stage_times["4_llm_analysis"] = round(time.time() - t0, 1)

        # Build data provenance for report attestation
        data_provenance = self._build_provenance(use_mock, flow_enabled, flow_df,
                                                  flow_degraded_note, ranked)

        # 6. Save reports
        t0 = time.time()
        logger.info("=== Step 5: Saving reports ===")
        md_path, json_path = save_report(
            ranked, resolved_date, total_stocks, self.config,
            pattern_stats, industry_dist, flow_degraded_note=flow_degraded_note,
            data_provenance=data_provenance,
        )
        stage_times["5_save_reports"] = round(time.time() - t0, 1)

        total_elapsed = round(time.time() - t_start, 1)
        stage_summary = " | ".join(f"{k}={v}s" for k, v in stage_times.items())
        logger.info("Pipeline completed in %.1fs [%s]", total_elapsed, stage_summary)

        return PipelineResult(
            trade_date=resolved_date,
            total_stocks=total_stocks,
            ranked_stocks=ranked,
            pattern_stats=pattern_stats,
            industry_dist=industry_dist,
            flow_degraded_note=flow_degraded_note,
            flow_passed_count=len(flow_passed) if flow_enabled else total_stocks,
            cross_validated=sum(pattern_stats.values()),
            md_path=md_path,
            json_path=json_path,
            stage_times=stage_times,
        )

    # ── helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_provenance(
        use_mock: bool,
        flow_enabled: bool,
        flow_df: pd.DataFrame,
        flow_degraded_note: str | None,
        ranked: list[dict],
    ) -> dict:
        """Build a data provenance record for report attestation.

        Returns a dict mapping each data category to its source label
        and whether the data is real (not mock/fallback).
        """
        prov = {}

        # K-line
        if use_mock:
            prov["kline"] = {"source": "Mock 生成（合成K线数据）", "is_real": False}
        else:
            prov["kline"] = {"source": "Pandadata get_stock_daily", "is_real": True}

        # Stock info
        if use_mock:
            prov["stock_info"] = {"source": "Mock 生成（合成股票池）", "is_real": False}
        else:
            prov["stock_info"] = {"source": "Pandadata get_stock_detail / get_trade_list", "is_real": True}

        # Fund flow — check degradation first (may have set flow_enabled=False)
        if flow_degraded_note:
            if use_mock:
                prov["flow"] = {"source": "Mock 合成（AKShare 不可用，降级为模拟数据）", "is_real": False}
            else:
                prov["flow"] = {"source": "AKShare 不可用（仅做形态扫描，未经过资金过滤）", "is_real": False}
        elif not flow_enabled:
            prov["flow"] = {"source": "已跳过（--no-flow 纯形态扫描）", "is_real": False}
        else:
            # Resolve actual flow source label
            flow_source = flow_df.attrs.get("flow_source", "")
            label_map = {
                "akshare": "AKShare stock_individual_fund_flow_rank (东方财富)",
                "eastmoney": "东方财富 push2 直连 API",
                "tonghuashun": "同花顺 stock_fund_flow_individual (10jqka)",
            }
            source_label = label_map.get(flow_source, "未知来源")
            prov["flow"] = {"source": source_label, "is_real": True}

        # LLM — aggregate from per-stock markers
        llm_sources = set()
        for s in ranked:
            llm_sources.add(s.get("llm_source", "unknown"))
        if "real" in llm_sources and len(llm_sources) == 1:
            prov["llm"] = {"source": "Claude / DeepSeek API（真实大模型分析）", "is_real": True}
        elif "real" in llm_sources:
            prov["llm"] = {"source": f"部分真实 + 部分回退（{llm_sources}）", "is_real": False}
        elif "dry_run" in llm_sources:
            prov["llm"] = {"source": "已跳过（--dry-run）", "is_real": False}
        elif "fallback" in llm_sources:
            prov["llm"] = {"source": "模板回退（API不可用，使用规则生成）", "is_real": False}
        else:
            prov["llm"] = {"source": "不可用", "is_real": False}

        # Overall: all sources must be real
        prov["all_real"] = all(v["is_real"] for v in prov.values())

        return prov

    @staticmethod
    def _get_row(df: pd.DataFrame, symbol: str) -> dict:
        """Return first matching row as dict, or empty dict."""
        if df.empty:
            return {}
        rows = df[df["symbol"] == symbol]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()

    @staticmethod
    def _merge_turnover_from_kline(kline_df: pd.DataFrame, flow_df: pd.DataFrame) -> None:
        """Merge turnover from K-line amount column into flow_df (yuan → 万元)."""
        date_col = "date" if "date" in kline_df.columns else "date_idx"
        kline_sorted = kline_df.sort_values(date_col)
        latest = kline_sorted.groupby("symbol").last().reset_index()
        if "amount" in latest.columns:
            turnover_map = dict(zip(
                latest["symbol"],
                pd.to_numeric(latest["amount"], errors="coerce") / 10000,
            ))
            flow_df["turnover"] = flow_df["symbol"].map(turnover_map).fillna(0)
