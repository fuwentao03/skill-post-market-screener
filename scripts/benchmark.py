#!/usr/bin/env python3
"""Performance benchmark for the post-market screener pipeline.

Measures throughput, per-stage timing, and scalability across different
stock universe sizes (50 → 500 → 5000). Projects expected runtime for
the full 5189-stock A-share universe.

Usage:
    python scripts/benchmark.py              # Run all benchmarks
    python scripts/benchmark.py --quick      # Quick benchmark (50 stocks only)
    python scripts/benchmark.py --size 1000  # Specific size
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

logging.basicConfig(
    level=logging.WARNING,  # suppress pipeline logs during benchmark
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

# Full A-share universe size (approximate)
FULL_UNIVERSE = 5189

# Benchmark sizes (small → large)
DEFAULT_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]


@dataclass
class BenchmarkResult:
    """Timing result for a single benchmark run."""
    n_stocks: int
    total_elapsed: float
    stage_times: dict[str, float]
    stocks_per_second: float
    ranked_count: int
    test_count: int = 1

    @property
    def projected_5189_time(self) -> float:
        """Linear projection to 5189 stocks."""
        if self.n_stocks == 0:
            return 0
        return (self.total_elapsed / self.n_stocks) * FULL_UNIVERSE


def run_benchmark(n_stocks: int, dry_run: bool = True) -> BenchmarkResult:
    """Run pipeline with mock data at a given scale and measure performance."""
    import logging as std_logging

    from core.mock_data import generate_mock_data
    from core.pipeline import ScreenerPipeline

    logger.info("Benchmark: %d stocks ...", n_stocks)

    # Suppress pipeline logging for clean timing
    std_logging.disable(std_logging.CRITICAL)

    try:
        pipeline = ScreenerPipeline()

        t0 = time.perf_counter()

        # Generate mock data at the requested scale
        kline_df, flow_df, info_df = generate_mock_data(
            trade_date="20260629", n_stocks=n_stocks, n_days=120,
        )
        symbols = kline_df["symbol"].dropna().unique()

        # Run scan & score directly (skip AKShare + LLM)
        ranked, pat_stats, ind_dist = pipeline.scan_stocks(
            kline_df, flow_df, info_df,
            flow_enabled=False,
            flow_passed_symbols=set(symbols),
            top_n=min(20, n_stocks),
        )

        # Dry-run LLM
        ranked = pipeline.analyze(ranked, dry_run=True)

        elapsed = time.perf_counter() - t0
    finally:
        std_logging.disable(std_logging.NOTSET)

    actual_stocks = len(symbols)
    stocks_per_sec = actual_stocks / elapsed if elapsed > 0 else 0
    logger.info(
        "  %d stocks: %.2fs (%.0f stocks/s) → %d ranked",
        actual_stocks, elapsed, stocks_per_sec, len(ranked),
    )

    return BenchmarkResult(
        n_stocks=actual_stocks,
        total_elapsed=round(elapsed, 3),
        stage_times={},
        stocks_per_second=round(stocks_per_sec, 1),
        ranked_count=len(ranked),
    )


def run_stage_breakdown(n_stocks: int = 500) -> dict[str, float]:
    """Run pipeline once with timing instrumentation for per-stage breakdown."""
    import logging as std_logging

    from core.mock_data import generate_mock_data
    from core.pipeline import ScreenerPipeline
    from core.reporter import save_report

    std_logging.disable(std_logging.CRITICAL)

    try:
        pipeline = ScreenerPipeline()

        # ── Stage 1: Data generation ──
        t0 = time.perf_counter()
        kline_df, flow_df, info_df = generate_mock_data(
            trade_date="20260629", n_stocks=n_stocks, n_days=120,
        )
        t1 = time.perf_counter()

        symbols = kline_df["symbol"].dropna().unique()
        total_stocks = len(symbols)
        flow_passed = set(symbols)

        # ── Stage 2: Scan & score ──
        ranked, pat_stats, ind_dist = pipeline.scan_stocks(
            kline_df, flow_df, info_df,
            flow_enabled=False,
            flow_passed_symbols=flow_passed,
            top_n=20,
        )
        t2 = time.perf_counter()

        # ── Stage 3: LLM analysis (dry run) ──
        ranked = pipeline.analyze(ranked, dry_run=True)
        t3 = time.perf_counter()

        # ── Stage 4: Save reports ──
        md_path, json_path = save_report(
            ranked, "20260629", total_stocks, pipeline.config,
            pat_stats, {"_bench": total_stocks},
        )
        t4 = time.perf_counter()

        stage_times = {
            "1_data_generation": round(t1 - t0, 3),
            "2_scan_score": round(t2 - t1, 3),
            "3_llm_analysis": round(t3 - t2, 3),
            "4_save_reports": round(t4 - t3, 3),
        }
        stage_times["total"] = round(t4 - t0, 3)

        return stage_times
    finally:
        std_logging.disable(std_logging.NOTSET)


def project_full_universe(results: list[BenchmarkResult]) -> dict:
    """Compute linear and sub-linear projections to 5189 stocks."""
    if not results:
        return {}

    # Linear regression: time = a * n + b
    sizes = np.array([r.n_stocks for r in results])
    times = np.array([r.total_elapsed for r in results])

    # Linear fit (through origin-ish — overhead is small for large n)
    slope_linear = np.sum(sizes * times) / np.sum(sizes * sizes)

    # Sub-linear fit: time = a * n^b  →  log(time) = log(a) + b * log(n)
    # Only use sizes >= 100 for log fit
    mask = sizes >= 100
    if mask.sum() >= 3:
        log_n = np.log(sizes[mask])
        log_t = np.log(times[mask])
        b = np.polyfit(log_n, log_t, 1)[0]
        a = np.exp(np.mean(log_t - b * log_n))
        projected_sublinear = a * (FULL_UNIVERSE ** b)
    else:
        b = 1.0
        projected_sublinear = slope_linear * FULL_UNIVERSE

    # Largest actual measurement
    largest = results[-1]

    return {
        "full_universe_size": FULL_UNIVERSE,
        "linear_projection_seconds": round(slope_linear * FULL_UNIVERSE, 1),
        "linear_rate_stocks_per_second": round(1.0 / slope_linear, 1) if slope_linear > 0 else 0,
        "sublinear_projection_seconds": round(projected_sublinear, 1),
        "sublinear_exponent": round(b, 3),
        "largest_measured_n": largest.n_stocks,
        "largest_measured_seconds": largest.total_elapsed,
        "largest_measured_rate": largest.stocks_per_second,
    }


def print_benchmark_report(
    results: list[BenchmarkResult],
    stage_breakdown: dict[str, float],
    projection: dict,
) -> None:
    """Print a formatted performance benchmark report."""
    print("\n" + "=" * 78)
    print("   Post-Market Screener — Performance Benchmark Report")
    print("=" * 78)

    # ── Throughput table ──
    print(f"\n{'Stocks':>8s}  {'Time (s)':>10s}  {'Rate (stk/s)':>14s}  "
          f"{'Ranked':>8s}  {'Proj. 5189':>12s}")
    print("-" * 78)
    for r in results:
        proj = f"{r.projected_5189_time:.0f}s" if r.n_stocks > 0 else "N/A"
        print(f"{r.n_stocks:>8d}  {r.total_elapsed:>10.2f}  "
              f"{r.stocks_per_second:>14.1f}  {r.ranked_count:>8d}  "
              f"{proj:>12s}")

    # ── Stage breakdown ──
    if stage_breakdown:
        print(f"\n{'=' * 78}")
        print("   Per-Stage Timing Breakdown")
        print(f"{'=' * 78}")
        total = stage_breakdown.get("total", 1)
        for stage, t in stage_breakdown.items():
            if stage == "total":
                continue
            pct = t / total * 100 if total > 0 else 0
            bar = "=" * int(pct / 2)
            print(f"  {stage:<30s} {t:>8.2f}s ({pct:>5.1f}%) {bar}")
        print(f"  {'─' * 68}")
        print(f"  {'TOTAL':<30s} {total:>8.2f}s")

    # ── Projections ──
    if projection:
        print(f"\n{'=' * 78}")
        print("   Full Universe (5189 stocks) Projections")
        print(f"{'=' * 78}")
        print(f"  Linear projection:        {projection['linear_projection_seconds']:.1f}s "
              f"({projection['linear_projection_seconds']/60:.1f} min)")
        print(f"  Sub-linear projection:    {projection['sublinear_projection_seconds']:.1f}s "
              f"({projection['sublinear_projection_seconds']/60:.1f} min)")
        print(f"  Sub-linear exponent:      {projection['sublinear_exponent']:.3f} "
              f"({'sub-linear' if projection['sublinear_exponent'] < 1 else 'linear+'})")
        print(f"  Throughput:               {projection['linear_rate_stocks_per_second']:.0f} stocks/s")
        print(f"  Largest measured:         {projection['largest_measured_n']} stocks "
              f"in {projection['largest_measured_seconds']:.1f}s "
              f"({projection['largest_measured_rate']:.0f} stk/s)")

    # ── Bottleneck analysis ──
    if stage_breakdown:
        print(f"\n{'=' * 78}")
        print("   Bottleneck Analysis")
        print(f"{'=' * 78}")
        stages = [(k, v) for k, v in stage_breakdown.items() if k != "total"]
        stages.sort(key=lambda x: x[1], reverse=True)
        for i, (name, t) in enumerate(stages, 1):
            pct = t / total * 100 if total > 0 else 0
            marker = " ← PRIMARY BOTTLENECK" if i == 1 else ""
            print(f"  {i}. {name:<28s} {t:>8.2f}s ({pct:>5.1f}%){marker}")

        # Improvement recommendations
        top_stage = stages[0][0]
        print(f"\n  Recommendations:")
        if "data_generation" in top_stage or "data_acquisition" in top_stage:
            print("    - Use pre-cached data for repeated scans")
            print("    - Fetch K-line in larger chunks (currently 200/symbol)")
            print("    - Mock data generation is I/O-free — real API calls will dominate")
        elif "scan_score" in top_stage:
            print("    - ThreadPoolExecutor is already in use (max 8 workers)")
            print("    - Consider ProcessPoolExecutor for CPU-bound detector loops")
            print("    - Batch pattern detection with numpy vectorization")
        elif "llm_analysis" in top_stage:
            print("    - Increase LLM request concurrency")
            print("    - Reduce max_tokens or use smaller model")
            print("    - Only analyze top-N stocks with LLM")
        elif "save_reports" in top_stage:
            print("    - Use async I/O for file writes")
            print("    - Reduce report verbosity")

    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screener performance benchmark")
    p.add_argument("--quick", action="store_true", help="Quick benchmark (50 stocks only)")
    p.add_argument("--size", type=int, default=0, help="Benchmark a specific size")
    p.add_argument("--json", action="store_true", help="Output results as JSON")
    p.add_argument("--no-stage-breakdown", action="store_true",
                   help="Skip detailed stage breakdown measurement")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.quick:
        sizes = [50]
    elif args.size > 0:
        sizes = [args.size]
    else:
        sizes = DEFAULT_SIZES

    logger.info("Starting performance benchmark...")
    logger.info("Sizes: %s", sizes)

    results: list[BenchmarkResult] = []
    for n in sizes:
        result = run_benchmark(n, dry_run=True)
        results.append(result)

    # Stage breakdown at a representative size
    stage_times: dict[str, float] = {}
    if not args.no_stage_breakdown:
        breakdown_size = min(500, sizes[-1]) if sizes else 50
        logger.info("Measuring per-stage breakdown at %d stocks ...", breakdown_size)
        stage_times = run_stage_breakdown(breakdown_size)

    projection = project_full_universe(results)

    if args.json:
        output = {
            "results": [
                {
                    "n_stocks": r.n_stocks,
                    "total_elapsed": r.total_elapsed,
                    "stocks_per_second": r.stocks_per_second,
                    "projected_5189_time_s": round(r.projected_5189_time, 1),
                }
                for r in results
            ],
            "stage_breakdown": stage_times,
            "projection": projection,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_benchmark_report(results, stage_times, projection)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
