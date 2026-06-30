#!/usr/bin/env python3
"""Backtest framework for pattern detector weight validation.

Computes per-detector hit rate, IC, and excess return from historical
K-line data. Generates statistically-grounded weight recommendations.

Usage:
    python scripts/analyze_weights.py --mock             # Demo with mock data (50 stocks × 120d)
    python scripts/analyze_weights.py --mock --n-stocks 200 --n-days 250
    python scripts/analyze_weights.py --data <parquet_dir>   # Real data (future)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure skill root on path
SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))

from core.pattern_detector import DETECTOR_REGISTRY, run_all_detectors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analyze_weights")

FORWARD_HORIZONS = [1, 3, 5, 10, 20]

# Rolling window defaults
DEFAULT_WINDOW_SIZE = 60   # trading days per window
DEFAULT_WINDOW_STEP = 20   # trading days between window starts


def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    mask = a.notna() & b.notna()
    a_clean, b_clean = a[mask], b[mask]
    if len(a_clean) < 3:
        return 0.0
    return a_clean.rank().corr(b_clean.rank())


# ── data structures ────────────────────────────────────────────

@dataclass
class DetectorStats:
    """Aggregated backtest statistics for one detector."""
    key: str
    label: str
    current_weight: int
    trigger_count: int = 0
    total_signals: int = 0
    # hit_rate[h] = fraction of triggers where forward h-day return > 0
    hit_rate: dict[int, float] = field(default_factory=dict)
    # mean_excess[h] = mean(forward return - cross-sectional avg) at horizon h
    mean_excess: dict[int, float] = field(default_factory=dict)
    # ic_mean[h] = mean rank IC at horizon h
    ic_mean: dict[int, float] = field(default_factory=dict)
    # ic_ir[h] = IC / IC_std (information ratio)
    ic_ir: dict[int, float] = field(default_factory=dict)


@dataclass
class RollingWindowResult:
    """Result for a single rolling window."""
    window_label: str
    start_date: str
    end_date: str
    n_stocks: int
    n_records: int
    # {detector_key: {horizon: ic}}
    ic_matrix: dict[str, dict[int, float]] = field(default_factory=dict)
    # {detector_key: {horizon: hit_rate}}
    hit_matrix: dict[str, dict[int, float]] = field(default_factory=dict)


@dataclass
class RollingStabilityStats:
    """Stability summary across rolling windows for one detector."""
    key: str
    label: str
    current_weight: int
    n_windows: int = 0
    # horizon → (mean_ic, std_ic, min_ic, max_ic)
    ic_stability: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)
    # horizon → (mean_hit, std_hit)
    hit_stability: dict[int, tuple[float, float]] = field(default_factory=dict)


# ── engine ─────────────────────────────────────────────────────

class BacktestEngine:
    """Run pattern detectors across historical data and compute statistics."""

    def __init__(self, kline_data: dict[str, pd.DataFrame]):
        """
        Args:
            kline_data: {symbol: DataFrame with OHLCV columns, sorted by date asc}
        """
        self._data = kline_data
        self._stats: dict[str, DetectorStats] = {}
        self._detector_keys = list(DETECTOR_REGISTRY.keys())

    # ── record building ───────────────────────────────────────

    @staticmethod
    def _resolve_date_col(df: pd.DataFrame) -> str | None:
        """Find the date column in a DataFrame (date > date_idx > None)."""
        for col in ["date", "date_idx"]:
            if col in df.columns:
                return col
        return None

    def _build_records(self, kline_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build backtest records DataFrame from kline data.

        Each row = one (stock, day) observation with detector signals and
        forward returns.
        """
        records: list[dict] = []
        total = len(kline_data)
        processed = 0

        for sym, df in kline_data.items():
            if len(df) < 70:
                continue

            close = df["close"].values
            n = len(close)
            date_col = self._resolve_date_col(df)

            for t in range(60, n - max(FORWARD_HORIZONS)):
                window = df.iloc[: t + 1]
                det_results = run_all_detectors(window)

                fwd_rets = {}
                for h in FORWARD_HORIZONS:
                    if t + h < n:
                        fwd_rets[h] = (close[t + h] - close[t]) / close[t]

                record = {
                    "symbol": sym,
                    "t": t,
                    "forward_rets": fwd_rets,
                    **det_results,
                }
                if date_col:
                    record["_date"] = str(df.iloc[t][date_col])
                records.append(record)

            processed += 1
            if processed % 500 == 0:
                logger.info("Backtest records: %d/%d stocks (%d records)",
                            processed, total, len(records))

        if not records:
            logger.warning("No backtest records generated — insufficient data")

        return pd.DataFrame(records)

    # ── stats computation ─────────────────────────────────────

    def _init_stats(self) -> dict[str, DetectorStats]:
        """Create fresh DetectorStats for each detector."""
        stats: dict[str, DetectorStats] = {}
        for key in self._detector_keys:
            entry = DETECTOR_REGISTRY[key]
            stats[key] = DetectorStats(
                key=key,
                label=entry["label"],
                current_weight=entry["weight"],
            )
        return stats

    def _compute_stats_from_records(
        self, results_df: pd.DataFrame, stats: dict[str, DetectorStats] | None = None
    ) -> dict[str, DetectorStats]:
        """Fill DetectorStats from a records DataFrame."""
        if stats is None:
            stats = self._init_stats()

        if results_df.empty:
            return stats

        for key in self._detector_keys:
            if key not in results_df.columns:
                continue
            # Strength > 0 = triggered (continuous signal, not binary)
            triggered = results_df[results_df[key] > 0]
            st = stats[key]
            st.trigger_count = len(triggered)
            st.total_signals = len(results_df)

            if st.trigger_count < 10:
                continue

            for h in FORWARD_HORIZONS:
                col = f"fwd_{h}d"
                fwd_col = results_df["forward_rets"].apply(
                    lambda d, horizon=h: d.get(horizon, np.nan)
                )
                results_df[col] = fwd_col.astype(float)

                # Cross-sectional mean per day for excess return
                daily_mean = results_df.groupby("t")[col].transform("mean")
                results_df[f"{col}_excess"] = results_df[col] - daily_mean

                # Hit rate
                triggered_fwd = results_df.loc[triggered.index, col].dropna()
                if len(triggered_fwd) > 0:
                    st.hit_rate[h] = round((triggered_fwd > 0).mean(), 4)
                    st.mean_excess[h] = round(
                        results_df.loc[triggered.index, f"{col}_excess"].dropna().mean(), 6
                    )

                # Rank IC — use continuous strength values (not binary 0/1)
                # This enables proper Spearman rank correlation instead of
                # degenerate point-biserial correlation on binary signals.
                signal_vals = results_df[key]  # raw float strength
                valid = results_df[col].notna() & signal_vals.notna()
                if valid.sum() >= 30:
                    ic = _spearman_corr(signal_vals[valid], results_df.loc[valid, col])
                    st.ic_mean[h] = round(ic, 6)

                    # IC IR: bootstrap estimate
                    ic_vals = []
                    rng = np.random.default_rng(42)
                    for _ in range(200):
                        idx = rng.choice(valid[valid].index, size=valid.sum() // 2, replace=False)
                        if len(idx) >= 10:
                            ic_boot = _spearman_corr(signal_vals.loc[idx], results_df.loc[idx, col])
                            ic_vals.append(ic_boot)
                    if ic_vals:
                        st.ic_ir[h] = round(np.mean(ic_vals) / max(np.std(ic_vals), 1e-10), 4)

        return stats

    def run(self) -> dict[str, DetectorStats]:
        """Execute backtest across all stocks and horizons."""
        logger.info("Running backtest on %d stocks", len(self._data))
        self._stats = self._init_stats()
        results_df = self._build_records(self._data)
        logger.info("Backtest records: %d", len(results_df))
        self._stats = self._compute_stats_from_records(results_df, self._stats)
        return self._stats

    # ── rolling window ────────────────────────────────────────

    def run_rolling(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        step: int = DEFAULT_WINDOW_STEP,
    ) -> tuple[list[RollingWindowResult], dict[str, RollingStabilityStats]]:
        """Run backtest across rolling time windows to assess weight stability.

        Splits the time axis into overlapping windows, runs the full backtest
        on each, and computes per-detector stability metrics (mean/std/min/max
        of IC and hit rate across windows).

        Args:
            window_size: Trading days per window.
            step: Trading days between window starts.

        Returns:
            (window_results, stability_stats)
        """
        # Determine global date range
        # Use (str_value, sort_key) tuples so numeric dates sort correctly
        date_pairs: list[tuple[str, int | str]] = []
        seen = set()
        for sym, df in self._data.items():
            date_col = self._resolve_date_col(df)
            if date_col:
                for v in df[date_col].dropna().unique():
                    str_v = str(v)
                    if str_v not in seen:
                        seen.add(str_v)
                        try:
                            sort_key = int(v)
                        except (ValueError, TypeError):
                            sort_key = str_v
                        date_pairs.append((str_v, sort_key))
            else:
                for i in range(len(df)):
                    str_v = str(i)
                    if str_v not in seen:
                        seen.add(str_v)
                        date_pairs.append((str_v, i))

        if not date_pairs:
            logger.error("No date data available for rolling windows")
            return [], {}

        date_pairs.sort(key=lambda x: x[1])
        unique_dates = [d[0] for d in date_pairs]
        logger.info("Rolling windows: %d unique dates, window=%d step=%d",
                     len(unique_dates), window_size, step)

        # Generate windows
        windows: list[tuple[int, int, str, str]] = []
        for start_idx in range(0, len(unique_dates) - window_size, step):
            end_idx = start_idx + window_size
            windows.append((
                start_idx, end_idx,
                unique_dates[start_idx], unique_dates[end_idx - 1],
            ))

        if not windows:
            logger.error("Not enough data for even one window (need >= %d days, have %d)",
                         window_size, len(unique_dates))
            return [], {}

        logger.info("Generated %d rolling windows", len(windows))

        # Per-window results
        window_results: list[RollingWindowResult] = []
        # Per-detector IC collections: {key: {horizon: [ic_values]}}
        ic_collect: dict[str, dict[int, list[float]]] = {
            key: {h: [] for h in FORWARD_HORIZONS} for key in self._detector_keys
        }
        hit_collect: dict[str, dict[int, list[float]]] = {
            key: {h: [] for h in FORWARD_HORIZONS} for key in self._detector_keys
        }

        for wi, (start_i, end_i, start_d, end_d) in enumerate(windows):
            date_set = set(unique_dates[start_i:end_i])
            label = f"W{wi+1}"

            # Filter data to this window's date range
            window_data: dict[str, pd.DataFrame] = {}
            for sym, df in self._data.items():
                date_col = self._resolve_date_col(df)
                if date_col:
                    mask = df[date_col].astype(str).isin(date_set)
                else:
                    # Use day indices
                    mask = pd.Series(
                        [str(i) in date_set for i in range(len(df))],
                        index=df.index,
                    )
                filtered = df[mask].reset_index(drop=True)
                if len(filtered) >= 70:
                    window_data[sym] = filtered

            if len(window_data) < 5:
                logger.warning("Window %s: only %d stocks — skipping", label, len(window_data))
                continue

            # Run backtest on this window
            records_df = self._build_records(window_data)
            w_stats = self._compute_stats_from_records(records_df)

            # Extract IC/hit matrix
            ic_matrix: dict[str, dict[int, float]] = {}
            hit_matrix: dict[str, dict[int, float]] = {}
            for key in self._detector_keys:
                st = w_stats.get(key)
                if st:
                    ic_matrix[key] = dict(st.ic_mean)
                    hit_matrix[key] = dict(st.hit_rate)
                    for h in FORWARD_HORIZONS:
                        if h in st.ic_mean:
                            ic_collect[key][h].append(st.ic_mean[h])
                        if h in st.hit_rate:
                            hit_collect[key][h].append(st.hit_rate[h])

            wr = RollingWindowResult(
                window_label=label,
                start_date=start_d,
                end_date=end_d,
                n_stocks=len(window_data),
                n_records=len(records_df),
                ic_matrix=ic_matrix,
                hit_matrix=hit_matrix,
            )
            window_results.append(wr)
            logger.info("Window %s [%s → %s]: %d stocks, %d records",
                        label, start_d, end_d, wr.n_stocks, wr.n_records)

        # Compute stability stats
        stability: dict[str, RollingStabilityStats] = {}
        for key in self._detector_keys:
            entry = DETECTOR_REGISTRY[key]
            ss = RollingStabilityStats(
                key=key,
                label=entry["label"],
                current_weight=entry["weight"],
                n_windows=len(window_results),
            )
            for h in FORWARD_HORIZONS:
                ic_vals = ic_collect[key][h]
                hit_vals = hit_collect[key][h]
                if len(ic_vals) >= 2:
                    ss.ic_stability[h] = (
                        round(np.mean(ic_vals), 6),
                        round(np.std(ic_vals), 6),
                        round(np.min(ic_vals), 6),
                        round(np.max(ic_vals), 6),
                    )
                if len(hit_vals) >= 2:
                    ss.hit_stability[h] = (
                        round(np.mean(hit_vals), 4),
                        round(np.std(hit_vals), 4),
                    )
            stability[key] = ss

        return window_results, stability

    def recommend_weights(self) -> dict[str, dict]:
        """Generate weight recommendations based on backtest statistics.

        Rules:
        - IC (5d) > 0.02 → +1 weight
        - IC (5d) > 0.04 → +2 weight
        - IC (5d) < -0.02 → -1 weight
        - Hit rate (5d) < 0.45 → cap at max(1, current-1)
        - Final weight clamped to [1, 5]
        """
        recommendations = {}
        for key, st in self._stats.items():
            h = 5  # primary horizon for decisions
            ic = st.ic_mean.get(h, 0)
            hit = st.hit_rate.get(h, 0.5)
            cur = st.current_weight

            adjustment = 0
            reasons = []

            if ic > 0.04 and st.trigger_count >= 20:
                adjustment = 2
                reasons.append(f"IC(5d)={ic:.4f} > 0.04")
            elif ic > 0.02 and st.trigger_count >= 20:
                adjustment = 1
                reasons.append(f"IC(5d)={ic:.4f} > 0.02")
            elif ic < -0.02 and st.trigger_count >= 20:
                adjustment = -1
                reasons.append(f"IC(5d)={ic:.4f} < -0.02")

            if hit < 0.45 and st.trigger_count >= 20:
                if adjustment == 0:
                    adjustment = -1
                reasons.append(f"hit_rate(5d)={hit:.2%} < 45%")

            new_weight = max(1, min(5, cur + adjustment))
            if new_weight == cur and st.trigger_count >= 20:
                reasons.append("stable — no change needed")
            elif st.trigger_count < 10:
                reasons.append("insufficient data — keep current")

            recommendations[key] = {
                "label": st.label,
                "current_weight": cur,
                "recommended_weight": new_weight,
                "adjustment": new_weight - cur,
                "ic_5d": ic,
                "hit_rate_5d": hit,
                "trigger_count": st.trigger_count,
                "reasons": reasons,
            }

        # Normalize: preserve relative ratios approximately
        total_new = sum(r["recommended_weight"] for r in recommendations.values())
        total_old = sum(r["current_weight"] for r in recommendations.values())
        scale_factor = total_old / max(total_new, 1)
        for rec in recommendations.values():
            rec["normalized_weight"] = round(rec["recommended_weight"] * scale_factor, 1)

        return recommendations


# ── report ─────────────────────────────────────────────────────

def print_report(stats: dict[str, DetectorStats], recommendations: dict[str, dict]) -> None:
    """Print a formatted weight analysis report to stdout."""

    print("\n" + "=" * 80)
    print("   Pattern Detector Weight Analysis Report")
    print("=" * 80)

    print(f"\n{'Detector':<22s} {'Cur':>3s} {'Rec':>3s} {'Norm':>5s}  "
          f"{'IC(5d)':>8s} {'Hit(5d)':>8s} {'N':>6s}  Rationale")
    print("-" * 80)

    for key in DETECTOR_REGISTRY:
        rec = recommendations.get(key, {})
        st = stats.get(key)
        n = st.trigger_count if st else 0
        print(
            f"{rec.get('label', key):<22s} "
            f"{rec.get('current_weight', 0):>3d} "
            f"{rec.get('recommended_weight', 0):>3d} "
            f"{rec.get('normalized_weight', 0):>5.1f}  "
            f"{rec.get('ic_5d', 0):>8.4f} "
            f"{rec.get('hit_rate_5d', 0):>8.2%} "
            f"{n:>6d}  "
            f"{'; '.join(rec.get('reasons', []))}"
        )

    total_cur = sum(DETECTOR_REGISTRY[k]["weight"] for k in DETECTOR_REGISTRY)
    total_rec = sum(rec.get("recommended_weight", 0) for rec in recommendations.values())
    print("-" * 80)
    print(f"{'TOTAL':<22s} {total_cur:>3d} {total_rec:>3d}")
    print("\nNotes:")
    print("  Cur  = current weight in DETECTOR_REGISTRY")
    print("  Rec  = recommended integer weight [1,5]")
    print("  Norm = normalized to preserve current total weight sum")
    print("  IC   = Spearman rank correlation (signal vs 5d forward return)")
    print("  N    = number of pattern triggers in backtest")
    print()

    # Per-horizon detail table
    print("=" * 80)
    print("   Per-Horizon IC Detail")
    print("=" * 80)
    header = f"{'Detector':<22s}"
    for h in FORWARD_HORIZONS:
        header += f"  {h}d_IC"
    print(header)
    print("-" * 80)
    for key in DETECTOR_REGISTRY:
        st = stats.get(key)
        if st is None:
            continue
        row = f"{st.label:<22s}"
        for h in FORWARD_HORIZONS:
            ic = st.ic_mean.get(h, float("nan"))
            row += f"  {ic:>6.4f}" if not np.isnan(ic) else f"  {'N/A':>6s}"
        print(row)
    print()


# ── main ───────────────────────────────────────────────────────

def print_rolling_report(
    window_results: list[RollingWindowResult],
    stability: dict[str, RollingStabilityStats],
) -> None:
    """Print rolling window stability report."""
    if not window_results:
        print("\nNo rolling window results to display.")
        return

    print("\n" + "=" * 90)
    print("   Rolling Window Weight Stability Report")
    print("=" * 90)
    print(f"   Windows: {len(window_results)}  |  "
          f"Range: {window_results[0].start_date} → {window_results[-1].end_date}")

    # ── Per-window IC summary table ──
    h = 5  # primary horizon
    print(f"\n{'Window':<10s} {'Stocks':>6s} {'Recs':>8s}", end="")
    for key in DETECTOR_REGISTRY:
        label = DETECTOR_REGISTRY[key]["label"]
        print(f"  {label:<8s}", end="")
    print(f"\n{'-' * 90}")

    for wr in window_results:
        print(f"{wr.window_label:<10s} {wr.n_stocks:>6d} {wr.n_records:>8d}", end="")
        for key in DETECTOR_REGISTRY:
            ic = wr.ic_matrix.get(key, {}).get(h, float("nan"))
            if np.isnan(ic):
                print(f"  {'N/A':>8s}", end="")
            else:
                sign = "+" if ic > 0 else ""
                print(f"  {sign}{ic:>7.4f}", end="")
        print()

    # ── Stability summary ──
    print(f"\n{'=' * 90}")
    print(f"   Stability Summary (IC {h}d — mean ± std across windows)")
    print(f"{'=' * 90}")
    print(f"{'Detector':<22s} {'Wgt':>3s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} "
          f"{'Max':>8s} {'IR':>7s} {'Verdict':<s}")
    print("-" * 90)

    for key in DETECTOR_REGISTRY:
        ss = stability.get(key)
        if ss is None or h not in ss.ic_stability:
            continue
        mean_ic, std_ic, min_ic, max_ic = ss.ic_stability[h]
        ir = mean_ic / max(std_ic, 1e-10)

        # Verdict
        if mean_ic > 0.02 and std_ic < abs(mean_ic) * 1.5:
            verdict = "STABLE ++"
        elif mean_ic > 0.01 and std_ic < abs(mean_ic) * 2.0:
            verdict = "MODERATE ~"
        elif mean_ic < -0.01:
            verdict = "NEGATIVE --"
        else:
            verdict = "UNSTABLE --"

        print(f"{ss.label:<22s} {ss.current_weight:>3d} "
              f"{mean_ic:>8.4f} {std_ic:>8.4f} {min_ic:>8.4f} {max_ic:>8.4f} "
              f"{ir:>7.2f} {verdict}")

    # ── All-horizon stability heatmap ──
    print(f"\n{'=' * 90}")
    print(f"   Stability Heatmap (IC mean/std per horizon)")
    print(f"{'=' * 90}")
    header = f"{'Detector':<22s}"
    for h in FORWARD_HORIZONS:
        header += f"  {h}d_IC     "
    print(header)
    print("-" * 90)
    for key in DETECTOR_REGISTRY:
        ss = stability.get(key)
        if ss is None:
            continue
        row = f"{ss.label:<22s}"
        for h in FORWARD_HORIZONS:
            if h in ss.ic_stability:
                mean_ic, std_ic, _, _ = ss.ic_stability[h]
                row += f"  {mean_ic:+.4f}/{std_ic:.4f}"
            else:
                row += f"  {'N/A':>12s}"
        print(row)
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pattern detector weight analysis")
    p.add_argument("--mock", action="store_true", help="Use mock data for demo")
    p.add_argument("--n-stocks", type=int, default=50, help="Mock: number of stocks")
    p.add_argument("--n-days", type=int, default=250, help="Mock: days per stock")
    p.add_argument("--data", type=str, default=None, help="Path to parquet cache dir")
    p.add_argument("--sample", type=int, default=0, help="Randomly sample N stocks from data (0=all)")
    p.add_argument("--json", action="store_true", help="Output recommendations as JSON")
    p.add_argument("--rolling", action="store_true", help="Run rolling window stability analysis")
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
                   help=f"Trading days per rolling window (default: {DEFAULT_WINDOW_SIZE})")
    p.add_argument("--window-step", type=int, default=DEFAULT_WINDOW_STEP,
                   help=f"Trading days between window starts (default: {DEFAULT_WINDOW_STEP})")
    return p.parse_args()


def _generate_mock_kline_data(n_stocks: int = 50, n_days: int = 120) -> dict[str, pd.DataFrame]:
    """Generate diverse mock K-line data for backtesting."""
    from core.mock_data import generate_mock_data

    kline_df, _, _ = generate_mock_data(n_stocks=n_stocks, n_days=n_days)
    result: dict[str, pd.DataFrame] = {}
    for sym, group in kline_df.groupby("symbol"):
        df = group.sort_values("date_idx") if "date_idx" in group.columns else group.reset_index(drop=True)
        # Ensure OHLC columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 10.0 if col != "volume" else 1_000_000.0
        result[sym] = df.reset_index(drop=True)
    return result


def _load_cache_data(cache_root: str) -> dict[str, pd.DataFrame]:
    """Load K-line data from cache parquet files.

    Args:
        cache_root: Path to cache directory (e.g. ``cache/``).
            Each subdirectory named YYYYMMDD should contain kline.parquet.

    Returns:
        {symbol: DataFrame} keyed by stock symbol, sorted by date.
    """
    root = Path(cache_root)
    if not root.exists():
        logger.error("Cache directory not found: %s", root)
        return {}

    result: dict[str, pd.DataFrame] = {}
    date_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and len(d.name) == 8 and d.name.isdigit()]
    )

    if not date_dirs:
        logger.error("No date folders found in %s", root)
        return {}

    logger.info("Loading K-line data from %d date folders ...", len(date_dirs))
    for date_dir in date_dirs:
        parquet_path = date_dir / "kline.parquet"
        if not parquet_path.exists():
            continue

        df = pd.read_parquet(parquet_path)
        if "symbol" not in df.columns:
            continue

        date_col = "date" if "date" in df.columns else "date_idx"
        for sym, group in df.groupby("symbol"):
            group_sorted = group.sort_values(date_col) if date_col in group.columns else group.reset_index(drop=True)
            if sym in result:
                result[sym] = pd.concat([result[sym], group_sorted], ignore_index=True)
            else:
                result[sym] = group_sorted.reset_index(drop=True)

    logger.info("Loaded %d unique stocks", len(result))
    return result


def main() -> int:
    args = parse_args()

    if args.data:
        logger.info("Loading data from cache: %s", args.data)
        kline_data = _load_cache_data(args.data)
        if not kline_data:
            logger.error("No valid K-line data found. Run a scan first to populate the cache, or use --mock.")
            return 1
    elif args.mock:
        logger.info("Generating mock data: %d stocks × %d days", args.n_stocks, args.n_days)
        kline_data = _generate_mock_kline_data(n_stocks=args.n_stocks, n_days=args.n_days)
    else:
        logger.error("Specify --mock for demo or --data <cache_dir> for real backtest data.")
        return 1

    # Sample if requested
    if args.sample > 0 and len(kline_data) > args.sample:
        import random
        sampled_keys = random.sample(sorted(kline_data.keys()), args.sample)
        kline_data = {k: kline_data[k] for k in sampled_keys}
        logger.info("Sampled %d stocks from %d total", args.sample, len(kline_data) + args.sample)

    engine = BacktestEngine(kline_data)

    if args.rolling:
        # Rolling window stability analysis
        logger.info("Running rolling window analysis (size=%d, step=%d)",
                     args.window_size, args.window_step)
        window_results, stability = engine.run_rolling(
            window_size=args.window_size,
            step=args.window_step,
        )

        if args.json:
            output = {
                "windows": [
                    {
                        "label": wr.window_label,
                        "start_date": wr.start_date,
                        "end_date": wr.end_date,
                        "n_stocks": wr.n_stocks,
                        "n_records": wr.n_records,
                        "ic_5d": {
                            key: wr.ic_matrix.get(key, {}).get(5)
                            for key in DETECTOR_REGISTRY
                        },
                    }
                    for wr in window_results
                ],
                "stability": {
                    key: {
                        "label": ss.label,
                        "current_weight": ss.current_weight,
                        "ic_stability_5d": (
                            list(ss.ic_stability[5])
                            if 5 in ss.ic_stability else None
                        ),
                    }
                    for key, ss in stability.items()
                },
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print_rolling_report(window_results, stability)

        # Also print standard recommendation for comparison
        logger.info("Running full-period backtest for comparison...")
        stats = engine.run()
        recommendations = engine.recommend_weights()
        print_report(stats, recommendations)
    else:
        # Standard single-period backtest
        stats = engine.run()
        recommendations = engine.recommend_weights()

        if args.json:
            output = {
                key: {
                    "label": rec["label"],
                    "current_weight": rec["current_weight"],
                    "recommended_weight": rec["recommended_weight"],
                    "normalized_weight": rec["normalized_weight"],
                    "ic_5d": rec["ic_5d"],
                    "hit_rate_5d": rec["hit_rate_5d"],
                    "trigger_count": rec["trigger_count"],
                    "reasons": rec["reasons"],
                }
                for key, rec in recommendations.items()
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print_report(stats, recommendations)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
