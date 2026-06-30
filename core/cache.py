"""Local Parquet cache for K-line, fund flow, and stock info data.

Avoids re-fetching the full ~5189-stock universe from Pandadata on every run
during the same day. Cache is keyed by trade date (YYYYMMDD).

Cache directory::

    cache/
      20260629/
        kline.parquet
        flow.parquet
        info.parquet

Usage::

    mgr = CacheManager()
    if mgr.has(date_str):
        kline, flow, info = mgr.load(date_str)
    else:
        kline, flow, info = fetcher.fetch_all_data(date_str)
        mgr.save(date_str, kline, flow, info)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

META_FILENAME = ".cache_meta.json"


class CacheManager:
    """Date-keyed Parquet cache for screener data."""

    def __init__(self, cache_root: str | Path = "cache") -> None:
        self._root = Path(cache_root)

    # ── public API ────────────────────────────────────────────

    def has(self, trade_date: str) -> bool:
        """Check if cached data exists for *trade_date*."""
        if not self._root.exists():
            return False
        date_dir = self._root / trade_date
        return date_dir.is_dir() and (date_dir / "kline.parquet").exists()

    def load(self, trade_date: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load cached (kline, flow, info) for *trade_date*.

        Returns empty DataFrames on corrupt or missing cache files
        rather than crashing the pipeline.
        """
        date_dir = self._root / trade_date
        try:
            kline = pd.read_parquet(date_dir / "kline.parquet")
        except Exception:
            logger.warning("Corrupt or unreadable kline cache for %s, ignoring", trade_date)
            kline = pd.DataFrame()
        try:
            flow = pd.read_parquet(date_dir / "flow.parquet")
        except Exception:
            logger.warning("Corrupt or unreadable flow cache for %s, ignoring", trade_date)
            flow = pd.DataFrame()
        try:
            info = pd.read_parquet(date_dir / "info.parquet")
        except Exception:
            logger.warning("Corrupt or unreadable info cache for %s, ignoring", trade_date)
            info = pd.DataFrame()
        logger.info("Cache hit: %s (%d kline rows)", trade_date, len(kline))
        return kline, flow, info

    def load_meta(self, trade_date: str) -> dict:
        """Load cache metadata (including flow_source) without reading parquet files."""
        date_dir = self._root / trade_date
        meta_path = date_dir / META_FILENAME
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save(
        self,
        trade_date: str,
        kline_df: pd.DataFrame,
        flow_df: pd.DataFrame,
        info_df: pd.DataFrame,
    ) -> None:
        """Persist data to cache."""
        date_dir = self._root / trade_date
        date_dir.mkdir(parents=True, exist_ok=True)
        kline_df.to_parquet(date_dir / "kline.parquet", index=False)
        flow_df.to_parquet(date_dir / "flow.parquet", index=False)
        info_df.to_parquet(date_dir / "info.parquet", index=False)
        # Write metadata
        flow_source = flow_df.attrs.get("flow_source", "unknown") if not flow_df.empty else "none"
        meta = {
            "trade_date": trade_date,
            "kline_rows": len(kline_df),
            "flow_rows": len(flow_df),
            "info_rows": len(info_df),
            "kline_columns": list(kline_df.columns),
            "flow_source": flow_source,
        }
        (date_dir / META_FILENAME).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Cache saved: %s", date_dir)

    def clear_old(self, keep_days: int = 30) -> int:
        """Remove cached dates older than *keep_days* calendar days.

        Returns:
            Number of directories removed.
        """
        from datetime import datetime, timedelta

        if not self._root.exists():
            return 0

        cutoff = datetime.now() - timedelta(days=keep_days)
        removed = 0
        for child in self._root.iterdir():
            if not child.is_dir():
                continue
            try:
                dt = datetime.strptime(child.name, "%Y%m%d")
                if dt < cutoff:
                    for f in child.iterdir():
                        f.unlink()
                    child.rmdir()
                    removed += 1
                    logger.info("Cache cleanup: removed %s", child.name)
            except ValueError:
                continue
        return removed

    @property
    def cached_dates(self) -> list[str]:
        """Return sorted list of cached dates."""
        if not self._root.exists():
            return []
        dates = []
        for child in self._root.iterdir():
            if child.is_dir() and (child / "kline.parquet").exists():
                dates.append(child.name)
        return sorted(dates)
