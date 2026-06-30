"""Fund flow fetcher: Tonghuashun primary + East Money fallbacks.

Three-path architecture (in priority order):

1. **Tonghuashun (同花顺)** via AKShare ``stock_fund_flow_individual`` —
   independent, reliable data source. No order-size breakdown (proxied).
2. **AKShare** ``stock_individual_fund_flow_rank`` — East Money-backed,
   convenient but fragile (one page failure kills entire fetch).
3. **East Money direct** — page-level parallel fetch with retry.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# AKShare → project column mapping for fund flow
_AK_FLOW_RENAME = {
    "代码": "symbol_raw",
    "今日主力净流入-净额": "main_net_inflow",
    "今日主力净流入-净占比": "main_inflow_rate",
    "今日超大单净流入-净额": "super_large_net_inflow",
    "今日大单净流入-净额": "large_net_inflow",
}

# East Money API field mapping (f12=code, f14=name, f62=main_net_inflow,
# f184=main_inflow_rate, f66=super_large_net_inflow, f72=large_net_inflow)
_EM_FIELDS = "f12,f14,f62,f184,f66,f72"
_EM_URL = "https://push2.eastmoney.com/api/qt/clist/get"
_EM_MARKET_FILTER = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"
_EM_PAGE_SIZE = 100  # East Money hard cap per page
_EM_MAX_CONCURRENT = 4  # parallel page fetches


def _add_exchange_suffix(code: str) -> str:
    """Add .SH / .SZ suffix to a bare A-share code."""
    code = str(code).zfill(6)
    if code.startswith("6"):
        return f"{code}.SH"
    return f"{code}.SZ"


def _build_http_session() -> requests.Session:
    """Create a requests Session with retry adapter for transient errors."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://data.eastmoney.com/",
    })
    return session


def _fetch_em_page(session: requests.Session, page: int, page_size: int = _EM_PAGE_SIZE,
                   timeout: int = 15) -> list[dict] | None:
    """Fetch a single page from the East Money fund flow API.

    Returns list of row dicts on success, None on failure.
    """
    params = {
        "fid": "f62", "po": "1", "pz": str(page_size), "pn": str(page),
        "np": "1", "fltt": "2", "invt": "2",
        "fs": _EM_MARKET_FILTER,
        "fields": _EM_FIELDS,
    }
    try:
        resp = session.get(_EM_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("rc") != 0 or data.get("data") is None:
            logger.warning("East Money page %d returned rc=%s", page, data.get("rc"))
            return None
        return data["data"].get("diff") or []
    except Exception:
        logger.debug("East Money page %d failed", page, exc_info=True)
        return None


def _fetch_all_em(session: requests.Session, max_workers: int = _EM_MAX_CONCURRENT,
                  page_retries: int = 2, timeout: int = 15) -> pd.DataFrame:
    """Fetch all A-share fund flow data directly from East Money.

    Parallelises page fetches with page-level retry. Returns a DataFrame
    with the same schema as the AKShare path.
    """
    # First request: get total count
    first_page = _fetch_em_page(session, 1, page_size=1, timeout=timeout)
    if first_page is None:
        raise RuntimeError("East Money fund flow API unreachable (first-page probe failed)")

    # Probe total — use a small request to get metadata
    probe = session.get(_EM_URL, params={
        "fid": "f62", "po": "1", "pz": "1", "pn": "1",
        "np": "1", "fltt": "2", "invt": "2",
        "fs": _EM_MARKET_FILTER, "fields": _EM_FIELDS,
    }, timeout=timeout)
    probe.raise_for_status()
    total = probe.json()["data"]["total"]
    total_pages = (total + _EM_PAGE_SIZE - 1) // _EM_PAGE_SIZE
    logger.info("East Money fund flow: %d stocks, %d pages", total, total_pages)

    all_rows: list[dict] = []
    failed_pages: set[int] = set()

    # Fetch pages in parallel batches
    pages_to_fetch = list(range(1, total_pages + 1))
    batch_size = max_workers * 2  # fetch in overlapping batches

    for batch_start in range(0, len(pages_to_fetch), batch_size):
        batch = pages_to_fetch[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_em_page, session, p, _EM_PAGE_SIZE, timeout): p
                for p in batch
            }
            for future in as_completed(futures):
                page = futures[future]
                try:
                    rows = future.result()
                except Exception:
                    rows = None
                if rows is not None:
                    all_rows.extend(rows)
                else:
                    failed_pages.add(page)

    # Retry failed pages sequentially (with backoff)
    for page in sorted(failed_pages):
        for retry_attempt in range(page_retries):
            time.sleep(1.0 * (retry_attempt + 1))
            rows = _fetch_em_page(session, page, _EM_PAGE_SIZE, timeout=timeout)
            if rows is not None:
                all_rows.extend(rows)
                logger.debug("East Money page %d recovered on retry %d", page, retry_attempt + 1)
                break
        else:
            logger.warning("East Money page %d failed after %d retries", page, page_retries)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={
        "f12": "symbol_raw", "f14": "name",
        "f62": "main_net_inflow", "f184": "main_inflow_rate",
        "f66": "super_large_net_inflow", "f72": "large_net_inflow",
    })

    # Add exchange suffix
    df["symbol"] = df["symbol_raw"].astype(str).str.zfill(6).apply(_add_exchange_suffix)
    df = df.drop(columns=["symbol_raw"])

    # Convert yuan → 万元
    for col in ["main_net_inflow", "super_large_net_inflow", "large_net_inflow"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 10000

    if "main_inflow_rate" in df.columns:
        df["main_inflow_rate"] = pd.to_numeric(df["main_inflow_rate"], errors="coerce")

    df["turnover"] = 0.0
    df = df.reset_index(drop=True)
    logger.info("East Money direct fetch: %d stocks (pages with errors: %d)",
                len(df), len(failed_pages))
    return df


class FlowFetcher:
    """Fetches A-share fund flow data.

    Primary path: Tonghuashun (同花顺) via ``stock_fund_flow_individual`` —
    independent data source that does not depend on East Money infrastructure.

    Fallback paths: AKShare (East Money) → direct East Money API.

    All monetary values are converted to 万元.
    """

    def __init__(
        self,
        max_retries: int = 2,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        """
        Args:
            max_retries: Retry attempts for the AKShare path (before fallback).
            base_delay: Initial delay between retries in seconds.
            backoff_factor: Multiplier for successive delays.
        """
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._backoff_factor = backoff_factor

    def fetch(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch today's fund flow.

        Priority: Tonghuashun (同花顺) → AKShare → East Money direct.

        Args:
            symbols: List of stock symbols with exchange suffix (e.g. 600519.SH).

        Returns:
            DataFrame with columns: symbol, main_net_inflow, main_inflow_rate,
            super_large_net_inflow, large_net_inflow, turnover.
            Empty DataFrame if all three paths fail.
        """
        if not symbols:
            return pd.DataFrame()

        # Path 1: Tonghuashun (同花顺) — primary, independent data source
        try:
            df = self._fetch_via_tonghuashun(symbols)
            if not df.empty:
                df.attrs["flow_source"] = "tonghuashun"
                return df
        except Exception as e:
            logger.warning("Tonghuashun path failed: %s", e)

        # Path 2: AKShare (East Money)
        logger.info("Tonghuashun path failed — falling back to AKShare (East Money)")
        df = self._fetch_via_akshare(symbols)
        if not df.empty:
            df.attrs["flow_source"] = "akshare"
            return df

        # Path 3: Direct East Money API
        logger.info("AKShare path failed — falling back to direct East Money API")
        try:
            df = self._fetch_via_eastmoney(symbols)
            if not df.empty:
                df.attrs["flow_source"] = "eastmoney"
                return df
        except Exception as e:
            logger.warning("East Money direct fallback also failed: %s", e)

        logger.warning("All fund flow paths exhausted — returning empty DataFrame")
        return pd.DataFrame()

    # ── AKShare path ───────────────────────────────────────────

    def _fetch_via_akshare(self, symbols: list[str]) -> pd.DataFrame:
        """Attempt fund flow via AKShare with retry."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                return self._do_akshare_fetch(symbols)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    delay = self._base_delay * (self._backoff_factor ** (attempt - 1))
                    logger.warning(
                        "AKShare fetch attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt, self._max_retries, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "AKShare fetch failed after %d attempts: %s",
                        self._max_retries, e,
                    )

        return pd.DataFrame()

    def _do_akshare_fetch(self, symbols: list[str]) -> pd.DataFrame:
        """Single AKShare fetch attempt."""
        import akshare as ak

        logger.info("Fetching fund flow from AKShare (East Money) ...")
        raw = ak.stock_individual_fund_flow_rank(indicator="今日")

        if raw.empty:
            logger.warning("AKShare returned empty fund flow data")
            return pd.DataFrame()

        # Rename Chinese columns → English, keep only what we need
        df = raw.rename(columns=_AK_FLOW_RENAME)
        keep_cols = [
            "symbol_raw", "main_net_inflow", "main_inflow_rate",
            "super_large_net_inflow", "large_net_inflow",
        ]
        df = df[[c for c in keep_cols if c in df.columns]].copy()

        # Add exchange suffix
        df["symbol"] = df["symbol_raw"].apply(_add_exchange_suffix)
        df = df.drop(columns=["symbol_raw"])

        # Filter to requested universe
        df = df[df["symbol"].isin(symbols)]

        # Convert yuan → 万元
        for col in ["main_net_inflow", "super_large_net_inflow", "large_net_inflow"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 10000

        # Inflow rate: keep as percentage (e.g. 11.87 → 11.87%)
        if "main_inflow_rate" in df.columns:
            df["main_inflow_rate"] = pd.to_numeric(df["main_inflow_rate"], errors="coerce")

        # turnover placeholder — filled later from K-line
        df["turnover"] = 0.0

        logger.info("AKShare fund flow: %d stocks", len(df))
        return df.reset_index(drop=True)

    # ── East Money direct path ─────────────────────────────────

    def _fetch_via_eastmoney(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch fund flow directly from East Money API with page-level retry."""
        session = _build_http_session()
        try:
            df = _fetch_all_em(session)
        finally:
            session.close()

        if df.empty:
            return pd.DataFrame()

        # Filter to requested universe
        df = df[df["symbol"].isin(symbols)]
        logger.info("East Money direct: %d stocks matched universe", len(df))
        return df.reset_index(drop=True)

    # ── Tonghuashun (同花顺) path ───────────────────────────────

    def _fetch_via_tonghuashun(self, symbols: list[str]) -> pd.DataFrame:
        """Fetch fund flow from Tonghuashun (同花顺) via AKShare.

        Uses ``ak.stock_fund_flow_individual()`` which returns ~5186 stocks
        with total inflow/outflow/net/turnover. This source does NOT provide
        order-size breakdown (super_large/large/medium/small), so we proxy
        those from the net amount.

        All monetary values are converted to 万元.
        """
        import akshare as ak

        logger.info("Fetching fund flow from Tonghuashun (同花顺) ...")
        try:
            raw = ak.stock_fund_flow_individual()
        except Exception as e:
            logger.warning("Tonghuashun fetch failed: %s", e)
            return pd.DataFrame()

        if raw.empty:
            logger.warning("Tonghuashun returned empty fund flow data")
            return pd.DataFrame()

        df = raw.copy()

        # Map columns: 股票代码→symbol, 净额→main_net_inflow, 成交额→turnover
        col_map = {
            "股票代码": "symbol_raw",
            "股票简称": "name",
            "净额": "net_str",
            "成交额": "turnover_str",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if "symbol_raw" not in df.columns:
            logger.warning("Tonghuashun: missing 股票代码 column")
            return pd.DataFrame()

        # Add exchange suffix
        df["symbol"] = df["symbol_raw"].astype(str).str.zfill(6).apply(_add_exchange_suffix)

        # Parse monetary strings (e.g. "1.09亿" → 10900万, "7316.85万" → 7316.85万)
        def _parse_amount(val) -> float:
            if pd.isna(val):
                return 0.0
            s = str(val).strip()
            if not s or s == "-":
                return 0.0
            try:
                if s.endswith("亿"):
                    return float(s[:-1]) * 10000
                elif s.endswith("万"):
                    return float(s[:-1])
                else:
                    return float(s) / 10000  # assume raw yuan
            except (ValueError, TypeError):
                return 0.0

        net_series = df["net_str"].apply(_parse_amount) if "net_str" in df.columns else pd.Series(0.0, index=df.index)
        turnover_series = df["turnover_str"].apply(_parse_amount) if "turnover_str" in df.columns else pd.Series(0.0, index=df.index)

        df["main_net_inflow"] = net_series
        df["turnover"] = turnover_series

        # Compute inflow rate: net / turnover * 100
        df["main_inflow_rate"] = 0.0
        mask = turnover_series > 0
        df.loc[mask, "main_inflow_rate"] = (
            net_series[mask] / turnover_series[mask] * 100
        )

        # No order-size breakdown: proxy super_large and large as net/2 each
        df["super_large_net_inflow"] = net_series * 0.5
        df["large_net_inflow"] = net_series * 0.5

        # Filter to requested universe
        df = df[df["symbol"].isin(symbols)]

        # Keep standard columns
        keep = [
            "symbol", "main_net_inflow", "main_inflow_rate",
            "super_large_net_inflow", "large_net_inflow", "turnover",
        ]
        df = df[[c for c in keep if c in df.columns]]

        logger.info("Tonghuashun: %d stocks matched universe", len(df))
        return df.reset_index(drop=True)
