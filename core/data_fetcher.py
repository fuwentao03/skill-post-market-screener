"""Data fetcher: wraps panda_data SDK for the post-market screener."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Load .env file if present (for DEFAULT_USERNAME, DEFAULT_PASSWORD, etc.)
try:
    from dotenv import load_dotenv
    _env_file = Path(__file__).resolve().parent.parent / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass
import panda_data as pdd

logger = logging.getLogger(__name__)

DEFAULT_FIELDS_KLINE = [
    "open", "high", "low", "close", "volume", "amount", "pre_close",
]


_config_cache: dict | None = None


def _load_config() -> dict:
    """Load config.json with in-memory caching to avoid repeated disk reads."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    if config_path.exists():
        _config_cache = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        _config_cache = {}
    return _config_cache


def _retry_api_call(
    func,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: float = 30.0,
    description: str = "API call",
    **kwargs,
):
    """Call *func* with retry and exponential backoff.

    Args:
        func: Callable to invoke.
        max_retries: Maximum retry attempts (3 = 1 initial + 2 retries).
        base_delay: Initial delay in seconds before first retry.
        backoff_factor: Multiplier for successive retry delays.
        timeout: Per-call timeout in seconds. Not enforced at the
                 callable level (depends on underlying library support);
                 logged as a guideline.
        description: Human-readable label for log messages.
    """
    import time as _time

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    description, attempt + 1, max_retries + 1, e, delay,
                )
                _time.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    description, max_retries + 1, e,
                )
                raise


class DataFetcher:
    """Encapsulates all Pandadata API calls for the screener."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or _load_config()
        self._initialized = False

    def init_api(self) -> None:
        """Log into Pandadata service.

        Credentials are resolved in order:
        1. Environment variables: DEFAULT_USERNAME / DEFAULT_PASSWORD
        2. config.json: pandadata.username / pandadata.password

        Note: username must use "86" prefix (e.g. 8618046753943).
        """
        if self._initialized:
            return
        pd_cfg = self._config.get("pandadata", {})

        # Resolve credentials: env vars take precedence
        username = os.getenv("DEFAULT_USERNAME") or pd_cfg.get("username", "")
        password = os.getenv("DEFAULT_PASSWORD") or pd_cfg.get("password", "")
        base_url = pd_cfg.get("base_url", "http://pandadata.pandaaiquant.com")

        # Auto-add 86 prefix if missing
        if username and not username.startswith("86"):
            username = "86" + username
            logger.info("Auto-added 86 prefix: %s", username)

        if not username or not password:
            raise RuntimeError(
                "Pandadata credentials not configured. Set either:\n"
                "  - Environment variables: DEFAULT_USERNAME / DEFAULT_PASSWORD\n"
                "  - config.json: pandadata.username / pandadata.password\n"
                "Note: username must be 86+phone (e.g. 8618046753943)."
            )

        pdd.init_token(username=username, password=password, base_url=base_url)
        self._initialized = True
        logger.info("Pandadata API initialized (base_url=%s)", base_url)

    # --- Trading calendar ---

    def get_last_trade_date(self, exchange: str = "sh") -> str:
        """Return the latest completed A-share trading day as 'YYYYMMDD'."""
        return pdd.get_last_trade_date(exchange=exchange)

    def is_trading_day(self, date_str: str) -> bool:
        """Check if date_str is an A-share trading day."""
        cal = pdd.get_trade_cal(
            start_date=date_str, end_date=date_str, exchange="sh"
        )
        if cal.empty:
            return False
        # Column name is 'is_trade' in current SDK version
        val = cal.iloc[0].get("is_trade", cal.iloc[0].get("is_trading_day", "0"))
        return val in ("1", 1, True)

    # --- Stock universe ---

    def get_stock_universe(self, date_str: str) -> pd.DataFrame:
        """Return tradable A-share stocks on a given date.

        Columns include: symbol, name, market_cap, industry, list_status.
        """
        raw = _retry_api_call(
            pdd.get_trade_list,
            date=date_str,
            description="Stock universe",
        )
        if raw.empty:
            return raw
        return raw

    # --- K-line ---

    def fetch_kline_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        fields: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fetch daily K-line for a batch of symbols.

        Returns DataFrame with columns: symbol, date, open, high, low, close,
        volume, amount, pre_close (depending on fields).

        Chunks of 200 symbols are fetched in parallel (ThreadPoolExecutor, max 4
        workers) to reduce total data-acquisition time for the full ~5189-stock
        universe.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not symbols:
            return pd.DataFrame()
        fields = fields or DEFAULT_FIELDS_KLINE
        chunk_size = 200
        frames: list[pd.DataFrame] = []

        def _fetch_chunk(chunk_symbols: list[str], chunk_idx: int) -> pd.DataFrame:
            try:
                df = _retry_api_call(
                    pdd.get_stock_daily,
                    symbol=chunk_symbols,
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields,
                    description=f"K-line chunk {chunk_idx}",
                )
                if not df.empty:
                    return df
                return pd.DataFrame()
            except Exception as e:
                logger.warning("K-line fetch failed for chunk %d after retries: %s", chunk_idx, e)
                return pd.DataFrame()

        # Build chunk list
        chunks = [
            (symbols[i : i + chunk_size], i)
            for i in range(0, len(symbols), chunk_size)
        ]

        max_workers = min(4, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_chunk, chunk, idx): idx
                for chunk, idx in chunks
            }
            for future in as_completed(futures):
                df = future.result()
                if not df.empty:
                    frames.append(df)

        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame()

    # --- Stock info ---

    def fetch_stock_details_batch(
        self,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Fetch stock basic info: name, industry, list_status.

        Pandadata get_stock_detail returns: name, sector_code_name, status.
        We remap: sector_code_name → industry, status → list_status.
        Market cap is computed separately from share_float × close price.
        """
        if not symbols:
            return pd.DataFrame()

        chunk_size = 200
        frames: list[pd.DataFrame] = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            try:
                df = _retry_api_call(
                    pdd.get_stock_detail,
                    symbol=chunk,
                    description=f"Stock detail chunk {i}",
                )
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning("Stock detail fetch failed for chunk %d after retries: %s", i, e)
        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)

        # Remap columns to our expected names
        col_map = {
            "sector_code_name": "industry",
            "status": "list_status",
        }
        for old, new in col_map.items():
            if old in result.columns and new not in result.columns:
                result[new] = result[old]

        # Ensure required columns exist
        for col in ["name", "industry", "list_status"]:
            if col not in result.columns:
                result[col] = "未知"

        # Keep only needed columns
        keep = ["symbol", "name", "industry", "list_status"]
        result = result[[c for c in keep if c in result.columns]]

        logger.info("Stock details: %d stocks", len(result))
        return result

    def fetch_market_cap_batch(
        self,
        symbols: list[str],
        trade_date: str,
    ) -> dict[str, float]:
        """Fetch total shares and compute market cap (close × total_a).

        Returns dict[symbol] = market_cap in 亿元.
        """
        if not symbols:
            return {}

        logger.info("Fetching share float for %d stocks ...", len(symbols))
        chunk_size = 200
        frames: list[pd.DataFrame] = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i : i + chunk_size]
            try:
                # Use a date range (API requires start < end for share_float)
                range_start = (datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
                df = _retry_api_call(
                    pdd.get_share_float,
                    symbol=chunk,
                    start_date=range_start,
                    end_date=trade_date,
                    description=f"Share float chunk {i}",
                )
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning("Share float fetch failed for chunk %d after retries: %s", i, e)

        if not frames:
            logger.warning("No share float data — market_cap will be 0")
            return {}

        sf = pd.concat(frames, ignore_index=True)
        # Keep latest per symbol
        sf = sf.sort_values("date" if "date" in sf.columns else sf.columns[1])
        sf = sf.groupby("symbol").last().reset_index()

        # total_a = A股总股本 (shares), close from K-line gives price
        # market_cap = close × total_a, stored later after K-line is fetched
        shares_map = {}
        for _, row in sf.iterrows():
            sym = row.get("symbol", "")
            total_a = float(row.get("total_a", row.get("total", 0)) or 0)
            if sym and total_a > 0:
                shares_map[sym] = total_a

        logger.info("Share float: %d stocks", len(shares_map))
        return shares_map

    # --- High-level convenience ---

    def fetch_all_data(
        self, trade_date: str, lookback_days: int = 120
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch K-line and stock info for the entire universe.

        Returns:
            (kline_df, info_df)
        info_df columns: symbol, name, industry, list_status, market_cap.
        market_cap is in 亿元.

        Note: Fund flow data is fetched separately by the pipeline layer
        (via FlowFetcher) for a clean separation of concerns.
        """
        # Determine lookback start date (approximate)
        dt = datetime.strptime(trade_date, "%Y%m%d")
        start_dt = dt - timedelta(days=lookback_days * 2)
        start_date = start_dt.strftime("%Y%m%d")

        # 1. Get stock universe
        logger.info("Fetching stock universe for %s ...", trade_date)
        universe = self.get_stock_universe(trade_date)
        if universe.empty:
            logger.warning("Empty stock universe for %s", trade_date)
            return pd.DataFrame(), pd.DataFrame()

        if "symbol" in universe.columns:
            symbols = universe["symbol"].dropna().unique().tolist()
        else:
            symbols = universe.iloc[:, 0].dropna().unique().tolist()

        logger.info("Universe: %d stocks", len(symbols))

        # 2. Fetch K-line
        logger.info("Fetching K-line data (%d stocks, %s ~ %s) ...",
                     len(symbols), start_date, trade_date)
        kline_df = self.fetch_kline_batch(symbols, start_date, trade_date)

        # 3. Fetch share float → compute market_cap (close × total_a → 亿元)
        shares_map = self.fetch_market_cap_batch(symbols, trade_date)

        # 4. Fetch stock info
        logger.info("Fetching stock details ...")
        info_df = self.fetch_stock_details_batch(symbols)

        # Compute market_cap from close_price × total_shares
        if shares_map and not kline_df.empty:
            date_col = "date" if "date" in kline_df.columns else "date_idx"
            kline_sorted = kline_df.sort_values(date_col)
            latest = kline_sorted.groupby("symbol").last().reset_index()
            mc_data: dict[str, float] = {}
            for _, row in latest.iterrows():
                sym = row.get("symbol", "")
                close = float(row.get("close", 0) or 0)
                shares = shares_map.get(sym, 0)
                if sym and close > 0 and shares > 0:
                    mc_data[sym] = round(close * shares / 1e8, 2)  # → 亿元
            info_df["market_cap"] = info_df["symbol"].map(mc_data).fillna(0)
            logger.info("Market cap computed for %d stocks", len(mc_data))
        else:
            info_df["market_cap"] = 0

        return kline_df, info_df
