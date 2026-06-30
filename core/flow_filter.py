"""Capital flow filter for the post-market screener."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_inflow_rate(rate: float) -> float:
    """Convert inflow_rate from percentage (0-100) to decimal (0-1) if needed.

    AKShare may return rates as percentages (e.g. 12.5 = 12.5%).
    This normalizes to decimal form (e.g. 0.125) for consistent downstream use.
    """
    if rate > 1:
        return rate / 100.0
    return rate


class FlowFilter:
    """Filters stocks by capital flow criteria."""

    def __init__(
        self,
        main_inflow_rate_min: float = 0.05,
        min_turnover: float = 5000,  # 5000万 (in 万元)
        super_large_positive: bool = True,
    ):
        self.main_inflow_rate_min = main_inflow_rate_min
        self.min_turnover = min_turnover
        self.super_large_positive = super_large_positive

    def filter_single(
        self,
        inflow_rate: float,
        turnover: float,
        super_large_inflow: float,
    ) -> tuple[bool, list[str]]:
        """Check if a single stock passes flow criteria.

        Returns:
            (passed, reasons_list)
        """
        reasons: list[str] = []

        if turnover <= self.min_turnover:
            reasons.append(f"成交额 {turnover:.0f} <= {self.min_turnover:.0f} 阈值")
            return False, reasons

        if inflow_rate <= self.main_inflow_rate_min:
            reasons.append(f"主力流入率 {inflow_rate:.1%} <= {self.main_inflow_rate_min:.0%} 阈值")
            return False, reasons

        if self.super_large_positive and super_large_inflow <= 0:
            reasons.append(f"超大单净流入 {super_large_inflow:.0f} <= 0")
            return False, reasons

        return True, ["通过"]

    def filter_dataframe(
        self,
        flow_df: pd.DataFrame,
        symbol_col: str = "symbol",
        inflow_rate_col: str = "main_inflow_rate",
        turnover_col: str = "turnover",
        super_large_col: str = "super_large_net_inflow",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply flow filter to a DataFrame of fund flow data.

        Args:
            flow_df: DataFrame with per-stock fund flow data.

        Returns:
            (passed_df, rejected_df)
        """
        if flow_df.empty:
            return flow_df, flow_df

        rates = flow_df.get(inflow_rate_col, pd.Series([0] * len(flow_df)))
        rates = rates.apply(normalize_inflow_rate)

        turnover = flow_df.get(turnover_col, pd.Series([0] * len(flow_df)))
        super_large = flow_df.get(super_large_col, pd.Series([0] * len(flow_df)))

        mask_turnover = turnover > self.min_turnover
        mask_rate = rates > self.main_inflow_rate_min
        mask_super = super_large > 0 if self.super_large_positive else pd.Series([True] * len(flow_df))

        passed_mask = mask_turnover & mask_rate & mask_super

        passed = flow_df[passed_mask].copy()
        rejected = flow_df[~passed_mask].copy()

        logger.info("Flow filter: %d passed, %d rejected", len(passed), len(rejected))
        return passed, rejected
