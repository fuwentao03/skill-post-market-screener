"""Scoring and ranking for the post-market screener."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .pattern_detector import get_pattern_score

logger = logging.getLogger(__name__)


def compute_score(
    pattern_results: dict[str, float],
    inflow_rate: float,
    market_cap: float,   # in 亿元
    turnover: float,     # in 万元
) -> dict:
    """Compute composite score for a single stock.

    Returns:
        {score, pattern_score, flow_score, quality_bonus}
    """
    pattern_score = get_pattern_score(pattern_results)
    # Flow score: 1 point per 5% inflow rate, capped at 3
    # inflow_rate is in decimal form (e.g. 0.11 = 11%), so divide by 0.05
    flow_score = min(inflow_rate / 0.05, 3.0) if inflow_rate > 0 else 0
    # Quality bonus
    quality_bonus = 0
    if market_cap >= 50:   # >= 50 亿 CNY
        quality_bonus += 1
    if turnover >= 20000:  # >= 2 亿 CNY (20000 万)
        quality_bonus += 1

    total = round(pattern_score + flow_score + quality_bonus, 2)
    return {
        "score": total,
        "pattern_score": pattern_score,
        "flow_score": round(flow_score, 2),
        "quality_bonus": quality_bonus,
    }


def rank_stocks(
    stocks: list[dict],
    score_key: str = "score",
    top_n: int = 20,
) -> list[dict]:
    """Sort stocks by score descending and return top N."""
    sorted_stocks = sorted(stocks, key=lambda s: s.get(score_key, 0), reverse=True)
    ranked = []
    for i, stock in enumerate(sorted_stocks[:top_n], 1):
        stock["rank"] = i
        ranked.append(stock)
    return ranked
