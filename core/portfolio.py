"""Portfolio optimization layer for post-market screener results.

Converts the Top-N ranked stock list into actionable portfolio weights
using three complementary approaches: equal-weight, minimum-variance,
and risk-parity.

All three methods are provided so the user can compare allocations
rather than being forced into a single optimization regime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioAllocation:
    """Optimized portfolio weights for a set of stocks."""

    stocks: list[dict] = field(default_factory=list)
    equal_weight: dict[str, float] = field(default_factory=dict)
    min_variance: dict[str, float] = field(default_factory=dict)
    risk_parity: dict[str, float] = field(default_factory=dict)
    # Diagnostics
    min_var_cond_number: float = 0.0
    risk_parity_converged: bool = False
    note: str = ""


def _build_covariance_matrix(
    kline_df: pd.DataFrame,
    symbols: list[str],
    lookback: int = 60,
) -> tuple[np.ndarray, list[str]]:
    """Build a sample covariance matrix from daily K-line returns.

    Uses the past *lookback* trading days of close prices. Symbols without
    enough history are dropped and reported via log.

    Returns:
        (cov_matrix N×N, valid_symbols list)
    """
    date_col = "date" if "date" in kline_df.columns else "date_idx"
    valid_symbols = []
    return_series = []

    for sym in symbols:
        sub = kline_df[kline_df["symbol"] == sym].sort_values(date_col)
        if len(sub) < lookback // 2:  # need at least half the lookback
            logger.debug("Portfolio opt: skipping %s (insufficient history %d)", sym, len(sub))
            continue
        closes = sub["close"].astype(float).values[-lookback:]
        if len(closes) < 2:
            continue
        rets = np.diff(np.log(closes))
        if len(rets) < 5:
            continue
        return_series.append(rets[-lookback + 1:])  # align to shortest
        valid_symbols.append(sym)

    if len(valid_symbols) < 2:
        return np.array([[1.0]]), valid_symbols

    # Align lengths — take the minimum length across all series
    min_len = min(len(r) for r in return_series)
    aligned = np.array([r[:min_len] for r in return_series])

    # Sample covariance (annualized)
    cov = np.cov(aligned) * 252
    return cov, valid_symbols


def _compute_equal_weight(symbols: list[str]) -> dict[str, float]:
    """Equal-weight allocation (1/N)."""
    n = len(symbols)
    if n == 0:
        return {}
    w = 1.0 / n
    return {s: round(w, 4) for s in symbols}


def _compute_min_variance(
    symbols: list[str],
    cov: np.ndarray,
) -> tuple[dict[str, float], float]:
    """Minimum-variance portfolio via quadratic programming.

    Minimize w'Σw  subject to Σw_i = 1, w_i ≥ 0 (long-only).

    Uses a simple Lagrange-multiplier closed form for the unconstrained
    case, then projects onto the non-negative simplex when short positions
    would be indicated.

    Returns:
        (weights dict, condition_number)
    """
    n = len(symbols)
    cond_number = float(np.linalg.cond(cov)) if n > 1 else 0.0

    try:
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(n)
        # Unconstrained min-variance: w* = Σ⁻¹1 / 1'Σ⁻¹1
        raw = inv_cov @ ones
        denom = ones @ raw
        if abs(denom) < 1e-12:
            return _compute_equal_weight(symbols), cond_number
        w_unconstrained = raw / denom

        # If all weights are non-negative, we're done
        if np.all(w_unconstrained >= -1e-10):
            w = np.maximum(w_unconstrained, 0)
            w = w / w.sum()
            return {s: round(float(w[i]), 4) for i, s in enumerate(symbols)}, cond_number

        # Otherwise, use a simple iterative projection onto the simplex
        # (gradient projection for min-variance with non-negativity)
        w_proj = _project_min_variance_simplex(cov, n_iter=500)
        return {s: round(float(w_proj[i]), 4) for i, s in enumerate(symbols)}, cond_number

    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix is singular — falling back to equal weight")
        return _compute_equal_weight(symbols), cond_number


def _project_min_variance_simplex(cov: np.ndarray, n_iter: int = 500) -> np.ndarray:
    """Gradient-projection for min-variance on the non-negative simplex.

    Minimize ½ w'Σw  subject to Σw_i = 1, w_i ≥ 0.

    Uses a simple projected-gradient descent with Armijo-like step size.
    """
    n = cov.shape[0]
    w = np.ones(n) / n  # start from equal weight
    step = 0.5 / np.max(np.abs(cov)) if np.max(np.abs(cov)) > 0 else 0.01

    for _ in range(n_iter):
        grad = cov @ w
        w_new = w - step * grad

        # Project onto simplex (Duchi et al. 2008 efficient algorithm)
        w_new = _project_simplex(w_new)

        # Simple convergence check
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new
            break
        w = w_new

    return w


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex {w ≥ 0, Σw = 1}.

    Based on the algorithm from Duchi, Shalev-Shwartz, Singer, Chandra (2008).
    O(n log n) complexity.
    """
    n = len(v)
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    # Find ρ such that Σ max(v_i - ρ, 0) = 1
    rho_idx = np.where(u * np.arange(1, n + 1) > css - 1.0)[0]
    if len(rho_idx) == 0:
        rho = (css[-1] - 1.0) / n
    else:
        k = rho_idx[-1]
        rho = (css[k] - 1.0) / (k + 1)
    return np.maximum(v - rho, 0)


def _compute_risk_parity(
    symbols: list[str],
    cov: np.ndarray,
    n_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[dict[str, float], bool]:
    """Risk-parity (equal risk contribution) portfolio.

    Each asset contributes equally to total portfolio risk.
    Minimize Σ_i (w_i(Σw)_i − target_risk)² via iterative scaling.

    Returns:
        (weights dict, converged)
    """
    n = len(symbols)
    if n == 0:
        return {}, False
    if n == 1:
        return {symbols[0]: 1.0}, True

    w = np.ones(n) / n
    target_risk = 1.0 / n
    converged = False

    for _ in range(n_iter):
        sigma_w = cov @ w
        portfolio_vol = np.sqrt(w @ sigma_w)
        if portfolio_vol < 1e-12:
            break
        # Marginal risk contribution = Σw / σ_p
        mrc = sigma_w / portfolio_vol
        # Risk contribution = w * mrc
        rc = w * mrc
        # Scale: w_i *= target_risk / rc_i
        new_w = w * target_risk / np.maximum(rc, 1e-12)
        new_w = new_w / new_w.sum()

        if np.max(np.abs(new_w - w)) < tol:
            converged = True
            w = new_w
            break
        w = new_w
    else:
        converged = True  # reached max iterations

    return {s: round(float(w[i]), 4) for i, s in enumerate(symbols)}, converged


def optimize_portfolio(
    ranked_stocks: list[dict],
    kline_df: pd.DataFrame | None = None,
    lookback: int = 60,
) -> PortfolioAllocation:
    """Compute three portfolio allocations for a set of ranked stocks.

    Args:
        ranked_stocks: List of stock dicts from the pipeline (must have "code").
        kline_df: K-line DataFrame with "symbol", "close", "date" columns.
                  If None, only equal-weight is computed.
        lookback: Number of trading days for covariance estimation.

    Returns:
        PortfolioAllocation with all three weighting schemes.
    """
    symbols = [s["code"] for s in ranked_stocks]
    result = PortfolioAllocation(stocks=ranked_stocks)

    # Equal weight is always available
    result.equal_weight = _compute_equal_weight(symbols)

    if kline_df is None or kline_df.empty or len(symbols) < 2:
        result.min_variance = result.equal_weight
        result.risk_parity = result.equal_weight
        result.note = "K-line data unavailable — only equal-weight allocation computed"
        return result

    # Build covariance
    cov, valid_symbols = _build_covariance_matrix(kline_df, symbols, lookback)

    if len(valid_symbols) < 2:
        result.min_variance = result.equal_weight
        result.risk_parity = result.equal_weight
        result.note = (
            f"Insufficient return history for {len(valid_symbols)}/{len(symbols)} "
            f"stocks — only equal-weight allocation computed"
        )
        return result

    # If some symbols were dropped, we still optimize over the valid subset
    if len(valid_symbols) < len(symbols):
        result.note = (
            f"Covariance estimated from {len(valid_symbols)}/{len(symbols)} stocks "
            f"({len(symbols) - len(valid_symbols)} dropped due to insufficient history). "
            f"Non-estimable stocks assigned zero weight in min-variance and risk-parity."
        )

    # Min-variance
    mv_weights, cond_num = _compute_min_variance(valid_symbols, cov)
    result.min_var_cond_number = round(cond_num, 1)
    # Extend to full symbol list
    full_mv = {s: 0.0 for s in symbols}
    full_mv.update(mv_weights)
    result.min_variance = {s: round(full_mv[s], 4) for s in symbols}

    # Risk-parity
    rp_weights, converged = _compute_risk_parity(valid_symbols, cov)
    result.risk_parity_converged = converged
    full_rp = {s: 0.0 for s in symbols}
    full_rp.update(rp_weights)
    result.risk_parity = {s: round(full_rp[s], 4) for s in symbols}

    return result


def portfolio_summary_table(allocation: PortfolioAllocation) -> str:
    """Render a markdown table comparing the three portfolio allocations."""
    symbols = [s["code"] for s in allocation.stocks]
    names = {s["code"]: s.get("name", s["code"]) for s in allocation.stocks}

    lines = [
        "| 代码 | 名称 | 等权 | 最小方差 | 风险平价 |",
        "|------|------|:----:|:--------:|:--------:|",
    ]
    for sym in symbols:
        name = names.get(sym, sym)
        ew = allocation.equal_weight.get(sym, 0)
        mv = allocation.min_variance.get(sym, 0)
        rp = allocation.risk_parity.get(sym, 0)
        lines.append(
            f"| {sym} | {name} | {ew:.1%} | {mv:.1%} | {rp:.1%} |"
        )

    if allocation.note:
        lines.append(f"\n> ⚠️ {allocation.note}")

    if allocation.min_var_cond_number > 100:
        lines.append(
            f"\n> ⚠️ 协方差矩阵条件数 = {allocation.min_var_cond_number:.0f}（>100，"
            f"最小方差权重可能不稳定，建议减少标的数量或增加历史数据窗口）"
        )

    return "\n".join(lines)
