"""LLM analyst: generates per-stock Chinese analysis via Claude or DeepSeek API.

Supports both Anthropic Claude and DeepSeek (via Anthropic-compatible endpoint).
Set ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic for DeepSeek.
DeepSeek ThinkingBlock responses are automatically handled (non-text blocks skipped).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

ANALYSIS_PROMPT_TEMPLATE = """你是一位资深 A 股分析师。以下股票今日触发了技术形态和资金信号的双重验证。
请用 150-300 字解释它为什么值得关注。

## 股票
- 名称：{name}（{code}）
- 行业：{industry} | 市值：{market_cap}亿
- 今日涨跌幅：{pct_change}% | 收盘价：{close}

## 触发形态
{triggered_patterns}

## 形态技术细节（基于今日收盘数据）
{pattern_details}

## 资金信号
- 主力净流入：{main_inflow}万（流入率 {inflow_rate}%）
- 超大单净流入：{super_large}万
- 成交额：{turnover}万 | 量比：{vol_ratio}

## 要求
1. 参考「形态技术细节」中的具体数值，分析触发原因和可靠性（如：金叉是刚发生还是已运行多日？量能是否真实放大？）
2. 解读资金信号的强弱和方向
3. 结合行业背景给出综合判断
4. 提示后续观察点（压力位、确认信号、风险）
5. 用「值得关注」「可跟踪」等措辞，不说「推荐买入」"""


def build_prompt(stock: dict) -> str:
    """Build the LLM analysis prompt for a single stock."""
    triggered = stock.get("triggered_patterns", [])
    if isinstance(triggered, list):
        triggered = "、".join(triggered)

    details = stock.get("pattern_details", {})
    if isinstance(details, dict) and details:
        detail_lines = []
        for key, text in details.items():
            entry = None
            # Resolve key to label
            from core.pattern_detector import DETECTOR_REGISTRY
            entry = DETECTOR_REGISTRY.get(key, {})
            label = entry.get("label", key) if entry else key
            detail_lines.append(f"- {label}: {text}")
        pattern_details = "\n".join(detail_lines)
    else:
        pattern_details = "（无详细数据）"

    return ANALYSIS_PROMPT_TEMPLATE.format(
        name=stock.get("name", "?"),
        code=stock.get("code", "?"),
        industry=stock.get("industry", "未知"),
        market_cap=stock.get("market_cap", "?"),
        pct_change=stock.get("pct_change", 0),
        close=stock.get("close", "?"),
        triggered_patterns=triggered,
        pattern_details=pattern_details,
        main_inflow=stock.get("main_inflow", "?"),
        inflow_rate=stock.get("inflow_rate", "?"),
        super_large=stock.get("super_large_inflow", "?"),
        turnover=stock.get("turnover", "?"),
        vol_ratio=stock.get("vol_ratio", "?"),
    )


def _generate_fallback(stock: dict) -> str:
    """Generate a fallback analysis when LLM is unavailable."""
    patterns = stock.get("triggered_patterns", "多个形态")
    if isinstance(patterns, list):
        patterns = "、".join(patterns)

    # Extract key metrics from pattern_details
    details = stock.get("pattern_details", {})
    metrics = []
    if details.get("volume_breakout"):
        # Extract vol ratio
        import re
        m = re.search(r'量比([\d.]+)倍', details["volume_breakout"])
        if m:
            metrics.append(f"量比{m.group(1)}倍")
    if details.get("ma_golden_cross"):
        m = re.search(r'MA5\(([\d.]+)\)', details["ma_golden_cross"])
        if m:
            metrics.append(f"MA5={m.group(1)}")
    if details.get("rsi_oversold"):
        m = re.search(r'RSI从昨日(\d+).*反弹至今日(\d+)', details["rsi_oversold"])
        if m:
            metrics.append(f"RSI从{m.group(1)}回升至{m.group(2)}")
    metrics_str = "，".join(metrics) if metrics else ""

    return (
        f"{stock.get('name', '?')}今日触发{patterns}。"
        f"{metrics_str + '。' if metrics_str else ''}"
        f"主力资金净流入{stock.get('main_inflow', '?')}万（流入率{stock.get('inflow_rate', '?')}%），"
        f"超大单净流入{stock.get('super_large_inflow', '?')}万。"
        f"技术面与资金面形成共振，可跟踪后续确认信号。"
    )


class LLMAnalyst:
    """Generate per-stock analysis using Claude or DeepSeek API.

    Set ANTHROPIC_BASE_URL to use DeepSeek or other Anthropic-compatible
    endpoints. ANTHROPIC_MODEL selects the model (default: claude-sonnet-4-6).
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        # Model: env ANTHROPIC_MODEL > config > default
        self._model = (
            os.getenv("ANTHROPIC_MODEL")
            or self._config.get("model", "claude-sonnet-4-6")
        )
        self._max_tokens = int(self._config.get("max_tokens", 1536))
        self._base_url = os.getenv("ANTHROPIC_BASE_URL") or None
        self._client = None
        self._api_key: Optional[str] = None

    @property
    def client(self):
        """Lazy-init the Anthropic client.

        API key resolution (in order):
        1. ANTHROPIC_API_KEY env var
        2. ANTHROPIC_AUTH_TOKEN env var (used by DeepSeek deployments)
        3. anthropic_api_key in config.json
        """
        if self._client is None:
            import anthropic

            # Resolve API key: multiple env vars, then config
            self._api_key = (
                os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("ANTHROPIC_AUTH_TOKEN")
                or self._config.get("anthropic_api_key", "")
            )

            if not self._api_key:
                raise RuntimeError(
                    "API key not set. Set one of:\n"
                    "  - ANTHROPIC_API_KEY\n"
                    "  - ANTHROPIC_AUTH_TOKEN\n"
                    "  - anthropic_api_key in config.json"
                )

            # Create client with optional custom base_url (for DeepSeek etc.)
            client_kwargs = {"api_key": self._api_key}
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            self._client = anthropic.Anthropic(**client_kwargs)
        return self._client

    def analyze(self, stock: dict) -> str:
        """Generate analysis for a single stock via Claude API.

        Falls back to a simple template if the API is unavailable.
        """
        prompt = build_prompt(stock)

        try:
            client = self.client
        except RuntimeError as e:
            logger.warning("Claude API not available: %s — using fallback", e)
            return _generate_fallback(stock)

        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract text from response (skip ThinkingBlock from DeepSeek)
            text_parts: list[str] = []
            for block in resp.content:
                if hasattr(block, "text") and block.text:
                    text_parts.append(block.text)
            text = "".join(text_parts).strip()
            if not text:
                logger.warning("LLM returned empty text for %s, using fallback", stock.get("code", "?"))
                return _generate_fallback(stock)
            return text

        except Exception as e:
            logger.warning(
                "Claude API call failed for %s: %s — using fallback",
                stock.get("code", "?"), e,
            )
            return _generate_fallback(stock)

    def analyze_batch(
        self, stocks: list[dict], delay: float = 0.5, max_workers: int = 3,
    ) -> list[dict]:
        """Analyze a batch of stocks concurrently.

        Args:
            stocks: List of stock dicts to analyze.
            delay: Seconds between API calls within a worker to respect rate limits.
            max_workers: Max concurrent LLM requests (default 3).

        Returns:
            The same list with llm_analysis field populated.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _analyze_with_delay(idx_stock: tuple[int, dict]) -> tuple[int, str]:
            idx, stock = idx_stock
            # Stagger start within the pool to reduce burst pressure
            time.sleep(idx % max_workers * delay)
            result = self.analyze(stock)
            return idx, result

        indexed = list(enumerate(stocks))
        workers = min(max_workers, len(stocks)) if stocks else 1
        results_map: dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_analyze_with_delay, item): item[0]
                for item in indexed
            }
            for future in as_completed(futures):
                try:
                    idx, analysis = future.result()
                    results_map[idx] = analysis
                except Exception:
                    idx = futures[future]
                    logger.exception("LLM analysis failed for %s", stocks[idx].get("code", "?"))
                    results_map[idx] = _generate_fallback(stocks[idx])

        for i, stock in enumerate(stocks):
            analysis = results_map.get(i)
            if analysis is None:
                analysis = _generate_fallback(stock)
                stock["llm_analysis"] = analysis
                stock["llm_source"] = "fallback"
            else:
                stock["llm_analysis"] = analysis
                stock["llm_source"] = "real"

        return stocks
