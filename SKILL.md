---
name: post-market-screener
description: Daily A-share end-of-day quantitative screener that finds stocks with both
  technical-pattern breakouts and institutional capital inflows. Covers 8 pattern detectors
  (MA/MACD cross, bullish alignment, volume breakout, Bollinger breakout, hammer, morning
  star, RSI oversold), main-capital flow filtering (Tonghuashun primary, East Money fallback),
  dual-factor cross-validation, z-score industry neutralization, and per-stock LLM analysis
  (Claude/DeepSeek). Use when the user asks for 收盘扫描, 选股扫描, 形态选股,
  post-market screening, 资金+形态筛选, or to set up automated daily stock screening.
license: MIT
metadata:
  organization: QuantSkills
  organization_url: https://github.com/quantskills
  repository: skill-post-market-screener
  project_type: skill
  collection: post-market-screener
  creator: Tao
quantSkills:
  project_type: skill
  category: analysis
  tags:
  - a-share
  - technical-patterns
  - capital-flow
  - screening
  - pandadata
  - tonghuashun
  platforms:
  - claude-code
  - codex
  - hermes
  - openclaw
  - cursor
  status: dev
  validation_level: runnable
  maintainer_type: community
  summary_zh: 每日收盘后自动扫描全市场，8 个技术形态检测器（连续强度信号）+ 同花顺主力资金流入做双因子交叉验证，Z-score 行业中性化排名，Top N 个股 DeepSeek LLM 分析，输出 Markdown 日报 + JSON 数据。资金流向 3 路径容灾：同花顺→东方财富 AKShare→东方财富直连。
  summary_en: Daily A-share post-market screener: 8 technical pattern detectors (continuous
    float strength) x Tonghuashun capital inflow filter = dual-factor cross-validated
    stock picks with DeepSeek LLM analysis and z-score industry neutralization.
    Fund flow uses 3-path fallback: Tonghuashun → East Money (AKShare) → East Money direct.
  license: MIT
  requires:
  - skill-pandadata-api
---

# Post-Market Screener

After each A-share trading day, scan the entire market with dual-factor (technical pattern + capital flow) cross-validation, then generate a ranked daily screening report with per-stock LLM analysis.

## Workflow

1. **Determine the target date.** If the user does not provide one, use the latest completed A-share trading day. Check `get_last_trade_date` and `get_trade_cal`; if the target date is closed, return a short "今日休市" note instead of running the scan.
2. **Load `pandadata-api`** before making real API calls. Use its method index or search script to confirm parameters and fields; do not guess Pandadata signatures.
3. **Collect data** in this order:
   - Trading calendar and stock universe: `get_last_trade_date`, `get_trade_cal`, `get_trade_list`.
   - Daily K-line (120 days for pattern calculation window): `get_stock_daily`.
   - Fund flow per stock: **同花顺 (Tonghuashun/10jqka)** `stock_fund_flow_individual` as primary source. Falls back to AKShare `stock_individual_fund_flow_rank` (东方财富), then direct East Money push2 API on failure. The flow source is tracked via `df.attrs["flow_source"]` and persisted in cache metadata.
   - Stock basic info (name, industry): `get_stock_detail`; market cap computed from `get_share_float` × close price.
4. **Run 8 pattern detectors** against each stock's K-line data (formulas in `references/pattern-formulas.md`). Each detector returns a continuous float strength value (> 0 = triggered, ≤ 0 = not triggered). The strength magnitude is used for backtest IC calculation; scoring uses binary weights.
   - MA Golden Cross: MA5 crosses above MA20
   - MACD Golden Cross: DIF crosses above DEA
   - Bullish Alignment: MA5 > MA10 > MA20 > MA60
   - Volume Breakout: close at 20-day high + volume > 1.5x 5-day avg volume
   - Bollinger Breakout: bandwidth expansion + price breaks upper band
   - Hammer: lower shadow >= 2x body + in downtrend
   - Morning Star: bear candle → small body → bull candle (3-day reversal)
   - RSI Oversold Rebound: RSI(14) < 30 + today closes bullish
5. **Filter by capital flow:**
   - Main capital net inflow rate > 5%
   - Turnover > 50M CNY (exclude illiquid stocks)
   - Super-large order net inflow > 0
   Note: Tonghuashun does not provide order-size breakdown (super_large/large/medium/small). Super-large and large net inflows are proxied as `main_net_inflow / 2` each.
6. **Cross-validate and rank.** A stock must pass at least one pattern AND the flow filter. Compute `stock_score = neutralized_pattern_score + flow_score + quality_bonus`, then take the Top N.
7. **Generate per-stock LLM analysis** using `references/report-template.md` for the analysis prompt. Each stock gets detailed Chinese analysis (technical breakdown + fund verification + sector context + risk notes).
8. **Render the report** with data provenance table showing the actual source for each data category. Save to `output/YYYY-MM-DD/daily_screener_YYYYMMDD.md` and `output/YYYY-MM-DD/daily_screener_YYYYMMDD.json` unless the user gives another path.
9. **Run `scripts/validate_screener.py <md-path> <json-path>`** after writing the output. Production runs use `--strict` mode. Fix missing sections, missing data-source labels, or missing pattern descriptions before presenting the result.

## Pandadata Reference

Read `references/pandadata-map.md` when planning calls, selecting fields, or deciding how to degrade if a data interface is unavailable. The map is a routing aid only; the exact call contract must still come from `pandadata-api`.

## Scoring Formula

```
pattern_score = sum(triggered_pattern_weights)   # see pattern-formulas.md for weights
flow_score    = min(main_inflow_rate_decimal / 0.05, 3)  # 1 point per 5% inflow rate, capped at 3
quality_bonus:
  +1  if market_cap >= 50B CNY
  +1  if turnover >= 200M CNY

Industry neutralization (z-score within each industry):
  z = (raw_pattern_score − μ_industry) / σ_industry
  neutralized = max(0, z)   # floor at 0

Final score = neutralized_pattern_score + flow_score + quality_bonus
```

## Report Rules

- Write in Chinese unless the user requests another language.
- Use absolute dates such as `2026-06-30`; avoid ambiguous "today" in the final report body.
- Each selected stock must show: triggered patterns, capital flow data, and the LLM-generated analysis.
- Use cautious language: "值得关注", "可跟踪", "信号偏多"; never say "推荐买入", "目标价", "必涨".
- When a data call fails, keep the report useful by skipping that stock (not the whole scan) and adding a missing-data count under "数据说明".
- Include scoring breakdown in the JSON output for auditability.
- Always include a **数据来源 (data provenance)** table showing the actual source and authenticity status for each data category.

## Automation

When the user asks for automated daily screening, create an after-close task for trading days only, preferably after `15:30 Asia/Shanghai` (30 min after A-share close). A `scripts/daily_screener.bat` entry point for Windows Task Scheduler is provided. Make the task idempotent: if `output/YYYY-MM-DD/daily_screener_YYYYMMDD.md` already exists, regenerate and overwrite it.

## Directory Structure

```
skill-post-market-screener/
├── SKILL.md                         # Agent workflow entry point
├── README.md                        # Human documentation
├── OPERATION_MANUAL.md              # Detailed operation manual
├── LICENSE                          # GPLv3
├── config.json                      # Runtime configuration
├── pyproject.toml                   # Python project metadata + linting config
├── .env.example                     # Environment variable template
├── requirements.txt                 # Python dependencies
├── run.py                           # CLI entry point (real data only, no --mock/--dry-run)
├── mcp_server.py                    # MCP server
├── core/
│   ├── data_fetcher.py              # Pandadata data acquisition (with retry + backoff)
│   ├── flow_fetcher.py              # Fund flow: 同花顺 → AKShare → East Money direct (3-path)
│   ├── pattern_detector.py          # 8 technical pattern detectors (float strength)
│   ├── flow_filter.py               # Capital flow filter + inflow rate normalization
│   ├── scorer.py                    # Scoring and ranking (z-score industry neutralization)
│   ├── reporter.py                  # Markdown + JSON report generator (data provenance)
│   ├── pipeline.py                  # End-to-end pipeline orchestrator
│   ├── cache.py                     # Date-keyed Parquet cache (with meta.json)
│   └── mock_data.py                 # Mock data generator (internal testing only)
├── llm/
│   └── analyst.py                   # LLM analysis (Claude/DeepSeek, concurrent API calls)
├── tests/
│   ├── test_pattern_detector.py     # 45 pattern detector tests
│   ├── test_integration.py          # 32 pipeline/integration tests
│   ├── test_analyst.py              # 24 LLM analyst tests
│   ├── test_api_integration.py      # 25 mocked API integration tests (3-path fallback)
│   └── test_reporter.py             # 28 report generation tests
├── references/
│   ├── pandadata-map.md             # Data routing reference
│   ├── pattern-formulas.md          # Detector formulas and weights
│   └── report-template.md           # Report template + LLM prompt
├── scripts/
│   ├── validate_screener.py         # Output integrity validator
│   ├── daily_screener.bat           # Windows Task Scheduler entry point
│   ├── benchmark.py                 # Performance benchmark
│   └── analyze_weights.py           # IC analysis and weight optimization
├── agents/
│   ├── openai.yaml                  # OpenAI/Codex adapter
│   ├── cursor-rule.mdc              # Cursor IDE adapter
│   └── portable-loader.md           # Generic loader for any agent
└── output/                          # Reports grouped by date
    └── YYYY-MM-DD/
        ├── daily_screener_YYYYMMDD.md
        └── daily_screener_YYYYMMDD.json
```

## Core Constraints

| Constraint | Description |
|---|---|
| Dual-factor must hold | Stock passes only if pattern AND flow both trigger |
| Liquidity threshold | Turnover < 50M CNY → auto-exclude |
| No stock recommendation | Use "值得关注" / "可跟踪", forbid "买入" / "目标价" |
| Trading-day aware | Skip holidays, check trade calendar |
| Data fault tolerance | Skip individual failing stocks, do not halt the full scan |
| Audit trail | JSON output must include scoring breakdown per stock (raw + neutralized) |
| Data provenance | Report must label each data category's actual source and authenticity |

## Disclaimer

This skill's output is for research reference only. It does not constitute any investment advice. Investors should make independent judgments and bear trading risks.

## License

GPLv3
