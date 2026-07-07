# validate_screener.py — 输出完整性校验规则

`scripts/validate_screener.py <md-path> <json-path>` 对每日报告进行自动化校验。生产运行使用 `--strict` 模式（所有规则均为 ERROR 级别）。

## 校验规则清单

### 结构完整性 (5 条)

| # | 规则 | 检查内容 | 级别 |
|---|------|----------|:----:|
| S1 | 报告头部 | 必须包含 `# 收盘扫描日报` 标题 + `扫描时间` + `全市场 X 只` + `入选 Y 只` 摘要行 | ERROR |
| S2 | 扫描配置节 | 必须包含「扫描配置」表格，含 `最小市值`、`最小成交额`、`形态检测器`、`主力净流入率阈值` 行 | ERROR |
| S3 | 今日精选节 | 必须包含「今日精选」节，入选股票数量 > 0（`--strict` 模式下为 0 时报 ERROR） | ERROR |
| S4 | 统计节 | 必须包含「形态触发统计」和「行业分布」两个表格 | WARNING |
| S5 | 数据说明节 | 必须包含「数据说明」节，含 `使用接口`、`缺失或降级数据` 字段 | ERROR |

### 数据溯源 (3 条)

| # | 规则 | 检查内容 | 级别 |
|---|------|----------|:----:|
| D1 | 每只入选股票含资金数据源标签 | `data_source` 字段存在且为 `tonghuashun` / `akshare_eastmoney` / `eastmoney_direct` 之一 | ERROR |
| D2 | K线数据源标注 | 报告「数据说明」节中必须标注 K 线来源（`pandadata` 或 `degraded`） | WARNING |
| D3 | 降级标注 | 任一数据源发生降级时，`missing_data_note` 必须非空并说明降级原因 | ERROR |

### 评分可审计性 (4 条)

| # | 规则 | 检查内容 | 级别 |
|---|------|----------|:----:|
| A1 | JSON 含分项评分 | 每只入选股票的 JSON 条目必须包含 `pattern_score`、`flow_score`、`quality_bonus`、`raw_pattern_score`、`industry_neutralized_score`、`final_score` 字段 | ERROR |
| A2 | 双因子约束 | 每只入选股票的 `pattern_score > 0` 且 `flow_score > 0`（双因子交叉验证必须成立） | ERROR |
| A3 | 行业字段 | 每只股票含 `industry` 字段，非空字符串 | WARNING |
| A4 | 评分一致性 | `final_score = industry_neutralized_score + flow_score + quality_bonus`（浮点误差 ±0.01） | ERROR |

### 内容合规 (3 条)

| # | 规则 | 检查内容 | 级别 |
|---|------|----------|:----:|
| C1 | 禁止投资建议措辞 | 全文不得包含「推荐买入」「推荐卖出」「目标价」「必涨」「必跌」「加仓」「减仓」 | ERROR |
| C2 | 允许的谨慎措辞 | 至少出现「值得关注」「可跟踪」「信号偏多」「信号偏空」之一（确保报告有实质性判断） | WARNING |
| C3 | LLM 分析覆盖 | 每只入选股票必须有 `llm_analysis` 字段且字数 ≥ 80 字（`--strict` 模式） | WARNING |

### 数据质量 (2 条)

| # | 规则 | 检查内容 | 级别 |
|---|------|----------|:----:|
| Q1 | 去重检查 | JSON 中 `code` 字段无重复 | ERROR |
| Q2 | 排序一致性 | JSON 中股票按 `final_score` 降序排列 | WARNING |

## 退出码

| 退出码 | 含义 |
|:---:|---|
| 0 | 全部通过 |
| 1 | WARNING 级别问题（`--strict` 模式下 WARNING 升级为 ERROR） |
| 2 | ERROR 级别问题 |
| 3 | 文件不存在或 JSON 解析失败 |

## 使用方式

```bash
# 开发模式（WARNING 不阻塞）
python scripts/validate_screener.py output/2026-07-02/daily_screener_20260702.md output/2026-07-02/daily_screener_20260702.json

# 生产模式（全部规则按 ERROR 处理）
python scripts/validate_screener.py output/2026-07-02/daily_screener_20260702.md output/2026-07-02/daily_screener_20260702.json --strict

# CI 模式（JSON 输出）
python scripts/validate_screener.py output/2026-07-02/daily_screener_20260702.md output/2026-07-02/daily_screener_20260702.json --strict --json
```
