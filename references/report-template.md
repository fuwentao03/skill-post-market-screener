# 收盘扫描日报 — {{trade_date}}

> 扫描时间：{{scan_time}} | 全市场 {{total_stocks}} 只 | 入选 {{selected_count}} 只

## 扫描配置

| 参数 | 值 |
|---|---|
| 最小市值 | {{min_market_cap}}亿 |
| 最小成交额 | {{min_turnover}}万 |
| 形态检测器 | {{active_detectors}} |
| 主力净流入率阈值 | {{main_inflow_rate_min}}% |

## 今日精选

{{#each stocks}}
### {{rank}}. {{name}}（{{code}}） 得分 {{score}} | 收盘 {{close}} | {{pct_change}}%

> {{llm_analysis}}

#### 触发形态：{{triggered_patterns}}
#### 资金信号：主力净流入 {{main_inflow}}万 | 流入率 {{inflow_rate}}% | 超大单 {{super_large}}万

---
{{/each}}

## 形态触发统计

| 形态 | 触发次数 | 占比 |
|---|---|---|
{{#each pattern_stats}}
| {{pattern_name}} | {{trigger_count}} | {{trigger_pct}}% |
{{/each}}

## 行业分布

| 行业 | 入选数量 |
|---|---|
{{#each industry_distribution}}
| {{industry}} | {{count}} |
{{/each}}

## 数据说明

- 使用接口：{{api_list}}
- 数据截止时间：{{data_cutoff}}
- 缺失或降级数据：{{missing_data_note}}
- 评分公式：pattern_score + flow_score + quality_bonus（详见 JSON 输出）
- 统计口径：{{calculation_notes}}

---

## LLM 逐股分析 Prompt

For each selected stock, send the following prompt to the LLM:

```markdown
你是一位资深 A 股分析师。以下股票今日触发了技术形态和资金信号的双重验证。
请用 150-300 字解释它为什么值得关注。

## 股票
- 名称：{name}（{code}）
- 行业：{industry} | 市值：{market_cap}亿
- 今日：{pct_change}% | 收盘：{close}

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
5. 用「值得关注」「可跟踪」等措辞，不说「推荐买入」
```
