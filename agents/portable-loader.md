# Portable Loader Prompt

Use this prompt in agents that do not natively discover `SKILL.md` folders.

```text
You have access to a local skill named post-market-screener at:
<POST_MARKET_SCREENER_SKILL_ROOT>

When the user request matches this skill's SKILL.md description:
1. Read <POST_MARKET_SCREENER_SKILL_ROOT>/SKILL.md.
2. Follow the workflow and guardrails in that file exactly.
3. Load referenced files under <POST_MARKET_SCREENER_SKILL_ROOT>/references/ only when needed.
4. Run bundled scripts from the skill root only after reading the relevant instructions.
5. Preserve documented API names, parameters, file paths, formulas, validation limits, and freshness notes.
6. Do not invent data interfaces, credentials, factor definitions, or runtime behavior that is not supported by the skill files.
```

## MCP Server Integration

The skill can also be called by LLMs via its MCP server. Start the server with:

```bash
cd <SKILL_ROOT>
python mcp_server.py
```

### Claude Code (claude.ai/code)

Add to `.claude/mcp.json` or `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "post-market-screener": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "<SKILL_ROOT>",
      "env": {
        "ANTHROPIC_AUTH_TOKEN": "sk-xxx",
        "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
        "ANTHROPIC_MODEL": "deepseek-v4-pro",
        "DEFAULT_USERNAME": "86xxxxxxxxxxx",
        "DEFAULT_PASSWORD": "xxx"
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
|---|---|
| `run_screener` | Run full daily dual-factor scan with optional params `date`, `no_flow`, `top_n` |
| `get_latest_report` | Read the most recent Markdown report |
| `check_trading_day` | Check if a date is an A-share trading day |

### Direct CLI

```bash
python run.py                    # Full scan, latest trading day
python run.py --date 20260629    # Specific date
python run.py --no-flow          # Pattern-only scan (skip fund flow)
python run.py --top-n 10         # Limit report to top 10
```

All data is from real sources (Pandadata + 同花顺 + LLM API). No mock/dry-run mode.
