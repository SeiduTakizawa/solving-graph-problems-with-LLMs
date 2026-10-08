Claude Code (headless, `claude -p`, on a subscription) answering one dataset question with only our graph tools via MCP.

    claude -p "<system line> What is a shortest path from node 11 to node 21? ..." \
      --model sonnet --tools "" --mcp-config mcp.json --strict-mcp-config --allowedTools mcp__graph \
      --output-format stream-json --verbose < /dev/null

- `--tools ""`: no built-in tools (no Bash/Read, so it can't open the graph file); `--strict-mcp-config`: only our server.
- Run outside the repo so no CLAUDE.md is loaded. `--model sonnet` resolved to `claude-sonnet-5`; pin the full ID next time.
- `stream.jsonl`: the full event stream (init, tool calls, results, usage). `mcp.json`: the server config (paths shortened).
