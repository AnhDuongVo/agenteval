# agenteval

**Framework-agnostic evaluation for LLM agents.** One run schema, adapters for the common agent frameworks (LangGraph, LlamaIndex, OpenAI tool-calling), and one set of behavioral metrics. Point it at traces from any of them and get a comparable scorecard.

## Why

Every agent team needs to answer the same questions, whatever framework they built on: does the agent pick the right tools, do the tool calls succeed, is the final answer grounded in what it retrieved, does it get stuck in loops, and how many steps does it take? Most eval tooling is tied to one framework. `agenteval` normalises traces into one schema first, so the metrics (and the leaderboard) work across all of them.

## Metrics

| Metric | Question | Better |
|---|---|---|
| Tool success rate | Do tool calls run without error? | higher |
| Tool selection accuracy | Were the expected tools actually used? | higher |
| Grounding rate | Is every cited source one that was retrieved? | higher |
| Task success rate | Did the run accomplish the task (gold label)? | higher |
| Mean steps | How many steps per run? | context |
| Loop rate | Does it repeat the same tool call back to back? | lower |

## One schema, three adapters

Each adapter takes the plain trace dict a framework emits and returns a common `AgentRun`, so agenteval does not depend on those frameworks being installed:

- `from_openai` — OpenAI chat messages with `tool_calls` and `role: "tool"` results.
- `from_langgraph` — a list of node `events` (`tool` / `retriever` / `llm`).
- `from_llamaindex` — agent `steps` (`tool`, `is_error`, `observation`).

The expected shapes are documented in `adapters.py`, with a runnable example per framework in `data/sample_traces.jsonl`.

## Quickstart

```bash
pip install -e ".[dev]"
agenteval data/sample_traces.jsonl        # prints a leaderboard, overall and per framework
```

## Development

```bash
ruff check .
pytest
```

## License

Apache-2.0.
