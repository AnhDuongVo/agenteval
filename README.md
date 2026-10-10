# agenteval

## What this project demonstrates

Demonstrates selected trace normalization and explicit evaluation metrics for tool use, task outcomes, citations and numerical consistency.

## Watch the demo

![Demo](docs/demo.gif)

[Portfolio videos](https://anhduongvo.github.io/projects/agentic-tooling/). Clinical recordings use the separate simplified interactive demo.

## Try it offline

Python 3.11–3.13. In a fresh virtual environment, from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
agenteval behavior data/sample_traces.jsonl
pytest -q
```

## Run with NVIDIA or another configured backend

No NVIDIA endpoint is required. Capture supported trace shapes from your own configured agent; the adapters do not support every framework/version automatically. Calibration reports evaluate supplied labels and do not calibrate the underlying model.

Model IDs in `.env.example` and NAT configs are examples, not a current availability guarantee. Check your endpoint before running; the validation below does not include live model execution.

## What is verified

One logical OpenAI-style tool request/response pair, exact tool-set agreement, call precision, expected-set coverage, citation integrity and numerical screening. Legacy `grounding_rate` fields mean citation integrity, not semantic support.

| Validation layer | Status |
|---|---|
| Unit/regression tests | Executed locally on Python 3.12; see `docs/validation.md` |
| Mocked/simulated integrations | Executed locally; scope documented in tests |
| Live hosted endpoints | Not executed; access and appropriate inputs required |
| Self-hosted GPU endpoints | Not executed |
| Domain-specific validation | Not completed; synthetic examples only |

See [validation details](docs/validation.md). The architecture and detailed workflows follow.

## Architecture and detailed workflows

**Evaluate LLM agents at two levels.** Level 1 scores how an agent behaves (tool calls, loops, task success) from traces of LangGraph, LlamaIndex or OpenAI tool-calling runs. Level 2 scores what it says (citations that exist, numbers that match the source, confidence calibration metrics). Most evaluation tooling covers one of the two, for one framework.

| Level | Question | Input | Command |
|---|---|---|---|
| **Behavior** | Did the agent act well? | Traces from LangGraph, LlamaIndex or OpenAI tool-calling | `agenteval behavior` |
| **Claims** | Can what it said be trusted? | The agent's claims with citations, numbers and confidence | `agenteval claims` |

## Demo

![agenteval demo](docs/demo.gif)

Both levels on the bundled samples: the behavior scorecard per framework, then the claims leaderboard, which picks up a planted wrong number and a planted citation that does not exist. The video is on [anhduongvo.github.io](https://anhduongvo.github.io/projects/agentic-tooling/).

## Level 1: behavior (framework-agnostic)

One run schema (`AgentRun`), adapters that normalise each framework's trace into it, and behavioral metrics:

| Metric | Question | Better |
|---|---|---|
| Tool success rate | Do tool calls run without error? | higher |
| Tool selection accuracy | Does the used tool set exactly match the expected set? | higher |
| Expected-tool recall (coverage) | Were all expected tools called, regardless of extras? | higher |
| Tool precision | What fraction of calls used an expected tool? | higher |
| Citation integrity (`grounding_rate` legacy alias) | Is every cited source one that was retrieved? | higher |
| Task success rate | Did the run accomplish the task (gold label)? | higher |
| Mean steps | How many steps per run? | context |
| Loop rate | Does it repeat the same tool call back to back? | lower |

Adapters take the plain trace dict each framework emits, so agenteval does not depend on those frameworks being installed: `from_openai` (chat messages with `tool_calls` and `role: "tool"` results), `from_langgraph` (node `events`), `from_llamaindex` (agent `steps`). Shapes are documented in `adapters.py`, with a runnable example per framework in `data/sample_traces.jsonl`.

## Level 2: claims (citation integrity, numbers, calibration)

For clinical use the output has to be checkable claim by claim. The `agenteval.clinical` layer scores each claim an agent makes against the sources it had:

| Metric | Question | Better |
|---|---|---|
| Citation integrity (`grounding_rate` legacy alias) | Does every claim cite at least one source, with all cited IDs present? | higher |
| Hallucinated-citation rate | How many cited ids do not exist? | lower |
| Number accuracy | Do asserted numbers match the cited source value (rounding tolerated, sign flips not)? | higher |
| Calibration (ECE, Brier) | Does a stated confidence match how often it is right? | lower |

One small schema (`Record` with `sources` and `Claim`s carrying `citations`, `numbers`, `confidence`, `label`) can represent claims from selected agent workflows: transcript lines, FHIR facts, table rows or literature ids are the sources; note sentences, eligibility verdicts, drafted claims or ranked statements are the claims. Bundled samples for four clinical agents are in `data/clinical_samples/`, and `agenteval claims` renders a per-task leaderboard (markdown or HTML).

## Quickstart

```bash
pip install -e ".[dev]"

agenteval behavior data/sample_traces.jsonl                        # per-framework behavior scorecard
agenteval claims data/clinical_samples --md leaderboard.md         # per-task claims leaderboard
```

## Development

```bash
ruff check .
pytest            # behavior adapters + metrics, and claim-level metrics with known-value tests
```

## License

Apache-2.0.

## Developer workshop

[40-minute workshop](docs/workshop.md): tested installation, a trace-mutation exercise, automated checks and troubleshooting.
