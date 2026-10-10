# Workshop: evaluate an agent without overclaiming reliability

**Duration:** 40 minutes. **Prerequisites:** Python 3.11–3.13, a terminal, and a clone of this repository. Installation downloads Python packages; the exercise requires no API key, GPU or model endpoint.

## 0–8 minutes: install and establish a baseline

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
agenteval behavior data/sample_traces.jsonl
```

Inspect one OpenAI-style trace. Identify its tool request, response, call ID, expected tools, retrieved IDs and cited IDs.

## 8–18 minutes: reconcile calls and interpret the scorecard

```bash
python examples/workshop.py
```

The example has two logical calls, complete expected-tool coverage, precision 0.5, exact selection accuracy 0, citation integrity 0 and no repeated-call loop. Explain why two tool requests and two responses are two calls rather than four. Identify which score penalizes the unnecessary tool.

## 18–30 minutes: learner exercise

Copy `examples/workshop.py` to a scratch file. Remove the unnecessary call and response; change the citation to `table:1`. Update its assertions to expect one call, precision 1, exact selection accuracy 1 and citation integrity 1. Run it. Then keep the valid citation but change the answer to an unsupported sentence. Explain why reference validity cannot detect that semantic error. Propose one expert-labelled test that would.

## 30–40 minutes: verify and discuss limits

Run `pytest -q tests/test_regressions.py` and the original workshop script. Explain what happens to tool-success rate when a response is absent, when the response arrives out of order, and when the API returns an error. Discuss why calibration metrics need representative labels and cannot be generalized from five synthetic traces.

## Troubleshooting

- `command not found`: activate `.venv` and repeat editable installation.
- Import error: run from the repository root using the virtual environment's Python.
- Failed learner assertions: compare the expected metric with the trace mutation; do not delete assertions.
- Framework trace mismatch: normalize it to the documented shapes in `adapters.py`; arbitrary framework logs are not supported.

The regression tests and script are the automated checks. This workshop teaches software evaluation and its limits; it does not establish clinical correctness.
