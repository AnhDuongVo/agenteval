"""Adapters normalise each framework and metrics return known values."""

from __future__ import annotations

import json
from pathlib import Path

from agenteval.adapters import load
from agenteval.metrics import evaluate, grounding_rate, loop_rate, tool_success_rate

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "sample_traces.jsonl"


def _runs():
    return [load(json.loads(line)) for line in SAMPLES.read_text().splitlines() if line.strip()]


def test_adapters_normalise_all_frameworks():
    runs = _runs()
    fws = {r.framework for r in runs}
    assert fws == {"openai", "langgraph", "llamaindex"}
    assert all(r.id for r in runs)


def test_tool_success_rate_counts_errors():
    runs = _runs()
    # one web_search tool result has error=True; the rest succeed
    assert tool_success_rate(runs) < 1.0


def test_grounding_detects_bad_citation():
    runs = _runs()
    # oa-2 cites lbl:xyz which is not retrieved
    assert grounding_rate(runs) < 1.0


def test_loop_rate_detects_repeat():
    runs = _runs()
    # lg-2 calls fhir_lookup twice with same (empty) input in a row
    assert loop_rate(runs) and loop_rate(runs) > 0.0


def test_evaluate_keys():
    rep = evaluate(_runs())
    for k in ["tool_success_rate", "tool_selection_accuracy", "grounding_rate", "task_success_rate", "mean_steps", "loop_rate"]:
        assert k in rep
