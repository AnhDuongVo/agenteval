"""Behavioral metrics over AgentRuns. Each returns None when it has no applicable denominator."""

from __future__ import annotations

from .schema import AgentRun


def tool_success_rate(runs: list[AgentRun]) -> float | None:
    total = ok = 0
    for r in runs:
        for s in r.tool_calls():
            total += 1
            ok += int(s.ok)
    return ok / total if total else None


def expected_tool_recall(runs: list[AgentRun]) -> float | None:
    """Legacy expected-set coverage: extra tools do not reduce this score."""
    scored = [r for r in runs if r.expected_tools is not None]
    if not scored:
        return None
    good = 0
    for r in scored:
        used = {s.name for s in r.tool_calls()}
        good += int(set(r.expected_tools).issubset(used))
    return good / len(scored)


def citation_integrity_rate(runs: list[AgentRun]) -> float | None:
    """Of runs that cite anything, the fraction whose every citation is in the retrieved set."""
    scored = [r for r in runs if r.citations]
    if not scored:
        return None
    good = sum(int(set(r.citations).issubset(set(r.retrieved))) for r in scored)
    return good / len(scored)


def tool_selection_accuracy(runs: list[AgentRun]) -> float | None:
    """Exact expected/used set agreement, penalizing extra tools."""
    scored = [r for r in runs if r.expected_tools is not None]
    return (
        sum({s.name for s in r.tool_calls()} == set(r.expected_tools) for r in scored) / len(scored) if scored else None
    )


def tool_precision(runs: list[AgentRun]) -> float | None:
    scored = [r for r in runs if r.expected_tools is not None]
    total = sum(len(r.tool_calls()) for r in scored)
    good = sum(s.name in set(r.expected_tools) for r in scored for s in r.tool_calls())
    return good / total if total else None


# Compatibility alias: this checks reference existence, not semantic grounding.
grounding_rate = citation_integrity_rate


def task_success_rate(runs: list[AgentRun]) -> float | None:
    labeled = [r for r in runs if r.success is not None]
    return sum(int(r.success) for r in labeled) / len(labeled) if labeled else None


def mean_steps(runs: list[AgentRun]) -> float | None:
    return sum(len(r.steps) for r in runs) / len(runs) if runs else None


def loop_rate(runs: list[AgentRun]) -> float | None:
    """Fraction of runs that call the same tool with the same input twice in a row (a sign of a stuck loop)."""
    if not runs:
        return None
    looped = 0
    for r in runs:
        calls = [(s.name, s.input) for s in r.tool_calls()]
        looped += int(any(calls[i] == calls[i - 1] for i in range(1, len(calls))))
    return looped / len(runs)


def evaluate(runs: list[AgentRun]) -> dict:
    return {
        "n_runs": len(runs),
        "tool_success_rate": tool_success_rate(runs),
        "tool_selection_accuracy": tool_selection_accuracy(runs),
        "expected_tool_recall": expected_tool_recall(runs),
        "tool_precision": tool_precision(runs),
        "citation_integrity_rate": citation_integrity_rate(runs),
        "grounding_rate": grounding_rate(runs),
        "task_success_rate": task_success_rate(runs),
        "mean_steps": mean_steps(runs),
        "loop_rate": loop_rate(runs),
    }
