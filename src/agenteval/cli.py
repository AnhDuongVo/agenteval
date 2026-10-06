"""Command line: score agent traces and print a leaderboard across frameworks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .adapters import load
from .metrics import evaluate

app = typer.Typer(add_completion=False, help="Framework-agnostic evaluation for LLM agents.")
console = Console()


def _load_runs(paths: list[str]):
    runs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            line = line.strip()
            if line:
                runs.append(load(json.loads(line)))
    return runs


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


@app.command()
def score(files: list[str] = typer.Argument(...), out: str = typer.Option("", help="write JSON report")):
    """Score traces (JSONL, one trace per line) overall and by framework."""
    runs = _load_runs(files)
    by_fw = defaultdict(list)
    for r in runs:
        by_fw[r.framework].append(r)
    report = {"overall": evaluate(runs), "by_framework": {fw: evaluate(rs) for fw, rs in sorted(by_fw.items())}}
    if out:
        Path(out).write_text(json.dumps(report, indent=2))
    table = Table(title="Agent evaluation")
    for c in ["Framework", "Runs", "Tool success", "Tool selection", "Grounding", "Task success", "Mean steps", "Loop rate"]:
        table.add_column(c)
    rows = [{"fw": fw, **s} for fw, s in report["by_framework"].items()] + [{"fw": "overall", **report["overall"]}]
    for r in rows:
        ms = r.get("mean_steps")
        table.add_row(r["fw"], str(r.get("n_runs", "")), _pct(r.get("tool_success_rate")),
                      _pct(r.get("tool_selection_accuracy")), _pct(r.get("grounding_rate")),
                      _pct(r.get("task_success_rate")), "n/a" if ms is None else f"{ms:.1f}", _pct(r.get("loop_rate")))
    console.print(table)


if __name__ == "__main__":
    app()
