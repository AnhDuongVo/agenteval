"""Command line. Two levels of evaluation:

agenteval behavior <traces.jsonl ...>      how the agent behaves: tool success, selection, grounding, loops
agenteval claims   <records.jsonl|dir ...> whether what it says holds: grounding, numbers, calibration
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .adapters import load
from .clinical.harness import evaluate_by_task, load_records
from .clinical.leaderboard import to_html, to_markdown
from .metrics import evaluate

app = typer.Typer(
    add_completion=False,
    help="Evaluate LLM agents at two levels: behavior (traces) and claims (citation integrity, numbers, calibration).",
)
console = Console()


def _expand(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        path = Path(p)
        out.extend(str(f) for f in sorted(path.glob("*.jsonl"))) if path.is_dir() else out.append(p)
    return out


def _pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _dec(v):
    return "n/a" if v is None else f"{v:.3f}"


@app.command()
def behavior(
    files: list[str] = typer.Argument(..., help="JSONL traces, one per line (openai | langgraph | llamaindex)"),
    out: str = typer.Option("", help="write JSON report"),
):
    """Score agent traces: tool success, tool selection, grounding, task success, mean steps, loop rate."""
    runs = [load(json.loads(line)) for p in _expand(files) for line in Path(p).read_text().splitlines() if line.strip()]
    by_fw = defaultdict(list)
    for r in runs:
        by_fw[r.framework].append(r)
    report = {"overall": evaluate(runs), "by_framework": {fw: evaluate(rs) for fw, rs in sorted(by_fw.items())}}
    if out:
        Path(out).write_text(json.dumps(report, indent=2))
    table = Table(title="Agent behavior")
    for c in [
        "Framework",
        "Runs",
        "Tool success",
        "Tool selection",
        "Citation integrity",
        "Task success",
        "Mean steps",
        "Loop rate",
    ]:
        table.add_column(c)
    rows = [{"fw": fw, **s} for fw, s in report["by_framework"].items()] + [{"fw": "overall", **report["overall"]}]
    for r in rows:
        ms = r.get("mean_steps")
        table.add_row(
            r["fw"],
            str(r.get("n_runs", "")),
            _pct(r.get("tool_success_rate")),
            _pct(r.get("tool_selection_accuracy")),
            _pct(r.get("grounding_rate")),
            _pct(r.get("task_success_rate")),
            "n/a" if ms is None else f"{ms:.1f}",
            _pct(r.get("loop_rate")),
        )
    console.print(table)


@app.command()
def claims(
    files: list[str] = typer.Argument(
        ..., help="JSONL records (claims with citations, numbers, confidence) or directories"
    ),
    md: str = typer.Option("", help="write a markdown leaderboard"),
    html: str = typer.Option("", help="write an HTML leaderboard"),
    out: str = typer.Option("", help="write JSON report"),
    rel_tol: float = typer.Option(0.01, help="relative tolerance for number matching"),
    bins: int = typer.Option(10, help="bins for calibration ECE"),
):
    """Score claims: grounding rate, hallucinated-citation rate, number accuracy, calibration (ECE, Brier)."""
    data = evaluate_by_task(load_records(_expand(files)), rel_tol=rel_tol, bins=bins)
    if out:
        Path(out).write_text(json.dumps(data, indent=2))
    if md:
        Path(md).write_text(to_markdown(data))
    if html:
        Path(html).write_text(to_html(data))
    table = Table(title="Agent claims")
    for c in ["Task", "Items", "Claims", "Citations valid", "Hallucinated cites", "Number acc.", "ECE", "Brier"]:
        table.add_column(c)
    rows = [{"task": t, **s} for t, s in data["by_task"].items()] + [{"task": "overall", **data["overall"]}]
    for r in rows:
        table.add_row(
            r["task"],
            str(r.get("n_records", "")),
            str(r.get("n_claims", "")),
            _pct(r.get("grounding_rate")),
            _pct(r.get("hallucinated_citation_rate")),
            _pct(r.get("number_accuracy")),
            _dec(r.get("calibration_ece")),
            _dec(r.get("calibration_brier")),
        )
    console.print(table)


if __name__ == "__main__":
    app()
