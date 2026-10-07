"""Load records from JSONL and evaluate them, overall and grouped by task."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .metrics import evaluate
from .schemas import Record


def load_records(paths: list[str | Path]) -> list[Record]:
    records: list[Record] = []
    for path in paths:
        text = Path(path).read_text()
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(Record.model_validate_json(line))
    return records


def evaluate_by_task(records: list[Record], rel_tol: float = 0.01, bins: int = 10) -> dict:
    by_task: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_task[r.task].append(r)
    result = {"overall": evaluate(records, rel_tol=rel_tol, bins=bins), "by_task": {}}
    for task, recs in sorted(by_task.items()):
        result["by_task"][task] = evaluate(recs, rel_tol=rel_tol, bins=bins)
    return result


def run(paths: list[str | Path], out: str | Path | None = None, rel_tol: float = 0.01, bins: int = 10) -> dict:
    report = evaluate_by_task(load_records(paths), rel_tol=rel_tol, bins=bins)
    if out:
        Path(out).write_text(json.dumps(report, indent=2))
    return report
