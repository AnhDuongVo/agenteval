"""Render an evaluation report as a markdown or HTML leaderboard."""

from __future__ import annotations

_COLUMNS = [
    ("task", "Task"),
    ("n_records", "Items"),
    ("n_claims", "Claims"),
    ("grounding_rate", "Citations valid"),
    ("hallucinated_citation_rate", "Hallucinated cites"),
    ("number_accuracy", "Number accuracy"),
    ("calibration_ece", "ECE"),
    ("calibration_brier", "Brier"),
]
_PERCENT = {"grounding_rate", "hallucinated_citation_rate", "number_accuracy"}


def _fmt(key: str, value) -> str:
    if value is None:
        return "n/a"
    if key in {"task", "n_records", "n_claims"}:
        return str(value)
    if key in _PERCENT:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _rows(report: dict) -> list[dict]:
    rows = []
    for task, scores in report.get("by_task", {}).items():
        rows.append({"task": task, **scores})
    rows.append({"task": "overall", **report.get("overall", {})})
    return rows


def to_markdown(report: dict, title: str = "Clinical agent evaluation") -> str:
    header = "| " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [f"# {title}", "", header, sep]
    for row in _rows(report):
        cells = [_fmt(key, row.get(key)) for key, _ in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Citations valid: claims with at least one citation and every cited ID present in the source set; this is not semantic entailment. Hallucinated cites: cited ids that do "
        "not exist (lower is better). Number accuracy: asserted numbers matching the cited source. ECE/Brier: "
        "confidence calibration (lower is better).",
    ]
    return "\n".join(lines) + "\n"


def to_html(report: dict, title: str = "Clinical agent evaluation") -> str:
    head = "".join(f"<th>{label}</th>" for _, label in _COLUMNS)
    body = ""
    for row in _rows(report):
        tds = "".join(f"<td>{_fmt(key, row.get(key))}</td>" for key, _ in _COLUMNS)
        strong = ' style="font-weight:600"' if row.get("task") == "overall" else ""
        body += f"<tr{strong}>{tds}</tr>"
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;color:#10212b}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #d0d7de;padding:8px 12px;text-align:left}"
        "th{background:#f6f8fa}</style></head><body>"
        f"<h1>{title}</h1>{'<table>'}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></body></html>"
    )
