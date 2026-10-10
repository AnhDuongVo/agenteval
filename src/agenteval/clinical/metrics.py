"""The metrics. Pure functions over a list of records, so they are easy to test and to trust.

Every metric returns None when it has no applicable denominator, rather than a misleading zero.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import Record


def _is_number_correct(value: float, source_value: float, rel_tol: float, abs_tol: float) -> bool:
    diff = abs(value - source_value)
    if diff <= abs_tol:
        return True
    denom = max(abs(source_value), 1e-9)
    return diff / denom <= rel_tol


def citation_integrity_rate(records: list[Record]) -> float | None:
    """Fraction of claims that cite at least one source and whose every cited id exists in the record."""
    total = grounded = 0
    for r in records:
        src = set(r.sources)
        for c in r.claims:
            total += 1
            if c.citations and all(cid in src for cid in c.citations):
                grounded += 1
    return grounded / total if total else None


# Compatibility alias: existence checks do not establish semantic grounding.
grounding_rate = citation_integrity_rate


def hallucinated_citation_rate(records: list[Record]) -> float | None:
    """Fraction of all cited ids that do not exist in their record's sources."""
    total = bad = 0
    for r in records:
        src = set(r.sources)
        for c in r.claims:
            for cid in c.citations:
                total += 1
                if cid not in src:
                    bad += 1
    return bad / total if total else None


def number_accuracy(records: list[Record], rel_tol: float = 0.01, abs_tol: float = 1e-6) -> float | None:
    """Fraction of asserted numbers that match the cited source value (a number from an ungrounded source fails)."""
    total = correct = 0
    for r in records:
        src = set(r.sources)
        for c in r.claims:
            for n in c.numbers:
                if n.source_value is None:
                    continue
                total += 1
                grounded = n.source_id is None or n.source_id in src
                if grounded and _is_number_correct(n.value, n.source_value, rel_tol, abs_tol):
                    correct += 1
    return correct / total if total else None


@dataclass
class Calibration:
    ece: float | None  # expected calibration error (lower is better)
    brier: float | None  # Brier score (lower is better)
    accuracy: float | None
    n: int


def calibration(records: list[Record], bins: int = 10) -> Calibration:
    """ECE, Brier and accuracy over claims that carry both a confidence and a gold label."""
    pairs = [
        (c.confidence, 1.0 if c.label else 0.0)
        for r in records
        for c in r.claims
        if c.confidence is not None and c.label is not None
    ]
    n = len(pairs)
    if not n:
        return Calibration(ece=None, brier=None, accuracy=None, n=0)
    brier = sum((conf - lab) ** 2 for conf, lab in pairs) / n
    accuracy = sum(lab for _, lab in pairs) / n
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [(conf, lab) for conf, lab in pairs if (conf >= lo and (conf < hi or (b == bins - 1 and conf <= hi)))]
        if not in_bin:
            continue
        avg_conf = sum(conf for conf, _ in in_bin) / len(in_bin)
        avg_acc = sum(lab for _, lab in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(avg_conf - avg_acc)
    return Calibration(ece=ece, brier=brier, accuracy=accuracy, n=n)


def evaluate(records: list[Record], rel_tol: float = 0.01, bins: int = 10) -> dict:
    """Run every metric over a set of records and return a plain dict of scores."""
    cal = calibration(records, bins=bins)
    return {
        "n_records": len(records),
        "n_claims": sum(len(r.claims) for r in records),
        "citation_integrity_rate": citation_integrity_rate(records),
        "grounding_rate": grounding_rate(records),
        "hallucinated_citation_rate": hallucinated_citation_rate(records),
        "number_accuracy": number_accuracy(records, rel_tol=rel_tol),
        "calibration_ece": cal.ece,
        "calibration_brier": cal.brier,
        "calibration_accuracy": cal.accuracy,
        "calibration_n": cal.n,
    }
