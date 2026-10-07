"""Claim-level metrics return known values, and the harness scores the bundled clinical samples."""

from __future__ import annotations

from pathlib import Path

from agenteval.clinical import (
    Claim,
    NumberClaim,
    Record,
    calibration,
    evaluate_by_task,
    grounding_rate,
    hallucinated_citation_rate,
    load_records,
    number_accuracy,
)

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "clinical_samples"


def _rec(claims, sources=("s1", "s2")):
    return Record(id="x", task="t", sources=list(sources), claims=claims)


def test_grounding_rate():
    recs = [_rec([Claim(citations=["s1"]), Claim(citations=["s2", "s1"]), Claim(citations=["s9"]), Claim(citations=[])])]
    assert grounding_rate(recs) == 0.5


def test_hallucinated_citation_rate():
    assert abs(hallucinated_citation_rate([_rec([Claim(citations=["s1", "s9", "s8"])])]) - 2 / 3) < 1e-9


def test_number_accuracy_with_rounding():
    recs = [_rec([
        Claim(numbers=[NumberClaim(value=1.21, source_id="s1", source_value=1.21)]),
        Claim(numbers=[NumberClaim(value=12.0, source_id="s2", source_value=12.004)]),
        Claim(numbers=[NumberClaim(value=-1.4, source_id="s1", source_value=-1.21)]),
        Claim(numbers=[NumberClaim(value=5.0, source_id="s9", source_value=5.0)]),
    ])]
    assert number_accuracy(recs, rel_tol=0.01) == 0.5


def test_calibration_perfect_and_empty():
    cal = calibration([_rec([Claim(confidence=1.0, label=True), Claim(confidence=0.0, label=False)])])
    assert cal.n == 2 and abs(cal.brier) < 1e-9 and abs(cal.ece) < 1e-9
    assert calibration([_rec([Claim(text="no conf")])]).n == 0


def test_samples_score_and_surface_errors():
    report = evaluate_by_task(load_records(sorted(SAMPLES.glob("*.jsonl"))))
    for task in ["consult-to-note", "trial-matcher", "csr-assistant", "ai-scientist"]:
        assert task in report["by_task"]
    assert report["by_task"]["csr-assistant"]["number_accuracy"] < 1.0        # planted wrong number
    assert report["by_task"]["ai-scientist"]["hallucinated_citation_rate"] > 0  # planted bad citation
