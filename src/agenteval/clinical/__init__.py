"""Claim-level evaluation for clinical agents: is each claim grounded, are the numbers right, is confidence calibrated.

This is the second level of agenteval. The top-level package scores how an agent *behaves* (tool use, loops,
task success); this subpackage scores whether what it *says* can be trusted.
"""

from .harness import evaluate_by_task, load_records, run
from .leaderboard import to_html, to_markdown
from .metrics import calibration, evaluate, grounding_rate, hallucinated_citation_rate, number_accuracy
from .schemas import Claim, NumberClaim, Record

__all__ = [
    "Claim",
    "NumberClaim",
    "Record",
    "calibration",
    "evaluate",
    "evaluate_by_task",
    "grounding_rate",
    "hallucinated_citation_rate",
    "load_records",
    "number_accuracy",
    "run",
    "to_html",
    "to_markdown",
]
