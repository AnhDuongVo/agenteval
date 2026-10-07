"""The evaluation schema. Any clinical agent can emit records in this shape, so one harness scores them all.

A record is one evaluated item (a generated note, a screening decision, a drafted CSR section, a ranked
candidate). A claim is one checkable statement the agent produced, with the source ids it cited, any
numbers it asserted (with the value it should match from the source), and optionally a confidence and a
gold label for calibration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class NumberClaim(BaseModel):
    value: float
    source_id: str | None = None          # the source row/fact this number should come from
    source_value: float | None = None     # the true value in that source, when known


class Claim(BaseModel):
    text: str = ""
    citations: list[str] = Field(default_factory=list)   # source ids the claim cites
    numbers: list[NumberClaim] = Field(default_factory=list)
    confidence: float | None = None       # the agent's confidence in [0, 1]
    label: bool | None = None             # gold: was the claim correct? (for calibration)


class Record(BaseModel):
    id: str
    task: str                              # consult-to-note | trial-matcher | csr-assistant | ai-scientist | ...
    sources: list[str] = Field(default_factory=list)   # the source ids that actually exist for this item
    claims: list[Claim] = Field(default_factory=list)
