"""The common run schema. Every framework adapter produces this, so one set of metrics scores them all."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Step(BaseModel):
    kind: str                       # "tool_call" | "retrieval" | "message"
    name: str = ""                  # tool or node name
    ok: bool = True                 # did the step succeed (no error)?
    input: str = ""
    output: str = ""


class AgentRun(BaseModel):
    id: str
    framework: str = "unknown"      # langgraph | llamaindex | openai | ...
    task: str = ""
    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    citations: list[str] = Field(default_factory=list)   # source ids the answer cites
    retrieved: list[str] = Field(default_factory=list)    # source ids available to the run
    expected_tools: list[str] | None = None              # gold: tools the task should use
    success: bool | None = None                          # gold: did the run accomplish the task?

    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool_call"]
