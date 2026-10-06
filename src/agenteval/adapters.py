"""Adapters from each framework's trace into the common AgentRun.

They accept plain dicts/lists (the shape each framework emits), so agenteval does not depend on langgraph,
llamaindex or openai being installed. The docstrings give the expected shape; see tests and data/samples.
"""

from __future__ import annotations

from .schema import AgentRun, Step


def from_openai(trace: dict) -> AgentRun:
    """OpenAI tool-calling trace.

    Expected shape: {"id", "task", "messages": [...], "retrieved": [...], "citations": [...],
    "expected_tools": [...], "success": bool}. Assistant messages may carry `tool_calls`
    (OpenAI shape: [{"function": {"name", "arguments"}}]); role "tool" messages are their results.
    """
    steps: list[Step] = []
    for m in trace.get("messages", []):
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                steps.append(Step(kind="tool_call", name=fn.get("name", ""), input=str(fn.get("arguments", ""))))
            if m.get("content"):
                steps.append(Step(kind="message", output=str(m["content"])))
        elif role == "tool":
            ok = not bool(m.get("error"))
            steps.append(Step(kind="tool_call", name=m.get("name", ""), ok=ok, output=str(m.get("content", ""))))
    final = next((str(m.get("content", "")) for m in reversed(trace.get("messages", [])) if m.get("role") == "assistant" and m.get("content")), "")
    return AgentRun(
        id=trace["id"], framework="openai", task=trace.get("task", ""), steps=steps,
        final_answer=trace.get("final_answer", final), citations=trace.get("citations", []),
        retrieved=trace.get("retrieved", []), expected_tools=trace.get("expected_tools"),
        success=trace.get("success"),
    )


def from_langgraph(trace: dict) -> AgentRun:
    """LangGraph-style trace: {"id","task","events":[{"node","type","name","ok","output"}...], ...}.

    `type` is "tool"|"retriever"|"llm"; everything else maps to a message step.
    """
    kind_map = {"tool": "tool_call", "retriever": "retrieval"}
    steps = [
        Step(kind=kind_map.get(e.get("type", ""), "message"), name=e.get("name", e.get("node", "")),
             ok=e.get("ok", True), output=str(e.get("output", "")))
        for e in trace.get("events", [])
    ]
    return AgentRun(
        id=trace["id"], framework="langgraph", task=trace.get("task", ""), steps=steps,
        final_answer=trace.get("final_answer", ""), citations=trace.get("citations", []),
        retrieved=trace.get("retrieved", []), expected_tools=trace.get("expected_tools"),
        success=trace.get("success"),
    )


def from_llamaindex(trace: dict) -> AgentRun:
    """LlamaIndex-style trace: {"id","task","steps":[{"tool","is_error","observation"}...], ...}."""
    steps = [
        Step(kind="tool_call", name=s.get("tool", ""), ok=not s.get("is_error", False), output=str(s.get("observation", "")))
        for s in trace.get("steps", [])
    ]
    return AgentRun(
        id=trace["id"], framework="llamaindex", task=trace.get("task", ""), steps=steps,
        final_answer=trace.get("final_answer", ""), citations=trace.get("citations", []),
        retrieved=trace.get("retrieved", []), expected_tools=trace.get("expected_tools"),
        success=trace.get("success"),
    )


ADAPTERS = {"openai": from_openai, "langgraph": from_langgraph, "llamaindex": from_llamaindex}


def load(trace: dict, framework: str | None = None) -> AgentRun:
    """Dispatch to the right adapter. `framework` overrides the trace's own 'framework' field."""
    fw = framework or trace.get("framework")
    if fw not in ADAPTERS:
        raise ValueError(f"unknown framework {fw!r}; expected one of {sorted(ADAPTERS)}")
    return ADAPTERS[fw](trace)
