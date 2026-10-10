"""Workshop exercise and automated check: python examples/workshop.py."""

from agenteval.adapters import from_openai
from agenteval.metrics import evaluate

TRACE = {
    "id": "workshop",
    "expected_tools": ["lookup"],
    "retrieved": ["table:1"],
    "citations": ["invented"],
    "messages": [
        {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "source"},
        {"role": "assistant", "tool_calls": [{"id": "b", "function": {"name": "unnecessary", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": "extra"},
    ],
}


def check():
    run = from_openai(TRACE)
    metrics = evaluate([run])
    assert len(run.tool_calls()) == 2
    assert metrics["expected_tool_recall"] == 1
    assert metrics["tool_precision"] == 0.5
    assert metrics["tool_selection_accuracy"] == 0
    assert metrics["citation_integrity_rate"] == 0
    assert metrics["loop_rate"] == 0
    print(metrics)
    print("Workshop checks passed")


if __name__ == "__main__":
    check()
