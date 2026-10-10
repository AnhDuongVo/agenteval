from agenteval.adapters import from_openai
from agenteval.metrics import evaluate, loop_rate


def trace(messages, expected=None):
    return from_openai({"id": "x", "messages": messages, "expected_tools": expected})


def request(id, name="search"):
    return {"role": "assistant", "tool_calls": [{"id": id, "function": {"name": name, "arguments": "{}"}}]}


def response(id, error=False):
    return {"role": "tool", "tool_call_id": id, "content": "result", "error": error}


def test_one_logical_call_and_no_false_loop():
    run = trace([request("a"), response("a")])
    assert len(run.tool_calls()) == 1 and run.tool_calls()[0].ok
    assert run.tool_calls()[0].output == "result" and loop_rate([run]) == 0


def test_out_of_order_parallel_results_and_error():
    msg = request("a")
    msg["tool_calls"] += request("b", "lookup")["tool_calls"]
    run = trace([msg, response("b", True), response("a")])
    assert [s.ok for s in run.tool_calls()] == [True, False]
    assert evaluate([run])["tool_success_rate"] == 0.5


def test_missing_response_is_not_success():
    assert not trace([request("a")]).tool_calls()[0].ok


def test_extra_tool_penalizes_precision_and_exact_accuracy():
    run = trace([request("a"), response("a"), request("b", "extra"), response("b")], ["search"])
    metrics = evaluate([run])
    assert metrics["expected_tool_recall"] == 1
    assert metrics["tool_selection_accuracy"] == 0
    assert metrics["tool_precision"] == 0.5


def test_citation_integrity_does_not_establish_semantic_support():
    run = trace([])
    run.final_answer = "Unsupported claim"
    run.citations = ["doc"]
    run.retrieved = ["doc"]
    assert evaluate([run])["citation_integrity_rate"] == 1
