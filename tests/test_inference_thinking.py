"""Thinking-policy regressions for Module 16's provided Ollama transport.

Suggested order to implement & turn green:
1. Implement OllamaBackend.complete (Module 16) for the generate cases.
2. Implement the RAG prompt assembler (17) and plan parser (19) for callers.
Native-chat and raw-transport cases exercise provided boilerplate directly.
All requests are local fakes: no running Ollama or model download is needed.
"""

import io
import json

import pytest

from g2c.inference import OllamaBackend, OllamaError


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize(
    "default,override,expected",
    [
        (False, None, False),
        (True, None, True),
        (None, None, None),
        (False, True, True),
        (True, False, False),
        (None, False, False),
    ],
)
def test_policy_and_explicit_override_on_both_endpoints(chat, default, override, expected):
    requests = []

    def transport(req, timeout=None):
        requests.append(json.loads(req.data))
        return io.BytesIO(json.dumps({"response": "ok", "message": {"content": "ok"}}).encode())

    backend = OllamaBackend("custom-model-alias", default_think=default, urlopen=transport)
    if chat:
        result = backend.chat_with_tools([{"role": "user", "content": "hi"}], think=override)
    else:
        result = backend.complete("hi", think=override)
    if expected is None:
        assert "think" not in requests[0]
    else:
        assert requests[0]["think"] is expected
    assert result.metadata["sampling"]["think"] is expected


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize(
    "content,tools,reason,raises",
    [
        ("", [], "length", True),
        (" \n", [], "length", True),
        ("partial answer", [], "length", False),
        ("", [], "stop", False),
        (
            "",
            [{"function": {"name": "calculator", "arguments": {"expression": "2+2"}}}],
            "length",
            False,
        ),
    ],
)
def test_thinking_exhaustion_does_not_masquerade_as_answer(chat, content, tools, reason, raises):
    if tools and not chat:
        pytest.skip("generate does not return native tool calls")
    payload = {
        "thinking": "some reasoning",
        "response": content,
        "done_reason": reason,
        "message": {"thinking": "some reasoning", "content": content, "tool_calls": tools},
    }
    backend = OllamaBackend(
        urlopen=lambda req, timeout=None: io.BytesIO(json.dumps(payload).encode())
    )

    def call():
        if chat:
            return backend.chat_with_tools([{"role": "user", "content": "hi"}], think=True)
        return backend.complete("hi", think=True)

    if raises:
        with pytest.raises(OllamaError, match="exhausted max_new_tokens.*think=False"):
            call()
    else:
        result = call()
        assert result.metadata["thinking"] == "some reasoning"
        assert result.metadata["done_reason"] == reason
        assert (result.content if chat else result.completion) == content


def test_rag_and_planning_inherit_transport_policy():
    from g2c.agent.planner import make_plan
    from g2c.rag import RAGPipeline
    from g2c.tools import ToolRegistry

    requests = []

    def transport(req, timeout=None):
        requests.append(json.loads(req.data))
        return io.BytesIO(json.dumps({"response": "1. Read the context.\n2. Answer."}).encode())

    class EmptyRetriever:
        def retrieve(self, query, *, k):
            return []

    backend = OllamaBackend("qwen3.5:4b", urlopen=transport)
    RAGPipeline(EmptyRetriever(), backend).answer("Who owns Cedar?")
    make_plan(backend, "Who owns Cedar?", ToolRegistry())
    assert len(requests) == 2
    assert all(r["think"] is False for r in requests)


def test_raw_transport_policy_preserves_student_mapping_and_input():
    requests = []

    def transport(req, timeout=None):
        requests.append(json.loads(req.data))
        return io.BytesIO(b'{"response": "hello"}')

    backend = OllamaBackend("qwen3.5:4b", urlopen=transport)
    body = {"model": "qwen3.5:4b", "prompt": "Hi", "options": {"num_predict": 128}}
    backend._post_generate(body)
    assert "think" not in body
    assert requests[0]["think"] is False
    assert requests[0]["options"] == {"num_predict": 128}
