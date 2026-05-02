"""Tests for g2c.assistant — Module 20 (Capstone: a tiny ChatGPT).

Suggested order to implement & turn green:

    1. `Conversation.format_for_prompt` in
       `g2c/assistant/conversation.py`. Pure logic — render the chat
       history as a multi-line "User: / Assistant:" block. Independent
       of everything else. Turns green:
         TestConversationFormat
         TestConversationFormatTruncation

    2. `Assistant.chat` in `g2c/assistant/assistant.py`. The
       orchestration. Composes Module 16 (Backend), Module 17 (RAG),
       Module 18 (tools), Module 19 (Agent), and Module 20's new
       Conversation primitive. Turns green:
         TestAssistantChat
         TestAssistantChatRAG
         TestAssistantChatHistory
         TestAssistantChatErrorHandling
         TestEvalSuite
         TestCLI

The two scaffolds are independent. Suggested order is "conversation
first" because the assistant's chat depends on `format_for_prompt`,
but you can implement either first — the tests for the unimplemented
one will keep failing.

Boilerplate tests (`TestMessage`, `TestAssistantConfig`,
`TestAssistantConstruction`, etc.) pass from the start.
"""
from __future__ import annotations

import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from g2c.agent import Agent, AgentRunResult
from g2c.assistant import (
    CLI_HELP,
    Assistant,
    AssistantConfig,
    AssistantError,
    AssistantTurn,
    Conversation,
    EvalCase,
    EvalCaseResult,
    EvalReport,
    Message,
    run_cli,
    run_evaluation,
)
from g2c.assistant.conversation import ASSISTANT_ROLE, USER_ROLE
from g2c.inference import Backend, BackendInfo, InferenceResult
from g2c.tools import ToolRegistry, make_calculator

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _FakeBackend(Backend):
    """Backend that returns pre-recorded completions in order.

    Records every call's prompt for assertion. Each `complete` pops
    the next completion off the list. If the list runs dry, the
    test has called too many times — fail loudly.
    """

    def __init__(
        self,
        completions: Iterable[str],
        *,
        info: BackendInfo | None = None,
    ) -> None:
        self._completions = list(completions)
        self._info = info or BackendInfo(name="fake", model_id="fake-model")
        self.calls: list[dict[str, Any]] = []

    @property
    def info(self) -> BackendInfo:
        return self._info

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> InferenceResult:
        if not self._completions:
            raise AssertionError(
                f"FakeBackend exhausted: prompt was {prompt!r}"
            )
        completion = self._completions.pop(0)
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )
        return InferenceResult(
            prompt=prompt,
            completion=completion,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(completion.split()),
            latency_ms=1.0,
            backend=self._info,
        )


def _final_answer_completion(answer: str) -> str:
    """A canned ReAct completion that emits `Final Answer: <answer>`."""
    return f"Thought: I have the answer.\nFinal Answer: {answer}"


def _action_completion(tool: str, args_json: str, thought: str = "go") -> str:
    """A canned ReAct completion that emits an Action."""
    return (
        f"Thought: {thought}\n"
        f"Action: {tool}\n"
        f"Action Input: {args_json}"
    )


def _planning_completion() -> str:
    """A canned planning-prompt completion. Used when `plan=True`."""
    return "Goal: do the thing\n1. step one\n2. step two"


def _trivial_chat(answer: str = "hi back") -> list[str]:
    """Sequence of completions for the simplest chat: plan + final answer."""
    return [_planning_completion(), _final_answer_completion(answer)]


def _no_plan_chat(answer: str = "hi back") -> list[str]:
    """Sequence of completions when plan=False: just the final answer."""
    return [_final_answer_completion(answer)]


@dataclass
class _FakeChunk:
    text: str
    source: str = "doc.md"


@dataclass
class _FakeRetrieved:
    chunk: _FakeChunk
    score: float = 0.9
    rank: int = 1


class _FakeRetriever:
    """A retriever that returns pre-recorded chunks per call.

    Useful for asserting "RAG fired" / "RAG didn't fire" without
    standing up a real embedder + vector store.
    """

    def __init__(self, chunks: Iterable[_FakeRetrieved] | None = None) -> None:
        self._chunks = list(chunks) if chunks is not None else []
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, *, k: int = 5) -> list[_FakeRetrieved]:
        self.queries.append((query, k))
        return list(self._chunks[:k])


class _BrokenRetriever:
    """A retriever that always raises. Used to test error propagation."""

    def retrieve(self, query: str, *, k: int = 5) -> list[_FakeRetrieved]:
        raise RuntimeError("intentional retrieval failure")


def _make_assistant(
    completions: Iterable[str],
    *,
    config: AssistantConfig | None = None,
    registry: ToolRegistry | None = None,
    retriever: Any = None,
    conversation: Conversation | None = None,
) -> tuple[Assistant, _FakeBackend]:
    """Build an Assistant with a fake backend and trivial registry.

    Returns the (assistant, backend) pair so tests can assert on
    backend.calls.
    """
    backend = _FakeBackend(completions)
    if registry is None:
        registry = ToolRegistry()
    assistant = Assistant(
        backend,
        registry,
        config=config,
        retriever=retriever,
        conversation=conversation,
    )
    return assistant, backend


# ---------------------------------------------------------------------------
# Boilerplate / dataclass tests — pass from the start.
# ---------------------------------------------------------------------------


class TestMessage:
    def test_construction(self) -> None:
        m = Message(role="user", content="hi")
        assert m.role == "user"
        assert m.content == "hi"

    def test_assistant_role_works(self) -> None:
        Message(role="assistant", content="ok")

    def test_invalid_role_raises(self) -> None:
        with pytest.raises(AssistantError, match="role must be one of"):
            Message(role="system", content="hi")

    def test_empty_content_raises(self) -> None:
        with pytest.raises(AssistantError, match="non-empty"):
            Message(role="user", content="")

    def test_non_string_content_raises(self) -> None:
        with pytest.raises(AssistantError, match="non-empty"):
            Message(role="user", content=123)  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        m = Message(role="user", content="hi")
        with pytest.raises(Exception):  # FrozenInstanceError
            m.content = "bye"  # type: ignore[misc]

    def test_role_constants(self) -> None:
        assert USER_ROLE == "user"
        assert ASSISTANT_ROLE == "assistant"


class TestConversationConstruction:
    def test_empty(self) -> None:
        conv = Conversation()
        assert len(conv) == 0
        assert conv.messages == []
        assert conv.max_messages is None

    def test_with_max_messages(self) -> None:
        conv = Conversation(max_messages=5)
        assert conv.max_messages == 5

    def test_max_messages_zero_raises(self) -> None:
        with pytest.raises(AssistantError, match="must be > 0"):
            Conversation(max_messages=0)

    def test_max_messages_negative_raises(self) -> None:
        with pytest.raises(AssistantError, match="must be > 0"):
            Conversation(max_messages=-1)

    def test_seed_with_messages(self) -> None:
        seeds = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        conv = Conversation(seeds)
        assert len(conv) == 2
        assert conv.messages == seeds

    def test_seed_with_non_message_raises(self) -> None:
        with pytest.raises(AssistantError, match="non-Message"):
            Conversation([{"role": "user", "content": "hi"}])  # type: ignore[list-item]


class TestConversationOps:
    def test_add_user(self) -> None:
        conv = Conversation()
        m = conv.add_user("hi")
        assert m.role == "user"
        assert m.content == "hi"
        assert len(conv) == 1

    def test_add_assistant(self) -> None:
        conv = Conversation()
        m = conv.add_assistant("hello")
        assert m.role == "assistant"
        assert m.content == "hello"

    def test_clear(self) -> None:
        conv = Conversation()
        conv.add_user("a")
        conv.add_assistant("b")
        conv.add_user("c")
        assert len(conv) == 3
        conv.clear()
        assert len(conv) == 0
        assert conv.messages == []

    def test_clear_is_reusable(self) -> None:
        conv = Conversation()
        conv.add_user("a")
        conv.clear()
        conv.add_user("b")
        assert len(conv) == 1

    def test_messages_returns_snapshot(self) -> None:
        conv = Conversation()
        conv.add_user("a")
        snap = conv.messages
        snap.append(Message(role="user", content="b"))
        assert len(conv) == 1  # underlying not mutated

    def test_iter(self) -> None:
        conv = Conversation()
        conv.add_user("a")
        conv.add_assistant("b")
        roles = [m.role for m in conv]
        assert roles == ["user", "assistant"]

    def test_last_user_message_empty(self) -> None:
        conv = Conversation()
        assert conv.last_user_message() is None

    def test_last_user_message(self) -> None:
        conv = Conversation()
        conv.add_user("first")
        conv.add_assistant("response")
        conv.add_user("second")
        last = conv.last_user_message()
        assert last is not None
        assert last.content == "second"

    def test_last_user_message_no_user_msgs(self) -> None:
        conv = Conversation()
        conv.add_assistant("opening line")
        assert conv.last_user_message() is None

    def test_repr_contains_count(self) -> None:
        conv = Conversation(max_messages=10)
        conv.add_user("a")
        r = repr(conv)
        assert "n_messages=1" in r
        assert "max=10" in r


# ---------------------------------------------------------------------------
# Conversation.format_for_prompt — SCAFFOLDED.
# ---------------------------------------------------------------------------


class TestConversationFormat:
    def test_empty_returns_empty_string(self) -> None:
        conv = Conversation()
        assert conv.format_for_prompt() == ""

    def test_one_user_message(self) -> None:
        conv = Conversation()
        conv.add_user("hi")
        out = conv.format_for_prompt()
        assert out == "User: hi"

    def test_user_assistant_pair(self) -> None:
        conv = Conversation()
        conv.add_user("hi")
        conv.add_assistant("hello")
        out = conv.format_for_prompt()
        assert out == "User: hi\nAssistant: hello"

    def test_three_messages(self) -> None:
        conv = Conversation()
        conv.add_user("a")
        conv.add_assistant("b")
        conv.add_user("c")
        out = conv.format_for_prompt()
        assert out == "User: a\nAssistant: b\nUser: c"

    def test_role_prefixes_used(self) -> None:
        conv = Conversation()
        conv.add_user("user-side")
        conv.add_assistant("assistant-side")
        out = conv.format_for_prompt()
        assert "User: user-side" in out
        assert "Assistant: assistant-side" in out

    def test_multiline_content_preserved(self) -> None:
        conv = Conversation()
        conv.add_user("line1\nline2")
        out = conv.format_for_prompt()
        # Prefix appears once at the start; embedded newline is kept.
        assert out.startswith("User: line1")
        assert "\nline2" in out

    def test_ordering_preserved(self) -> None:
        conv = Conversation()
        for i in range(5):
            conv.add_user(f"u{i}")
            conv.add_assistant(f"a{i}")
        out = conv.format_for_prompt()
        # Find positions of each — they must be monotonically increasing.
        positions = [out.index(f"User: u{i}") for i in range(5)]
        assert positions == sorted(positions)


class TestConversationFormatTruncation:
    def test_unlimited_keeps_all(self) -> None:
        conv = Conversation(max_messages=None)
        for i in range(10):
            conv.add_user(f"msg{i}")
        out = conv.format_for_prompt()
        for i in range(10):
            assert f"msg{i}" in out

    def test_max_messages_keeps_recent(self) -> None:
        conv = Conversation(max_messages=2)
        conv.add_user("oldest")
        conv.add_assistant("middle")
        conv.add_user("newest")
        out = conv.format_for_prompt()
        assert "newest" in out
        assert "middle" in out
        assert "oldest" not in out

    def test_max_messages_exact_fit(self) -> None:
        conv = Conversation(max_messages=3)
        conv.add_user("a")
        conv.add_assistant("b")
        conv.add_user("c")
        out = conv.format_for_prompt()
        for token in ("a", "b", "c"):
            assert token in out

    def test_max_messages_one(self) -> None:
        conv = Conversation(max_messages=1)
        conv.add_user("first")
        conv.add_assistant("second")
        out = conv.format_for_prompt()
        assert "second" in out
        assert "first" not in out


# ---------------------------------------------------------------------------
# AssistantConfig
# ---------------------------------------------------------------------------


class TestAssistantConfig:
    def test_defaults(self) -> None:
        c = AssistantConfig()
        assert c.name == "g2c-assistant"
        assert c.max_steps == 8
        assert c.plan is True
        assert c.loop_detection is True
        assert c.halt_on_stuck is False
        assert c.rag_enabled is True
        assert c.rag_k == 5

    def test_custom(self) -> None:
        c = AssistantConfig(
            name="custom",
            max_steps=4,
            plan=False,
            temperature=0.5,
            rag_k=3,
        )
        assert c.name == "custom"
        assert c.max_steps == 4
        assert c.plan is False
        assert c.temperature == 0.5
        assert c.rag_k == 3

    def test_empty_name_raises(self) -> None:
        with pytest.raises(AssistantError, match="non-empty"):
            AssistantConfig(name="")

    def test_max_steps_zero_raises(self) -> None:
        with pytest.raises(AssistantError, match="max_steps"):
            AssistantConfig(max_steps=0)

    def test_max_steps_negative_raises(self) -> None:
        with pytest.raises(AssistantError, match="max_steps"):
            AssistantConfig(max_steps=-1)

    def test_max_new_tokens_zero_raises(self) -> None:
        with pytest.raises(AssistantError, match="max_new_tokens"):
            AssistantConfig(max_new_tokens=0)

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(AssistantError, match="temperature"):
            AssistantConfig(temperature=-0.1)

    def test_temperature_zero_ok(self) -> None:
        c = AssistantConfig(temperature=0.0)
        assert c.temperature == 0.0

    def test_rag_k_zero_raises(self) -> None:
        with pytest.raises(AssistantError, match="rag_k"):
            AssistantConfig(rag_k=0)

    def test_max_history_zero_raises(self) -> None:
        with pytest.raises(AssistantError, match="max_history"):
            AssistantConfig(max_history_messages=0)

    def test_max_history_none_ok(self) -> None:
        c = AssistantConfig(max_history_messages=None)
        assert c.max_history_messages is None

    def test_scratchpad_zero_raises(self) -> None:
        with pytest.raises(AssistantError, match="scratchpad"):
            AssistantConfig(scratchpad_max_chars=0)


# ---------------------------------------------------------------------------
# AssistantTurn boilerplate.
# ---------------------------------------------------------------------------


class TestAssistantTurnBoilerplate:
    def test_field_set(self) -> None:
        # Construct directly — most fields are filled by Assistant.chat,
        # but a unit test of the dataclass shape lives here.
        info = BackendInfo(name="fake", model_id="x")
        infres = InferenceResult(
            prompt="p",
            completion="c",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
            backend=info,
        )
        run = AgentRunResult(
            user_message="hi",
            plan=None,
            final_answer="bye",
            steps=[],
            stopped_reason="final_answer",
            metadata={"n_steps": 0, "n_tool_calls": 0},
        )
        del infres  # silence unused; only here to confirm imports.
        turn = AssistantTurn(
            user_message="hi",
            final_answer="bye",
            agent_run=run,
            retrieved_context="",
            contextualized_message="hi",
        )
        assert turn.final_answer == "bye"
        assert turn.metadata == {}


# ---------------------------------------------------------------------------
# Assistant construction.
# ---------------------------------------------------------------------------


class TestAssistantConstruction:
    def test_minimal(self) -> None:
        backend = _FakeBackend([])
        registry = ToolRegistry()
        a = Assistant(backend, registry)
        assert a.backend is backend
        assert a.registry is registry
        assert a.retriever is None
        assert isinstance(a.conversation, Conversation)
        assert isinstance(a.agent, Agent)
        assert a.config.name == "g2c-assistant"

    def test_with_config(self) -> None:
        backend = _FakeBackend([])
        registry = ToolRegistry()
        cfg = AssistantConfig(name="custom", max_steps=4)
        a = Assistant(backend, registry, config=cfg)
        assert a.config.name == "custom"
        assert a.agent.max_steps == 4

    def test_with_retriever(self) -> None:
        backend = _FakeBackend([])
        registry = ToolRegistry()
        retr = _FakeRetriever()
        a = Assistant(backend, registry, retriever=retr)
        assert a.retriever is retr

    def test_with_existing_conversation(self) -> None:
        backend = _FakeBackend([])
        registry = ToolRegistry()
        conv = Conversation()
        conv.add_user("seeded")
        a = Assistant(backend, registry, conversation=conv)
        assert a.conversation is conv
        assert len(a.conversation) == 1

    def test_with_existing_agent(self) -> None:
        backend = _FakeBackend([])
        registry = ToolRegistry()
        ag = Agent(backend, registry, max_steps=3)
        a = Assistant(backend, registry, agent=ag)
        assert a.agent is ag

    def test_bad_backend_type_raises(self) -> None:
        with pytest.raises(AssistantError, match="Backend"):
            Assistant("not-a-backend", ToolRegistry())  # type: ignore[arg-type]

    def test_bad_registry_type_raises(self) -> None:
        with pytest.raises(AssistantError, match="ToolRegistry"):
            Assistant(_FakeBackend([]), "not-a-registry")  # type: ignore[arg-type]

    def test_bad_config_type_raises(self) -> None:
        with pytest.raises(AssistantError, match="AssistantConfig"):
            Assistant(
                _FakeBackend([]),
                ToolRegistry(),
                config="not-a-config",  # type: ignore[arg-type]
            )

    def test_bad_retriever_type_raises(self) -> None:
        class _NoRetrieve:
            pass

        with pytest.raises(AssistantError, match="retrieve"):
            Assistant(
                _FakeBackend([]),
                ToolRegistry(),
                retriever=_NoRetrieve(),  # type: ignore[arg-type]
            )

    def test_bad_conversation_type_raises(self) -> None:
        with pytest.raises(AssistantError, match="Conversation"):
            Assistant(
                _FakeBackend([]),
                ToolRegistry(),
                conversation=[],  # type: ignore[arg-type]
            )

    def test_bad_agent_type_raises(self) -> None:
        with pytest.raises(AssistantError, match="Agent"):
            Assistant(
                _FakeBackend([]),
                ToolRegistry(),
                agent="not-an-agent",  # type: ignore[arg-type]
            )

    def test_repr(self) -> None:
        a, _ = _make_assistant([])
        r = repr(a)
        assert "Assistant" in r
        assert "g2c-assistant" in r

    def test_default_max_history_propagated(self) -> None:
        cfg = AssistantConfig(max_history_messages=7)
        a, _ = _make_assistant([], config=cfg)
        assert a.conversation.max_messages == 7

    def test_explicit_conversation_overrides_max_history(self) -> None:
        # If the caller passes their own conversation, the assistant
        # should not silently rebuild it.
        conv = Conversation(max_messages=3)
        cfg = AssistantConfig(max_history_messages=99)
        a, _ = _make_assistant([], config=cfg, conversation=conv)
        assert a.conversation is conv
        assert a.conversation.max_messages == 3


class TestAssistantReset:
    def test_reset_clears_conversation(self) -> None:
        a, _ = _make_assistant([])
        a.conversation.add_user("a")
        a.conversation.add_assistant("b")
        assert len(a.conversation) == 2
        a.reset()
        assert len(a.conversation) == 0

    def test_reset_clears_turns(self) -> None:
        # Use a config that disables planning so the trivial chat
        # only consumes one backend completion.
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("hello")] * 3,
            config=cfg,
        )
        # Trigger one chat, populate turns.
        a.chat("hi")
        assert len(a.turns) == 1
        a.reset()
        assert a.turns == []
        assert len(a.conversation) == 0


# ---------------------------------------------------------------------------
# Assistant.chat — the main scaffold.
# ---------------------------------------------------------------------------


class TestAssistantChat:
    def test_returns_assistant_turn(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("hello")], config=cfg
        )
        turn = a.chat("hi")
        assert isinstance(turn, AssistantTurn)
        assert turn.final_answer == "hello"
        assert turn.user_message == "hi"

    def test_records_user_and_assistant_in_conversation(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("hello")], config=cfg
        )
        a.chat("hi")
        msgs = a.conversation.messages
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].content == "hi"
        assert msgs[1].content == "hello"

    def test_appends_to_turns(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [
                _final_answer_completion("a1"),
                _final_answer_completion("a2"),
            ],
            config=cfg,
        )
        a.chat("q1")
        a.chat("q2")
        assert len(a.turns) == 2
        assert a.turns[0].user_message == "q1"
        assert a.turns[1].user_message == "q2"

    def test_metadata_populated(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("hi")], config=cfg
        )
        turn = a.chat("hi")
        meta = turn.metadata
        assert meta["turn_index"] == 0
        assert meta["rag_fired"] is False
        assert meta["had_plan"] is False
        assert meta["stopped_reason"] == "final_answer"
        assert meta["n_agent_steps"] >= 1
        assert meta["n_tool_calls"] == 0

    def test_turn_index_increments(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("a")] * 3, config=cfg
        )
        a.chat("q1")
        a.chat("q2")
        a.chat("q3")
        assert [t.metadata["turn_index"] for t in a.turns] == [0, 1, 2]

    def test_empty_message_raises(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant([], config=cfg)
        with pytest.raises(AssistantError, match="non-empty"):
            a.chat("")

    def test_non_string_message_raises(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant([], config=cfg)
        with pytest.raises(AssistantError, match="non-empty"):
            a.chat(123)  # type: ignore[arg-type]

    def test_planning_phase_uses_extra_completion(self) -> None:
        # plan=True consumes one extra backend call before the loop.
        cfg = AssistantConfig(plan=True)
        a, backend = _make_assistant(
            [_planning_completion(), _final_answer_completion("done")],
            config=cfg,
        )
        a.chat("hi")
        assert len(backend.calls) == 2

    def test_no_plan_uses_one_completion(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, backend = _make_assistant(
            [_final_answer_completion("done")], config=cfg
        )
        a.chat("hi")
        assert len(backend.calls) == 1

    def test_contextualized_message_includes_user_message(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("ok")], config=cfg
        )
        turn = a.chat("the question")
        assert "the question" in turn.contextualized_message


class TestAssistantChatHistory:
    def test_first_turn_has_no_history(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("ok")], config=cfg
        )
        turn = a.chat("hi")
        # No "Previous conversation:" header on the first turn.
        assert "Previous conversation" not in turn.contextualized_message

    def test_second_turn_includes_first_turn_history(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [
                _final_answer_completion("hello"),
                _final_answer_completion("yes"),
            ],
            config=cfg,
        )
        a.chat("hi")
        turn2 = a.chat("again")
        assert "Previous conversation" in turn2.contextualized_message
        assert "User: hi" in turn2.contextualized_message
        assert "Assistant: hello" in turn2.contextualized_message

    def test_history_does_not_double_include_current_message(self) -> None:
        # Pin the "render history BEFORE adding current user message" rule.
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("ok")], config=cfg
        )
        turn = a.chat("the unique question 7XK")
        # The current message should appear once in the contextualized
        # message — not in a "User: the unique question" history line.
        assert turn.contextualized_message.count("the unique question 7XK") == 1
        assert "User: the unique question 7XK" not in turn.contextualized_message

    def test_history_truncated_by_max_history(self) -> None:
        # max_history_messages=2 keeps only the last two messages of history.
        cfg = AssistantConfig(plan=False, max_history_messages=2)
        a, _ = _make_assistant(
            [_final_answer_completion(f"ans{i}") for i in range(4)],
            config=cfg,
        )
        a.chat("q1")
        a.chat("q2")
        a.chat("q3")
        turn4 = a.chat("q4")
        # Turn 4 sees the truncated history. The conversation has
        # 6 messages (3 user + 3 assistant) before q4 is added; the
        # history rendered for turn 4 keeps only the last two — the
        # latest user message q3 and its assistant reply ans2.
        assert "q1" not in turn4.contextualized_message
        assert "ans0" not in turn4.contextualized_message
        # The most-recent prior exchange should be there.
        assert "q3" in turn4.contextualized_message
        assert "ans2" in turn4.contextualized_message

    def test_history_visible_to_backend_in_prompt(self) -> None:
        # Pin the wiring: history arrives at backend.complete via the
        # agent's prompt, not just via the AssistantTurn.
        cfg = AssistantConfig(plan=False)
        a, backend = _make_assistant(
            [
                _final_answer_completion("first reply"),
                _final_answer_completion("second reply"),
            ],
            config=cfg,
        )
        a.chat("first question")
        a.chat("second question")
        # The second backend call's prompt must contain the first
        # exchange (via the rendered history block).
        second_prompt = backend.calls[-1]["prompt"]
        assert "first question" in second_prompt
        assert "first reply" in second_prompt


# ---------------------------------------------------------------------------
# RAG integration.
# ---------------------------------------------------------------------------


class TestAssistantChatRAG:
    def test_rag_off_when_no_retriever(self) -> None:
        cfg = AssistantConfig(plan=False, rag_enabled=True)
        a, _ = _make_assistant(
            [_final_answer_completion("ok")], config=cfg
        )
        turn = a.chat("hi")
        assert turn.retrieved_context == ""
        assert turn.metadata["rag_fired"] is False

    def test_rag_on_with_retriever_splices_context(self) -> None:
        cfg = AssistantConfig(plan=False, rag_enabled=True, rag_k=2)
        retr = _FakeRetriever([
            _FakeRetrieved(_FakeChunk(text="Madrid is in Spain.", source="geo.md")),
            _FakeRetrieved(_FakeChunk(text="Paris is in France.", source="geo.md")),
        ])
        a, _ = _make_assistant(
            [_final_answer_completion("Madrid")],
            config=cfg,
            retriever=retr,
        )
        turn = a.chat("where is Madrid?")
        assert "Madrid is in Spain" in turn.retrieved_context
        assert "Paris is in France" in turn.retrieved_context
        # rag_fired metadata is True when context is non-empty.
        assert turn.metadata["rag_fired"] is True

    def test_rag_off_per_call_overrides_config(self) -> None:
        cfg = AssistantConfig(plan=False, rag_enabled=True, rag_k=2)
        retr = _FakeRetriever([
            _FakeRetrieved(_FakeChunk(text="some doc", source="x.md")),
        ])
        a, _ = _make_assistant(
            [_final_answer_completion("ok")],
            config=cfg,
            retriever=retr,
        )
        turn = a.chat("hi", use_rag=False)
        assert turn.retrieved_context == ""
        assert turn.metadata["rag_fired"] is False
        # Retriever was NOT called.
        assert retr.queries == []

    def test_rag_on_per_call_overrides_disabled_config(self) -> None:
        cfg = AssistantConfig(plan=False, rag_enabled=False, rag_k=2)
        retr = _FakeRetriever([
            _FakeRetrieved(_FakeChunk(text="forced doc", source="x.md")),
        ])
        a, _ = _make_assistant(
            [_final_answer_completion("ok")],
            config=cfg,
            retriever=retr,
        )
        turn = a.chat("hi", use_rag=True)
        assert "forced doc" in turn.retrieved_context

    def test_rag_query_is_user_message(self) -> None:
        cfg = AssistantConfig(plan=False, rag_enabled=True, rag_k=3)
        retr = _FakeRetriever([])
        a, _ = _make_assistant(
            [_final_answer_completion("ok")], config=cfg, retriever=retr
        )
        a.chat("the precise question")
        assert retr.queries == [("the precise question", 3)]

    def test_rag_passed_to_backend_prompt(self) -> None:
        cfg = AssistantConfig(plan=False, rag_enabled=True, rag_k=1)
        retr = _FakeRetriever([
            _FakeRetrieved(_FakeChunk(text="THE_RAG_TOKEN", source="x.md")),
        ])
        a, backend = _make_assistant(
            [_final_answer_completion("ok")],
            config=cfg,
            retriever=retr,
        )
        a.chat("hi")
        prompt = backend.calls[0]["prompt"]
        assert "THE_RAG_TOKEN" in prompt

    def test_empty_retrieval_does_not_emit_context_block(self) -> None:
        cfg = AssistantConfig(plan=False, rag_enabled=True, rag_k=1)
        retr = _FakeRetriever([])  # empty
        a, _ = _make_assistant(
            [_final_answer_completion("ok")], config=cfg, retriever=retr
        )
        turn = a.chat("hi")
        assert turn.retrieved_context == ""
        # rag_fired False because the block was empty.
        assert turn.metadata["rag_fired"] is False

    def test_broken_retriever_raises_assistant_error(self) -> None:
        cfg = AssistantConfig(plan=False, rag_enabled=True)
        a, _ = _make_assistant(
            [_final_answer_completion("ok")],
            config=cfg,
            retriever=_BrokenRetriever(),
        )
        with pytest.raises(AssistantError, match="retriever"):
            a.chat("hi")


# ---------------------------------------------------------------------------
# Error handling — agent failures surface as data, not exceptions.
# ---------------------------------------------------------------------------


class TestAssistantChatErrorHandling:
    def test_max_steps_returns_none_final_answer(self) -> None:
        # Backend always emits an action; loop runs out of steps.
        cfg = AssistantConfig(plan=False, max_steps=2, loop_detection=False)
        completions = [
            _action_completion("calc", '{"expression": "2+2"}'),
            _action_completion("calc", '{"expression": "3+3"}'),
        ]
        registry = ToolRegistry([make_calculator()])
        a, _ = _make_assistant(
            completions, config=cfg, registry=registry
        )
        turn = a.chat("hi")
        assert turn.final_answer is None
        assert turn.agent_run.stopped_reason == "max_steps"

    def test_failed_run_records_synthetic_assistant_message(self) -> None:
        # Even though final_answer is None, the conversation gets a
        # placeholder so the next turn's history is coherent.
        cfg = AssistantConfig(plan=False, max_steps=1, loop_detection=False)
        registry = ToolRegistry([make_calculator()])
        a, _ = _make_assistant(
            [_action_completion("calc", '{"expression": "2+2"}')],
            config=cfg,
            registry=registry,
        )
        a.chat("hi")
        msgs = a.conversation.messages
        assert msgs[-1].role == "assistant"
        # The placeholder mentions the stop reason.
        assert "stopped" in msgs[-1].content.lower() or "max_steps" in msgs[-1].content

    def test_duplicate_action_returns_none_final_answer(self) -> None:
        cfg = AssistantConfig(plan=False, max_steps=4, loop_detection=True)
        completions = [
            _action_completion("calc", '{"expression": "2+2"}'),
            _action_completion("calc", '{"expression": "2+2"}'),
        ]
        registry = ToolRegistry([make_calculator()])
        a, _ = _make_assistant(
            completions, config=cfg, registry=registry
        )
        turn = a.chat("hi")
        assert turn.final_answer is None
        assert turn.agent_run.stopped_reason == "duplicate_action"

    def test_agent_run_is_threaded_through(self) -> None:
        # The AssistantTurn must carry the agent's run result intact.
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("answer")], config=cfg
        )
        turn = a.chat("hi")
        assert isinstance(turn.agent_run, AgentRunResult)
        assert turn.agent_run.final_answer == "answer"


# ---------------------------------------------------------------------------
# Eval harness.
# ---------------------------------------------------------------------------


class TestEvalCase:
    def test_construction(self) -> None:
        c = EvalCase(name="basic", question="what is 2+2?", expected_substring="4")
        assert c.name == "basic"
        assert c.question == "what is 2+2?"
        assert c.expected_substring == "4"
        assert c.expected_tool is None
        assert c.rag is None

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            EvalCase(name="", question="q")

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValueError, match="question"):
            EvalCase(name="x", question="")

    def test_with_expected_tool(self) -> None:
        c = EvalCase(name="x", question="q", expected_tool="calc")
        assert c.expected_tool == "calc"

    def test_frozen(self) -> None:
        c = EvalCase(name="x", question="q")
        with pytest.raises(Exception):
            c.name = "y"  # type: ignore[misc]


class TestEvalSuite:
    def test_passes_when_substring_matches(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("the answer is 4")] * 2, config=cfg
        )
        cases = [
            EvalCase(name="c1", question="2+2?", expected_substring="4"),
        ]
        report = run_evaluation(a, cases)
        assert report.n_passed == 1
        assert report.n_total == 1
        assert report.pass_rate == 1.0

    def test_fails_when_substring_missing(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("I don't know")] * 2, config=cfg
        )
        cases = [
            EvalCase(name="c1", question="2+2?", expected_substring="4"),
        ]
        report = run_evaluation(a, cases)
        assert report.n_failed == 1
        assert "expected substring" in report.failures[0].failure_reason.lower()

    def test_fails_on_no_final_answer(self) -> None:
        cfg = AssistantConfig(plan=False, max_steps=1, loop_detection=False)
        registry = ToolRegistry([make_calculator()])
        a, _ = _make_assistant(
            [_action_completion("calc", '{"expression": "2+2"}')] * 5,
            config=cfg,
            registry=registry,
        )
        cases = [
            EvalCase(name="c1", question="2+2?", expected_substring="4"),
        ]
        report = run_evaluation(a, cases)
        assert report.n_failed == 1
        assert "no final_answer" in report.failures[0].failure_reason.lower()

    def test_passes_when_expected_tool_called(self) -> None:
        cfg = AssistantConfig(plan=False, max_steps=4, loop_detection=False)
        registry = ToolRegistry([make_calculator()])
        completions = [
            _action_completion("calc", '{"expression": "2+2"}'),
            _final_answer_completion("4"),
        ]
        a, _ = _make_assistant(
            completions, config=cfg, registry=registry
        )
        cases = [
            EvalCase(
                name="c1",
                question="2+2?",
                expected_substring="4",
                expected_tool="calc",
            ),
        ]
        report = run_evaluation(a, cases)
        assert report.n_passed == 1

    def test_fails_when_expected_tool_not_called(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("4")], config=cfg
        )
        cases = [
            EvalCase(
                name="c1",
                question="2+2?",
                expected_substring="4",
                expected_tool="calc",
            ),
        ]
        report = run_evaluation(a, cases)
        assert report.n_failed == 1
        assert "expected tool" in report.failures[0].failure_reason.lower()

    def test_substring_check_is_case_insensitive(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("Madrid is the capital")], config=cfg
        )
        cases = [
            EvalCase(name="c1", question="capital?", expected_substring="MADRID"),
        ]
        report = run_evaluation(a, cases)
        assert report.n_passed == 1

    def test_reset_each_isolates_cases(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion(f"a{i}") for i in range(3)], config=cfg
        )
        cases = [
            EvalCase(name=f"c{i}", question=f"q{i}", expected_substring=f"a{i}")
            for i in range(3)
        ]
        report = run_evaluation(a, cases, reset_each=True)
        assert report.n_passed == 3
        # After reset_each, each case sees an empty conversation.
        # The final assistant state has only the LAST case's exchange.
        assert len(a.conversation) == 2

    def test_reset_each_false_threads_state(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion(f"a{i}") for i in range(3)], config=cfg
        )
        cases = [
            EvalCase(name=f"c{i}", question=f"q{i}", expected_substring=f"a{i}")
            for i in range(3)
        ]
        run_evaluation(a, cases, reset_each=False)
        # All 3 turns accumulated in the same conversation.
        assert len(a.conversation) == 6

    def test_report_summary_includes_pass_count(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("42")] * 3, config=cfg
        )
        cases = [
            EvalCase(name="hit", question="q", expected_substring="42"),
            EvalCase(name="miss", question="q", expected_substring="not-here"),
        ]
        report = run_evaluation(a, cases)
        s = report.summary()
        assert "1/2" in s

    def test_report_failures_only_returns_failed(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("42")] * 3, config=cfg
        )
        cases = [
            EvalCase(name="hit", question="q", expected_substring="42"),
            EvalCase(name="miss", question="q", expected_substring="zzz"),
        ]
        report = run_evaluation(a, cases)
        fails = report.failures
        assert len(fails) == 1
        assert fails[0].case.name == "miss"

    def test_empty_cases_returns_empty_report(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant([], config=cfg)
        report = run_evaluation(a, [])
        assert report.n_total == 0
        assert report.pass_rate == 0.0


class TestEvalReport:
    def test_pass_rate_zero_when_empty(self) -> None:
        report = EvalReport()
        assert report.pass_rate == 0.0

    def test_n_failed_calculated(self) -> None:
        # Build a bare-bones report by hand.
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("yes")] * 2, config=cfg
        )
        turn = a.chat("hi")
        results = [
            EvalCaseResult(
                case=EvalCase(name="c1", question="q"),
                passed=True,
                final_answer="yes",
                failure_reason=None,
                turn=turn,
            ),
            EvalCaseResult(
                case=EvalCase(name="c2", question="q"),
                passed=False,
                final_answer="yes",
                failure_reason="missed",
                turn=turn,
            ),
        ]
        report = EvalReport(results=results)
        assert report.n_total == 2
        assert report.n_passed == 1
        assert report.n_failed == 1
        assert report.pass_rate == 0.5


# ---------------------------------------------------------------------------
# CLI loop.
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help_command(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant([], config=cfg)
        inp = io.StringIO("/help\n\n")
        out = io.StringIO()
        run_cli(a, inp=inp, out=out)
        text = out.getvalue()
        assert "/help" in text
        assert "/clear" in text

    def test_clear_command(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("hi")], config=cfg
        )
        a.conversation.add_user("orphan")
        inp = io.StringIO("/clear\n\n")
        out = io.StringIO()
        run_cli(a, inp=inp, out=out)
        assert len(a.conversation) == 0
        assert "cleared" in out.getvalue().lower()

    def test_blank_line_exits(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant([], config=cfg)
        inp = io.StringIO("\n")
        out = io.StringIO()
        run_cli(a, inp=inp, out=out)
        # Doesn't crash; doesn't loop forever; conversation untouched.
        assert len(a.conversation) == 0

    def test_exit_command(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant([], config=cfg)
        inp = io.StringIO("/exit\n")
        out = io.StringIO()
        run_cli(a, inp=inp, out=out)
        assert "exiting" in out.getvalue().lower()

    def test_unknown_command_continues_loop(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant([], config=cfg)
        inp = io.StringIO("/bogus\n\n")
        out = io.StringIO()
        run_cli(a, inp=inp, out=out)
        assert "unknown command" in out.getvalue().lower()

    def test_message_drives_chat(self) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("hello")], config=cfg
        )
        inp = io.StringIO("hi\n\n")
        out = io.StringIO()
        run_cli(a, inp=inp, out=out)
        text = out.getvalue()
        assert "hello" in text
        # Check the conversation grew.
        assert len(a.conversation) == 2

    def test_save_writes_transcript(self, tmp_path: Path) -> None:
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("response")], config=cfg
        )
        save_path = tmp_path / "transcript.json"
        # First chat, then save, then exit.
        cmds = f"hi\n/save {save_path}\n\n"
        inp = io.StringIO(cmds)
        out = io.StringIO()
        run_cli(a, inp=inp, out=out)
        assert save_path.exists()
        data = json.loads(save_path.read_text())
        assert data["name"] == cfg.name
        assert data["messages"][0]["content"] == "hi"
        assert data["messages"][1]["content"] == "response"

    def test_chat_error_caught_does_not_kill_loop(self) -> None:
        # If the assistant.chat raises (e.g., empty message), the CLI
        # prints the error and continues. In our test, send a valid
        # message after the bad one to confirm.
        cfg = AssistantConfig(plan=False)
        a, _ = _make_assistant(
            [_final_answer_completion("ok")], config=cfg
        )
        # We can't cleanly generate a chat error with valid input, but
        # we CAN start the assistant with a depleted backend so the
        # *agent* raises, and verify the CLI catches it.
        cfg2 = AssistantConfig(plan=False)
        backend = _FakeBackend([])  # empty
        registry = ToolRegistry()
        a2 = Assistant(backend, registry, config=cfg2)
        inp = io.StringIO("hi\nbye\n\n")
        out = io.StringIO()
        run_cli(a2, inp=inp, out=out)
        text = out.getvalue()
        assert "(error:" in text  # CLI caught the error gracefully

    def test_help_string_constant(self) -> None:
        assert "/help" in CLI_HELP
        assert "/exit" in CLI_HELP
        assert "/save" in CLI_HELP


# ---------------------------------------------------------------------------
# Integration smoke — full pipeline with a real calculator.
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    def test_calculator_through_assistant(self) -> None:
        # Plan → Action calc(2+2) → Final Answer "4".
        # Exercises the full Assistant → Agent → ToolRegistry → Backend
        # integration. NOTE: depends on Module 18's `validate_arguments`
        # and `calculator_evaluate` being implemented.
        cfg = AssistantConfig(plan=False, max_steps=4)
        registry = ToolRegistry([make_calculator()])
        completions = [
            _action_completion("calculator", '{"expression": "2+2"}'),
            _final_answer_completion("4"),
        ]
        a, _ = _make_assistant(
            completions, config=cfg, registry=registry
        )
        turn = a.chat("what's 2+2?")
        assert turn.final_answer == "4"
        assert turn.metadata["n_tool_calls"] == 1
        assert any(s.action and s.action.tool == "calculator"
                   for s in turn.agent_run.steps)

    def test_assistant_handles_multi_turn_calculator(self) -> None:
        cfg = AssistantConfig(plan=False, max_steps=4)
        registry = ToolRegistry([make_calculator()])
        # First turn: 2+2 → 4. Second turn: 3+3 → 6.
        completions = [
            _action_completion("calculator", '{"expression": "2+2"}'),
            _final_answer_completion("4"),
            _action_completion("calculator", '{"expression": "3+3"}'),
            _final_answer_completion("6"),
        ]
        a, _ = _make_assistant(
            completions, config=cfg, registry=registry
        )
        t1 = a.chat("2+2?")
        t2 = a.chat("3+3?")
        assert t1.final_answer == "4"
        assert t2.final_answer == "6"
        # Turn 2's contextualized message includes turn 1's exchange.
        assert "2+2?" in t2.contextualized_message
        assert "User: 2+2?" in t2.contextualized_message


# ---------------------------------------------------------------------------
# Module exports — sanity check the public surface.
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_assistant_exported(self) -> None:
        from g2c.assistant import Assistant
        assert Assistant is not None

    def test_conversation_exported(self) -> None:
        from g2c.assistant import Conversation, Message
        assert Conversation is not None
        assert Message is not None

    def test_eval_exported(self) -> None:
        from g2c.assistant import EvalCase, EvalReport, run_evaluation
        assert EvalCase is not None
        assert EvalReport is not None
        assert run_evaluation is not None

    def test_cli_exported(self) -> None:
        from g2c.assistant import CLI_HELP, run_cli
        assert CLI_HELP is not None
        assert run_cli is not None

    def test_config_exported(self) -> None:
        from g2c.assistant import AssistantConfig, AssistantError
        assert AssistantConfig is not None
        assert AssistantError is not None
