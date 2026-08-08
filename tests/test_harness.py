"""Tests for g2c/harness — Beyond module: agent harness engineering.

Suggested order to implement & turn green:

1. `classify_failure` — unblocks `test_classify_*`.
2. `compact_context` — unblocks `test_compact_*`. The three pinned
   guarantees: the task statement always survives, the newest events
   stay verbatim, the budget is respected.
3. `ToolRunner.execute` — unblocks `test_runner_*`. The decision
   ladder in order: dedupe, crash window, permissions, execute (with
   the tool_call event logged BEFORE execution).
4. `HarnessAgent.resume` — unblocks the crash drill
   (`test_crash_drill_*`), the module in miniature: kill the agent
   mid-task, resume from the log, finish the task, and leave the
   sandbox byte-identical to an uncrashed run.

Everything runs against a scripted fake backend and deliberately
misbehaving fake tools — no ProdLM, no network. (The loop-level tests
also require Module 18's `dispatch_tool_call` and Module 19's
`parse_react_step`, per the lesson page's prerequisites.)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from g2c.harness import (
    Budgets,
    Event,
    EventLog,
    HarnessAgent,
    Permission,
    ToolRunner,
    classify_failure,
    compact_context,
    estimate_tokens,
)
from g2c.harness.runner import UNKNOWN_OUTCOME_STATUS
from g2c.inference import Backend, BackendInfo, InferenceResult
from g2c.tools import Tool, ToolCall, ToolError, ToolRegistry, ToolResult


class _Crash(RuntimeError):
    """Injected process death."""


class _ScriptedBackend(Backend):
    """Replays a fixed list of completions; optionally dies on cue."""

    def __init__(self, completions: list[str], crash_after: int | None = None):
        self.completions = list(completions)
        self.crash_after = crash_after
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def info(self) -> BackendInfo:
        return BackendInfo("fake", "scripted")

    def complete(self, prompt, *, max_new_tokens=128, temperature=1.0,
                 top_k=None, top_p=None) -> InferenceResult:
        if self.crash_after is not None and self.calls >= self.crash_after:
            raise _Crash("injected crash")
        self.prompts.append(prompt)
        completion = self.completions[self.calls]
        self.calls += 1
        return InferenceResult(
            prompt=prompt,
            completion=completion,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=1.0,
            backend=self.info,
        )


def _sandbox_registry(sandbox: Path, counters: dict) -> ToolRegistry:
    notes = sandbox / "notes.txt"

    def append_line(text: str) -> str:
        counters["append_calls"] = counters.get("append_calls", 0) + 1
        with notes.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        return f"appended {text!r}"

    def flaky(text: str) -> str:
        counters["flaky_calls"] = counters.get("flaky_calls", 0) + 1
        if counters["flaky_calls"] <= counters.get("flaky_fail_first", 0):
            raise ToolError("connection timed out")
        return f"ok {text!r}"

    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "payload"}},
        "required": ["text"],
    }
    return ToolRegistry(
        [
            Tool("append_line", "Append a line to notes.txt", schema, append_line),
            Tool("flaky", "Sometimes times out", schema, flaky),
        ]
    )


_ACT = 'Action: append_line\nAction Input: {{"text": "{text}"}}'
_FINAL = "Final Answer: done"


# ---------------------------------------------------------------------------
# EventLog — provided plumbing, green from the start.
# ---------------------------------------------------------------------------


def test_event_log_roundtrip(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("task", {"content": "organize"})
    log.append("model_turn", {"completion": "thinking"}, call_id=None)
    events = log.replay()
    assert [e.type for e in events] == ["task", "model_turn"]
    assert [e.step for e in events] == [0, 1]
    # replay reads the FILE — a second log object sees the same history
    assert len(EventLog(log.path).replay()) == 2


# ---------------------------------------------------------------------------
# classify_failure
# ---------------------------------------------------------------------------


def _err(output: str) -> ToolResult:
    return ToolResult(call_id="c", name="t", output=output, is_error=True)


def test_classify_transient_retries():
    decision = classify_failure(_err("connection timed out after 30s"))
    assert decision.kind == "transient" and decision.retry


def test_classify_deterministic_does_not_retry():
    decision = classify_failure(_err("file not found: /tmp/x"))
    assert decision.kind == "deterministic" and not decision.retry


def test_classify_unrecognized_defaults_to_no_retry():
    decision = classify_failure(_err("zorp flubbed the grommet"))
    assert not decision.retry


def test_classify_rejects_success():
    ok = ToolResult(call_id="c", name="t", output="fine", is_error=False)
    with pytest.raises(ValueError):
        classify_failure(ok)


# ---------------------------------------------------------------------------
# compact_context
# ---------------------------------------------------------------------------


def _event(step: int, type: str, **payload) -> Event:
    return Event(step=step, type=type, payload=payload)


def _long_log() -> list[Event]:
    events = [_event(0, "task", content="organize the sandbox files by date")]
    for i in range(1, 21):
        events.append(_event(2 * i - 1, "model_turn", completion=f"thinking {i}"))
        events.append(
            _event(2 * i, "tool_result", output="x" * 300, is_error=False)
        )
    return events


def test_compact_keeps_everything_under_a_big_budget():
    events = _long_log()
    segments = compact_context(events, budget_tokens=100_000)
    assert segments[0] == "Task: organize the sandbox files by date"
    assert len(segments) == len(events)  # every event renders here


def test_compact_task_survives_any_budget():
    segments = compact_context(_long_log(), budget_tokens=30)
    assert segments[0] == "Task: organize the sandbox files by date"


def test_compact_respects_the_budget():
    budget = 400
    segments = compact_context(_long_log(), budget_tokens=budget)
    assert sum(estimate_tokens(s) for s in segments) <= budget


def test_compact_recent_events_stay_verbatim():
    events = _long_log()
    segments = compact_context(events, budget_tokens=400)
    assert segments[-1] == "Observation: " + "x" * 300


# ---------------------------------------------------------------------------
# ToolRunner
# ---------------------------------------------------------------------------


def _call(call_id: str, text: str = "alpha") -> ToolCall:
    return ToolCall(name="append_line", arguments={"text": text}, call_id=call_id)


def test_runner_executes_and_logs_intent_before_outcome(tmp_path):
    counters: dict = {}
    log = EventLog(tmp_path / "log.jsonl")
    runner = ToolRunner(_sandbox_registry(tmp_path, counters), log)
    result = runner.execute(_call("c1"))
    assert not result.is_error
    types = [e.type for e in log.replay()]
    assert types.index("tool_call") < types.index("tool_result")
    assert (tmp_path / "notes.txt").read_text() == "alpha\n"


def test_runner_dedupes_by_call_id(tmp_path):
    counters: dict = {}
    log = EventLog(tmp_path / "log.jsonl")
    runner = ToolRunner(_sandbox_registry(tmp_path, counters), log)
    first = runner.execute(_call("c1"))
    second = runner.execute(_call("c1"))
    assert counters["append_calls"] == 1  # the function ran ONCE
    assert second.output == first.output
    assert (tmp_path / "notes.txt").read_text() == "alpha\n"


def test_runner_never_reruns_a_crash_window_call(tmp_path):
    """Intent logged, outcome missing → unknown result, no execution."""
    counters: dict = {}
    log = EventLog(tmp_path / "log.jsonl")
    log.append(
        "tool_call",
        {"tool": "append_line", "arguments": {"text": "alpha"}},
        call_id="c9",
    )
    runner = ToolRunner(_sandbox_registry(tmp_path, counters), log)
    result = runner.execute(_call("c9"))
    assert result.is_error
    assert counters.get("append_calls", 0) == 0
    statuses = [
        e.payload.get("status")
        for e in log.replay()
        if e.type == "tool_result"
    ]
    assert UNKNOWN_OUTCOME_STATUS in statuses


def test_runner_does_not_duplicate_an_unrecorded_landed_effect(tmp_path):
    """The same log state can mean the side effect already landed."""
    counters: dict = {}
    log = EventLog(tmp_path / "log.jsonl")
    log.append(
        "tool_call",
        {"tool": "append_line", "arguments": {"text": "alpha"}},
        call_id="c9",
    )
    # Simulate: execute succeeded, then the process died before result logging.
    (tmp_path / "notes.txt").write_text("alpha\n", encoding="utf-8")

    runner = ToolRunner(_sandbox_registry(tmp_path, counters), log)
    result = runner.execute(_call("c9"))

    assert result.is_error and "unknown outcome" in result.output
    assert counters.get("append_calls", 0) == 0
    assert (tmp_path / "notes.txt").read_text() == "alpha\n"


def test_runner_denies_by_policy(tmp_path):
    counters: dict = {}
    log = EventLog(tmp_path / "log.jsonl")
    runner = ToolRunner(
        _sandbox_registry(tmp_path, counters),
        log,
        permissions={"append_line": Permission.DENY},
    )
    result = runner.execute(_call("c1"))
    assert result.is_error and "permission denied" in result.output
    assert counters.get("append_calls", 0) == 0


def test_runner_ask_requires_a_logged_approval(tmp_path):
    counters: dict = {}
    log = EventLog(tmp_path / "log.jsonl")
    runner = ToolRunner(
        _sandbox_registry(tmp_path, counters),
        log,
        permissions={"append_line": Permission.ASK},
    )
    refused = runner.execute(_call("c1"))
    assert refused.is_error
    log.append("approval", {"approved": True}, call_id="c2")
    approved = runner.execute(_call("c2"))
    assert not approved.is_error
    assert counters["append_calls"] == 1


# ---------------------------------------------------------------------------
# HarnessAgent end to end
# ---------------------------------------------------------------------------


def _agent(tmp_path, completions, counters, **kwargs) -> HarnessAgent:
    backend = _ScriptedBackend(completions, crash_after=kwargs.pop("crash_after", None))
    return HarnessAgent(
        backend,
        _sandbox_registry(tmp_path, counters),
        EventLog(tmp_path / "events.jsonl"),
        **kwargs,
    )


def test_agent_completes_a_two_step_task(tmp_path):
    counters: dict = {}
    agent = _agent(
        tmp_path,
        [_ACT.format(text="alpha"), _ACT.format(text="beta"), _FINAL],
        counters,
    )
    result = agent.run("write alpha then beta")
    assert result.stopped_reason == "final_answer"
    assert result.final_answer == "done"
    assert (tmp_path / "notes.txt").read_text() == "alpha\nbeta\n"
    types = [e.type for e in result.events]
    assert types[0] == "task" and "final_answer" in types


def test_agent_uses_the_injected_context_policy(tmp_path):
    counters: dict = {}
    seen: list[tuple[list[Event], int]] = []

    def policy(events: list[Event], budget: int) -> list[str]:
        seen.append((events, budget))
        return ["Task: policy-controlled prompt"]

    backend = _ScriptedBackend([_FINAL])
    agent = HarnessAgent(
        backend,
        _sandbox_registry(tmp_path, counters),
        EventLog(tmp_path / "events.jsonl"),
        budgets=Budgets(context_tokens=37),
        context_policy=policy,
    )
    agent.run("original task")

    assert seen and seen[0][1] == 37
    assert any(e.type == "task" for e in seen[0][0])
    assert "Task: policy-controlled prompt" in backend.prompts[0]
    assert "original task" not in backend.prompts[0]


def test_agent_retries_transient_failures(tmp_path):
    counters: dict = {"flaky_fail_first": 1}
    script = ['Action: flaky\nAction Input: {"text": "x"}', _FINAL]
    agent = _agent(tmp_path, script, counters)
    result = agent.run("poke the flaky tool")
    assert result.stopped_reason == "final_answer"
    assert counters["flaky_calls"] == 2  # one failure, one retry


def test_agent_stops_on_repeats(tmp_path):
    counters: dict = {}
    same = _ACT.format(text="alpha")
    agent = _agent(tmp_path, [same] * 6, counters)
    result = agent.run("loop forever")
    assert result.stopped_reason == "repeat_budget"


def test_crash_drill_resume_completes(tmp_path):
    """Kill mid-task, resume from the log, finish the task."""
    counters: dict = {}
    script = [_ACT.format(text="alpha"), _ACT.format(text="beta"), _FINAL]
    crashed = _agent(tmp_path, script, counters, crash_after=1)
    with pytest.raises(_Crash):
        crashed.run("write alpha then beta")

    # A new process: fresh backend continuing the script, same log,
    # same sandbox.
    revived = _agent(
        tmp_path, [_ACT.format(text="beta"), _FINAL], counters
    )
    result = revived.resume()
    assert result.stopped_reason == "final_answer"
    assert (tmp_path / "notes.txt").read_text() == "alpha\nbeta\n"


def test_crash_drill_no_duplicate_side_effects(tmp_path):
    """Recorded outcomes are not replayed after a later backend crash."""
    counters: dict = {}
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    ref_counters: dict = {}
    reference = _agent(
        reference_dir,
        [_ACT.format(text="alpha"), _ACT.format(text="beta"), _FINAL],
        ref_counters,
    )
    reference.run("write alpha then beta")

    crash_dir = tmp_path / "crashed"
    crash_dir.mkdir()
    crashed = _agent(
        crash_dir,
        [_ACT.format(text="alpha"), _ACT.format(text="beta"), _FINAL],
        counters,
        crash_after=2,
    )
    with pytest.raises(_Crash):
        crashed.run("write alpha then beta")
    # The resumed model sees the completed beta call in its context and
    # finishes; a model that re-issued the action would (correctly) get
    # a fresh call id and re-run it — the dedupe protects against the
    # HARNESS repeating work, not against the model choosing to.
    revived = _agent(crash_dir, [_FINAL], counters)
    revived.resume()

    assert (
        (crash_dir / "notes.txt").read_text()
        == (reference_dir / "notes.txt").read_text()
    )


def test_resume_refuses_a_fresh_or_finished_log(tmp_path):
    counters: dict = {}
    agent = _agent(tmp_path, [_FINAL], counters)
    with pytest.raises(ValueError):
        agent.resume()  # nothing to resume
    agent.run("trivial")
    with pytest.raises(ValueError):
        agent.resume()  # already finished
