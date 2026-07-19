"""Tests for g2c.agent — Module 19 (Agent loops).

Suggested order to implement & turn green:

    1. `parse_react_step` in `g2c/agent/parser.py`. Pure logic — regex
       + JSON shape validation, the agent's counterpart to Module 18's
       `parse_tool_calls`. Turns green:
         TestParseReactStep
         TestParseReactStepEdgeCases

    2. `extract_plan` in `g2c/agent/planner.py`. Parse a numbered plan
       out of a planning-prompt completion. Independent of the loop.
       Turns green:
         TestExtractPlan
         TestExtractPlanEdgeCases
         TestMakePlan (composition; needs extract_plan)

    3. `Scratchpad.render` in `g2c/agent/memory.py`. Format past steps
       as the next prompt's history block. Turns green:
         TestScratchpadRender
         TestScratchpadTruncation

    4. `Agent._decide_step` in `g2c/agent/agent.py`. The per-step
       policy. The `run`/`_run_loop` driver is provided; it parses each
       completion and hands the step to `_decide_step`, which decides
       act / finish / stop. Once 1-3 are done, this turns green and the
       integration smoke tests pass:
         TestDecideStep
         TestAgentRun
         TestAgentRunStopConditions
         TestAgentRunErrorRecovery
         TestAgentRunPlanning
         TestIntegrationSmoke

The four are independent — work in any order. Boilerplate tests
(`TestActionBoilerplate`, etc.) pass from the start.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import pytest

from g2c.agent import (
    DEFAULT_AGENT_SYSTEM,
    DEFAULT_PLANNING_PROMPT,
    Action,
    Agent,
    AgentError,
    AgentRunResult,
    AgentStep,
    Observation,
    ParsedStep,
    Plan,
    Scratchpad,
    extract_plan,
    make_plan,
    parse_react_step,
    render_plan_block,
    render_planning_prompt,
    render_system_prompt,
)
from g2c.inference import Backend, BackendInfo, InferenceResult
from g2c.tools import Tool, ToolError, ToolRegistry, make_calculator

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


def _echo_tool() -> Tool:
    """A trivial tool that echoes its argument back. Used for tests
    where we don't care what the tool does, just that it ran.
    """
    def _func(text: str) -> str:
        return f"echo: {text}"

    return Tool(
        name="echo",
        description="Echo the argument back.",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo."},
            },
            "required": ["text"],
        },
        func=_func,
    )


def _broken_tool() -> Tool:
    """A tool that always raises. Used to test error-recovery paths."""
    def _func(x: str) -> str:
        raise ValueError(f"intentional failure: {x}")

    return Tool(
        name="broken",
        description="Always raises ValueError.",
        parameters={
            "type": "object",
            "properties": {
                "x": {"type": "string", "description": "ignored"},
            },
            "required": ["x"],
        },
        func=_func,
    )


def _make_step(
    *,
    completion: str = "",
    thought: str = "",
    action: Action | None = None,
    observation: Observation | None = None,
    final_answer: str | None = None,
    parse_error: str | None = None,
) -> AgentStep:
    """Build an AgentStep with a minimal InferenceResult for tests."""
    info = BackendInfo(name="fake", model_id="fake-model")
    return AgentStep(
        completion=completion,
        thought=thought,
        action=action,
        observation=observation,
        final_answer=final_answer,
        parse_error=parse_error,
        inference=InferenceResult(
            prompt="prompt",
            completion=completion,
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
            backend=info,
        ),
    )


# ---------------------------------------------------------------------------
# Boilerplate / dataclass tests — pass from the start.
# ---------------------------------------------------------------------------


class TestActionBoilerplate:
    def test_construction(self) -> None:
        a = Action(tool="calculator", arguments={"expression": "2+2"})
        assert a.tool == "calculator"
        assert a.arguments == {"expression": "2+2"}

    def test_frozen(self) -> None:
        a = Action(tool="calc", arguments={})
        with pytest.raises(Exception):  # noqa: B017
            a.tool = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a1 = Action(tool="calc", arguments={"x": 1})
        a2 = Action(tool="calc", arguments={"x": 1})
        assert a1 == a2


class TestObservationBoilerplate:
    def test_default_is_not_error(self) -> None:
        o = Observation(output="ok")
        assert o.output == "ok"
        assert o.is_error is False

    def test_error_flag(self) -> None:
        o = Observation(output="boom", is_error=True)
        assert o.is_error is True


class TestPlanBoilerplate:
    def test_construction(self) -> None:
        p = Plan(goal="solve task", steps=["step a", "step b"])
        assert p.goal == "solve task"
        assert p.steps == ["step a", "step b"]

    def test_default_steps_empty(self) -> None:
        p = Plan(goal="just a goal")
        assert p.steps == []

    def test_empty_goal_rejected(self) -> None:
        with pytest.raises(AgentError):
            Plan(goal="", steps=["a"])

    def test_non_str_goal_rejected(self) -> None:
        with pytest.raises(AgentError):
            Plan(goal=123, steps=[])  # type: ignore[arg-type]

    def test_non_list_steps_rejected(self) -> None:
        with pytest.raises(AgentError):
            Plan(goal="g", steps="not a list")  # type: ignore[arg-type]

    def test_empty_step_rejected(self) -> None:
        with pytest.raises(AgentError):
            Plan(goal="g", steps=["valid", ""])

    def test_non_str_step_rejected(self) -> None:
        with pytest.raises(AgentError):
            Plan(goal="g", steps=["valid", 42])  # type: ignore[list-item]


class TestAgentStepBoilerplate:
    def test_construction(self) -> None:
        s = _make_step(thought="x", final_answer="42")
        assert s.thought == "x"
        assert s.final_answer == "42"
        assert s.action is None
        assert s.observation is None
        assert s.parse_error is None


class TestAgentRunResultBoilerplate:
    def test_construction(self) -> None:
        r = AgentRunResult(
            user_message="hi",
            plan=None,
            final_answer="hello",
            steps=[],
            stopped_reason="final_answer",
        )
        assert r.user_message == "hi"
        assert r.final_answer == "hello"
        assert r.metadata == {}

    def test_metadata_default_factory(self) -> None:
        r1 = AgentRunResult(
            user_message="a", plan=None, final_answer=None,
            steps=[], stopped_reason="max_steps",
        )
        r2 = AgentRunResult(
            user_message="b", plan=None, final_answer=None,
            steps=[], stopped_reason="max_steps",
        )
        r1.metadata["foo"] = 1
        assert "foo" not in r2.metadata


class TestParsedStepBoilerplate:
    def test_construction(self) -> None:
        p = ParsedStep(
            thought="x",
            action=Action(tool="t", arguments={}),
            final_answer=None,
            parse_error=None,
        )
        assert p.thought == "x"
        assert p.action is not None
        assert p.action.tool == "t"


# ---------------------------------------------------------------------------
# parse_react_step
# ---------------------------------------------------------------------------


class TestParseReactStep:
    def test_final_answer_only(self) -> None:
        p = parse_react_step("Final Answer: 42")
        assert p.final_answer == "42"
        assert p.action is None
        assert p.parse_error is None
        assert p.thought == ""

    def test_thought_plus_final_answer(self) -> None:
        p = parse_react_step("Thought: easy\nFinal Answer: 4")
        assert p.thought == "easy"
        assert p.final_answer == "4"
        assert p.action is None
        assert p.parse_error is None

    def test_thought_plus_action(self) -> None:
        text = (
            "Thought: I need to compute this\n"
            "Action: calculator\n"
            'Action Input: {"expression": "2+2"}'
        )
        p = parse_react_step(text)
        assert p.thought == "I need to compute this"
        assert p.action is not None
        assert p.action.tool == "calculator"
        assert p.action.arguments == {"expression": "2+2"}
        assert p.final_answer is None
        assert p.parse_error is None

    def test_action_only_no_thought(self) -> None:
        text = 'Action: echo\nAction Input: {"text": "hi"}'
        p = parse_react_step(text)
        assert p.thought == ""
        assert p.action is not None
        assert p.action.tool == "echo"
        assert p.action.arguments == {"text": "hi"}

    def test_final_answer_wins_over_action(self) -> None:
        # If both appear, Final Answer is authoritative.
        text = (
            "Thought: I have an answer\n"
            "Action: calculator\n"
            'Action Input: {"expression": "2+2"}\n'
            "Final Answer: 4"
        )
        p = parse_react_step(text)
        assert p.action is None
        assert p.final_answer == "4"
        assert p.parse_error is None

    def test_multiline_action_input(self) -> None:
        text = (
            "Action: echo\n"
            "Action Input: {\n"
            '  "text": "hi"\n'
            "}"
        )
        p = parse_react_step(text)
        assert p.action is not None
        assert p.action.arguments == {"text": "hi"}

    def test_returns_parsedstep(self) -> None:
        p = parse_react_step("Final Answer: x")
        assert isinstance(p, ParsedStep)

    def test_type_error_on_non_str(self) -> None:
        with pytest.raises(TypeError):
            parse_react_step(123)  # type: ignore[arg-type]


class TestParseReactStepEdgeCases:
    def test_no_structure_is_parse_error(self) -> None:
        p = parse_react_step("I don't know what to do.")
        assert p.action is None
        assert p.final_answer is None
        assert p.parse_error is not None

    def test_action_without_action_input_is_error(self) -> None:
        p = parse_react_step("Action: calculator")
        assert p.action is None
        assert p.parse_error is not None
        assert "Action Input" in p.parse_error or "input" in p.parse_error.lower()

    def test_action_input_invalid_json_is_error(self) -> None:
        p = parse_react_step(
            "Action: calculator\nAction Input: {not valid json}"
        )
        assert p.action is None
        assert p.parse_error is not None
        assert "JSON" in p.parse_error or "json" in p.parse_error.lower()

    def test_action_input_non_dict_is_error(self) -> None:
        p = parse_react_step("Action: calculator\nAction Input: [1, 2, 3]")
        assert p.action is None
        assert p.parse_error is not None

    def test_empty_final_answer_treated_as_no_answer(self) -> None:
        p = parse_react_step("Final Answer:   ")
        assert p.final_answer is None
        # Should be a parse error or stuck — but NOT a valid empty answer.
        assert p.action is None or p.parse_error is not None or p.final_answer is None

    def test_code_fenced_action_input_tolerated(self) -> None:
        # Some models wrap JSON in ```json ... ``` fences.
        text = (
            "Action: calculator\n"
            "Action Input:\n"
            "```json\n"
            '{"expression": "1+1"}\n'
            "```"
        )
        p = parse_react_step(text)
        assert p.action is not None
        assert p.action.arguments == {"expression": "1+1"}

    def test_trailing_period_on_action_name_stripped(self) -> None:
        p = parse_react_step(
            'Action: calculator.\nAction Input: {"expression": "2+2"}'
        )
        assert p.action is not None
        assert p.action.tool == "calculator"

    def test_case_insensitive_markers(self) -> None:
        # Models occasionally lowercase or vary the case of markers.
        p = parse_react_step("thought: hmm\nfinal answer: 7")
        assert p.thought == "hmm"
        assert p.final_answer == "7"

    def test_extra_whitespace_around_colons(self) -> None:
        p = parse_react_step("Thought : hello\nFinal Answer  :  world")
        assert p.thought == "hello"
        assert p.final_answer == "world"

    def test_thought_does_not_eat_next_marker(self) -> None:
        text = (
            "Thought: first\n"
            "Action: echo\n"
            'Action Input: {"text": "x"}'
        )
        p = parse_react_step(text)
        assert p.thought == "first"
        assert p.action is not None

    def test_prose_around_markers_tolerated(self) -> None:
        # The model adds prose before the structured part.
        text = (
            "Let me think about this.\n"
            "Thought: I should call calculator.\n"
            "Action: calculator\n"
            'Action Input: {"expression": "1+1"}'
        )
        p = parse_react_step(text)
        assert p.action is not None
        assert p.action.tool == "calculator"

    def test_empty_action_name_is_error(self) -> None:
        p = parse_react_step("Action:   \nAction Input: {}")
        assert p.action is None
        assert p.parse_error is not None


# ---------------------------------------------------------------------------
# extract_plan
# ---------------------------------------------------------------------------


class TestExtractPlan:
    def test_goal_plus_three_steps(self) -> None:
        text = (
            "Goal: solve the task\n"
            "1. read the file\n"
            "2. extract numbers\n"
            "3. compute the sum"
        )
        p = extract_plan(text, "user task")
        assert p is not None
        assert p.goal == "solve the task"
        assert p.steps == ["read the file", "extract numbers", "compute the sum"]

    def test_no_goal_falls_back_to_user_message(self) -> None:
        text = "1. step one\n2. step two"
        p = extract_plan(text, "the original task")
        assert p is not None
        assert p.goal == "the original task"
        assert p.steps == ["step one", "step two"]

    def test_no_steps_returns_none(self) -> None:
        text = "Goal: solve everything\nThis is just prose, no list."
        p = extract_plan(text, "task")
        assert p is None

    def test_none_when_no_structure(self) -> None:
        p = extract_plan("blah blah blah", "task")
        assert p is None

    def test_returns_plan_or_none(self) -> None:
        p = extract_plan("Goal: x\n1. a", "task")
        assert isinstance(p, Plan)


class TestExtractPlanEdgeCases:
    def test_paren_style_numbering(self) -> None:
        text = "Goal: g\n1) first\n2) second"
        p = extract_plan(text, "task")
        assert p is not None
        assert p.steps == ["first", "second"]

    def test_dash_style_numbering(self) -> None:
        text = "Goal: g\n1 - first\n2 - second"
        p = extract_plan(text, "task")
        assert p is not None
        assert p.steps == ["first", "second"]

    def test_out_of_order_indices_dropped(self) -> None:
        text = "Goal: g\n1. first\n2. second\n1. RESTART\n3. third"
        p = extract_plan(text, "task")
        assert p is not None
        # Second "1." is dropped (not increasing). "3. third" comes
        # after the dropped "1. RESTART"; whether it appears depends
        # on the recipe — we just check that "RESTART" is gone.
        assert "RESTART" not in p.steps

    def test_capped_at_5_steps(self) -> None:
        text = (
            "Goal: g\n"
            "1. a\n2. b\n3. c\n4. d\n5. e\n6. f\n7. g"
        )
        p = extract_plan(text, "task")
        assert p is not None
        assert len(p.steps) == 5
        assert "f" not in p.steps and "g" not in p.steps

    def test_empty_step_body_dropped(self) -> None:
        text = "Goal: g\n1. \n2. real step"
        p = extract_plan(text, "task")
        assert p is not None
        assert p.steps == ["real step"]

    def test_type_error_on_non_str(self) -> None:
        with pytest.raises(TypeError):
            extract_plan(123, "task")  # type: ignore[arg-type]

    def test_value_error_on_empty_user_message(self) -> None:
        with pytest.raises(ValueError):
            extract_plan("Goal: g\n1. a", "")

    def test_goal_strips_whitespace(self) -> None:
        p = extract_plan("Goal:   solve   \n1. a", "task")
        assert p is not None
        assert p.goal == "solve"


class TestMakePlan:
    def test_make_plan_uses_planning_prompt(self) -> None:
        backend = _FakeBackend(["Goal: solve\n1. step one\n2. step two"])
        registry = ToolRegistry([_echo_tool()])
        p = make_plan(backend, "do something", registry)
        assert p is not None
        assert p.goal == "solve"
        assert p.steps == ["step one", "step two"]
        # Verify the backend was called with the planning prompt.
        assert len(backend.calls) == 1
        prompt = backend.calls[0]["prompt"]
        assert "do something" in prompt
        assert "Goal:" in prompt

    def test_make_plan_returns_none_when_unparseable(self) -> None:
        backend = _FakeBackend(["I don't know how to plan."])
        registry = ToolRegistry()
        p = make_plan(backend, "a task", registry)
        assert p is None

    def test_make_plan_rejects_empty_user_message(self) -> None:
        backend = _FakeBackend([])
        registry = ToolRegistry()
        with pytest.raises(ValueError):
            make_plan(backend, "", registry)

    def test_make_plan_includes_tools_in_prompt(self) -> None:
        backend = _FakeBackend(["Goal: x\n1. step"])
        registry = ToolRegistry([_echo_tool()])
        make_plan(backend, "task", registry)
        prompt = backend.calls[0]["prompt"]
        assert "echo" in prompt


# ---------------------------------------------------------------------------
# Scratchpad
# ---------------------------------------------------------------------------


class TestScratchpadConstruction:
    def test_starts_empty(self) -> None:
        sp = Scratchpad()
        assert len(sp) == 0
        assert sp.steps == []

    def test_max_chars_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            Scratchpad(max_chars=0)
        with pytest.raises(ValueError):
            Scratchpad(max_chars=-1)

    def test_max_chars_none_is_allowed(self) -> None:
        sp = Scratchpad(max_chars=None)
        assert sp._max_chars is None

    def test_append_rejects_non_step(self) -> None:
        sp = Scratchpad()
        with pytest.raises(TypeError):
            sp.append("not a step")  # type: ignore[arg-type]

    def test_repr(self) -> None:
        sp = Scratchpad()
        assert "Scratchpad" in repr(sp)
        assert "steps=0" in repr(sp)

    def test_steps_returns_snapshot(self) -> None:
        sp = Scratchpad()
        sp.append(_make_step(thought="t1"))
        snap = sp.steps
        snap.append("garbage")  # type: ignore[arg-type]
        assert len(sp) == 1


class TestScratchpadRender:
    def test_empty_renders_to_empty_string(self) -> None:
        sp = Scratchpad()
        assert sp.render() == ""

    def test_single_action_step(self) -> None:
        sp = Scratchpad()
        sp.append(_make_step(
            thought="use calc",
            action=Action(tool="calculator", arguments={"expression": "2+2"}),
            observation=Observation(output="4"),
        ))
        out = sp.render()
        assert "Thought: use calc" in out
        assert "Action: calculator" in out
        # JSON of arguments — order doesn't matter, but the keys do.
        assert '"expression"' in out
        assert '"2+2"' in out
        assert "Observation: 4" in out

    def test_final_answer_step(self) -> None:
        sp = Scratchpad()
        sp.append(_make_step(thought="done", final_answer="42"))
        out = sp.render()
        assert "Thought: done" in out
        assert "Final Answer: 42" in out
        assert "Action:" not in out

    def test_two_steps_separated_by_blank_line(self) -> None:
        sp = Scratchpad()
        sp.append(_make_step(
            thought="t1",
            action=Action(tool="echo", arguments={"text": "a"}),
            observation=Observation(output="echo: a"),
        ))
        sp.append(_make_step(
            thought="t2",
            action=Action(tool="echo", arguments={"text": "b"}),
            observation=Observation(output="echo: b"),
        ))
        out = sp.render()
        # Two blocks separated by a blank line ("\n\n").
        assert "\n\n" in out
        assert out.count("Thought:") == 2
        assert out.count("Action:") == 2
        assert out.count("Observation:") == 2

    def test_error_observation_marked(self) -> None:
        sp = Scratchpad()
        sp.append(_make_step(
            thought="try",
            action=Action(tool="broken", arguments={"x": "y"}),
            observation=Observation(output="ValueError: boom", is_error=True),
        ))
        out = sp.render()
        assert "[error]" in out
        assert "ValueError: boom" in out

    def test_parse_error_step_renders_as_observation(self) -> None:
        sp = Scratchpad()
        sp.append(_make_step(
            thought="confused",
            parse_error="no Action, Action Input, or Final Answer found",
        ))
        out = sp.render()
        # The parse error should be visible somewhere in the rendered
        # block so the model knows what went wrong.
        assert "parse error" in out.lower() or "no Action" in out

    def test_arguments_rendered_as_json_not_python_repr(self) -> None:
        sp = Scratchpad()
        sp.append(_make_step(
            thought="t",
            action=Action(tool="echo", arguments={"text": "hi"}),
            observation=Observation(output="echo: hi"),
        ))
        out = sp.render()
        # Python repr would give single quotes; JSON gives double.
        assert "{'text': 'hi'}" not in out
        assert '"text"' in out

    def test_step_without_thought_omits_thought_line(self) -> None:
        sp = Scratchpad()
        sp.append(_make_step(
            thought="",
            action=Action(tool="echo", arguments={"text": "x"}),
            observation=Observation(output="echo: x"),
        ))
        out = sp.render()
        # The Thought: line should be absent (or empty).
        assert not re.search(r"^Thought:\s*$", out, re.MULTILINE)

    def test_render_returns_string(self) -> None:
        sp = Scratchpad()
        assert isinstance(sp.render(), str)


class TestScratchpadTruncation:
    def test_under_cap_returns_full_text(self) -> None:
        sp = Scratchpad(max_chars=10000)
        for i in range(3):
            sp.append(_make_step(
                thought=f"step{i}",
                action=Action(tool="echo", arguments={"text": str(i)}),
                observation=Observation(output=f"echo: {i}"),
            ))
        out = sp.render()
        assert "step0" in out and "step1" in out and "step2" in out

    def test_over_cap_drops_earliest_blocks(self) -> None:
        # Make a small cap that forces dropping early blocks.
        sp = Scratchpad(max_chars=200)
        for i in range(10):
            sp.append(_make_step(
                thought=f"this is step {i} with some words",
                action=Action(tool="echo", arguments={"text": f"value{i}"}),
                observation=Observation(output=f"echo result for step {i}"),
            ))
        out = sp.render()
        assert len(out) <= 200 or sp.render().count("Thought:") == 1
        # The latest step should still be visible.
        assert "step 9" in out

    def test_keeps_at_least_last_block_even_if_over_cap(self) -> None:
        # Single block is bigger than the cap — we still keep it
        # rather than rendering empty.
        sp = Scratchpad(max_chars=10)
        sp.append(_make_step(
            thought="this is a long thought far exceeding the cap",
            action=Action(tool="echo", arguments={"text": "x"}),
            observation=Observation(output="long output that exceeds cap"),
        ))
        out = sp.render()
        # Rendered, even though it's over the cap.
        assert "Thought:" in out


# ---------------------------------------------------------------------------
# render_system_prompt / render_planning_prompt / render_plan_block
# ---------------------------------------------------------------------------


class TestRenderSystemPrompt:
    def test_empty_tool_list(self) -> None:
        out = render_system_prompt([])
        assert "(no tools registered)" in out
        assert "Thought:" in out
        assert "Action:" in out

    def test_includes_tool_names_and_descriptions(self) -> None:
        tools = [_echo_tool()]
        out = render_system_prompt(tools)
        assert "echo:" in out or "echo: Echo" in out
        assert "Echo the argument back" in out

    def test_lists_tool_names_in_action_hint(self) -> None:
        tools = [_echo_tool()]
        out = render_system_prompt(tools)
        # The system prompt should mention "echo" as one of the
        # allowed action names.
        assert "echo" in out

    def test_returns_string(self) -> None:
        assert isinstance(render_system_prompt([]), str)


class TestRenderPlanningPrompt:
    def test_includes_user_message(self) -> None:
        out = render_planning_prompt("a hard task", [_echo_tool()])
        assert "a hard task" in out
        assert "echo" in out

    def test_rejects_empty_user_message(self) -> None:
        with pytest.raises(ValueError):
            render_planning_prompt("", [_echo_tool()])

    def test_includes_format_template(self) -> None:
        out = render_planning_prompt("task", [])
        assert "Goal:" in out
        # Numbered list format prompt.
        assert "1." in out


class TestRenderPlanBlock:
    def test_basic(self) -> None:
        out = render_plan_block("solve x", ["a", "b"])
        assert "Plan:" in out
        assert "Goal: solve x" in out
        assert "1. a" in out
        assert "2. b" in out

    def test_empty_steps(self) -> None:
        out = render_plan_block("just a goal", [])
        assert "Goal: just a goal" in out
        assert "1." not in out


class TestModuleConstants:
    def test_default_agent_system_is_a_string(self) -> None:
        assert isinstance(DEFAULT_AGENT_SYSTEM, str)
        assert len(DEFAULT_AGENT_SYSTEM) > 100  # non-trivial

    def test_default_planning_prompt_is_a_string(self) -> None:
        assert isinstance(DEFAULT_PLANNING_PROMPT, str)
        assert "Goal:" in DEFAULT_PLANNING_PROMPT


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


class TestAgentConstruction:
    def test_basic_construction(self) -> None:
        backend = _FakeBackend([])
        registry = ToolRegistry()
        agent = Agent(backend, registry, plan=False)
        assert agent.backend is backend
        assert agent.registry is registry
        assert agent.max_steps == 8

    def test_rejects_non_backend(self) -> None:
        with pytest.raises(AgentError):
            Agent("not a backend", ToolRegistry(), plan=False)  # type: ignore[arg-type]

    def test_rejects_non_registry(self) -> None:
        with pytest.raises(AgentError):
            Agent(_FakeBackend([]), "not a registry", plan=False)  # type: ignore[arg-type]

    def test_rejects_zero_max_steps(self) -> None:
        with pytest.raises(AgentError):
            Agent(_FakeBackend([]), ToolRegistry(), max_steps=0, plan=False)

    def test_rejects_negative_max_steps(self) -> None:
        with pytest.raises(AgentError):
            Agent(_FakeBackend([]), ToolRegistry(), max_steps=-1, plan=False)

    def test_rejects_zero_scratchpad_max_chars(self) -> None:
        with pytest.raises(AgentError):
            Agent(
                _FakeBackend([]), ToolRegistry(),
                scratchpad_max_chars=0, plan=False,
            )

    def test_repr(self) -> None:
        agent = Agent(_FakeBackend([]), ToolRegistry(), plan=False)
        assert "Agent" in repr(agent)
        assert "fake" in repr(agent)


# ---------------------------------------------------------------------------
# Agent.run — happy path
# ---------------------------------------------------------------------------


class TestAgentRun:
    def test_one_shot_final_answer(self) -> None:
        backend = _FakeBackend(["Final Answer: 42"])
        agent = Agent(backend, ToolRegistry(), plan=False, max_steps=3)
        result = agent.run("What is the meaning of life?")
        assert result.final_answer == "42"
        assert result.stopped_reason == "final_answer"
        assert len(result.steps) == 1
        assert result.steps[0].final_answer == "42"
        assert result.steps[0].action is None

    def test_returns_agent_run_result(self) -> None:
        backend = _FakeBackend(["Final Answer: x"])
        agent = Agent(backend, ToolRegistry(), plan=False)
        result = agent.run("anything")
        assert isinstance(result, AgentRunResult)

    def test_user_message_recorded(self) -> None:
        backend = _FakeBackend(["Final Answer: x"])
        agent = Agent(backend, ToolRegistry(), plan=False)
        result = agent.run("the original question")
        assert result.user_message == "the original question"

    def test_rejects_empty_user_message(self) -> None:
        backend = _FakeBackend([])
        agent = Agent(backend, ToolRegistry(), plan=False)
        with pytest.raises(ValueError):
            agent.run("")

    def test_rejects_non_string_user_message(self) -> None:
        backend = _FakeBackend([])
        agent = Agent(backend, ToolRegistry(), plan=False)
        with pytest.raises(ValueError):
            agent.run(123)  # type: ignore[arg-type]

    def test_metadata_populated(self) -> None:
        backend = _FakeBackend(["Final Answer: ok"])
        agent = Agent(backend, ToolRegistry([_echo_tool()]), plan=False)
        result = agent.run("hi")
        assert result.metadata["n_steps"] >= 1
        assert result.metadata["backend_name"] == "fake"
        assert result.metadata["backend_model_id"] == "fake-model"
        assert "echo" in result.metadata["tools_available"]
        assert result.metadata["had_plan"] is False

    def test_two_step_action_then_final(self) -> None:
        # Turn 1: call echo. Turn 2: final answer.
        backend = _FakeBackend([
            'Thought: use echo\nAction: echo\nAction Input: {"text": "hi"}',
            "Thought: done\nFinal Answer: echo: hi",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=4,
        )
        result = agent.run("echo hi")
        assert result.final_answer == "echo: hi"
        assert result.stopped_reason == "final_answer"
        assert len(result.steps) == 2
        assert result.steps[0].action is not None
        assert result.steps[0].action.tool == "echo"
        assert result.steps[0].observation is not None
        assert result.steps[0].observation.output == "echo: hi"
        assert result.steps[0].observation.is_error is False
        assert result.steps[1].final_answer == "echo: hi"

    def test_n_tool_calls_in_metadata(self) -> None:
        backend = _FakeBackend([
            'Thought: a\nAction: echo\nAction Input: {"text": "1"}',
            'Thought: b\nAction: echo\nAction Input: {"text": "2"}',
            "Final Answer: done",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=5,
        )
        result = agent.run("x")
        assert result.metadata["n_tool_calls"] == 2

    def test_scratchpad_grows_into_prompt(self) -> None:
        # On step 2, the prompt should contain step 1's tool result.
        backend = _FakeBackend([
            'Action: echo\nAction Input: {"text": "first"}',
            "Final Answer: done",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=4,
        )
        agent.run("test")
        assert len(backend.calls) == 2
        prompt2 = backend.calls[1]["prompt"]
        # Step 1's observation should be in the prompt.
        assert "echo: first" in prompt2

    def test_prompt_ends_with_thought_marker(self) -> None:
        backend = _FakeBackend(["Final Answer: x"])
        agent = Agent(backend, ToolRegistry(), plan=False)
        agent.run("q")
        # The prompt should nudge the model toward the ReAct format.
        prompt = backend.calls[0]["prompt"]
        assert prompt.rstrip().endswith("Thought:")


# ---------------------------------------------------------------------------
# Agent.run — stop conditions
# ---------------------------------------------------------------------------


class TestAgentRunStopConditions:
    def test_max_steps_when_no_final_answer(self) -> None:
        # Backend keeps emitting actions; max_steps trips.
        backend = _FakeBackend([
            'Action: echo\nAction Input: {"text": "1"}',
            'Action: echo\nAction Input: {"text": "2"}',
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=2, loop_detection=False,
        )
        result = agent.run("x")
        assert result.final_answer is None
        assert result.stopped_reason == "max_steps"
        assert len(result.steps) == 2

    def test_loop_detection_stops_on_duplicate_action(self) -> None:
        # Same action with same args twice — loop detection should fire.
        backend = _FakeBackend([
            'Action: echo\nAction Input: {"text": "same"}',
            'Action: echo\nAction Input: {"text": "same"}',
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=10, loop_detection=True,
        )
        result = agent.run("x")
        assert result.stopped_reason == "duplicate_action"
        assert len(result.steps) == 2

    def test_loop_detection_off_allows_repeats(self) -> None:
        backend = _FakeBackend([
            'Action: echo\nAction Input: {"text": "same"}',
            'Action: echo\nAction Input: {"text": "same"}',
            "Final Answer: ok",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=5, loop_detection=False,
        )
        result = agent.run("x")
        assert result.final_answer == "ok"
        assert result.stopped_reason == "final_answer"

    def test_loop_detection_distinguishes_different_args(self) -> None:
        # Same tool, DIFFERENT args. Should NOT trigger.
        backend = _FakeBackend([
            'Action: echo\nAction Input: {"text": "a"}',
            'Action: echo\nAction Input: {"text": "b"}',
            "Final Answer: ok",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=5, loop_detection=True,
        )
        result = agent.run("x")
        assert result.stopped_reason == "final_answer"

    def test_halt_on_stuck(self) -> None:
        # Model emits prose with no structure — stuck step.
        backend = _FakeBackend(["I have no idea."])
        agent = Agent(
            backend, ToolRegistry(), plan=False,
            max_steps=5, halt_on_stuck=True,
        )
        result = agent.run("x")
        assert result.stopped_reason == "no_progress"
        assert result.final_answer is None
        assert len(result.steps) == 1
        assert result.steps[0].parse_error is not None

    def test_default_does_not_halt_on_stuck(self) -> None:
        # Two stuck steps then a final answer — should recover.
        backend = _FakeBackend([
            "I have no idea.",
            "Final Answer: figured it out",
        ])
        agent = Agent(
            backend, ToolRegistry(), plan=False,
            max_steps=5, halt_on_stuck=False,
        )
        result = agent.run("x")
        assert result.final_answer == "figured it out"
        assert result.stopped_reason == "final_answer"
        assert len(result.steps) == 2
        assert result.steps[0].parse_error is not None


# ---------------------------------------------------------------------------
# Agent.run — error recovery
# ---------------------------------------------------------------------------


class TestAgentRunErrorRecovery:
    def test_unknown_tool_surfaces_as_observation_error(self) -> None:
        backend = _FakeBackend([
            'Action: nonexistent\nAction Input: {"x": "y"}',
            "Final Answer: gave up",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=5,
        )
        result = agent.run("x")
        assert result.steps[0].observation is not None
        assert result.steps[0].observation.is_error is True
        # The model should have seen the error on step 2's prompt.
        assert "nonexistent" in backend.calls[1]["prompt"]

    def test_tool_runtime_exception_surfaces(self) -> None:
        backend = _FakeBackend([
            'Action: broken\nAction Input: {"x": "y"}',
            "Final Answer: oh well",
        ])
        agent = Agent(
            backend, ToolRegistry([_broken_tool()]),
            plan=False, max_steps=5,
        )
        result = agent.run("x")
        assert result.steps[0].observation is not None
        assert result.steps[0].observation.is_error is True
        assert "intentional failure" in result.steps[0].observation.output

    def test_bad_args_surfaces_as_error(self) -> None:
        # echo requires "text" but we send something else.
        backend = _FakeBackend([
            'Action: echo\nAction Input: {"wrong_key": "x"}',
            "Final Answer: gave up",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=5,
        )
        result = agent.run("x")
        assert result.steps[0].observation is not None
        assert result.steps[0].observation.is_error is True

    def test_loop_continues_after_tool_error(self) -> None:
        backend = _FakeBackend([
            'Action: nonexistent\nAction Input: {"x": "y"}',
            'Action: echo\nAction Input: {"text": "now I get it"}',
            "Final Answer: now I get it",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=5,
        )
        result = agent.run("x")
        assert result.final_answer == "now I get it"
        assert len(result.steps) == 3

    def test_error_observation_marked_in_subsequent_prompt(self) -> None:
        backend = _FakeBackend([
            'Action: nonexistent\nAction Input: {"x": "y"}',
            "Final Answer: ok",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=5,
        )
        agent.run("x")
        # Step 2's prompt should mark the error observation as an error.
        prompt2 = backend.calls[1]["prompt"]
        assert "[error]" in prompt2

    def test_parse_error_step_does_not_crash(self) -> None:
        backend = _FakeBackend([
            "Action: echo",  # Action without Action Input
            "Final Answer: recovered",
        ])
        agent = Agent(
            backend, ToolRegistry([_echo_tool()]),
            plan=False, max_steps=5,
        )
        result = agent.run("x")
        assert result.final_answer == "recovered"
        assert result.steps[0].parse_error is not None


# ---------------------------------------------------------------------------
# Agent.run — planning phase
# ---------------------------------------------------------------------------


class TestAgentRunPlanning:
    def test_plan_called_when_enabled(self) -> None:
        # First call is the planning phase, second is the main loop's step.
        backend = _FakeBackend([
            "Goal: solve\n1. think\n2. answer",
            "Final Answer: done",
        ])
        agent = Agent(
            backend, ToolRegistry(),
            plan=True, max_steps=3,
        )
        result = agent.run("a task")
        assert result.plan is not None
        assert result.plan.goal == "solve"
        assert result.plan.steps == ["think", "answer"]
        assert result.metadata["had_plan"] is True

    def test_plan_skipped_when_disabled(self) -> None:
        backend = _FakeBackend(["Final Answer: ok"])
        agent = Agent(
            backend, ToolRegistry(),
            plan=False, max_steps=3,
        )
        result = agent.run("a task")
        assert result.plan is None
        assert len(backend.calls) == 1  # just the main loop, no planning call.
        assert result.metadata["had_plan"] is False

    def test_plan_unparseable_falls_through_gracefully(self) -> None:
        # First call (planning) returns nonsense, second (main loop) succeeds.
        backend = _FakeBackend([
            "no structured plan here",
            "Final Answer: still worked",
        ])
        agent = Agent(
            backend, ToolRegistry(),
            plan=True, max_steps=3,
        )
        result = agent.run("a task")
        assert result.plan is None
        assert result.final_answer == "still worked"

    def test_plan_appears_in_main_prompt(self) -> None:
        backend = _FakeBackend([
            "Goal: do x\n1. do x first",
            "Final Answer: done",
        ])
        agent = Agent(
            backend, ToolRegistry(),
            plan=True, max_steps=3,
        )
        agent.run("a task")
        # The second backend call (the main loop's first step) should
        # have the plan in its prompt.
        main_prompt = backend.calls[1]["prompt"]
        assert "Plan:" in main_prompt
        assert "do x first" in main_prompt


# ---------------------------------------------------------------------------
# _decide_step — the pure per-step policy, tested in isolation
# ---------------------------------------------------------------------------


def _inf(completion: str = "") -> InferenceResult:
    """Minimal InferenceResult for driving `_decide_step` directly."""
    return InferenceResult(
        prompt="prompt",
        completion=completion,
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
        backend=BackendInfo(name="fake", model_id="fake-model"),
    )


def _policy_agent(**kwargs: Any) -> Agent:
    """Agent wired with a calculator; backend is never called by the policy."""
    return Agent(
        _FakeBackend([]),
        ToolRegistry([make_calculator()]),
        plan=False,
        **kwargs,
    )


class TestDecideStep:
    """The policy is pure: feed it a ParsedStep, assert the StepOutcome.

    No loop, no scratchpad, no backend completion — these exercise the
    Module-19 decision logic directly, which is only possible because
    `_decide_step` returns a value instead of mutating loop state.
    """

    def test_final_answer_stops_and_is_not_remembered(self) -> None:
        agent = _policy_agent()
        parsed = ParsedStep(thought="done", action=None,
                            final_answer="42", parse_error=None)
        outcome = agent._decide_step(parsed, _inf("Final Answer: 42"), [])
        assert outcome.stop_reason == "final_answer"
        assert outcome.remember is False
        assert outcome.step.final_answer == "42"
        assert outcome.step.action is None

    def test_action_dispatches_observes_and_continues(self) -> None:
        agent = _policy_agent()
        action = Action(tool="calculator", arguments={"expression": "2 + 2"})
        parsed = ParsedStep(thought="compute", action=action,
                            final_answer=None, parse_error=None)
        outcome = agent._decide_step(parsed, _inf(), [])
        assert outcome.stop_reason is None
        assert outcome.remember is True
        assert outcome.step.action == action
        assert outcome.step.observation is not None
        assert outcome.step.observation.is_error is False
        assert "4" in outcome.step.observation.output

    def test_repeat_action_with_loop_detection_stops(self) -> None:
        agent = _policy_agent(loop_detection=True)
        action = Action(tool="calculator", arguments={"expression": "1 + 1"})
        prior = _make_step(action=action,
                           observation=Observation(output="2"))
        parsed = ParsedStep(thought="again", action=action,
                            final_answer=None, parse_error=None)
        outcome = agent._decide_step(parsed, _inf(), [prior])
        assert outcome.stop_reason == "duplicate_action"
        assert outcome.remember is True  # the duplicate is still recorded

    def test_repeat_action_without_loop_detection_continues(self) -> None:
        agent = _policy_agent(loop_detection=False)
        action = Action(tool="calculator", arguments={"expression": "1 + 1"})
        prior = _make_step(action=action,
                           observation=Observation(output="2"))
        parsed = ParsedStep(thought="again", action=action,
                            final_answer=None, parse_error=None)
        outcome = agent._decide_step(parsed, _inf(), [prior])
        assert outcome.stop_reason is None

    def test_loop_detection_scans_past_stuck_step(self) -> None:
        # A, stuck, A -> the second A is still a duplicate of the first.
        agent = _policy_agent(loop_detection=True)
        action = Action(tool="calculator", arguments={"expression": "1 + 1"})
        a_step = _make_step(action=action, observation=Observation(output="2"))
        stuck_step = _make_step(parse_error="no parse")
        parsed = ParsedStep(thought="again", action=action,
                            final_answer=None, parse_error=None)
        outcome = agent._decide_step(parsed, _inf(), [a_step, stuck_step])
        assert outcome.stop_reason == "duplicate_action"

    def test_stuck_with_halt_stops_and_is_not_remembered(self) -> None:
        agent = _policy_agent(halt_on_stuck=True)
        parsed = ParsedStep(thought="", action=None,
                            final_answer=None, parse_error="bad json")
        outcome = agent._decide_step(parsed, _inf(), [])
        assert outcome.stop_reason == "no_progress"
        assert outcome.remember is False
        assert outcome.step.parse_error == "bad json"

    def test_stuck_without_halt_continues_and_remembers(self) -> None:
        agent = _policy_agent(halt_on_stuck=False)
        parsed = ParsedStep(thought="", action=None,
                            final_answer=None, parse_error="bad json")
        outcome = agent._decide_step(parsed, _inf(), [])
        assert outcome.stop_reason is None
        assert outcome.remember is True

    def test_stuck_falls_back_to_no_parse_label(self) -> None:
        agent = _policy_agent(halt_on_stuck=True)
        parsed = ParsedStep(thought="", action=None,
                            final_answer=None, parse_error=None)
        outcome = agent._decide_step(parsed, _inf(), [])
        assert outcome.step.parse_error == "no parse"

    def test_unknown_tool_observation_is_error_and_augmented(self) -> None:
        agent = _policy_agent()
        action = Action(tool="nonexistent", arguments={})
        parsed = ParsedStep(thought="try", action=action,
                            final_answer=None, parse_error=None)
        outcome = agent._decide_step(parsed, _inf(), [])
        assert outcome.step.observation.is_error is True
        assert "calculator" in outcome.step.observation.output
        assert outcome.stop_reason is None

    def test_policy_does_not_mutate_the_steps_list(self) -> None:
        # The whole point of the pure design: the policy decides, the
        # driver records. _decide_step must not append to `steps`.
        agent = _policy_agent()
        parsed = ParsedStep(thought="done", action=None,
                            final_answer="x", parse_error=None)
        steps: list[AgentStep] = []
        agent._decide_step(parsed, _inf(), steps)
        assert steps == []


# ---------------------------------------------------------------------------
# Integration smoke
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    def test_calculator_through_agent(self) -> None:
        backend = _FakeBackend([
            'Thought: I need to compute it.\n'
            'Action: calculator\n'
            'Action Input: {"expression": "21 * 2"}',
            "Thought: I have the answer.\nFinal Answer: The result is 42.",
        ])
        agent = Agent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=3,
        )
        result = agent.run("What is 21 times 2?")
        assert result.final_answer == "The result is 42."
        assert len(result.steps) == 2
        assert result.steps[0].action is not None
        assert result.steps[0].action.tool == "calculator"
        assert result.steps[0].observation is not None
        assert result.steps[0].observation.is_error is False
        assert result.steps[0].observation.output == "42"

    def test_calculator_recovers_from_bad_expression(self) -> None:
        # Step 1: bad expression — tool returns is_error.
        # Step 2: corrected.
        # Step 3: final answer.
        backend = _FakeBackend([
            'Action: calculator\nAction Input: {"expression": "abc"}',
            'Action: calculator\nAction Input: {"expression": "1+1"}',
            "Final Answer: 2",
        ])
        agent = Agent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=5,
        )
        result = agent.run("compute 1+1")
        assert result.final_answer == "2"
        assert result.steps[0].observation is not None
        assert result.steps[0].observation.is_error is True
        assert result.steps[1].observation is not None
        assert result.steps[1].observation.is_error is False
        assert result.steps[1].observation.output == "2"


# ---------------------------------------------------------------------------
# NativeAgent — structured-tool-calling channel.
# ---------------------------------------------------------------------------


class _FakeChatBackend(Backend):
    """Backend that satisfies `chat_with_tools` with canned responses.

    Each entry in `responses` is `(content: str, tool_calls: list[dict])`.
    `tool_calls` items are `{"name": str, "arguments": dict}`; the fake
    will add a `call_id` if absent (mirroring real Ollama behavior).
    """

    def __init__(
        self,
        responses: Iterable[tuple[str, list[dict[str, Any]]]],
        *,
        info: BackendInfo | None = None,
    ) -> None:
        self._responses = list(responses)
        self._info = info or BackendInfo(name="fake-chat", model_id="fake-chat-model")
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
        # Planning phase (Agent.run-style) uses complete; emit a
        # syntactically-valid empty plan so callers can also test the
        # planning path without needing both fakes.
        completion = "Goal: do the thing.\n1. step one"
        return InferenceResult(
            prompt=prompt,
            completion=completion,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(completion.split()),
            latency_ms=1.0,
            backend=self._info,
        )

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_k: int | None = None,
        top_p: float | None = None,
        think: bool | None = None,
    ):
        from g2c.inference import ChatResult

        if not self._responses:
            raise AssertionError(
                f"FakeChatBackend exhausted: last messages={messages!r}"
            )
        content, raw_calls = self._responses.pop(0)
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "think": think,
            }
        )
        parsed: list[dict[str, Any]] = []
        for i, raw in enumerate(raw_calls):
            parsed.append(
                {
                    "name": raw["name"],
                    "arguments": raw.get("arguments", {}),
                    "call_id": raw.get("call_id", f"call_{i}_fake"),
                }
            )
        return ChatResult(
            messages=messages,
            content=content,
            tool_calls=parsed,
            prompt_tokens=len(str(messages)),
            completion_tokens=len(content.split()) if content else 0,
            latency_ms=1.0,
            backend=self._info,
        )


class TestNativeAgentConstruction:
    def test_rejects_backend_without_chat_with_tools(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeBackend(["irrelevant"])
        with pytest.raises(AgentError, match="chat_with_tools"):
            NativeAgent(backend, ToolRegistry([make_calculator()]))

    def test_accepts_chat_capable_backend(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([])
        agent = NativeAgent(backend, ToolRegistry([make_calculator()]), plan=False)
        assert isinstance(agent, NativeAgent)

    def test_rejects_bad_max_steps(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([])
        with pytest.raises(AgentError):
            NativeAgent(backend, ToolRegistry([]), max_steps=0, plan=False)


class TestNativeAgentRun:
    def test_dispatches_a_tool_call(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            ("", [{"name": "calculator", "arguments": {"expression": "12 * 9"}}]),
            ("12 * 9 is 108.", []),
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=5, temperature=0.0,
        )
        result = agent.run("what is 12*9?")
        assert result.stopped_reason == "final_answer"
        assert result.final_answer == "12 * 9 is 108."
        assert len(result.steps) == 2
        assert result.steps[0].action is not None
        assert result.steps[0].action.tool == "calculator"
        assert result.steps[0].observation is not None
        assert result.steps[0].observation.output == "108"
        assert result.steps[0].observation.is_error is False

    def test_final_answer_from_content_when_no_tool_calls(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("the answer is 42", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4, temperature=0.0,
        )
        result = agent.run("x")
        assert result.final_answer == "the answer is 42"
        assert result.stopped_reason == "final_answer"
        assert len(result.steps) == 1
        assert result.steps[0].action is None

    def test_thought_carried_from_chat_content(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            (
                "I should compute this.",
                [{"name": "calculator", "arguments": {"expression": "1+1"}}],
            ),
            ("done", []),
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4, temperature=0.0,
        )
        result = agent.run("compute 1+1")
        assert result.steps[0].thought == "I should compute this."

    def test_loop_detection_stops_on_duplicate_action(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            ("", [{"name": "calculator", "arguments": {"expression": "2+2"}}]),
            ("", [{"name": "calculator", "arguments": {"expression": "2+2"}}]),
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=5, loop_detection=True,
        )
        result = agent.run("x")
        assert result.stopped_reason == "duplicate_action"
        assert len(result.steps) == 2

    def test_max_steps_hit(self) -> None:
        from g2c.agent import NativeAgent
        # Always emit a tool call, never a final answer.
        backend = _FakeChatBackend([
            ("", [{"name": "calculator", "arguments": {"expression": str(i)}}])
            for i in range(10)
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=3, loop_detection=False,
        )
        result = agent.run("x")
        assert result.stopped_reason == "max_steps"
        assert result.final_answer is None
        assert len(result.steps) == 3

    def test_empty_response_stuck_step(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            ("", []),                  # empty everything = stuck
            ("recovered", []),         # next turn produces an answer
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4, halt_on_stuck=False,
        )
        result = agent.run("x")
        assert result.stopped_reason == "final_answer"
        assert result.final_answer == "recovered"
        assert len(result.steps) == 2
        assert result.steps[0].parse_error is not None
        assert result.steps[0].action is None
        assert result.steps[0].final_answer is None

    def test_halt_on_stuck(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4, halt_on_stuck=True,
        )
        result = agent.run("x")
        assert result.stopped_reason == "no_progress"
        assert len(result.steps) == 1

    def test_messages_grow_with_history(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            ("", [{"name": "calculator", "arguments": {"expression": "1+1"}}]),
            ("ok done", []),
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4,
        )
        agent.run("hi")
        # Turn 1: system + user.
        assert len(backend.calls[0]["messages"]) == 2
        assert backend.calls[0]["messages"][0]["role"] == "system"
        assert backend.calls[0]["messages"][1]["role"] == "user"
        # Turn 2: + assistant tool_call + tool result.
        turn2 = backend.calls[1]["messages"]
        roles = [m["role"] for m in turn2]
        assert roles == ["system", "user", "assistant", "tool"]
        assert turn2[2]["tool_calls"][0]["function"]["name"] == "calculator"
        assert turn2[3]["name"] == "calculator"
        assert turn2[3]["content"] == "2"

    def test_tools_passed_as_ollama_specs(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("hi", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]), plan=False,
        )
        agent.run("x")
        specs = backend.calls[0]["tools"]
        assert isinstance(specs, list) and len(specs) == 1
        assert specs[0]["type"] == "function"
        assert specs[0]["function"]["name"] == "calculator"

    def test_think_param_forwarded(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("hi", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, think=False,
        )
        agent.run("x")
        assert backend.calls[0]["think"] is False

    def test_think_default_none(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("hi", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False,
        )
        agent.run("x")
        assert backend.calls[0]["think"] is None

    def test_metadata_includes_channel_native(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("done", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]), plan=False,
        )
        result = agent.run("x")
        assert result.metadata["channel"] == "native"
        assert "n_tool_calls" in result.metadata
        assert "backend_name" in result.metadata

    def test_recovers_from_validation_error(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            # Step 1: wrong arg name → validation error
            ("", [{"name": "calculator", "arguments": {"expr": "1+1"}}]),
            # Step 2: model corrects → success
            ("", [{"name": "calculator", "arguments": {"expression": "1+1"}}]),
            ("the answer is 2", []),
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=5,
        )
        result = agent.run("compute 1+1")
        assert result.final_answer == "the answer is 2"
        assert result.steps[0].observation.is_error is True
        assert result.steps[1].observation.is_error is False
        assert result.steps[1].observation.output == "2"

    def test_default_system_used_when_not_overridden(self) -> None:
        from g2c.agent import NATIVE_DEFAULT_AGENT_SYSTEM, NativeAgent
        backend = _FakeChatBackend([("hi", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]), plan=False,
        )
        agent.run("x")
        system_content = backend.calls[0]["messages"][0]["content"]
        # Default system prompt should be substring of what got sent
        # (a plan may be appended after, but the default text is there).
        assert NATIVE_DEFAULT_AGENT_SYSTEM in system_content
        # And explicitly NOT the ReAct system prompt — that would
        # confuse the model about what format to emit.
        assert "Thought:" not in system_content
        assert "Action Input:" not in system_content

    def test_custom_system_override(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("ok", [])])
        agent = NativeAgent(
            backend, ToolRegistry([]),
            plan=False, system="CUSTOM SYSTEM",
        )
        agent.run("x")
        assert backend.calls[0]["messages"][0]["content"].startswith("CUSTOM SYSTEM")

    def test_rescues_tool_call_emitted_as_content(self) -> None:
        # Small models sometimes emit a tool call as JSON in content
        # rather than via the structured `tool_calls` field. The rescue
        # parser pulls the call out so the loop can dispatch it.
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            (
                '{"name": "calculator", "parameters": {"expression": "2 + 2"}}',
                [],
            ),
            ("the answer is 4", []),
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4,
        )
        result = agent.run("what is 2+2?")
        assert result.stopped_reason == "final_answer"
        assert result.final_answer == "the answer is 4"
        assert result.steps[0].action is not None
        assert result.steps[0].action.tool == "calculator"
        assert result.steps[0].observation is not None
        assert result.steps[0].observation.output == "4"
        assert result.metadata["n_rescued_calls"] == 1

    def test_rescue_uses_arguments_or_parameters_key(self) -> None:
        # Both Llama-style ("parameters") and Qwen/Hermes-style
        # ("arguments") JSON shapes should be rescued.
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            (
                '{"name": "calculator", "arguments": {"expression": "5 + 5"}}',
                [],
            ),
            ("10", []),
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4,
        )
        result = agent.run("what is 5+5?")
        assert result.steps[0].action is not None
        assert result.steps[0].observation.output == "10"

    def test_no_rescue_when_content_is_plain_text(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("the answer is 42", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4,
        )
        result = agent.run("x")
        assert result.stopped_reason == "final_answer"
        assert result.final_answer == "the answer is 42"
        assert result.metadata["n_rescued_calls"] == 0

    def test_structured_calls_preferred_over_rescue(self) -> None:
        # If the backend already returned structured tool_calls, the
        # rescue should not also trigger on the content.
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([
            (
                '{"name": "calculator", "parameters": {"expression": "999"}}',
                [{"name": "calculator", "arguments": {"expression": "1 + 1"}}],
            ),
            ("done", []),
        ])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]),
            plan=False, max_steps=4,
        )
        result = agent.run("x")
        # The structured 1+1 ran, not the rescue 999.
        assert result.steps[0].action.arguments == {"expression": "1 + 1"}
        assert result.metadata["n_rescued_calls"] == 0

    def test_metadata_includes_rescue_count(self) -> None:
        from g2c.agent import NativeAgent
        backend = _FakeChatBackend([("hi", [])])
        agent = NativeAgent(
            backend, ToolRegistry([make_calculator()]), plan=False,
        )
        result = agent.run("x")
        assert result.metadata["n_rescued_calls"] == 0


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_all_exports_are_importable(self) -> None:
        # Smoke-test the top-level public API.
        from g2c.agent import (
            DEFAULT_AGENT_SYSTEM,
            DEFAULT_PLANNING_PROMPT,
            NATIVE_DEFAULT_AGENT_SYSTEM,
            Action,
            Agent,
            AgentError,
            AgentRunResult,
            AgentStep,
            NativeAgent,
            Observation,
            ParsedStep,
            Plan,
            Scratchpad,
            extract_plan,
            make_plan,
            parse_react_step,
            render_plan_block,
            render_planning_prompt,
            render_system_prompt,
        )
        # All present by virtue of import succeeding.
        assert all(x is not None for x in [
            DEFAULT_AGENT_SYSTEM, DEFAULT_PLANNING_PROMPT,
            NATIVE_DEFAULT_AGENT_SYSTEM,
            Action, Agent, AgentError, AgentRunResult, AgentStep,
            NativeAgent, Observation, ParsedStep, Plan, Scratchpad,
            extract_plan, make_plan, parse_react_step,
            render_plan_block, render_planning_prompt, render_system_prompt,
        ])

    def test_action_uses_attribute_named_tool(self) -> None:
        # The Action dataclass field is `tool`, not `name` — different
        # from ToolCall (Module 18). Pin it so we don't accidentally
        # rename and break the parser.
        a = Action(tool="x", arguments={})
        assert a.tool == "x"

    def test_observation_is_error_default_false(self) -> None:
        o = Observation(output="ok")
        assert o.is_error is False

    def test_tool_error_is_distinct_from_agent_error(self) -> None:
        # Sanity: AgentError doesn't accidentally subclass ToolError or
        # vice versa. The two modules' exceptions are distinct.
        assert not issubclass(AgentError, ToolError)
        assert not issubclass(ToolError, AgentError)
