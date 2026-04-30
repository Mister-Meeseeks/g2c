"""Tests for g2c.tools — Module 18 (Tool use).

Suggested order to implement & turn green:

    1. `validate_arguments` in `g2c/tools/schema.py`. Type-checks
       arguments against a JSON-schema-lite parameters dict. Pure
       logic — no external deps. Turns green:
         TestValidateArguments
         TestValidateArgumentsTypeChecks
         TestValidateArgumentsErrors
       (Also unblocks anything that goes through `dispatch_tool_call`,
       since the dispatcher calls it.)

    2. `parse_tool_calls` in `g2c/tools/parser.py`. Regex + JSON +
       shape validation. Turns green:
         TestParseToolCalls
         TestParseToolCallsEdgeCases

    3. `calculator_evaluate` in `g2c/tools/builtins.py`. AST-based
       safe arithmetic. Turns green:
         TestCalculatorEvaluate
         TestCalculatorRejection
         TestMakeCalculator

    4. `run_with_tools` in `g2c/tools/loop.py`. Orchestrates the
       call-parse-dispatch loop. Once 1-3 are done, this turns green
       and the integration smoke tests pass:
         TestRunWithTools
         TestRunWithToolsLoop
         TestRunWithToolsErrors
         TestIntegrationSmoke

The four are independent — work in any order. Boilerplate tests
(`TestToolBoilerplate`, etc.) pass from the start.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from g2c.inference import Backend, BackendInfo, InferenceResult
from g2c.tools import (
    DEFAULT_SYSTEM,
    Tool,
    ToolCall,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolRunResult,
    ToolStep,
    calculator_evaluate,
    dispatch_tool_call,
    format_tool_results,
    make_calculator,
    make_read_file,
    make_run_python,
    make_web_search,
    parse_tool_calls,
    render_tools_for_prompt,
    run_with_tools,
    validate_arguments,
)

# ---------------------------------------------------------------------------
# Test fixtures: FakeBackend that returns canned completions.
# ---------------------------------------------------------------------------


class _FakeBackend(Backend):
    """Backend that returns pre-recorded completions in order.

    Records the prompts it was called with for assertion. Each call
    pops the next completion off the list; if the list is exhausted,
    the test has called too many times and we fail loudly.
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


def _make_tool(
    *,
    name: str = "echo",
    description: str = "Echo the input",
    parameters: dict[str, Any] | None = None,
    func=None,
) -> Tool:
    """Quick Tool factory for tests."""
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "input"},
            },
            "required": ["text"],
        }
    if func is None:
        def _echo(text: str) -> str:
            return text
        func = _echo
    return Tool(name=name, description=description, parameters=parameters, func=func)


# ---------------------------------------------------------------------------
# Boilerplate: Tool, ToolCall, ToolResult dataclasses.
# ---------------------------------------------------------------------------


class TestToolBoilerplate:
    def test_construction(self):
        t = _make_tool()
        assert t.name == "echo"
        assert t.description == "Echo the input"
        assert callable(t.func)
        assert t.parameters["type"] == "object"

    def test_frozen(self):
        t = _make_tool()
        with pytest.raises((AttributeError, Exception)):
            t.name = "renamed"  # type: ignore[misc]

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name"):
            Tool(name="", description="x", parameters={}, func=lambda: None)

    def test_rejects_empty_description(self):
        with pytest.raises(ValueError, match="description"):
            Tool(name="x", description="", parameters={}, func=lambda: None)

    def test_rejects_non_dict_parameters(self):
        with pytest.raises(TypeError, match="parameters"):
            Tool(name="x", description="x", parameters="not a dict", func=lambda: None)  # type: ignore[arg-type]

    def test_rejects_non_callable_func(self):
        with pytest.raises(TypeError, match="callable"):
            Tool(name="x", description="x", parameters={}, func="not callable")  # type: ignore[arg-type]


class TestToolCallBoilerplate:
    def test_construction(self):
        c = ToolCall(name="calc", arguments={"x": 1}, call_id="call_0_abcd1234")
        assert c.name == "calc"
        assert c.arguments == {"x": 1}
        assert c.call_id == "call_0_abcd1234"

    def test_frozen(self):
        c = ToolCall(name="calc", arguments={}, call_id="x")
        with pytest.raises((AttributeError, Exception)):
            c.name = "renamed"  # type: ignore[misc]


class TestToolResultBoilerplate:
    def test_construction(self):
        r = ToolResult(call_id="x", name="calc", output="4")
        assert r.call_id == "x"
        assert r.name == "calc"
        assert r.output == "4"
        assert r.is_error is False

    def test_error_flag(self):
        r = ToolResult(call_id="x", name="calc", output="bad input", is_error=True)
        assert r.is_error is True

    def test_frozen(self):
        r = ToolResult(call_id="x", name="calc", output="4")
        with pytest.raises((AttributeError, Exception)):
            r.output = "5"  # type: ignore[misc]


class TestToolErrorBoilerplate:
    def test_is_exception(self):
        e = ToolError("msg")
        assert isinstance(e, Exception)
        assert str(e) == "msg"


class TestToolStepBoilerplate:
    def test_construction(self):
        info = BackendInfo(name="fake", model_id="m")
        ir = InferenceResult(
            prompt="p", completion="c", prompt_tokens=1, completion_tokens=1,
            latency_ms=1.0, backend=info,
        )
        step = ToolStep(
            completion="hello",
            tool_calls=[],
            tool_results=[],
            inference=ir,
        )
        assert step.completion == "hello"
        assert step.tool_calls == []
        assert step.tool_results == []
        assert step.inference is ir


class TestToolRunResultBoilerplate:
    def test_construction(self):
        r = ToolRunResult(
            user_message="hi",
            final_answer="hello",
            steps=[],
            stopped_reason="no_more_calls",
        )
        assert r.user_message == "hi"
        assert r.final_answer == "hello"
        assert r.steps == []
        assert r.stopped_reason == "no_more_calls"
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# ToolRegistry.
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_empty(self):
        r = ToolRegistry()
        assert len(r) == 0
        assert r.names() == []
        assert r.tools == []
        assert "calculator" not in r

    def test_register_one(self):
        r = ToolRegistry()
        t = _make_tool(name="calc")
        r.register(t)
        assert len(r) == 1
        assert "calc" in r
        assert r.get("calc") is t

    def test_register_via_constructor(self):
        a = _make_tool(name="a")
        b = _make_tool(name="b")
        r = ToolRegistry([a, b])
        assert r.names() == ["a", "b"]
        assert r.tools == [a, b]

    def test_register_duplicate_raises(self):
        r = ToolRegistry()
        r.register(_make_tool(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            r.register(_make_tool(name="dup"))

    def test_register_non_tool_raises(self):
        r = ToolRegistry()
        with pytest.raises(TypeError, match="Tool"):
            r.register("not a tool")  # type: ignore[arg-type]

    def test_get_unknown_raises_keyerror(self):
        r = ToolRegistry([_make_tool(name="a")])
        with pytest.raises(KeyError, match="no tool named"):
            r.get("unknown")

    def test_iteration_preserves_order(self):
        a = _make_tool(name="a")
        b = _make_tool(name="b")
        c = _make_tool(name="c")
        r = ToolRegistry([a, b, c])
        assert list(r) == [a, b, c]

    def test_repr(self):
        r = ToolRegistry([_make_tool(name="a"), _make_tool(name="b")])
        assert "a" in repr(r) and "b" in repr(r)

    def test_tools_property_returns_snapshot(self):
        r = ToolRegistry([_make_tool(name="a")])
        snap = r.tools
        snap.append(_make_tool(name="b"))
        # Mutating snapshot must not affect registry.
        assert r.names() == ["a"]


# ---------------------------------------------------------------------------
# validate_arguments — SCAFFOLDED.
# ---------------------------------------------------------------------------


class TestValidateArguments:
    """Tests for the validate_arguments scaffold once filled in."""

    def test_accepts_valid_arguments(self):
        t = _make_tool()
        out = validate_arguments(t, {"text": "hello"})
        assert out == {"text": "hello"}

    def test_returns_a_dict(self):
        t = _make_tool()
        out = validate_arguments(t, {"text": "hi"})
        assert isinstance(out, dict)

    def test_returns_copy_not_input(self):
        t = _make_tool()
        args = {"text": "hi"}
        out = validate_arguments(t, args)
        out["text"] = "modified"
        # Mutating the output must not mutate the input.
        assert args == {"text": "hi"}

    def test_empty_required_passes_with_empty_args(self):
        t = _make_tool(parameters={
            "type": "object",
            "properties": {},
            "required": [],
        })
        out = validate_arguments(t, {})
        assert out == {}

    def test_missing_required_raises(self):
        t = _make_tool()
        with pytest.raises(ToolError, match="missing required"):
            validate_arguments(t, {})

    def test_unknown_argument_raises(self):
        t = _make_tool()
        with pytest.raises(ToolError, match="unknown"):
            validate_arguments(t, {"text": "hi", "extra": "no"})

    def test_non_dict_arguments_raises(self):
        t = _make_tool()
        with pytest.raises(ToolError):
            validate_arguments(t, "not a dict")  # type: ignore[arg-type]

    def test_non_object_schema_raises(self):
        t = _make_tool(parameters={"type": "array"})
        with pytest.raises(ToolError, match="object"):
            validate_arguments(t, {})

    def test_optional_argument_omitted(self):
        # path required, max_chars optional
        t = _make_tool(parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["path"],
        })
        out = validate_arguments(t, {"path": "x"})
        assert out == {"path": "x"}

    def test_optional_argument_provided(self):
        t = _make_tool(parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_chars": {"type": "integer"},
            },
            "required": ["path"],
        })
        out = validate_arguments(t, {"path": "x", "max_chars": 100})
        assert out == {"path": "x", "max_chars": 100}


class TestValidateArgumentsTypeChecks:
    """Type-tag specific validation."""

    def _tool_for_type(self, type_tag: str) -> Tool:
        return _make_tool(parameters={
            "type": "object",
            "properties": {"v": {"type": type_tag}},
            "required": ["v"],
        })

    def test_string_accepts_str(self):
        t = self._tool_for_type("string")
        assert validate_arguments(t, {"v": "hello"}) == {"v": "hello"}

    def test_string_rejects_int(self):
        t = self._tool_for_type("string")
        with pytest.raises(ToolError, match="string"):
            validate_arguments(t, {"v": 42})

    def test_integer_accepts_int(self):
        t = self._tool_for_type("integer")
        assert validate_arguments(t, {"v": 42}) == {"v": 42}

    def test_integer_rejects_float(self):
        t = self._tool_for_type("integer")
        with pytest.raises(ToolError, match="integer"):
            validate_arguments(t, {"v": 3.14})

    def test_integer_rejects_str(self):
        t = self._tool_for_type("integer")
        with pytest.raises(ToolError, match="integer"):
            validate_arguments(t, {"v": "42"})

    def test_integer_rejects_bool(self):
        # The bool-vs-int trap: True is a valid int in Python. The
        # validator must reject it explicitly.
        t = self._tool_for_type("integer")
        with pytest.raises(ToolError, match="bool"):
            validate_arguments(t, {"v": True})

    def test_number_accepts_int(self):
        t = self._tool_for_type("number")
        assert validate_arguments(t, {"v": 42}) == {"v": 42}

    def test_number_accepts_float(self):
        t = self._tool_for_type("number")
        assert validate_arguments(t, {"v": 3.14}) == {"v": 3.14}

    def test_number_rejects_str(self):
        t = self._tool_for_type("number")
        with pytest.raises(ToolError):
            validate_arguments(t, {"v": "3.14"})

    def test_number_rejects_bool(self):
        t = self._tool_for_type("number")
        with pytest.raises(ToolError, match="bool"):
            validate_arguments(t, {"v": True})

    def test_boolean_accepts_true(self):
        t = self._tool_for_type("boolean")
        assert validate_arguments(t, {"v": True}) == {"v": True}

    def test_boolean_accepts_false(self):
        t = self._tool_for_type("boolean")
        assert validate_arguments(t, {"v": False}) == {"v": False}

    def test_boolean_rejects_int(self):
        t = self._tool_for_type("boolean")
        with pytest.raises(ToolError, match="boolean"):
            validate_arguments(t, {"v": 1})

    def test_unknown_type_tag_raises(self):
        t = self._tool_for_type("widget")
        with pytest.raises(ToolError, match="widget|type"):
            validate_arguments(t, {"v": "x"})


class TestValidateArgumentsErrors:
    def test_error_message_names_missing_key(self):
        t = _make_tool(parameters={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        })
        with pytest.raises(ToolError) as exc:
            validate_arguments(t, {"a": "x"})
        # Error message should mention the missing key.
        assert "b" in str(exc.value)

    def test_error_message_names_extra_key(self):
        t = _make_tool()
        with pytest.raises(ToolError) as exc:
            validate_arguments(t, {"text": "x", "junk": 1})
        assert "junk" in str(exc.value)

    def test_error_message_names_bad_type(self):
        t = _make_tool()
        with pytest.raises(ToolError) as exc:
            validate_arguments(t, {"text": 42})
        assert "text" in str(exc.value)


# ---------------------------------------------------------------------------
# render_tools_for_prompt.
# ---------------------------------------------------------------------------


class TestRenderToolsForPrompt:
    def test_empty(self):
        out = render_tools_for_prompt([])
        assert "none" in out.lower()
        assert "<tool_call>" in out

    def test_single_tool_includes_name_and_description(self):
        t = _make_tool(name="calc", description="A calculator")
        out = render_tools_for_prompt([t])
        assert "calc" in out
        assert "A calculator" in out

    def test_includes_call_format_marker(self):
        t = _make_tool()
        out = render_tools_for_prompt([t])
        assert "<tool_call>" in out

    def test_includes_schema_as_json(self):
        t = _make_tool()
        out = render_tools_for_prompt([t])
        # The properties dict should be JSON-stringified somewhere.
        assert "properties" in out
        assert "string" in out  # the param type tag

    def test_multiple_tools(self):
        a = _make_tool(name="a")
        b = _make_tool(name="b")
        out = render_tools_for_prompt([a, b])
        assert "a" in out and "b" in out
        # Both should appear; "a" before "b" in the output.
        assert out.index("- a:") < out.index("- b:")


# ---------------------------------------------------------------------------
# parse_tool_calls — SCAFFOLDED.
# ---------------------------------------------------------------------------


class TestParseToolCalls:
    def test_empty_string_returns_empty_list(self):
        assert parse_tool_calls("") == []

    def test_no_tool_call_returns_empty(self):
        assert parse_tool_calls("just plain text, no tools") == []

    def test_single_call(self):
        text = '<tool_call>{"name": "calc", "arguments": {"x": 1}}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "calc"
        assert calls[0].arguments == {"x": 1}
        assert calls[0].call_id  # non-empty

    def test_call_id_is_unique_within_parse(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0].call_id != calls[1].call_id

    def test_multiple_calls_in_order(self):
        text = (
            '<tool_call>{"name": "first", "arguments": {}}</tool_call>\n'
            "some text in between\n"
            '<tool_call>{"name": "second", "arguments": {}}</tool_call>'
        )
        calls = parse_tool_calls(text)
        assert [c.name for c in calls] == ["first", "second"]

    def test_multiline_json_body(self):
        text = """<tool_call>
{
  "name": "calc",
  "arguments": {"expression": "1 + 1"}
}
</tool_call>"""
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "calc"
        assert calls[0].arguments == {"expression": "1 + 1"}

    def test_arguments_default_to_empty_dict(self):
        text = '<tool_call>{"name": "ping"}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].arguments == {}

    def test_text_before_and_after_call(self):
        text = (
            "Let me think.\n"
            '<tool_call>{"name": "calc", "arguments": {"x": 1}}</tool_call>\n'
            "After the call."
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "calc"


class TestParseToolCallsEdgeCases:
    def test_malformed_json_skipped(self):
        text = '<tool_call>not even json</tool_call>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_missing_name_skipped(self):
        text = '<tool_call>{"arguments": {}}</tool_call>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_empty_name_skipped(self):
        text = '<tool_call>{"name": "", "arguments": {}}</tool_call>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_non_string_name_skipped(self):
        text = '<tool_call>{"name": 42, "arguments": {}}</tool_call>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_non_dict_arguments_skipped(self):
        text = '<tool_call>{"name": "x", "arguments": "string"}</tool_call>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_non_object_top_level_skipped(self):
        text = '<tool_call>[1, 2, 3]</tool_call>'
        calls = parse_tool_calls(text)
        assert calls == []

    def test_mixed_good_and_bad(self):
        text = (
            '<tool_call>not json</tool_call>'
            '<tool_call>{"name": "ok", "arguments": {}}</tool_call>'
            '<tool_call>{"arguments": {}}</tool_call>'  # no name
        )
        calls = parse_tool_calls(text)
        # Only the middle one is valid.
        assert len(calls) == 1
        assert calls[0].name == "ok"

    def test_non_str_input_raises(self):
        with pytest.raises((TypeError, AttributeError)):
            parse_tool_calls(None)  # type: ignore[arg-type]

    def test_whitespace_around_body_ok(self):
        text = '<tool_call>  \n  {"name": "x", "arguments": {}}  \n  </tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "x"


# ---------------------------------------------------------------------------
# format_tool_results.
# ---------------------------------------------------------------------------


class TestFormatToolResults:
    def test_empty(self):
        assert format_tool_results([]) == ""

    def test_single_success(self):
        r = ToolResult(call_id="cid", name="calc", output="4")
        out = format_tool_results([r])
        assert "<tool_result" in out
        assert "calc" in out
        assert "4" in out
        assert "cid" in out

    def test_single_error_uses_tool_error_tag(self):
        r = ToolResult(call_id="cid", name="calc", output="bad", is_error=True)
        out = format_tool_results([r])
        assert "<tool_error" in out
        assert "<tool_result" not in out

    def test_multiple_results_separated(self):
        a = ToolResult(call_id="a", name="x", output="1")
        b = ToolResult(call_id="b", name="y", output="2")
        out = format_tool_results([a, b])
        assert out.count("<tool_result") == 2
        assert out.index("x") < out.index("y")


# ---------------------------------------------------------------------------
# calculator_evaluate — SCAFFOLDED.
# ---------------------------------------------------------------------------


class TestCalculatorEvaluate:
    def test_simple_addition(self):
        assert calculator_evaluate("1 + 1") == 2

    def test_subtraction(self):
        assert calculator_evaluate("10 - 3") == 7

    def test_multiplication(self):
        assert calculator_evaluate("4 * 5") == 20

    def test_division(self):
        assert calculator_evaluate("7 / 2") == 3.5

    def test_floor_division(self):
        assert calculator_evaluate("7 // 2") == 3

    def test_modulo(self):
        assert calculator_evaluate("10 % 3") == 1

    def test_power(self):
        assert calculator_evaluate("2 ** 10") == 1024

    def test_unary_minus(self):
        assert calculator_evaluate("-5") == -5

    def test_unary_plus(self):
        assert calculator_evaluate("+5") == 5

    def test_parentheses(self):
        assert calculator_evaluate("(2 + 3) * 4") == 20

    def test_nested_parentheses(self):
        assert calculator_evaluate("((1 + 2) * (3 + 4))") == 21

    def test_float_constant(self):
        assert calculator_evaluate("3.14 * 2") == 6.28

    def test_combined_unary_and_binary(self):
        assert calculator_evaluate("-5 + 3") == -2

    def test_division_by_zero_propagates(self):
        # ZeroDivisionError isn't a ToolError; the dispatcher catches
        # all exceptions, but the bare evaluator should let it through
        # so callers can distinguish "rejected by safety check" from
        # "math error."
        with pytest.raises(ZeroDivisionError):
            calculator_evaluate("1 / 0")


class TestCalculatorRejection:
    """Anything outside the allowed AST node set raises ToolError."""

    def test_rejects_name(self):
        with pytest.raises(ToolError):
            calculator_evaluate("x + 1")

    def test_rejects_function_call(self):
        with pytest.raises(ToolError):
            calculator_evaluate("abs(-1)")

    def test_rejects_attribute_access(self):
        with pytest.raises(ToolError):
            calculator_evaluate("(1).bit_length()")

    def test_rejects_dunder_attribute(self):
        with pytest.raises(ToolError):
            calculator_evaluate("(1).__class__")

    def test_rejects_string_constant(self):
        with pytest.raises(ToolError):
            calculator_evaluate("'hello'")

    def test_rejects_boolean_constant(self):
        with pytest.raises(ToolError):
            calculator_evaluate("True")

    def test_rejects_subscript(self):
        with pytest.raises(ToolError):
            calculator_evaluate("[1, 2, 3][0]")

    def test_rejects_comparison(self):
        with pytest.raises(ToolError):
            calculator_evaluate("1 < 2")

    def test_rejects_lambda(self):
        with pytest.raises(ToolError):
            calculator_evaluate("(lambda x: x)(1)")

    def test_rejects_logical_not(self):
        with pytest.raises(ToolError):
            calculator_evaluate("not 0")

    def test_rejects_bitwise_not(self):
        with pytest.raises(ToolError):
            calculator_evaluate("~5")

    def test_rejects_matmul(self):
        # `5 @ 3` parses as MatMult, which we deliberately exclude.
        with pytest.raises(ToolError):
            calculator_evaluate("5 @ 3")

    def test_rejects_syntax_error(self):
        with pytest.raises(ToolError, match="parse"):
            calculator_evaluate("1 +")

    def test_rejects_walrus(self):
        with pytest.raises(ToolError):
            calculator_evaluate("(x := 5)")

    def test_rejects_dunder_import(self):
        with pytest.raises(ToolError):
            calculator_evaluate("__import__('os')")


# ---------------------------------------------------------------------------
# make_calculator (the wrapped tool).
# ---------------------------------------------------------------------------


class TestMakeCalculator:
    def test_returns_a_tool(self):
        t = make_calculator()
        assert isinstance(t, Tool)
        assert t.name == "calculator"

    def test_schema_requires_expression(self):
        t = make_calculator()
        assert "expression" in t.parameters["properties"]
        assert "expression" in t.parameters["required"]

    def test_callable_returns_string(self):
        # Once calculator_evaluate is implemented, the wrapper must
        # return a string.
        t = make_calculator()
        out = t.func(expression="2 + 2")
        assert isinstance(out, str)
        assert out == "4"


# ---------------------------------------------------------------------------
# make_read_file.
# ---------------------------------------------------------------------------


class TestMakeReadFile:
    def test_returns_a_tool(self, tmp_path: Path):
        t = make_read_file(root=tmp_path)
        assert isinstance(t, Tool)
        assert t.name == "read_file"

    def test_reads_existing_file(self, tmp_path: Path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        t = make_read_file(root=tmp_path)
        assert t.func(path="hello.txt") == "hello world"

    def test_reads_nested_file(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "nested.txt"
        f.write_text("nested content")
        t = make_read_file(root=tmp_path)
        assert t.func(path="sub/nested.txt") == "nested content"

    def test_missing_file_raises_toolerror(self, tmp_path: Path):
        t = make_read_file(root=tmp_path)
        with pytest.raises(ToolError, match="not found"):
            t.func(path="nope.txt")

    def test_directory_raises_toolerror(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        t = make_read_file(root=tmp_path)
        with pytest.raises(ToolError, match="not a file"):
            t.func(path="sub")

    def test_truncates_to_max_chars(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 10000)
        t = make_read_file(root=tmp_path)
        out = t.func(path="big.txt", max_chars=100)
        assert len(out) < 10000
        assert "truncated" in out.lower()

    def test_default_max_chars_returns_full(self, tmp_path: Path):
        f = tmp_path / "small.txt"
        f.write_text("small content")
        t = make_read_file(root=tmp_path)
        assert t.func(path="small.txt") == "small content"


class TestMakeReadFileSandbox:
    """Path-traversal protection: the read_file tool must NOT read
    files outside its sandboxed root.
    """

    def test_rejects_absolute_path(self, tmp_path: Path):
        t = make_read_file(root=tmp_path)
        with pytest.raises(ToolError):
            t.func(path="/etc/passwd")

    def test_rejects_dotdot_traversal(self, tmp_path: Path):
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret")
        try:
            t = make_read_file(root=tmp_path)
            with pytest.raises(ToolError, match="escape"):
                t.func(path="../outside.txt")
        finally:
            if outside.exists():
                outside.unlink()

    def test_rejects_empty_path(self, tmp_path: Path):
        t = make_read_file(root=tmp_path)
        with pytest.raises(ToolError):
            t.func(path="")


# ---------------------------------------------------------------------------
# make_web_search.
# ---------------------------------------------------------------------------


class TestMakeWebSearch:
    def test_returns_a_tool(self):
        t = make_web_search()
        assert isinstance(t, Tool)
        assert t.name == "web_search"

    def test_default_stub_returns_string(self):
        t = make_web_search()
        out = t.func(query="anything")
        assert isinstance(out, str)
        assert "stub" in out.lower()
        assert "anything" in out

    def test_injection_overrides_default(self):
        called: list[str] = []

        def fake(q: str) -> str:
            called.append(q)
            return f"results for {q}"

        t = make_web_search(search=fake)
        out = t.func(query="quantum gravity")
        assert called == ["quantum gravity"]
        assert out == "results for quantum gravity"

    def test_empty_query_raises(self):
        t = make_web_search()
        with pytest.raises(ToolError):
            t.func(query="")

    def test_whitespace_only_query_raises(self):
        t = make_web_search()
        with pytest.raises(ToolError):
            t.func(query="   ")


# ---------------------------------------------------------------------------
# make_run_python.
# ---------------------------------------------------------------------------


class TestMakeRunPython:
    def test_returns_a_tool(self):
        t = make_run_python()
        assert isinstance(t, Tool)
        assert t.name == "run_python"

    def test_runs_simple_print(self):
        t = make_run_python()
        out = t.func(code="print(2 + 2)")
        assert "4" in out

    def test_no_output_handled(self):
        t = make_run_python()
        out = t.func(code="x = 1")
        # No print → "(no output)" placeholder.
        assert "no output" in out

    def test_runtime_error_returned(self):
        t = make_run_python()
        out = t.func(code="raise ValueError('boom')")
        assert "exit" in out
        assert "ValueError" in out
        assert "boom" in out

    def test_timeout_returns_message(self):
        # Use a very short timeout to actually exercise the timeout path.
        t = make_run_python(timeout=0.5)
        out = t.func(code="import time; time.sleep(5)")
        assert "timed out" in out.lower()

    def test_zero_timeout_rejected(self):
        with pytest.raises(ValueError):
            make_run_python(timeout=0)

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValueError):
            make_run_python(timeout=-1)


# ---------------------------------------------------------------------------
# dispatch_tool_call (implemented, not scaffolded).
# Once validate_arguments is implemented, dispatch_tool_call will work.
# ---------------------------------------------------------------------------


class TestDispatchToolCall:
    def test_successful_call(self):
        t = _make_tool()
        registry = ToolRegistry([t])
        call = ToolCall(name="echo", arguments={"text": "hi"}, call_id="cid")
        result = dispatch_tool_call(registry, call)
        assert result.is_error is False
        assert result.output == "hi"
        assert result.call_id == "cid"
        assert result.name == "echo"

    def test_unknown_tool_returns_error(self):
        registry = ToolRegistry([_make_tool()])
        call = ToolCall(name="missing", arguments={}, call_id="cid")
        result = dispatch_tool_call(registry, call)
        assert result.is_error is True
        assert "missing" in result.output

    def test_invalid_args_returns_error(self):
        # Once validate_arguments is implemented, bad args become a
        # tool error (not a crash).
        registry = ToolRegistry([_make_tool()])
        call = ToolCall(name="echo", arguments={}, call_id="cid")  # missing required
        result = dispatch_tool_call(registry, call)
        assert result.is_error is True

    def test_tool_raise_returns_error(self):
        def boom(x: str) -> str:
            raise RuntimeError("kaboom")

        t = _make_tool(
            name="boom",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            func=boom,
        )
        registry = ToolRegistry([t])
        call = ToolCall(name="boom", arguments={"x": "yes"}, call_id="cid")
        result = dispatch_tool_call(registry, call)
        assert result.is_error is True
        assert "RuntimeError" in result.output
        assert "kaboom" in result.output

    def test_output_is_stringified(self):
        def returns_int(x: str) -> int:  # noqa: ARG001
            return 42

        t = _make_tool(
            name="i",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            func=returns_int,
        )
        registry = ToolRegistry([t])
        call = ToolCall(name="i", arguments={"x": "x"}, call_id="cid")
        result = dispatch_tool_call(registry, call)
        assert result.is_error is False
        assert result.output == "42"
        assert isinstance(result.output, str)

    def test_toolerror_returns_error_result(self):
        def raises_toolerror(x: str) -> str:  # noqa: ARG001
            raise ToolError("bad input")

        t = _make_tool(
            name="t",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            func=raises_toolerror,
        )
        registry = ToolRegistry([t])
        call = ToolCall(name="t", arguments={"x": "x"}, call_id="cid")
        result = dispatch_tool_call(registry, call)
        assert result.is_error is True
        assert "bad input" in result.output


# ---------------------------------------------------------------------------
# run_with_tools — SCAFFOLDED.
# ---------------------------------------------------------------------------


class TestRunWithTools:
    def test_no_tool_calls_returns_final_answer(self):
        backend = _FakeBackend(["the answer is 42"])
        registry = ToolRegistry([])
        result = run_with_tools(backend, registry, "what is the answer?")
        assert result.final_answer == "the answer is 42"
        assert result.stopped_reason == "no_more_calls"
        assert len(result.steps) == 1
        assert len(result.steps[0].tool_calls) == 0

    def test_single_tool_call_then_answer(self):
        # Step 1: model calls calculator(2+2). Step 2: model answers.
        backend = _FakeBackend([
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2"}}</tool_call>',
            "the result is 4",
        ])
        registry = ToolRegistry([make_calculator()])
        result = run_with_tools(backend, registry, "what is 2+2?")
        assert result.final_answer == "the result is 4"
        assert result.stopped_reason == "no_more_calls"
        assert len(result.steps) == 2
        assert len(result.steps[0].tool_calls) == 1
        assert result.steps[0].tool_calls[0].name == "calculator"
        assert result.steps[0].tool_results[0].output == "4"
        assert result.steps[0].tool_results[0].is_error is False

    def test_user_message_in_initial_prompt(self):
        backend = _FakeBackend(["done"])
        registry = ToolRegistry([])
        run_with_tools(backend, registry, "the user message")
        prompt = backend.calls[0]["prompt"]
        assert "the user message" in prompt

    def test_system_in_initial_prompt(self):
        backend = _FakeBackend(["done"])
        registry = ToolRegistry([])
        run_with_tools(backend, registry, "x", system="CUSTOM SYSTEM")
        prompt = backend.calls[0]["prompt"]
        assert "CUSTOM SYSTEM" in prompt

    def test_tool_block_in_initial_prompt(self):
        backend = _FakeBackend(["done"])
        registry = ToolRegistry([make_calculator()])
        run_with_tools(backend, registry, "x")
        prompt = backend.calls[0]["prompt"]
        assert "calculator" in prompt
        assert "<tool_call>" in prompt

    def test_tool_result_appears_in_subsequent_prompt(self):
        backend = _FakeBackend([
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2"}}</tool_call>',
            "the result is 4",
        ])
        registry = ToolRegistry([make_calculator()])
        run_with_tools(backend, registry, "what is 2+2?")
        # The second prompt should contain the tool_result block with "4".
        assert len(backend.calls) == 2
        second = backend.calls[1]["prompt"]
        assert "<tool_result" in second
        assert "4" in second

    def test_sampling_args_forwarded(self):
        backend = _FakeBackend(["done"])
        registry = ToolRegistry([])
        run_with_tools(
            backend, registry, "x",
            max_new_tokens=99, temperature=0.7, top_k=10, top_p=0.95,
        )
        call = backend.calls[0]
        assert call["max_new_tokens"] == 99
        assert call["temperature"] == 0.7
        assert call["top_k"] == 10
        assert call["top_p"] == 0.95

    def test_metadata_populated(self):
        backend = _FakeBackend(["done"])
        registry = ToolRegistry([make_calculator()])
        result = run_with_tools(backend, registry, "x")
        meta = result.metadata
        assert meta["n_steps"] == 1
        assert meta["n_tool_calls"] == 0
        assert "calculator" in meta["tools_available"]
        assert meta["backend_name"] == "fake"


class TestRunWithToolsLoop:
    def test_two_tool_calls_in_one_turn(self):
        # Model emits two calls in one completion → both dispatched
        # in a single step, then a final answer.
        backend = _FakeBackend([
            '<tool_call>{"name": "calculator", "arguments": {"expression": "1 + 1"}}</tool_call>'
            '<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2"}}</tool_call>',
            "got both",
        ])
        registry = ToolRegistry([make_calculator()])
        result = run_with_tools(backend, registry, "x")
        assert len(result.steps) == 2
        assert len(result.steps[0].tool_calls) == 2
        assert len(result.steps[0].tool_results) == 2
        assert result.steps[0].tool_results[0].output == "2"
        assert result.steps[0].tool_results[1].output == "4"
        assert result.final_answer == "got both"

    def test_chained_calls_across_steps(self):
        # Step 1: call A. Step 2: call B (informed by A's result).
        # Step 3: final answer.
        backend = _FakeBackend([
            '<tool_call>{"name": "calculator", "arguments": {"expression": "3 + 4"}}</tool_call>',
            '<tool_call>{"name": "calculator", "arguments": {"expression": "7 * 2"}}</tool_call>',
            "the answer is 14",
        ])
        registry = ToolRegistry([make_calculator()])
        result = run_with_tools(backend, registry, "x")
        assert len(result.steps) == 3
        assert result.final_answer == "the answer is 14"
        assert result.stopped_reason == "no_more_calls"

    def test_max_steps_stops_loop(self):
        # The model never stops calling tools; we hit max_steps.
        # Provide enough completions for the cap.
        backend = _FakeBackend([
            '<tool_call>{"name": "calculator", "arguments": {"expression": "1 + 1"}}</tool_call>'
        ] * 5)
        registry = ToolRegistry([make_calculator()])
        result = run_with_tools(backend, registry, "x", max_steps=3)
        assert result.stopped_reason == "max_steps"
        assert result.final_answer is None
        assert len(result.steps) == 3

    def test_max_steps_zero_rejected(self):
        backend = _FakeBackend(["x"])
        registry = ToolRegistry([])
        with pytest.raises(ValueError):
            run_with_tools(backend, registry, "x", max_steps=0)

    def test_empty_user_message_rejected(self):
        backend = _FakeBackend(["x"])
        registry = ToolRegistry([])
        with pytest.raises(ValueError):
            run_with_tools(backend, registry, "")


class TestRunWithToolsErrors:
    def test_unknown_tool_feedback(self):
        # Model calls a tool that isn't registered; the loop feeds
        # back an error result and the model recovers.
        backend = _FakeBackend([
            '<tool_call>{"name": "nonexistent", "arguments": {}}</tool_call>',
            "sorry, I tried a missing tool; the answer is 42",
        ])
        registry = ToolRegistry([make_calculator()])
        result = run_with_tools(backend, registry, "x")
        assert len(result.steps) == 2
        # First step's result should be is_error=True.
        assert result.steps[0].tool_results[0].is_error is True
        # Second prompt should contain the error tag.
        second_prompt = backend.calls[1]["prompt"]
        assert "<tool_error" in second_prompt
        assert result.final_answer.startswith("sorry")

    def test_bad_arguments_feedback(self):
        # Model calls calculator with wrong arg shape.
        backend = _FakeBackend([
            '<tool_call>{"name": "calculator", "arguments": {"wrong_arg": "x"}}</tool_call>',
            "I got a validation error; trying again is left to the agent loop in module 19",
        ])
        registry = ToolRegistry([make_calculator()])
        result = run_with_tools(backend, registry, "x")
        assert result.steps[0].tool_results[0].is_error is True


# ---------------------------------------------------------------------------
# Integration smoke: full pipeline end-to-end with all 4 scaffolds in.
# ---------------------------------------------------------------------------


class TestIntegrationSmoke:
    def test_full_pipeline_with_calculator(self, tmp_path: Path):
        """End-to-end: model calls calculator + read_file, gets results,
        produces a final answer."""
        # Set up a tmp file the read_file tool can fetch.
        f = tmp_path / "fact.txt"
        f.write_text("The answer is six.")

        backend = _FakeBackend([
            (
                'I will compute the value first.\n'
                '<tool_call>{"name": "calculator", '
                '"arguments": {"expression": "3 * 2"}}</tool_call>'
            ),
            (
                'Now let me check the doc.\n'
                '<tool_call>{"name": "read_file", '
                '"arguments": {"path": "fact.txt"}}</tool_call>'
            ),
            "Both tools agree: the answer is 6.",
        ])
        registry = ToolRegistry([
            make_calculator(),
            make_read_file(root=tmp_path),
        ])
        result = run_with_tools(backend, registry, "what is 3 * 2?")
        assert result.stopped_reason == "no_more_calls"
        assert result.final_answer.startswith("Both tools")
        assert len(result.steps) == 3
        # Step 1: calc → "6"
        assert result.steps[0].tool_results[0].output == "6"
        assert result.steps[0].tool_results[0].is_error is False
        # Step 2: read_file → file content
        assert result.steps[1].tool_results[0].output == "The answer is six."
        assert result.steps[1].tool_results[0].is_error is False

    def test_recovery_from_bad_call(self):
        """Model can recover from a bad tool call."""
        backend = _FakeBackend([
            # Bad: missing required arg
            '<tool_call>{"name": "calculator", "arguments": {}}</tool_call>',
            # Recovery: correct call
            '<tool_call>{"name": "calculator", "arguments": {"expression": "1 + 1"}}</tool_call>',
            # Final answer
            "the result is 2",
        ])
        registry = ToolRegistry([make_calculator()])
        result = run_with_tools(backend, registry, "x")
        assert result.steps[0].tool_results[0].is_error is True
        assert result.steps[1].tool_results[0].is_error is False
        assert result.steps[1].tool_results[0].output == "2"
        assert result.final_answer == "the result is 2"


# ---------------------------------------------------------------------------
# Module exports.
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_public_names_importable(self):
        import g2c.tools as mod
        for name in [
            "Tool",
            "ToolCall",
            "ToolError",
            "ToolRegistry",
            "ToolResult",
            "ToolRunResult",
            "ToolStep",
            "DEFAULT_SYSTEM",
            "calculator_evaluate",
            "dispatch_tool_call",
            "format_tool_results",
            "make_calculator",
            "make_read_file",
            "make_run_python",
            "make_web_search",
            "parse_tool_calls",
            "render_tools_for_prompt",
            "run_with_tools",
            "validate_arguments",
        ]:
            assert hasattr(mod, name), f"missing export: {name}"

    def test_default_system_is_string(self):
        assert isinstance(DEFAULT_SYSTEM, str)
        assert len(DEFAULT_SYSTEM) > 0
