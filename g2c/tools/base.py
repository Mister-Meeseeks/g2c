"""Tool — the dataclasses that describe a callable the model can use.

A `Tool` is a triple:
  * a NAME the model uses to invoke it,
  * a DESCRIPTION the model reads to decide whether to invoke it, and
  * a CALLABLE that runs when invoked.

Plus a JSON-schema-lite `parameters` dict that constrains the
arguments the model is allowed to pass.

The accompanying types in this file:

  * `ToolCall` — a parsed invocation. The model emits text; the parser
    turns each `<tool_call>{json}</tool_call>` block into a `ToolCall`.
    The `call_id` is assigned by the parser so the result can be
    matched back to its call later in the transcript.

  * `ToolResult` — the output of one execution. Carries either the
    successful output OR an error message; `is_error` says which.
    Crucially, errors are surfaced as ToolResults (not exceptions) so
    the loop can feed them back to the model and let it recover.

  * `ToolError` — raised internally by validators and the calculator's
    AST walker. The dispatcher catches it and wraps as `ToolResult`.

  * `ToolStep` and `ToolRunResult` — the per-step and end-of-run
    records that `run_with_tools` builds up. Useful for debugging:
    "what did the model say at step 3, what tools did it call, what
    came back?"

This file is pure dataclasses + one exception class. The interesting
code lives in:

  * `g2c/tools/registry.py` — collecting tools into a registry
  * `g2c/tools/schema.py` — validating arguments against the schema
  * `g2c/tools/parser.py` — extracting tool calls from model text
  * `g2c/tools/loop.py` — orchestrating the call → execute → feedback flow
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from g2c.inference import InferenceResult


class ToolError(Exception):
    """Raised when a tool argument fails validation, a schema is
    malformed, or a tool's internal logic refuses an input.

    The `dispatch_tool_call` wrapper catches this and converts it to
    a `ToolResult(is_error=True)` so the loop can continue. Direct
    callers (the calculator's AST walker, validators) raise it freely.
    """


@dataclass(frozen=True)
class Tool:
    """A registered callable the model can invoke by name.

    Attributes:
        name: stable identifier the model uses in `<tool_call>` JSON.
            Conventions: snake_case, no spaces. The model's prompt
            describes tools by name; the parser matches calls to
            registered tools by exact-string lookup.
        description: human-readable description the model reads to
            decide whether to use this tool. Embedded in the system
            prompt; the model sees it as part of its instructions.
            Worth a sentence — "Evaluates an arithmetic expression" —
            not a paragraph.
        parameters: JSON-schema-lite dict describing arguments. Shape:
            ```
            {
              "type": "object",
              "properties": {
                "<arg_name>": {
                  "type": "string|number|integer|boolean",
                  "description": "..."
                },
                ...
              },
              "required": ["<arg_name>", ...]
            }
            ```
            Only top-level objects with primitive properties are
            supported by `validate_arguments`. (No nested objects, no
            arrays. Real JSON schema is a much bigger spec.)
        func: the callable. Receives validated arguments as keyword
            arguments. Returns anything that can be coerced to str via
            `str(result)` — the dispatcher does that coercion before
            wrapping as a `ToolResult`.

    Frozen — once registered, a tool's contract should not change. If
    you want to swap an implementation, build a new Tool and re-add it
    to a new registry; mutating in place is a recipe for "the model
    sees one schema but executes against another."
    """

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Tool.name must be a non-empty str")
        if not isinstance(self.description, str) or not self.description:
            raise ValueError("Tool.description must be a non-empty str")
        if not isinstance(self.parameters, dict):
            raise TypeError("Tool.parameters must be a dict (JSON-schema-lite)")
        if not callable(self.func):
            raise TypeError("Tool.func must be callable")


@dataclass(frozen=True)
class ToolCall:
    """A parsed invocation extracted from model text.

    Attributes:
        name: the tool name the model asked to invoke.
        arguments: the dict the model emitted as `"arguments"`. NOT
            yet validated — the dispatcher runs `validate_arguments`
            before passing to the tool.
        call_id: a unique-ish id assigned by the parser. Used to match
            this call to its `ToolResult` later in the transcript. Format:
            `call_<step>_<hex>` (8 hex chars). Stable within a single
            parse, not globally unique.
    """

    name: str
    arguments: dict[str, Any]
    call_id: str


@dataclass(frozen=True)
class ToolResult:
    """The result of executing one `ToolCall`.

    Attributes:
        call_id: the id of the call this result is responding to.
            Lines up with `ToolCall.call_id`.
        name: the tool's name. Redundant with `call_id` lookup but
            convenient when formatting the result for feedback.
        output: stringified output. For successful calls, this is
            `str(tool.func(**args))`. For errors, this is the error
            message — wrapped as a string so the model can read it
            and (potentially) retry with corrected arguments.
        is_error: True if the call failed (unknown tool, bad arguments,
            tool raised). The loop feeds errors back to the model
            using a `<tool_error>` tag instead of `<tool_result>`,
            which the model is more likely to recognize as a recovery
            signal.

    Errors are surfaced as `ToolResult(is_error=True)` rather than as
    Python exceptions because the model is supposed to see them. A
    raised exception in the dispatcher would crash the loop and bury
    the error in a stack trace. A `ToolResult(is_error=True)` becomes
    a `<tool_error>` block in the next prompt, and the model can read
    it and try again.
    """

    call_id: str
    name: str
    output: str
    is_error: bool = False


@dataclass
class ToolStep:
    """One iteration of the call → execute → feedback loop.

    Attributes:
        completion: the model's text output at this step. Includes
            anything the model wrote alongside its tool calls (chain
            of thought, partial reasoning, etc.).
        tool_calls: the parsed `ToolCall`s. Empty if the model did not
            invoke any tools at this step (which is the loop's exit
            condition — no calls means "this is the final answer").
        tool_results: the `ToolResult`s from executing `tool_calls`.
            One per call, in the same order. Empty when `tool_calls`
            is empty.
        inference: the underlying `InferenceResult` from the backend.
            Carries token counts and latency for this step. Lets a
            tool-using session report its compute cost step-by-step.

    Mutable — the loop builds up a list of these as it iterates.
    """

    completion: str
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    inference: InferenceResult


@dataclass
class ToolRunResult:
    """The full result of one `run_with_tools` call.

    Attributes:
        user_message: the user's original message.
        final_answer: the model's last completion when it stopped
            calling tools. `None` if the loop hit `max_steps` without
            reaching a final answer.
        steps: every `ToolStep` of the run, in order. The last step's
            completion is `final_answer` (when the loop finished
            normally).
        stopped_reason: `"no_more_calls"` (model finished cleanly) or
            `"max_steps"` (loop ran out of iterations). Useful for
            telling apart "the model answered" from "we cut it off."
        metadata: free-form per-run extras. The loop fills in tool
            counts, total tokens, and the registry's tool list here.

    Mutable so callers can attach post-hoc analysis (e.g. an eval
    harness might add `metadata['eval'] = {'correct': True, ...}`).
    """

    user_message: str
    final_answer: str | None
    steps: list[ToolStep]
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
