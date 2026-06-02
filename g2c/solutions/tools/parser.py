# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.tools.parser pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import json  # noqa: F401 (used by parse_tool_calls scaffold)
import re
from collections.abc import Iterable
from uuid import uuid4  # noqa: F401 (used by parse_tool_calls scaffold)
from g2c.tools.base import ToolCall, ToolResult
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract `<tool_call>...</tool_call>` blocks from text.

    The contract:

      * Iterate over every `<tool_call>...</tool_call>` block in
        `text`, in order of appearance.
      * For each block, strip whitespace from the body and try to
        parse it as JSON. If that fails, skip the block.
      * The parsed value must be a dict with:
          - `"name"` → non-empty string
          - `"arguments"` → dict (defaults to `{}` if absent)
        If either condition fails, skip the block.
      * Build a `ToolCall(name=..., arguments=..., call_id=...)`. The
        `call_id` is `f"call_{i}_{uuid4().hex[:8]}"` where `i` is the
        zero-based block index in the input — guaranteeing uniqueness
        within a single parse.
      * Return the list of successful parses. Empty list is fine —
        if the model didn't emit any tool calls (or emitted only bad
        ones), the loop reads that as "this completion is the final
        answer."

    Args:
        text: the model's completion text. Typically the body of an
            assistant turn. May contain anything — chain-of-thought,
            partial reasoning, the final answer, or one or more tool
            call blocks.

    Returns:
        A list of `ToolCall`s, in order. Empty if no well-formed
        tool calls were found.

    Recipe:

        1. # Accept str only.
           if not isinstance(text, str):
               raise TypeError(f"parse_tool_calls expected str, got {type(text).__name__}")

        2. # Iterate over <tool_call> matches, indexed for the call_id.
           calls: list[ToolCall] = []
           for i, match in enumerate(_TOOL_CALL_RE.finditer(text)):
               body = match.group(1).strip()

        3. #     Try to parse JSON. Skip on failure.
               try:
                   obj = json.loads(body)
               except json.JSONDecodeError:
                   continue

        4. #     Validate shape: dict with str name + dict arguments.
               if not isinstance(obj, dict):
                   continue
               name = obj.get("name")
               args = obj.get("arguments", {})
               if not isinstance(name, str) or not name:
                   continue
               if not isinstance(args, dict):
                   continue

        5. #     Build the ToolCall. uuid4().hex[:8] is enough entropy
               #     for "unique within a single conversation step."
               calls.append(ToolCall(
                   name=name,
                   arguments=args,
                   call_id=f"call_{i}_{uuid4().hex[:8]}",
               ))

        6. return calls

    Implementation notes:

      * Why non-greedy `.*?`. Without `?`, `<tool_call>...A...</tool_call>...
        <tool_call>...B...</tool_call>` would match as one giant block
        spanning A and B. Non-greedy stops at the first closing tag.

      * Why `re.DOTALL`. Without it, `.` doesn't match newlines, and
        a multi-line JSON body wouldn't be captured.

      * Why `strip()` on the body. The model often emits `<tool_call>\n{...}\n</tool_call>`
        — newlines around the JSON. `json.loads` handles them fine,
        but `strip()` makes the captured body cleaner for logs.

      * Why we don't validate against a tool's schema here. The
        validator (`validate_arguments` in `schema.py`) needs the
        Tool object, which the parser doesn't have. The dispatch loop
        does the schema check after parsing.

      * Why we skip malformed blocks instead of raising. Models
        occasionally emit half-formed blocks. Raising would crash
        the loop and lose the conversation. Skipping degrades to
        "no calls this turn," which the loop reads as "the model
        finished" — a recoverable mode.

    Sanity values:

      * `parse_tool_calls("")` → `[]`. Empty input, no calls.
      * `parse_tool_calls("just plain text, no tags")` → `[]`.
      * `parse_tool_calls('<tool_call>{"name": "calc", "arguments": {"x": 1}}</tool_call>')`
        → list with one ToolCall(name="calc", arguments={"x": 1}, ...).
      * Two adjacent blocks → list of length 2, in order.
      * `<tool_call>not json</tool_call>` → `[]` (skipped).
      * `<tool_call>{"name": "x"}</tool_call>` → one ToolCall with
        empty arguments dict (the default).
      * `<tool_call>{"arguments": {}}</tool_call>` → `[]` (no name).
    """
    if not isinstance(text, str):
        raise TypeError(
            f"parse_tool_calls expected str, got {type(text).__name__}"
        )

    calls: list[ToolCall] = []
    for index, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        body = match.group(1).strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        arguments = obj.get("arguments", {})
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, dict):
            continue

        calls.append(
            ToolCall(
                name=name,
                arguments=arguments,
                call_id=f"call_{index}_{uuid4().hex[:8]}",
            )
        )

    return calls
