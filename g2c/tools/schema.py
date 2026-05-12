"""Schema validation + tool-prompt rendering.

Two pieces here:

  * `validate_arguments(tool, arguments)` — checks an `arguments` dict
    against `tool.parameters`. Implements a JSON-schema-LITE subset:
    object-with-primitive-properties only, no nesting, no arrays. The
    real JSON Schema spec is huge; we implement the corner that
    actually gets used in tool-calling prompts. **This is the lesson —
    SCAFFOLDED.**

  * `render_tools_for_prompt(tools)` — formats a list of tools as a
    text block suitable for embedding in the system prompt. Fully
    implemented. The format is a deliberate choice (named tools,
    JSON schemas, an explicit call-format reminder); the *content*
    matters more than the *layout*, but a layout the model has seen
    in pretraining gets followed more reliably than a custom one.

Why "JSON-schema-lite" and not full jsonschema? Two reasons:

  1. Pedagogy. The validation logic is short enough to read and
     explain. A real `jsonschema` library is a wonderful black box
     that doesn't help a student understand WHY their tool calls are
     being rejected.

  2. Realism. Production tool-calling prompts almost never use the
     full JSON Schema spec. They use a small subset — object with
     primitive properties, required list, descriptions on each
     property. Anthropic, OpenAI, and the function-calling fine-tuned
     open models all describe tools this way. Implementing the subset
     mirrors how production systems actually consume schemas.

If you outgrow the subset, the upgrade is one line: `import jsonschema;
jsonschema.validate(arguments, tool.parameters)`. We don't, and don't
need to for this module.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .base import Tool, ToolError  # noqa: F401 (ToolError used by validate_arguments scaffold)

# JSON-schema "type" tag → Python isinstance() target.
#
# Note `bool` comes BEFORE `int` for the "number" tag won't matter
# (the tag is "number"), but Python's `isinstance(True, int)` returns
# True — so for "integer", we explicitly reject booleans. JSON's `bool`
# is a separate type from `integer`; getting this wrong means the model
# could pass `true` for a numeric arg.
_TYPE_TAGS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
}


def validate_arguments(tool: Tool, arguments: Any) -> dict[str, Any]:
    """Validate `arguments` against `tool.parameters`. Return a dict.

    The schema accepted is JSON-schema-LITE:

      ```
      {
        "type": "object",
        "properties": {
          "<key>": {"type": "<string|number|integer|boolean>", "description": "..."},
          ...
        },
        "required": ["<key>", ...]
      }
      ```

    Validation rules:

      1. `arguments` must be a dict.
      2. Every key in `required` must appear in `arguments`.
      3. Every key in `arguments` must appear in `properties` (i.e. no
         unknown arguments — schemas are strict).
      4. Each value must satisfy the type tag of its corresponding
         property:
            - "string": isinstance(v, str)
            - "number": isinstance(v, (int, float)) AND not bool
            - "integer": isinstance(v, int) AND not bool
            - "boolean": isinstance(v, bool)
      5. The schema itself must look like an object schema. (We don't
         exhaustively validate the schema; we just refuse to operate
         on something obviously malformed.)

    Args:
        tool: the registered Tool whose parameters schema applies.
        arguments: the dict the model emitted. NOT yet trusted; this
            function is what makes it safe to pass to the tool.

    Returns:
        The validated arguments dict. Currently identical to the
        input on success — we don't coerce types, just check them.
        (A later version could coerce e.g. int → float for "number".)

    Raises:
        ToolError: on any validation failure. The error message names
            the offending key + the expected vs actual types so the
            model has a fighting chance of correcting on retry.

    Recipe:

        1. # Reject non-dict arguments.
           if not isinstance(arguments, dict):
               raise ToolError(...)

        2. # Pull schema, properties, required.
           schema = tool.parameters
           if schema.get("type") != "object":
               raise ToolError("schema must be type: object")
           properties = schema.get("properties", {})
           required = list(schema.get("required", []))

        3. # Required keys present?
           missing = [k for k in required if k not in arguments]
           if missing:
               raise ToolError(f"missing required arguments: {missing}")

        4. # No unknown keys?
           extra = [k for k in arguments if k not in properties]
           if extra:
               raise ToolError(f"unknown arguments: {extra}")

        5. # Type-check each present argument.
           for key, value in arguments.items():
               prop = properties[key]
               type_tag = prop.get("type", "string")
               expected = _TYPE_TAGS.get(type_tag)
               if expected is None:
                   raise ToolError(f"unknown type tag {type_tag!r}")
               # Reject bool when expecting numeric: True/False are int
               # subclasses in Python and would silently pass.
               if type_tag in ("number", "integer") and isinstance(value, bool):
                   raise ToolError(
                       f"argument {key!r}: expected {type_tag}, "
                       f"got bool"
                   )
               if not isinstance(value, expected):
                   raise ToolError(
                       f"argument {key!r}: expected {type_tag}, "
                       f"got {type(value).__name__}"
                   )

        6. # Return the (unchanged) arguments dict.
           return dict(arguments)

    Implementation notes:

      * Strict-mode schemas. We reject unknown keys. JSON Schema's
        `additionalProperties` defaults to True; ours defaults to
        False. The reason: if the model invents a parameter we don't
        consume, we'd rather catch it than silently drop it. Bad-arg
        feedback is what gets the model to fix the next call.

      * The bool-vs-int gotcha. `isinstance(True, int)` is True in
        Python because `bool` is a subclass of `int`. If you skip the
        explicit bool check, a tool that takes `count: int` would
        happily accept `True` and pass it through. The model wouldn't
        know it had passed garbage; the tool would compute weirdness;
        the user would see weirdness. The check is one line; pay it.

      * No coercion. We don't turn `"42"` into `42` for an integer
        param. This is a strict-mode choice; the model is supposed to
        emit JSON, and JSON has integers. If you find the model
        consistently emitting strings for numeric args, the right fix
        is a clearer prompt, not silent coercion.

    Sanity values:

      * Schema with `"required": ["expression"]` and no "expression"
        in arguments → ToolError("missing required arguments: ['expression']").

      * Schema with one "string" property and arguments containing an
        integer for that property → ToolError("argument 'x': expected
        string, got int").

      * Schema for `{"type":"object","properties":{"x":{"type":"integer"}}}`
        and arguments `{"x": True}` → ToolError naming the bool.

      * A correct call: passes through, returns a dict equal to
        `arguments` (a copy, not the original — so callers can mutate
        without affecting the validator's input).
    """
    if not isinstance(arguments, dict):
        raise ToolError(
            f"arguments must be a dict, got {type(arguments).__name__}"
        )

    schema = tool.parameters
    if schema.get("type") != "object":
        raise ToolError("tool parameters schema must be type: object")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ToolError("tool parameters schema must have object properties")

    required_raw = schema.get("required", [])
    if not isinstance(required_raw, list):
        raise ToolError("tool parameters schema required field must be a list")
    required = list(required_raw)

    missing = [key for key in required if key not in arguments]
    if missing:
        raise ToolError(f"missing required arguments: {missing}")

    extra = [key for key in arguments if key not in properties]
    if extra:
        raise ToolError(f"unknown arguments: {extra}")

    for key, value in arguments.items():
        prop = properties[key]
        if not isinstance(prop, dict):
            raise ToolError(f"schema for argument {key!r} must be an object")

        type_tag = prop.get("type", "string")
        expected = _TYPE_TAGS.get(type_tag)
        if expected is None:
            raise ToolError(f"unknown type tag {type_tag!r} for argument {key!r}")

        if type_tag in ("number", "integer") and isinstance(value, bool):
            raise ToolError(f"argument {key!r}: expected {type_tag}, got bool")
        if not isinstance(value, expected):
            raise ToolError(
                f"argument {key!r}: expected {type_tag}, "
                f"got {type(value).__name__}"
            )

    return dict(arguments)


DEFAULT_SYSTEM = (
    "You are a helpful assistant with access to tools. To call a tool, "
    "emit a <tool_call> block whose body is JSON of the form "
    '{"name": "<tool>", "arguments": {...}}. The arguments must match '
    "the tool's parameters schema exactly. After you emit the block, "
    "stop and wait for the result. When you have enough information "
    "to answer the user, give your final answer in plain text without "
    "any <tool_call> blocks."
)


def render_tools_for_prompt(tools: Iterable[Tool]) -> str:
    """Format a tool list as a text block for the system prompt.

    Args:
        tools: an iterable of `Tool`s. Empty allowed — the renderer
            emits a "no tools" block, which is a valid (if pointless)
            agent configuration.

    Returns:
        A multi-line string suitable for splicing into the prompt:

        ```
        Tools available:
        - calculator: Evaluates an arithmetic expression. Supports + - * / % ** //...
          parameters: { ... JSON schema ... }
        - read_file: Read a UTF-8 text file from the project root...
          parameters: { ... }

        Call format: <tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>
        ```

    The choice of layout is informed by what chat models actually
    follow. Most chat-tuned open models (Llama 3.x, Qwen 2.5, Mistral)
    have seen JSON-block tool-calling formats during instruction
    tuning. Anthropic recommends XML; OpenAI recommends a structured
    `tools=` array. We use a JSON-schema-described tool list inside
    a system prompt because it's the lowest common denominator across
    locally-hosted open models.

    Fully implemented. The lesson is the parser + validator + loop;
    the prompt template is a recipe.
    """
    tools_list = list(tools)
    if not tools_list:
        return "Tools available: (none)\n\nCall format: <tool_call>{...}</tool_call>"

    lines = ["Tools available:"]
    for t in tools_list:
        schema_str = json.dumps(t.parameters, indent=2)
        # Indent the schema body to align with the bullet.
        schema_str = "\n".join("  " + ln for ln in schema_str.splitlines())
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"  parameters:\n{schema_str}")
    lines.append("")
    lines.append(
        'Call format: <tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>'
    )
    return "\n".join(lines)
