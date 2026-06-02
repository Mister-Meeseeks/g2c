"""The tool-using loop — backend.complete → parse → execute → feed back.

`run_with_tools` turns a one-shot `Backend.complete` into a multi-step
"model can use tools" interaction. The shape:

  1. Build the system prompt: instructions + tool list + the user's
     message.
  2. Call `backend.complete(prompt)`.
  3. Parse the completion for `<tool_call>` blocks.
  4. If there are no tool calls → this completion is the final answer.
  5. Otherwise: dispatch each call, format the results back into the
     prompt as `<tool_result>` blocks, and loop.
  6. Stop when the model emits no more tool calls (success) or when
     `max_steps` is reached (timeout).

This is the **substrate** for Module 18 — the ReAct-style agent loop
with planning, scratchpad memory, and goal-tracking is Module 19.

`run_with_tools` (the public entry) is PROVIDED: it validates inputs
and selects the channel (native tool-calling when the backend exposes
`chat_with_tools`, else the text-format path). The **lesson** is
`run_text_loop` — the text-format complete → parse → dispatch →
feed-back loop, left SCAFFOLDED. Three small helpers
(`_init_transcript`, `_grow_transcript`, `_build_run_result`) hand you
the prompt-formatting and result bookkeeping so the loop body stays the
concept and nothing else.

`dispatch_tool_call` is the per-call wrapper:
  validate args → execute → wrap result.
Errors at every step (unknown tool, bad args, tool raised) get
surfaced as `is_error=True` ToolResults so the loop continues. This
is implemented (not scaffolded) — it's a small composition.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from g2c.inference import Backend

from .base import (
    Tool,
    ToolCall,
    ToolError,
    ToolResult,
    ToolRunResult,
    ToolStep,
)
from .parser import (
    format_tool_results,
    parse_tool_calls,  # noqa: F401 — used by run_text_loop, your deliverable
)
from .registry import ToolRegistry
from .schema import (
    DEFAULT_SYSTEM,
    NATIVE_DEFAULT_SYSTEM,
    render_tools_for_ollama,
    render_tools_for_prompt,
)


def dispatch_tool_call(registry: ToolRegistry, call: ToolCall) -> ToolResult:
    """Look up + validate + execute a single tool call.

    Three failure modes, each surfaced as `is_error=True`:
      * Unknown tool name → `KeyError` from the registry.
      * Argument validation failure → `ToolError` from `validate_arguments`.
      * Tool raised at execution → any exception wrapped.

    All three are caught and converted to `ToolResult` so the loop
    can feed them back to the model as recovery signals. A raised
    exception in the dispatcher would lose the conversation; a
    `ToolResult(is_error=True)` lets the model see the error and
    retry on the next step.

    Implemented. The composition is straightforward; the lesson is in
    the components (parser, validator) and the loop.
    """
    # Local import to avoid a circular dependency: schema.py imports
    # base.py for Tool/ToolError, and we want loop.py to also use the
    # validator without dragging schema's full surface into base.
    from .schema import validate_arguments

    try:
        tool: Tool = registry.get(call.name)
    except KeyError as e:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output=str(e),
            is_error=True,
        )
    try:
        validated = validate_arguments(tool, call.arguments)
    except ToolError as e:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output=str(e),
            is_error=True,
        )
    try:
        output = tool.func(**validated)
    except ToolError as e:
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output=str(e),
            is_error=True,
        )
    except Exception as e:  # noqa: BLE001 — tool errors are model-facing
        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            output=f"{type(e).__name__}: {e}",
            is_error=True,
        )
    return ToolResult(
        call_id=call.call_id,
        name=call.name,
        output=str(output),
        is_error=False,
    )


def backend_supports_native_tools(backend: Backend) -> bool:
    """Return True if the backend exposes a `chat_with_tools` method.

    The native-tool-calling path routes tool specs through the
    backend's chat endpoint (e.g. Ollama's `/api/chat` + `tools=`)
    rather than rendering them as text in the prompt. The backend is
    responsible for translating between the spec dict and the model's
    own post-training format.

    `hasattr(backend, "chat_with_tools")` is the duck-type check. We
    avoid an `isinstance(backend, OllamaBackend)` test so other
    backends (a future MLX chat backend, a thin OpenAI wrapper) can
    opt in without code changes here.
    """
    return callable(getattr(backend, "chat_with_tools", None))


def _extract_first_json_object(text: str) -> dict | None:
    """Find and JSON-decode the first balanced object in `text`.

    Skips leading prose and a single pair of surrounding markdown
    triple-backtick fences (with or without a language tag). Uses
    `json.JSONDecoder.raw_decode`, which correctly handles strings
    and escapes — a hand-rolled brace counter would mis-count braces
    that appear inside string values.

    Returns None if no parseable JSON object is found.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(s):
        start = s.find("{", pos)
        if start == -1:
            return None
        try:
            obj, _end = decoder.raw_decode(s[start:])
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(obj, dict):
            return obj
        pos = start + 1
    return None


def _try_parse_tool_call_from_content(content: str) -> ToolCall | None:
    """Rescue a tool call from plain content on the native channel.

    Small instruction-tuned models (notably Llama 3.2 3B) sometimes
    emit a syntactically valid tool-call JSON in their plain text
    output instead of using their post-training tool-call wire format
    (e.g. Llama's `<|python_tag|>` delimiter). Ollama then returns
    nothing in the structured `tool_calls` field, and the loop would
    otherwise stop with that JSON as the "final answer."

    This rescue scans `content` for an object shaped like
    `{"name": "<tool>", "parameters"|"arguments": {...}}` and
    synthesizes a `ToolCall` from it. We accept both `parameters`
    (Llama convention) and `arguments` (Qwen / Hermes / our own
    text-format), because on a rescue path we'd rather be permissive
    than miss the call.

    Returns None when no plausibly-shaped object is found.
    """
    if not isinstance(content, str) or not content.strip():
        return None
    candidate = _extract_first_json_object(content)
    if candidate is None:
        return None
    name = candidate.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = candidate.get("arguments")
    if args is None:
        args = candidate.get("parameters")
    if not isinstance(args, dict):
        return None
    return ToolCall(
        name=name,
        arguments=args,
        call_id=f"rescued_{uuid4().hex[:8]}",
    )


def run_with_tools(
    backend: Backend,
    registry: ToolRegistry,
    user_message: str,
    *,
    system: str | None = None,
    max_steps: int = 5,
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_k: int | None = None,
    top_p: float | None = None,
    use_native_tools: bool | None = None,
    think: bool | None = None,
) -> ToolRunResult:
    """Public entry point for the tool-using loop — PROVIDED.

    Validates inputs, picks the channel, defaults the system prompt to
    match, and dispatches to the loop:

      * native → `_run_with_native_tools` (provided; structured
        tool_calls via `backend.chat_with_tools`).
      * text   → `run_text_loop` (YOUR deliverable; the
        complete → parse → dispatch → feed-back loop).

    The pedagogical core is `run_text_loop`. This wrapper is the
    boilerplate around it — argument checks and channel selection —
    handed to you so your attention lands on the loop itself.

    Args:
        backend: an inference `Backend` from Module 16.
        registry: the `ToolRegistry` describing available tools.
        user_message: the user's question / instruction. Non-empty.
        system: the leading system prompt. Defaults (when None) to the
            channel-appropriate prompt: `DEFAULT_SYSTEM` for the text
            path, `NATIVE_DEFAULT_SYSTEM` for native.
        max_steps: cap on the number of complete-parse-dispatch
            iterations. Each step is one backend call. Defaults to 5
            — enough for "fetch + transform + answer," not enough for
            an agent to hang in a loop.
        max_new_tokens: forwarded to the backend each step.
        temperature: forwarded. Default 0.2 — tool-using models work
            best near-greedy because tool selection benefits from
            consistency.
        top_k, top_p: forwarded.
        use_native_tools: force the channel. None (default)
            auto-detects via `backend_supports_native_tools`.
        think: forwarded to the native channel's chat call (controls
            "thinking mode" on models that support it); ignored on the
            text path.

    Returns:
        `ToolRunResult` with `final_answer`, `steps`, `stopped_reason`,
        and `metadata` filled in.
    """
    if not isinstance(user_message, str) or not user_message:
        raise ValueError("user_message must be a non-empty str")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")

    # Channel selection: native when the backend exposes chat_with_tools,
    # unless the caller forces it one way.
    if use_native_tools is None:
        use_native_tools = backend_supports_native_tools(backend)
    if use_native_tools and not backend_supports_native_tools(backend):
        raise ValueError(
            f"use_native_tools=True but backend {type(backend).__name__} "
            "has no chat_with_tools method"
        )

    # Default the system prompt to the channel-appropriate one. The
    # text-format channel needs the model to be told about <tool_call>
    # blocks; the native channel actively does not — the chat template
    # handles the format and the model has been post-trained for it.
    if system is None:
        system = NATIVE_DEFAULT_SYSTEM if use_native_tools else DEFAULT_SYSTEM

    if use_native_tools:
        return _run_with_native_tools(
            backend=backend,
            registry=registry,
            user_message=user_message,
            system=system,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            think=think,
        )

    return run_text_loop(
        backend,
        registry,
        user_message,
        system=system,
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def run_text_loop(
    backend: Backend,
    registry: ToolRegistry,
    user_message: str,
    *,
    system: str,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> ToolRunResult:
    """The text-format tool-feedback loop — the Module-18 deliverable.

    This is the heart of tool use: turn a one-shot `backend.complete`
    into a multi-step interaction where the model can call tools and
    read their results. The provided `run_with_tools` wrapper has
    already validated inputs and chosen this (text) channel; you write
    the loop.

    The shape, one iteration per `max_steps`:

        complete → parse `<tool_call>` blocks →
            none?  this completion is the final answer; stop.
            some?  dispatch each, record the step, feed the results
                   back into the transcript, and loop.

    Three provided helpers do the plumbing so the loop body stays the
    concept:

        * `_init_transcript(system, registry, user_message)` builds the
          starting prompt (system + tool descriptions + the user turn +
          an `Assistant:` cue).
        * `_grow_transcript(transcript, completion, results)` appends one
          assistant turn + its formatted tool results and a fresh
          `Assistant:` cue — the monotonic growth that lets the model
          read its own past actions.
        * `_build_run_result(...)` assembles the `ToolRunResult` + its
          metadata.

    You still call `parse_tool_calls`, `dispatch_tool_call`, and build
    each `ToolStep` — those ARE the lesson.

    Args:
        backend, registry, user_message: as for `run_with_tools`.
        system: the resolved system prompt (already defaulted upstream).
        max_steps: iteration cap. The loop runs at most this many
            backend calls.
        max_new_tokens, temperature, top_k, top_p: forward each to
            `backend.complete`.

    Returns:
        `ToolRunResult`. Build it with `_build_run_result`, passing
        `stopped_reason="no_more_calls"` when the model produced a
        final answer, or `"max_steps"` when the loop ran out.

    Recipe:

        1. # Start the transcript and the step log.
           transcript = _init_transcript(system, registry, user_message)
           steps: list[ToolStep] = []

        2. # Up to max_steps passes; each pass is one backend call.
           for _ in range(max_steps):
               inference = backend.complete(
                   transcript,
                   max_new_tokens=max_new_tokens,
                   temperature=temperature,
                   top_k=top_k,
                   top_p=top_p,
               )
               tool_calls = parse_tool_calls(inference.completion)

        3. #     No tool calls -> this completion IS the final answer.
               if not tool_calls:
                   steps.append(ToolStep(
                       completion=inference.completion,
                       tool_calls=[],
                       tool_results=[],
                       inference=inference,
                   ))
                   return _build_run_result(
                       user_message=user_message,
                       final_answer=inference.completion,
                       steps=steps,
                       stopped_reason="no_more_calls",
                       registry=registry,
                       backend=backend,
                   )

        4. #     Otherwise dispatch every call, record the step, and
           #     feed the results back by growing the transcript.
               results = [
                   dispatch_tool_call(registry, c) for c in tool_calls
               ]
               steps.append(ToolStep(
                   completion=inference.completion,
                   tool_calls=tool_calls,
                   tool_results=results,
                   inference=inference,
               ))
               transcript = _grow_transcript(
                   transcript, inference.completion, results)

        5. # Fell out of the loop with no final answer -> max_steps.
           return _build_run_result(
               user_message=user_message,
               final_answer=None,
               steps=steps,
               stopped_reason="max_steps",
               registry=registry,
               backend=backend,
           )

    Implementation notes:

      * **Why "no tool calls = done."** Module 18's loop is purposely
        minimal: we trust the model to call tools when it needs them
        and to stop when it has the answer. The ReAct-style loop with
        explicit Thought / Action / Observation turns, planning, and
        scratchpad memory is Module 19; this is the substrate it sits
        on.

      * **Why the transcript grows monotonically.** Each step appends
        the assistant's completion + the tool results. The prompt
        grows; the model reads its own past actions. This is a tiny
        conversation, not a real chat template — a production loop
        would wrap each turn in the model's `<|user|>...<|assistant|>`
        markers.

      * **What `stopped_reason="max_steps"` looks like.** The loop ran
        out without the model emitting a final answer. The last step's
        `completion` is the model's last tool-calling output;
        `final_answer` is None.

    Sanity values:

      * Backend that always emits `"the answer is 42"` (no tool calls)
        → `final_answer="the answer is 42"`, one step,
        `stopped_reason="no_more_calls"`.

      * Backend that emits `<tool_call>{...calculator...}</tool_call>`
        on step 1 and `"the result is 4"` on step 2 → two steps; the
        first step has tool_calls + tool_results, the second is the
        final answer.

      * Backend that always emits a tool call with `max_steps=2` → two
        steps, `final_answer=None`, `stopped_reason="max_steps"`.
    """
    # TODO
    raise NotImplementedError


def _init_transcript(
    system: str, registry: ToolRegistry, user_message: str
) -> str:
    """Build the initial transcript for the text-format loop. PROVIDED.

    Layout: the system prompt, the rendered tool descriptions, the
    user's turn, and a trailing `Assistant:` cue for the model to
    continue from. The exact prompt format is plumbing, not the lesson.
    """
    tools_block = render_tools_for_prompt(registry.tools)
    return "\n\n".join(
        [
            system,
            tools_block,
            f"User: {user_message}",
            "Assistant:",
        ]
    )


def _grow_transcript(
    transcript: str, completion: str, results: list[ToolResult]
) -> str:
    """Append one assistant turn + its tool results to the transcript.

    Ends with a fresh `Assistant:` cue so the model continues from the
    results. This monotonic growth is what lets the model read its own
    past actions across steps; the exact delimiters are plumbing.
    PROVIDED.
    """
    return (
        transcript
        + " "
        + completion
        + "\n\n"
        + format_tool_results(results)
        + "\n\nAssistant:"
    )


def _build_run_result(
    *,
    user_message: str,
    final_answer: str | None,
    steps: list[ToolStep],
    stopped_reason: str,
    registry: ToolRegistry,
    backend: Backend,
    channel: str = "text-format",
) -> ToolRunResult:
    """Assemble the end-of-run `ToolRunResult` + metadata. PROVIDED.

    Counting steps and tool calls into a metadata dict is bookkeeping,
    not the lesson — so it's handed to you.
    """
    return ToolRunResult(
        user_message=user_message,
        final_answer=final_answer,
        steps=steps,
        stopped_reason=stopped_reason,
        metadata={
            "n_steps": len(steps),
            "n_tool_calls": sum(len(s.tool_calls) for s in steps),
            "tools_available": registry.names(),
            "backend_name": backend.info.name,
            "backend_model_id": backend.info.model_id,
            "channel": channel,
        },
    )


def _run_with_native_tools(
    *,
    backend: Backend,
    registry: ToolRegistry,
    user_message: str,
    system: str,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    think: bool | None = None,
) -> ToolRunResult:
    """Native-tool-calling loop using the backend's chat_with_tools.

    Parallel to the text-format loop but with three structural changes:

      1. **Conversation as messages, not a prompt string.** Each turn
         appends to a list of `{role, content, ...}` dicts. Tool results
         become `role="tool"` messages, not text appended after
         `<tool_result>` tags.
      2. **No `<tool_call>` parsing.** The backend returns a structured
         `tool_calls` list directly. We build `ToolCall` instances from
         the dicts the backend already parsed.
      3. **Stop condition is the same.** The model returning no
         tool_calls is still the success signal.

    Errors at parse/validate/dispatch are still surfaced as
    `is_error=True` `ToolResult`s — the harness behavior is identical;
    only the channel is different.
    """
    tool_specs = render_tools_for_ollama(registry.tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    steps: list[ToolStep] = []
    final_answer: str | None = None
    stopped_reason = "max_steps"
    n_rescued_calls = 0

    extra_chat_kwargs: dict[str, Any] = {}
    if think is not None:
        extra_chat_kwargs["think"] = think

    for _ in range(max_steps):
        chat = backend.chat_with_tools(  # type: ignore[attr-defined]
            messages,
            tool_specs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **extra_chat_kwargs,
        )
        inference = _chat_to_inference_result(chat, messages)
        tool_calls = [
            ToolCall(name=c["name"], arguments=c["arguments"], call_id=c["call_id"])
            for c in chat.tool_calls
        ]

        # Rescue: small models occasionally emit a tool call as plain
        # JSON in `content` instead of via the structured wire format.
        # If we have no structured calls but the content looks like a
        # tool-call object, synthesize one and continue the loop.
        if not tool_calls:
            rescued = _try_parse_tool_call_from_content(chat.content)
            if rescued is not None:
                tool_calls = [rescued]
                n_rescued_calls += 1

        if not tool_calls:
            final_answer = chat.content
            steps.append(
                ToolStep(
                    completion=chat.content,
                    tool_calls=[],
                    tool_results=[],
                    inference=inference,
                )
            )
            stopped_reason = "no_more_calls"
            break

        tool_results = [dispatch_tool_call(registry, call) for call in tool_calls]
        steps.append(
            ToolStep(
                completion=chat.content,
                tool_calls=tool_calls,
                tool_results=tool_results,
                inference=inference,
            )
        )

        # Append the assistant's tool-calling turn and each tool result
        # to the message list for the next iteration.
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": chat.content,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in tool_calls
            ],
        }
        messages = messages + [assistant_msg]
        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "name": result.name,
                    "content": result.output,
                }
            )

    return ToolRunResult(
        user_message=user_message,
        final_answer=final_answer,
        steps=steps,
        stopped_reason=stopped_reason,
        metadata={
            "n_steps": len(steps),
            "n_tool_calls": sum(len(step.tool_calls) for step in steps),
            "tools_available": registry.names(),
            "backend_name": backend.info.name,
            "backend_model_id": backend.info.model_id,
            "channel": "native",
            "n_rescued_calls": n_rescued_calls,
        },
    )


def _chat_to_inference_result(chat, messages):
    """Build an `InferenceResult` from a `ChatResult` for ToolStep parity.

    `ToolStep.inference` is typed as `InferenceResult`. The native path
    produces `ChatResult`s; this adapter packs the relevant fields so
    downstream consumers (notebooks, post-mortems) can treat the two
    channels uniformly. The chat metadata is preserved under
    `metadata["chat"]` for callers that need the structured tool_calls
    list.
    """
    from g2c.inference import InferenceResult

    # Pseudo-prompt: the user's last message content, for parity with
    # the text-format loop where `prompt` is the literal HTTP request
    # input. The full message history lives in metadata.
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user = msg.get("content", "")
            break
    meta = dict(chat.metadata)
    meta["chat"] = {
        "messages": chat.messages,
        "tool_calls": chat.tool_calls,
    }
    return InferenceResult(
        prompt=last_user,
        completion=chat.content,
        prompt_tokens=chat.prompt_tokens,
        completion_tokens=chat.completion_tokens,
        latency_ms=chat.latency_ms,
        backend=chat.backend,
        metadata=meta,
    )
