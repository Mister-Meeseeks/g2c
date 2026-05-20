"""The tool-using loop — backend.complete → parse → execute → feed back.

`run_with_tools` is the tiny dispatcher that turns a one-shot
`Backend.complete` into a multi-step "model can use tools" interaction.
The shape:

  1. Build the system prompt: instructions + tool list + the user's
     message.
  2. Call `backend.complete(prompt)`.
  3. Parse the completion for `<tool_call>` blocks.
  4. If there are no tool calls → this completion is the final answer.
  5. Otherwise: dispatch each call, format the results back into the
     prompt as `<tool_result>` blocks, and loop.
  6. Stop when the model emits no more tool calls (success) or when
     `max_steps` is reached (timeout).

This is the **lesson** for Module 18 — SCAFFOLDED. The ReAct-style
agent loop with planning, scratchpad memory, and goal-tracking is
Module 19. Module 18 is the *substrate*: the parse/dispatch/feedback
mechanism the agent will sit on.

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
    ToolStep,  # noqa: F401 (used by run_with_tools scaffold)
)
from .parser import (  # noqa: F401 (used by run_with_tools scaffold)
    format_tool_results,
    parse_tool_calls,
)
from .registry import ToolRegistry
from .schema import (  # noqa: F401 (used by run_with_tools scaffold)
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
    """Run the call → parse → dispatch → feedback loop until the model
    stops calling tools (or `max_steps` is hit).

    Args:
        backend: an inference `Backend` from Module 16.
        registry: the `ToolRegistry` describing available tools.
        user_message: the user's question / instruction. Non-empty.
        system: the leading system prompt. Defaults to `DEFAULT_SYSTEM`,
            which describes the tool-call format.
        max_steps: cap on the number of complete-parse-dispatch
            iterations. Each step is one backend call. Defaults to 5
            — enough for "fetch + transform + answer," not enough for
            an agent to hang in a loop.
        max_new_tokens: forwarded to `backend.complete`.
        temperature: forwarded. Default 0.2 — tool-using models work
            best near-greedy because tool selection benefits from
            consistency.
        top_k, top_p: forwarded.

    Returns:
        `ToolRunResult` with `final_answer`, `steps`, and
        `stopped_reason` filled in.

    Recipe:

        1. # Validate inputs.
           if not isinstance(user_message, str) or not user_message:
               raise ValueError("user_message must be a non-empty str")
           if max_steps <= 0:
               raise ValueError(f"max_steps must be > 0, got {max_steps}")

        2. # Build the initial transcript.
           tools_block = render_tools_for_prompt(registry.tools)
           transcript = "\n\n".join([
               system,
               tools_block,
               f"User: {user_message}",
               "Assistant:",
           ])

        3. # Loop. At each step: complete, parse, dispatch, append.
           steps: list[ToolStep] = []
           final_answer: str | None = None
           stopped_reason = "max_steps"

           for _ in range(max_steps):
               inference = backend.complete(
                   transcript,
                   max_new_tokens=max_new_tokens,
                   temperature=temperature,
                   top_k=top_k,
                   top_p=top_p,
               )
               completion = inference.completion
               tool_calls = parse_tool_calls(completion)

        4. #     No tool calls → this is the final answer.
               if not tool_calls:
                   final_answer = completion
                   steps.append(ToolStep(
                       completion=completion,
                       tool_calls=[],
                       tool_results=[],
                       inference=inference,
                   ))
                   stopped_reason = "no_more_calls"
                   break

        5. #     Dispatch each call; record the step.
               results = [dispatch_tool_call(registry, c) for c in tool_calls]
               steps.append(ToolStep(
                   completion=completion,
                   tool_calls=tool_calls,
                   tool_results=results,
                   inference=inference,
               ))

        6. #     Append the assistant's tool-calling output and the
               #     formatted results to the transcript, prompting
               #     for the next assistant turn.
               transcript = (
                   transcript
                   + " "
                   + completion
                   + "\n\n"
                   + format_tool_results(results)
                   + "\n\nAssistant:"
               )

        7. # Build and return the final result.
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
               },
           )

    Implementation notes:

      * **Why not a full ReAct loop here?** ReAct interleaves
        "Thought / Action / Observation" turns and trains the model
        to plan explicitly. Module 19 builds that. Module 18's loop
        is purposely minimal: we trust the model to use tools when
        it needs to and to stop when it has the answer. The "no tool
        calls = done" exit condition is the simplest contract that
        works.

      * **Why the transcript grows monotonically.** Each step appends
        the assistant's last completion + the tool results. The
        prompt grows; the model reads its own past actions. This is
        a tiny conversation, not a true chat template — for a real
        chat-template-aware loop, you'd wrap each turn in
        `<|user|>...<|assistant|>...` markers per the model's
        instruction-tuning.

      * **What `stopped_reason="max_steps"` looks like.** The loop
        ran out without the model emitting a final answer. The last
        step's `completion` is the model's last tool-calling output;
        `final_answer` is None. This is the "the model is stuck in
        a loop" mode — Module 19's agent has more sophisticated stop
        criteria, but for now `max_steps` is the safety net.

      * **Why not stream tokens?** Streaming a tool-calling loop
        requires server-sent events, partial-JSON parsing, and a
        much bigger interface. We do the synchronous version. Real
        production tool-calling streams; the conceptual shape is the
        same.

    Sanity values:

      * Backend that always emits `"the answer is 42"` (no tool
        calls) → `final_answer="the answer is 42"`, one step,
        `stopped_reason="no_more_calls"`.

      * Backend that emits `<tool_call>{...calculator...}</tool_call>`
        on step 1 and `"the result is 4"` on step 2 → two steps,
        first step has tool_calls + tool_results, second step has
        the final answer.

      * Backend that always emits a tool call and `max_steps=2` →
        two steps, `final_answer=None`, `stopped_reason="max_steps"`.
    """
    if not isinstance(user_message, str) or not user_message:
        raise ValueError("user_message must be a non-empty str")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")

    # Native path is opt-in by default-detection: if the user didn't say,
    # use it whenever the backend exposes chat_with_tools.
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

    tools_block = render_tools_for_prompt(registry.tools)
    transcript = "\n\n".join(
        [
            system,
            tools_block,
            f"User: {user_message}",
            "Assistant:",
        ]
    )

    steps: list[ToolStep] = []
    final_answer: str | None = None
    stopped_reason = "max_steps"

    for _ in range(max_steps):
        inference = backend.complete(
            transcript,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        completion = inference.completion
        tool_calls = parse_tool_calls(completion)

        if not tool_calls:
            final_answer = completion
            steps.append(
                ToolStep(
                    completion=completion,
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
                completion=completion,
                tool_calls=tool_calls,
                tool_results=tool_results,
                inference=inference,
            )
        )
        transcript = (
            transcript
            + " "
            + completion
            + "\n\n"
            + format_tool_results(tool_results)
            + "\n\nAssistant:"
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
            "channel": "text-format",
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
