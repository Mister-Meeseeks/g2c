"""NativeAgent — same loop, different wire format.

`Agent` (in `agent.py`) drives a ReAct-format conversation: each turn,
the model emits `Thought:` / `Action:` / `Action Input:` markers in
plain text, the harness regex-parses them, dispatches the action, and
re-renders the scratchpad as part of the next prompt. That's the
lesson of Module 19 — build the parser, build the loop, see where
small models struggle with text-format agents.

`NativeAgent` runs the *same loop* but speaks the modern structured
tool-calling protocol. Each turn:

  1. The conversation history is translated into the OpenAI/Ollama
     `messages` list with `role: "assistant"` (carrying `tool_calls`)
     and `role: "tool"` (carrying observations).
  2. `backend.chat_with_tools(messages, tools, ...)` is called. The
     model emits structured `tool_calls` directly; there is no text
     parser.
  3. The structured response is *unwrapped* back into a `AgentStep`
     so the rest of the agent (display, loop-detection, planner,
     postmortem) stays format-agnostic.

Why have both?

  * `Agent` is the buildable lesson — students implement the parser,
    the scratchpad rendering, the loop, the planner. ReAct is the
    historical agent format; its quirks (text markers, fragile
    parsing) are the pedagogy.
  * `NativeAgent` is what production looks like in 2026 — structured
    tool calling is what every major provider trained models to do.
    Same `AgentStep` history, same loop control, same planner. Only
    the I/O at the model boundary changes.

The wrap/unwrap functions (`_history_to_messages`, `_chat_to_step`)
are the entire mechanical difference. They're intentionally short and
readable so a student can see exactly where the format-specific code
lives.

Multi-tool-call per turn: the structured protocol supports it (a
single assistant turn can emit `tool_calls: [call_a, call_b]`),
but `AgentStep` carries one `action`. For now, this implementation
dispatches only the first call when several are returned and ignores
the rest; a richer version would emit one `AgentStep` per call. See
the comment in `_run_one_turn` below.
"""
from __future__ import annotations

import json
from typing import Any

from g2c.inference import Backend, ChatResult, InferenceResult
from g2c.tools import (
    ToolCall,
    ToolRegistry,
    dispatch_tool_call,
    render_tools_for_ollama,
)
from g2c.tools.loop import (
    _try_parse_tool_call_from_content,
    backend_supports_native_tools,
)

from .base import (
    Action,
    AgentError,
    AgentRunResult,
    AgentStep,
    Observation,
    Plan,
)
from .memory import Scratchpad  # noqa: F401 — kept for display compatibility
from .planner import make_plan
from .prompts import render_plan_block


# ---------------------------------------------------------------------------
# Default system prompt for the native channel.
# ---------------------------------------------------------------------------
#
# Why a separate prompt? `DEFAULT_AGENT_SYSTEM` describes the
# `Thought:`/`Action:`/`Action Input:` format that the ReAct parser
# expects. Sending that prompt to a model that's about to emit
# structured `tool_calls` actively mis-trains the model — it would
# either emit ReAct text (which we'd ignore) or hybridize (which
# parses badly). The native channel needs to be told "you have tools,
# call them when needed, give a plain final answer when done," and
# *nothing* about text markers.
NATIVE_DEFAULT_AGENT_SYSTEM = (
    "You are a careful agent that solves problems step by step using "
    "tools. When the user's question requires computation, file access, "
    "or live data, call one of the provided tools. Call one tool per "
    "turn, wait for the result, then decide what to do next. When you "
    "have enough information to answer, reply in plain text — no tool "
    "call is needed for the final answer.\n"
    "\n"
    "Once a tool result already contains what the user asked for, your "
    "next response MUST be the plain-text final answer. Do NOT call "
    "another tool to verify, double-check, convert, or reformat a value "
    "you already have."
)


# ---------------------------------------------------------------------------
# Wrap: AgentStep history → messages list.
# ---------------------------------------------------------------------------


def _history_to_messages(
    *,
    system: str,
    user_message: str,
    plan: Plan | None,
    steps: list[AgentStep],
) -> list[dict[str, Any]]:
    """Build the messages list that the next `chat_with_tools` call sees.

    Each prior `AgentStep` with an action expands into two messages:
    an `assistant` turn carrying the `tool_calls` payload, followed
    by a `tool` turn carrying the observation. Steps with no action
    (final-answer steps or stuck steps) are not included — they don't
    belong in the in-progress history.

    The plan, if present, is appended to the system prompt as a soft
    prior. It's not a separate message because the structured chat
    protocol has no "plan" role; sticking it in the system context is
    the natural place.
    """
    full_system = system
    if plan is not None:
        full_system = full_system + "\n\n" + render_plan_block(
            plan.goal, plan.steps
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_message},
    ]

    for i, step in enumerate(steps):
        if step.action is None:
            # No action means either a final-answer step (shouldn't
            # appear in mid-run history) or a stuck step (no useful
            # information to include). Skip either way.
            continue
        call_id = f"call_{i}"
        messages.append(
            {
                "role": "assistant",
                "content": step.thought or "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": step.action.tool,
                            "arguments": step.action.arguments,
                        },
                    }
                ],
            }
        )
        if step.observation is not None:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": step.action.tool,
                    "content": step.observation.output,
                }
            )
    return messages


# ---------------------------------------------------------------------------
# Unwrap: ChatResult → InferenceResult (for AgentStep parity).
# ---------------------------------------------------------------------------


def _chat_to_inference_result(
    chat: ChatResult, user_message: str
) -> InferenceResult:
    """Pack a `ChatResult` into an `InferenceResult` so it fits in
    `AgentStep.inference`. The full chat metadata is preserved under
    `metadata["chat"]` for callers that want to inspect the structured
    tool calls."""
    meta = dict(chat.metadata)
    meta["chat"] = {
        "content": chat.content,
        "tool_calls": chat.tool_calls,
    }
    return InferenceResult(
        prompt=user_message,
        completion=chat.content,
        prompt_tokens=chat.prompt_tokens,
        completion_tokens=chat.completion_tokens,
        latency_ms=chat.latency_ms,
        backend=chat.backend,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# NativeAgent.
# ---------------------------------------------------------------------------


class NativeAgent:
    """Structured-tool-calling agent runtime.

    Same arguments and behavior as `Agent`, but speaks the native
    chat-with-tools protocol instead of ReAct text. The backend must
    expose `chat_with_tools` (i.e., be an `OllamaBackend` or another
    backend that implements the duck-typed method).

    Args mirror `Agent`'s; see that class for full descriptions.

    Differences from `Agent`:

      * The system prompt defaults to `NATIVE_DEFAULT_AGENT_SYSTEM`
        (no ReAct markers). Override by passing `system=...`.
      * `think: bool | None = None` — forwarded to `chat_with_tools`.
        Set `False` to disable Ollama's thinking mode on Qwen3 /
        DeepSeek-R1 / etc. Set `None` to let the server pick.
      * `halt_on_stuck` here means "stop if the model returned neither
        tool_calls nor non-empty content" — the structured analog of
        "no parse."

    Backends without `chat_with_tools` are rejected at construction
    with `AgentError`.
    """

    def __init__(
        self,
        backend: Backend,
        registry: ToolRegistry,
        *,
        max_steps: int = 8,
        plan: bool = True,
        loop_detection: bool = True,
        halt_on_stuck: bool = False,
        scratchpad_max_chars: int | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_k: int | None = None,
        top_p: float | None = None,
        system: str | None = None,
        think: bool | None = None,
    ) -> None:
        if not isinstance(backend, Backend):
            raise AgentError(
                f"backend must be a Backend, got {type(backend).__name__}"
            )
        if not backend_supports_native_tools(backend):
            raise AgentError(
                f"NativeAgent requires a backend that supports "
                f"chat_with_tools, but {type(backend).__name__} does not. "
                "Use Agent (ReAct) instead, or wrap an Ollama backend."
            )
        if not isinstance(registry, ToolRegistry):
            raise AgentError(
                f"registry must be a ToolRegistry, got {type(registry).__name__}"
            )
        if max_steps <= 0:
            raise AgentError(f"max_steps must be > 0, got {max_steps}")
        if scratchpad_max_chars is not None and scratchpad_max_chars <= 0:
            raise AgentError(
                f"scratchpad_max_chars must be > 0 or None, "
                f"got {scratchpad_max_chars}"
            )

        self.backend = backend
        self.registry = registry
        self.max_steps = max_steps
        self.plan = plan
        self.loop_detection = loop_detection
        self.halt_on_stuck = halt_on_stuck
        self.scratchpad_max_chars = scratchpad_max_chars
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.system = system or NATIVE_DEFAULT_AGENT_SYSTEM
        self.think = think

    def __repr__(self) -> str:
        return (
            f"NativeAgent(backend={self.backend.info.name}, "
            f"tools={self.registry.names()}, max_steps={self.max_steps})"
        )

    def run(self, user_message: str) -> AgentRunResult:
        """Execute the structured-tool-calling loop on `user_message`.

        Loop logic is structurally identical to `Agent.run`: planning
        phase (optional), then up to `max_steps` iterations of "call
        model, dispatch tool, append step, check stop conditions."
        The only mechanical difference is wrapping the history into a
        messages list before each call and unwrapping the response
        into an `AgentStep` after.

        Stop conditions:

          * `final_answer` — model returned non-empty content with no
            `tool_calls`. The content IS the final answer.
          * `duplicate_action` — model emitted the same tool_call (same
            name + same arguments) two turns in a row. Same heuristic
            as `Agent`.
          * `no_progress` — model returned empty content AND no
            `tool_calls`, AND `halt_on_stuck=True`. By default the loop
            keeps going and the model gets another shot.
          * `max_steps` — safety net.
        """
        if not isinstance(user_message, str) or not user_message:
            raise ValueError("user_message must be a non-empty str")

        # Planning phase (same as Agent). Failure → run without plan.
        plan: Plan | None = None
        if self.plan:
            try:
                plan = make_plan(
                    self.backend,
                    user_message,
                    self.registry,
                    max_new_tokens=min(self.max_new_tokens, 256),
                    temperature=self.temperature,
                )
            except Exception:  # noqa: BLE001
                plan = None

        steps: list[AgentStep] = []
        final_answer: str | None = None
        stopped_reason = "max_steps"
        last_action_key: tuple[str, str] | None = None
        n_rescued_calls = 0

        tool_specs = render_tools_for_ollama(self.registry.tools)

        for _ in range(self.max_steps):
            messages = _history_to_messages(
                system=self.system,
                user_message=user_message,
                plan=plan,
                steps=steps,
            )

            extra_kwargs: dict[str, Any] = {}
            if self.think is not None:
                extra_kwargs["think"] = self.think

            chat: ChatResult = self.backend.chat_with_tools(  # type: ignore[attr-defined]
                messages,
                tool_specs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                **extra_kwargs,
            )
            inference = _chat_to_inference_result(chat, user_message)

            content = chat.content or ""
            tool_calls = list(chat.tool_calls)

            # Rescue: small models occasionally emit a tool call as plain
            # JSON in `content` instead of via the structured wire format.
            # If we have no structured calls but the content parses as a
            # tool-call object, synthesize one and continue the loop. Same
            # mechanism as run_with_tools' native channel.
            if not tool_calls and content:
                rescued = _try_parse_tool_call_from_content(content)
                if rescued is not None:
                    tool_calls = [
                        {
                            "name": rescued.name,
                            "arguments": rescued.arguments,
                            "call_id": rescued.call_id,
                        }
                    ]
                    n_rescued_calls += 1

            # Branch 1: no tool calls → either final answer (content
            # non-empty) or stuck (content empty).
            if not tool_calls:
                if content:
                    step = AgentStep(
                        completion=content,
                        thought="",
                        action=None,
                        observation=None,
                        final_answer=content,
                        parse_error=None,
                        inference=inference,
                    )
                    steps.append(step)
                    final_answer = content
                    stopped_reason = "final_answer"
                    break

                # Empty content + no tool_calls. The model returned
                # nothing useful. Mirror Agent's "stuck step" semantics.
                step = AgentStep(
                    completion="",
                    thought="",
                    action=None,
                    observation=None,
                    final_answer=None,
                    parse_error="empty response (no content, no tool_calls)",
                    inference=inference,
                )
                steps.append(step)
                if self.halt_on_stuck:
                    stopped_reason = "no_progress"
                    break
                continue

            # Branch 2: at least one tool_call. Dispatch the first one.
            # `AgentStep` carries a single action; richer parallel-call
            # support would emit one step per call. See module docstring.
            call_data = tool_calls[0]
            action = Action(
                tool=call_data["name"],
                arguments=call_data["arguments"],
            )
            action_key = (
                action.tool,
                json.dumps(action.arguments, sort_keys=True),
            )

            tool_call = ToolCall(
                name=action.tool,
                arguments=action.arguments,
                call_id=str(call_data.get("call_id") or f"call_{len(steps)}"),
            )
            tool_result = dispatch_tool_call(self.registry, tool_call)
            observation = Observation(
                output=tool_result.output,
                is_error=tool_result.is_error,
            )

            step = AgentStep(
                completion=content,
                thought=content,  # native: any free-form content is the "thought"
                action=action,
                observation=observation,
                final_answer=None,
                parse_error=None,
                inference=inference,
            )
            steps.append(step)

            if self.loop_detection and last_action_key == action_key:
                stopped_reason = "duplicate_action"
                break
            last_action_key = action_key

        return AgentRunResult(
            user_message=user_message,
            plan=plan,
            final_answer=final_answer,
            steps=steps,
            stopped_reason=stopped_reason,
            metadata={
                "n_steps": len(steps),
                "n_tool_calls": sum(1 for s in steps if s.action is not None),
                "tools_available": self.registry.names(),
                "backend_name": self.backend.info.name,
                "backend_model_id": self.backend.info.model_id,
                "had_plan": plan is not None,
                "channel": "native",
                "n_rescued_calls": n_rescued_calls,
            },
        )
