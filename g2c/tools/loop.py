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


def run_with_tools(
    backend: Backend,
    registry: ToolRegistry,
    user_message: str,
    *,
    system: str = DEFAULT_SYSTEM,
    max_steps: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_k: int | None = None,
    top_p: float | None = None,
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
    # TODO
    raise NotImplementedError
