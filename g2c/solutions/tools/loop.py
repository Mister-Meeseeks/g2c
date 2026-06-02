# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.tools.loop pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from g2c.inference import Backend
from g2c.tools.base import ToolRunResult, ToolStep
from g2c.tools.parser import parse_tool_calls
from g2c.tools.registry import ToolRegistry


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
    """The text-format tool-feedback loop — worked solution.

    `_init_transcript`, `_grow_transcript`, `_build_run_result`, and
    `dispatch_tool_call` are siblings in `g2c.tools.loop`; `apply()`
    rebinds this function's globals to that module so the bare names
    resolve to the provided helpers.
    """
    transcript = _init_transcript(system, registry, user_message)
    steps: list[ToolStep] = []

    for _ in range(max_steps):
        inference = backend.complete(
            transcript,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        tool_calls = parse_tool_calls(inference.completion)

        # No tool calls -> this completion is the final answer.
        if not tool_calls:
            steps.append(
                ToolStep(
                    completion=inference.completion,
                    tool_calls=[],
                    tool_results=[],
                    inference=inference,
                )
            )
            return _build_run_result(
                user_message=user_message,
                final_answer=inference.completion,
                steps=steps,
                stopped_reason="no_more_calls",
                registry=registry,
                backend=backend,
            )

        # Dispatch each call, record the step, feed the results back.
        results = [dispatch_tool_call(registry, c) for c in tool_calls]
        steps.append(
            ToolStep(
                completion=inference.completion,
                tool_calls=tool_calls,
                tool_results=results,
                inference=inference,
            )
        )
        transcript = _grow_transcript(transcript, inference.completion, results)

    # Ran out of steps without a final answer.
    return _build_run_result(
        user_message=user_message,
        final_answer=None,
        steps=steps,
        stopped_reason="max_steps",
        registry=registry,
        backend=backend,
    )
