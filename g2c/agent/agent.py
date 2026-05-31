"""Agent — the ReAct loop that ties everything together.

`Agent.run(user_message)` is the public entry point. It is split into a
provided DRIVER and a scaffolded POLICY:

  * `run` / `_run_loop` (PROVIDED) own the mechanics — the optional
    planning phase, the bounded `max_steps` iteration, the
    prompt → complete → parse pipeline, recording each step, and rolling
    up the `AgentRunResult`. This is the same loop you built in Module 18
    (`run_with_tools`); here it is handed to you.

  * `_decide_step` (SCAFFOLDED) is the per-step policy — and the whole
    Module-19 deliverable. Given one `ParsedStep`, it decides which of
    the three step shapes this is (final answer / action / stuck),
    dispatches a tool when needed, and returns a `StepOutcome` telling
    the driver whether to remember the step and whether to stop. This is
    where the agent's *behavior* lives: memory, tool use, and the
    stop-condition taxonomy.

The split keeps the lesson focused. The bounded loop is mechanical and
already familiar from Module 18; what's new in Module 19 — goal-directed
memory and the four stop conditions — is exactly what `_decide_step`
expresses, and nothing else.

Stop conditions, in order of priority:

    * `final_answer` — model emitted Final Answer (clean exit).
    * `duplicate_action` — model called the same action with the same
      arguments two steps in a row (loop detection). Optional; controlled
      by `loop_detection=True` (default).
    * `no_progress` — model emitted neither action nor final answer AND
      `halt_on_stuck=True`. Default is to feed the parse error back as an
      Observation and keep going.
    * `max_steps` — the safety net.

The supporting pieces (parse_react_step, Scratchpad.render, extract_plan)
are scaffolded in their own files; `_decide_step` is the orchestration
core.
"""
from __future__ import annotations

import json

from g2c.inference import Backend, InferenceResult
from g2c.tools import ToolCall, ToolRegistry, dispatch_tool_call

from .base import (
    Action,
    AgentError,
    AgentRunResult,
    AgentStep,
    Observation,
    Plan,
    StepOutcome,
)
from .memory import Scratchpad
from .parser import ParsedStep, parse_react_step
from .planner import make_plan
from .prompts import render_plan_block, render_system_prompt


def _action_key(action: Action) -> tuple[str, str]:
    """Canonical (tool, args) identity for loop detection.

    Arguments are serialized with sorted keys so two calls with the same
    arguments in a different dict order compare equal. PROVIDED — the
    canonicalization is plumbing; the *decision* to stop on a repeat is
    the lesson (see `_decide_step`).
    """
    return (action.tool, json.dumps(action.arguments, sort_keys=True))


def _prev_action_key(steps: list[AgentStep]) -> tuple[str, str] | None:
    """Key of the most recent prior ACTION step, or None.

    Scans back past stuck steps (which don't reset the "last action"),
    so an `A, stuck, A` sequence is still detected as a repeat. PROVIDED.
    """
    for step in reversed(steps):
        if step.action is not None:
            return _action_key(step.action)
    return None


class Agent:
    """ReAct-style agent runtime.

    Args:
        backend: an inference `Backend` (Module 16). Required.
        registry: a `ToolRegistry` (Module 18). Required, even if
            empty — the agent is allowed to run with no tools, in
            which case it can only emit a Final Answer.
        max_steps: cap on the main-loop iterations. Each step is one
            backend call. Defaults to 8 — enough for a typical 3-tool
            sequence with some recovery, not enough for an agent to
            burn unbounded compute.
        plan: whether to run the planning phase before the main loop.
            Default True. If the planner returns `None` (no parseable
            plan), the loop runs without one anyway.
        loop_detection: whether to stop when the same action is
            repeated with the same arguments two steps in a row. Default
            True. Disable for tasks where repeating an action is
            legitimate (e.g., paginated reads).
        halt_on_stuck: whether to stop on a stuck step (no action and
            no final answer). Default False — by default, the loop
            feeds the parse error back as an Observation and lets the
            model recover. Set True for stricter "no second chances."
        scratchpad_max_chars: cap on the rendered scratchpad length
            (characters). `None` (default) means unlimited.
        max_new_tokens: forwarded to backend.complete.
        temperature: forwarded. 0.2 is the ReAct default — near-greedy
            biases the model toward consistent format.
        top_k, top_p: forwarded.

    Raises:
        AgentError: on misconfigured args (bad backend type, bad
            max_steps).

    Use `agent.run(user_message)` to execute one task. The agent is
    re-entrant: `run` is stateless on the agent itself; one Agent
    instance can drive many independent runs.
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
    ) -> None:
        if not isinstance(backend, Backend):
            raise AgentError(
                f"backend must be a Backend, got {type(backend).__name__}"
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

    def __repr__(self) -> str:
        return (
            f"Agent(backend={self.backend.info.name}, "
            f"tools={self.registry.names()}, max_steps={self.max_steps})"
        )

    def _build_prompt(
        self,
        user_message: str,
        plan: Plan | None,
        scratchpad: Scratchpad,
    ) -> str:
        """Compose the next backend.complete prompt.

        Layout:

            <system prompt with tools>

            [Plan: ...]               # only if plan is not None

            Question: <user message>

            <scratchpad render>       # empty on the first turn

            Thought:                  # nudge for the model's next turn

        Implemented (not scaffolded). The lesson is the decision logic
        (in `_decide_step`); this is layout.
        """
        parts: list[str] = [render_system_prompt(self.registry.tools)]
        if plan is not None:
            parts.append(render_plan_block(plan.goal, plan.steps))
        parts.append(f"Question: {user_message}")
        rendered = scratchpad.render()
        if rendered:
            parts.append(rendered)
        # The trailing "Thought:" nudges the model into the right
        # format. Without it, models occasionally start with prose.
        return "\n\n".join(parts) + "\n\nThought:"

    def run(self, user_message: str) -> AgentRunResult:
        """Execute the ReAct loop on `user_message`.

        PROVIDED. The thin driver entry point: validate the input, run
        the optional planning phase, drive the main loop, and roll the
        result up. The per-step DECISION — the heart of Module 19 — lives
        in `_decide_step`, which you implement.

        Returns an `AgentRunResult` with `final_answer` (or None on
        timeout / stuck), `steps` (every iteration's record),
        `stopped_reason`, the parsed `plan` (or None), and `metadata`.

        Raises `ValueError` only on misuse (empty `user_message`). The
        loop itself NEVER raises on model wobble — bad parses, unknown
        tools, and tool exceptions all surface as data on the step
        records, courtesy of `_decide_step` + `_observe`.
        """
        if not isinstance(user_message, str) or not user_message:
            raise ValueError("user_message must be a non-empty str")

        plan = self._plan_phase(user_message)
        scratchpad = Scratchpad(max_chars=self.scratchpad_max_chars)
        steps, stopped_reason, final_answer = self._run_loop(
            user_message, plan, scratchpad
        )
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
            },
        )

    def _plan_phase(self, user_message: str) -> Plan | None:
        """PROVIDED. Optional planning phase with graceful degradation.

        Returns a `Plan` if planning is enabled and succeeds, else `None`
        (planning disabled, planner returned nothing, or the backend /
        parser raised — a failed plan is never fatal; the loop just runs
        without one).
        """
        if not self.plan:
            return None
        try:
            return make_plan(
                self.backend,
                user_message,
                self.registry,
                max_new_tokens=min(self.max_new_tokens, 256),
                temperature=self.temperature,
            )
        except Exception:  # noqa: BLE001 - failed planning is non-fatal
            return None

    def _run_loop(
        self,
        user_message: str,
        plan: Plan | None,
        scratchpad: Scratchpad,
    ) -> tuple[list[AgentStep], str, str | None]:
        """PROVIDED. The bounded ReAct driver.

        Identical in shape to the tool loop you built in Module 18: for
        up to `max_steps`, render the prompt, call the backend, parse the
        completion, then hand the parsed step to `_decide_step` (your
        policy). The driver owns only the mechanical parts — iteration,
        the prompt → complete → parse pipeline, recording each step,
        honoring the policy's `remember` / `stop_reason` signals, and the
        `max_steps` fall-through. Everything that makes this an *agent*
        lives in `_decide_step`.

        Returns `(steps, stopped_reason, final_answer)`.
        """
        steps: list[AgentStep] = []
        for _ in range(self.max_steps):
            prompt = self._build_prompt(user_message, plan, scratchpad)
            inference = self.backend.complete(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            parsed = parse_react_step(inference.completion)
            outcome = self._decide_step(parsed, inference, steps)
            steps.append(outcome.step)
            if outcome.remember:
                scratchpad.append(outcome.step)
            if outcome.stop_reason is not None:
                return steps, outcome.stop_reason, outcome.step.final_answer
        return steps, "max_steps", None

    def _observe(self, action: Action, step_index: int) -> Observation:
        """PROVIDED. Dispatch one Action through Module 18, wrap as Observation.

        Builds a `ToolCall`, dispatches it via `dispatch_tool_call`
        (which runs the validate → execute → wrap-result pipeline,
        including the unknown-tool / bad-args / runtime-error branches),
        and returns an `Observation`. On an unknown-tool error, the
        message is augmented with the list of registered tools so the
        model can correct on the next step.
        """
        call = ToolCall(
            name=action.tool,
            arguments=action.arguments,
            call_id=f"call_{step_index}",
        )
        result = dispatch_tool_call(self.registry, call)
        output = result.output
        if result.is_error and action.tool not in self.registry:
            allowed = ", ".join(self.registry.names()) or "(none)"
            output = (
                f"{output} Use Action with one of the registered tool "
                f"names: {allowed}. Put operators or other inputs inside "
                "the Action Input JSON."
            )
        return Observation(output=output, is_error=result.is_error)

    def _decide_step(
        self,
        parsed: ParsedStep,
        inference: InferenceResult,
        steps: list[AgentStep],
    ) -> StepOutcome:
        """Decide what one parsed step means — the Module-19 deliverable.

        This is the agent's policy. Given the `ParsedStep` the model just
        produced (and the steps so far, for loop detection), classify it,
        dispatch a tool if needed, and return a `StepOutcome` telling the
        driver what to record and whether to stop. The driver
        (`_run_loop`) owns the loop; you own the decision.

        Args:
            parsed: the `ParsedStep` from `parse_react_step`. On a
                well-formed step exactly one of `final_answer` / `action`
                is set; on a stuck step neither is (and `parse_error` is).
            inference: the `InferenceResult` that produced `parsed`. Pass
                it straight into the `AgentStep.*` factory.
            steps: every step recorded SO FAR this run (NOT including the
                current one). Used only for loop detection.

        Returns:
            A `StepOutcome(step, stop_reason=None, remember=True)`:
              * `step` — build it with `AgentStep.final/.act/.stuck`
                (these fill the mechanical fields for you).
              * `stop_reason` — set it to halt the loop (`"final_answer"`,
                `"duplicate_action"`, `"no_progress"`); leave `None` to
                keep going.
              * `remember` — leave True to append the step to the
                scratchpad (so the next prompt sees it); set False when
                the loop is ending and nothing will read it again.

        Do NOT mutate `steps` or the scratchpad here — return a value and
        let the driver act on it. That keeps this method pure and
        unit-testable: feed it a `ParsedStep`, assert the `StepOutcome`.

        Recipe:

            1. # Final answer -> clean exit. Nothing more will run, so
               #   there's no point remembering it.
               if parsed.final_answer is not None:
                   return StepOutcome(
                       AgentStep.final(inference, thought=parsed.thought,
                                       final_answer=parsed.final_answer),
                       stop_reason="final_answer", remember=False)

            2. # Action -> dispatch, observe, remember. Stop only if this
               #   repeats the previous action (loop detection).
               if parsed.action is not None:
                   observation = self._observe(parsed.action, len(steps))
                   step = AgentStep.act(inference, thought=parsed.thought,
                                        action=parsed.action,
                                        observation=observation)
                   if self.loop_detection and (
                           _prev_action_key(steps)
                           == _action_key(parsed.action)):
                       return StepOutcome(step, stop_reason="duplicate_action")
                   return StepOutcome(step)

            3. # Stuck -> neither action nor final answer. Halt if
               #   configured to; otherwise remember it and let the model
               #   recover on the next turn.
               step = AgentStep.stuck(
                   inference, thought=parsed.thought,
                   parse_error=parsed.parse_error or "no parse")
               if self.halt_on_stuck:
                   return StepOutcome(step, stop_reason="no_progress",
                                      remember=False)
               return StepOutcome(step)

        Sanity values (paired with the driver):

          * parsed.final_answer="42" -> stop_reason="final_answer",
            remember=False; the run ends with final_answer="42".
          * parsed.action=calculator(2+2), not a repeat -> stop_reason=None,
            remember=True; the observation ("4") lands in the scratchpad.
          * same action as the previous step, loop_detection=True ->
            stop_reason="duplicate_action" (the step is still recorded).
          * stuck step, halt_on_stuck=True -> stop_reason="no_progress".
          * stuck step, halt_on_stuck=False -> stop_reason=None,
            remember=True (parse error feeds back as an observation).
        """
        # TODO
        raise NotImplementedError
