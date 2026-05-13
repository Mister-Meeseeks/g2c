"""Agent — the ReAct loop that ties everything together.

`Agent.run(user_message)` orchestrates:

    1. (Optional) planning phase: call backend once with the planning
       prompt; parse a Plan out of the response.
    2. Build the initial transcript: system prompt + plan block + user
       question + "Thought:" prefix.
    3. Loop up to `max_steps`:
         a. Call backend.complete on the current transcript.
         b. Parse the completion via `parse_react_step`.
         c. If the parsed step has a Final Answer, return success.
         d. If it has an Action, dispatch it via the tool registry and
            observe the result.
         e. If it has neither (parse error / stuck step), record the
            error as an Observation and continue (or stop, per
            configuration).
         f. Append the step to the scratchpad.
         g. Re-render the transcript with the updated scratchpad and
            another "Thought:" prefix for the next turn.
    4. If `max_steps` is hit without a Final Answer, return with
       `stopped_reason="max_steps"`.

Stop conditions, in order of priority:

    * `final_answer` — model emitted Final Answer (clean exit).
    * `duplicate_action` — model called the same action with the
      same arguments two steps in a row (loop detection). Optional;
      controlled by `loop_detection=True` (default).
    * `no_progress` — model emitted neither action nor final answer
      AND `halt_on_stuck=True`. Default is to feed the parse error
      back as an Observation and keep going.
    * `max_steps` — the safety net.

`Agent.run` is **SCAFFOLDED**. It's the main lesson of Module 19 — the
orchestration that turns a one-shot backend into a multi-step agent.
The supporting pieces (parse_react_step, Scratchpad.render,
extract_plan) are also scaffolded but smaller and isolated.
"""
from __future__ import annotations

import json

from g2c.inference import Backend
from g2c.tools import (
    ToolCall,  # noqa: F401 (used by run scaffold)
    ToolRegistry,
    dispatch_tool_call,  # noqa: F401 (used by run scaffold)
)

from .base import (
    AgentError,
    AgentRunResult,
    AgentStep,  # noqa: F401 (used by run scaffold)
    Observation,  # noqa: F401 (used by run scaffold)
    Plan,
)
from .memory import Scratchpad  # noqa: F401 (used by run scaffold)
from .parser import parse_react_step  # noqa: F401 (used by run scaffold)
from .planner import make_plan  # noqa: F401 (used by run scaffold)
from .prompts import (  # noqa: F401 (used by run scaffold)
    render_plan_block,
    render_system_prompt,
)


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

        Implemented (not scaffolded). The lesson is the loop logic
        (in `run`); this is layout.
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

        Args:
            user_message: the user's task / question. Non-empty str.

        Returns:
            `AgentRunResult` with `final_answer` (or None on
            timeout/stuck), `steps` (every iteration's record),
            `stopped_reason`, the parsed `plan` (or None), and
            `metadata`.

        The loop NEVER raises on model wobble. Bad parses, unknown
        tools, tool exceptions — all surface as data on the step
        records. The only way `run` raises is misuse: empty
        `user_message`, etc.

        Recipe:

            1. # Validate.
               if not isinstance(user_message, str) or not user_message:
                   raise ValueError("user_message must be a non-empty str")

            2. # Optional planning phase. Failure → run without plan.
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
                   except Exception:
                       plan = None

            3. # Init state.
               scratchpad = Scratchpad(max_chars=self.scratchpad_max_chars)
               steps: list[AgentStep] = []
               final_answer: str | None = None
               stopped_reason = "max_steps"
               last_action_key: tuple[str, str] | None = None

            4. # Main loop.
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

            5. #     Final answer branch.
                   if parsed.final_answer is not None:
                       step = AgentStep(
                           completion=inference.completion,
                           thought=parsed.thought,
                           action=None,
                           observation=None,
                           final_answer=parsed.final_answer,
                           parse_error=None,
                           inference=inference,
                       )
                       steps.append(step)
                       final_answer = parsed.final_answer
                       stopped_reason = "final_answer"
                       break

            6. #     Action branch.
                   if parsed.action is not None:
                       # Loop detection — same action + same args twice in a row.
                       import json
                       action_key = (
                           parsed.action.tool,
                           json.dumps(parsed.action.arguments, sort_keys=True),
                       )
                       if self.loop_detection and last_action_key == action_key:
                           # Record the step, then stop.
                           call = ToolCall(
                               name=parsed.action.tool,
                               arguments=parsed.action.arguments,
                               call_id=f"call_{len(steps)}",
                           )
                           tool_result = dispatch_tool_call(self.registry, call)
                           obs = Observation(
                               output=tool_result.output,
                               is_error=tool_result.is_error,
                           )
                           step = AgentStep(
                               completion=inference.completion,
                               thought=parsed.thought,
                               action=parsed.action,
                               observation=obs,
                               final_answer=None,
                               parse_error=None,
                               inference=inference,
                           )
                           steps.append(step)
                           scratchpad.append(step)
                           stopped_reason = "duplicate_action"
                           break
                       last_action_key = action_key

                       # Dispatch via Module 18.
                       call = ToolCall(
                           name=parsed.action.tool,
                           arguments=parsed.action.arguments,
                           call_id=f"call_{len(steps)}",
                       )
                       tool_result = dispatch_tool_call(self.registry, call)
                       obs = Observation(
                           output=tool_result.output,
                           is_error=tool_result.is_error,
                       )
                       step = AgentStep(
                           completion=inference.completion,
                           thought=parsed.thought,
                           action=parsed.action,
                           observation=obs,
                           final_answer=None,
                           parse_error=None,
                           inference=inference,
                       )
                       steps.append(step)
                       scratchpad.append(step)
                       continue

            7. #     Stuck step — no action AND no final answer.
                   step = AgentStep(
                       completion=inference.completion,
                       thought=parsed.thought,
                       action=None,
                       observation=None,
                       final_answer=None,
                       parse_error=parsed.parse_error or "no parse",
                       inference=inference,
                   )
                   steps.append(step)
                   if self.halt_on_stuck:
                       stopped_reason = "no_progress"
                       break
                   # Otherwise: append to scratchpad (renders as
                   # parse-error observation) and let the model retry.
                   scratchpad.append(step)

            8. # Build result.
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

        Implementation notes:

          * **The "Thought:" prefix at end of prompt.** The
            `_build_prompt` helper adds `"\\n\\nThought:"` after the
            scratchpad. This nudges instruction-tuned models into the
            ReAct format. Without it, they often start with a sentence
            of prose ("I think we should...") that the parser then
            has to recover from.

          * **Why catch all exceptions in the planning phase?** The
            agent should not crash because the planner emitted bad
            JSON or the backend hiccuped. A failed plan just means
            "run without one" — graceful degradation.

          * **Why dispatch through Module 18's `dispatch_tool_call`?**
            Because Module 18 already implements the validate →
            execute → wrap-result pipeline, including the three error
            branches (unknown tool, bad args, runtime exception). The
            agent module shouldn't reimplement it.

          * **Why is `is_error` propagated to Observation?** So the
            scratchpad can render `"Observation: [error] ..."` instead
            of `"Observation: ..."`. Empirically, the `[error]` prefix
            is what makes the model reliably treat errors as recovery
            signals rather than parroting them as answers.

          * **Loop detection is a simple equality check.** Same tool,
            same arguments (compared by canonical JSON), TWO STEPS IN
            A ROW. A more sophisticated check would look at the last
            N steps, or use semantic similarity between actions. The
            simple check catches the most common loop pattern (model
            calls the same tool with the same args because it didn't
            understand the observation).

          * **Why does loop detection still record the duplicate
            step?** So the AgentRunResult has a complete picture of
            what happened. The model called the duplicate; we
            dispatched it; we just chose to stop afterward.

        Sanity values:

          * Backend that always emits "Final Answer: 42":
            run("anything") → final_answer="42", stopped_reason=
            "final_answer", 1 step (the planning step doesn't count;
            it's not in `steps`).

          * Backend that emits Action calculator(2+2) on step 1 and
            "Final Answer: 4" on step 2: 2 steps, final_answer="4".

          * Backend that always emits an Action: max_steps trips,
            final_answer=None, stopped_reason="max_steps".

          * Backend that emits the same Action twice with
            loop_detection=True: 2 steps recorded, stopped_reason=
            "duplicate_action", final_answer=None.

          * Backend that emits "I don't know" (no parse) twice with
            halt_on_stuck=True: 1 step, stopped_reason="no_progress".

          * Backend that emits "I don't know" (no parse) twice with
            halt_on_stuck=False: 2 steps, both with parse_error set,
            stopped_reason="max_steps" (or whatever the next branch
            triggers).
        """
        if not isinstance(user_message, str) or not user_message:
            raise ValueError("user_message must be a non-empty str")

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
            except Exception:  # noqa: BLE001 - failed planning is non-fatal
                plan = None

        scratchpad = Scratchpad(max_chars=self.scratchpad_max_chars)
        steps: list[AgentStep] = []
        final_answer: str | None = None
        stopped_reason = "max_steps"
        last_action_key: tuple[str, str] | None = None

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

            if parsed.final_answer is not None:
                step = AgentStep(
                    completion=inference.completion,
                    thought=parsed.thought,
                    action=None,
                    observation=None,
                    final_answer=parsed.final_answer,
                    parse_error=None,
                    inference=inference,
                )
                steps.append(step)
                final_answer = parsed.final_answer
                stopped_reason = "final_answer"
                break

            if parsed.action is not None:
                action_key = (
                    parsed.action.tool,
                    json.dumps(parsed.action.arguments, sort_keys=True),
                )
                call = ToolCall(
                    name=parsed.action.tool,
                    arguments=parsed.action.arguments,
                    call_id=f"call_{len(steps)}",
                )
                tool_result = dispatch_tool_call(self.registry, call)
                output = tool_result.output
                if tool_result.is_error and parsed.action.tool not in self.registry:
                    allowed = ", ".join(self.registry.names()) or "(none)"
                    output = (
                        f"{output} Use Action with one of the registered tool "
                        f"names: {allowed}. Put operators or other inputs inside "
                        "the Action Input JSON."
                    )
                observation = Observation(
                    output=output,
                    is_error=tool_result.is_error,
                )
                step = AgentStep(
                    completion=inference.completion,
                    thought=parsed.thought,
                    action=parsed.action,
                    observation=observation,
                    final_answer=None,
                    parse_error=None,
                    inference=inference,
                )
                steps.append(step)
                scratchpad.append(step)

                if self.loop_detection and last_action_key == action_key:
                    stopped_reason = "duplicate_action"
                    break
                last_action_key = action_key
                continue

            step = AgentStep(
                completion=inference.completion,
                thought=parsed.thought,
                action=None,
                observation=None,
                final_answer=None,
                parse_error=parsed.parse_error or "no parse",
                inference=inference,
            )
            steps.append(step)
            if self.halt_on_stuck:
                stopped_reason = "no_progress"
                break
            scratchpad.append(step)

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
