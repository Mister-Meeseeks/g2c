# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.agent.agent pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import json

from g2c.agent.agent import Agent
from g2c.agent.base import (
    Action,
    AgentError,
    AgentRunResult,
    AgentStep,
    Observation,
    Plan,
    StepOutcome,
)
from g2c.agent.memory import Scratchpad
from g2c.agent.parser import ParsedStep, parse_react_step
from g2c.agent.planner import make_plan
from g2c.agent.prompts import render_plan_block, render_system_prompt
from g2c.inference import Backend, InferenceResult
from g2c.tools import ToolCall, ToolRegistry, dispatch_tool_call


class _AgentImpl:  # patched onto Agent by apply()
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
        # Final answer -> clean exit; nothing more reads the scratchpad.
        if parsed.final_answer is not None:
            return StepOutcome(
                AgentStep.final(
                    inference,
                    thought=parsed.thought,
                    final_answer=parsed.final_answer,
                ),
                stop_reason="final_answer",
                remember=False,
            )

        # Action -> dispatch, observe, remember; stop on a repeat.
        if parsed.action is not None:
            observation = self._observe(parsed.action, len(steps))
            step = AgentStep.act(
                inference,
                thought=parsed.thought,
                action=parsed.action,
                observation=observation,
            )
            if self.loop_detection and (
                _prev_action_key(steps) == _action_key(parsed.action)
            ):
                return StepOutcome(step, stop_reason="duplicate_action")
            return StepOutcome(step)

        # Stuck -> halt if configured, else remember and let the model retry.
        step = AgentStep.stuck(
            inference,
            thought=parsed.thought,
            parse_error=parsed.parse_error or "no parse",
        )
        if self.halt_on_stuck:
            return StepOutcome(step, stop_reason="no_progress", remember=False)
        return StepOutcome(step)

