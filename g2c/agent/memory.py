"""Scratchpad — accumulate steps and render them back into the prompt.

The scratchpad is the agent's short-term memory. Every step adds a
record of `(thought, action, observation)` (or just `thought`, if the
step ended with a Final Answer). Before the next step, the scratchpad
renders all past records as text and splices them into the prompt so
the model sees its own history.

This is what makes the loop "ReAct" instead of "tool use" — the
explicit interleaving of THOUGHT, ACTION, OBSERVATION is what lets
the model reason about its own past actions on the next turn. Without
the scratchpad, every step would be the model "starting fresh" with
no awareness of what it had already tried.

Two pieces:

  * `Scratchpad.append(step)` — record one AgentStep. Implemented.

  * `Scratchpad.render() -> str` — produce the text block to splice
    into the next prompt. **SCAFFOLDED.** It's the lesson on "what
    does the model see at step N?"

The format the model sees:

    Thought: <step 1's thought>
    Action: <step 1's action.tool>
    Action Input: <step 1's action.arguments as JSON>
    Observation: <step 1's observation.output>
    Thought: <step 2's thought>
    Action: <step 2's action.tool>
    ...

Exactly the format the system prompt asked for. The model continues
the pattern by emitting its next Thought / Action / Action Input.

Why is `render` scaffolded when the format is so simple? Because the
scaffolding makes a precise pedagogical point: *the prompt grows by
exactly one (thought, action, observation) block per step*. Students
who skip the scaffold rarely build the right mental model of what
the model sees on the 5th turn vs. the 1st. Implementing it once,
even as a small recipe, makes the contract concrete.
"""
from __future__ import annotations

import json  # noqa: F401 (used by render scaffold)

from .base import AgentStep


class Scratchpad:
    """Accumulator + renderer for the agent's scratchpad.

    Args:
        max_chars: optional cap on the rendered length, in characters.
            When the rendered scratchpad would exceed `max_chars`, the
            renderer drops the earliest steps until it fits. `None`
            (the default) means unlimited — rely on the backend's
            context cap. The cap is in characters, not tokens, because
            this module deliberately stays tokenizer-agnostic; a real
            production agent would token-count.

    Use `append(step)` to add records, `render()` to produce the text
    block. `len(scratchpad)` returns the number of recorded steps.
    """

    def __init__(self, *, max_chars: int | None = None) -> None:
        if max_chars is not None and max_chars <= 0:
            raise ValueError(f"max_chars must be > 0 or None, got {max_chars}")
        self._steps: list[AgentStep] = []
        self._max_chars = max_chars

    def append(self, step: AgentStep) -> None:
        """Record one step. Order matters — render walks them in order.

        The `step` is stored by reference. If a caller mutates the
        step after appending, the scratchpad sees the mutation. This
        is fine in practice — `AgentStep` is built up across one
        iteration of the loop and then frozen via convention. If you
        want defensive copying, replace this with a deepcopy.
        """
        if not isinstance(step, AgentStep):
            raise TypeError(
                f"Scratchpad.append expected AgentStep, "
                f"got {type(step).__name__}"
            )
        self._steps.append(step)

    @property
    def steps(self) -> list[AgentStep]:
        """Snapshot of recorded steps. Safe to mutate without
        affecting the scratchpad.
        """
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"Scratchpad(steps={len(self._steps)})"

    def render(self) -> str:
        """Produce the scratchpad block to splice into the next prompt.

        Each step's contribution:

          * If the step has both an Action and an Observation:

                Thought: <thought>
                Action: <action.tool>
                Action Input: <json.dumps(action.arguments)>
                Observation: <observation.output, prefixed [error] if is_error>

          * If the step has a Final Answer (no action):

                Thought: <thought>
                Final Answer: <final_answer>

          * If the step is a parse-error step (no action, no final
            answer): include just the Thought (if any) plus a
            recovery hint:

                Thought: <thought>
                Observation: [parse error] <parse_error>

        Steps are joined with a single blank line between blocks. The
        final block does NOT have a trailing blank line — the caller
        is expected to splice the scratchpad immediately before
        "Thought:" (the model's next-turn prefix).

        If `max_chars` is set and the full rendering exceeds it, drop
        the earliest blocks one at a time until it fits. Steps from
        the end of the run are more useful for the model's next
        decision than steps from the start. (A more sophisticated
        truncation would summarize old steps; we just drop them.)

        Returns:
            The rendered scratchpad. Empty string when there are no
            steps yet (the first turn).

        Recipe:

            1. # Empty case.
               if not self._steps:
                   return ""

            2. # Render each step into a list of blocks.
               blocks: list[str] = []
               for step in self._steps:
                   lines: list[str] = []
                   if step.thought:
                       lines.append(f"Thought: {step.thought}")

                   if step.final_answer is not None:
                       lines.append(f"Final Answer: {step.final_answer}")
                   elif step.action is not None and step.observation is not None:
                       args_json = json.dumps(step.action.arguments)
                       lines.append(f"Action: {step.action.tool}")
                       lines.append(f"Action Input: {args_json}")
                       prefix = "[error] " if step.observation.is_error else ""
                       lines.append(f"Observation: {prefix}{step.observation.output}")
                   elif step.parse_error is not None:
                       lines.append(f"Observation: [parse error] {step.parse_error}")

                   blocks.append("\n".join(lines))

            3. # Join blocks with blank lines between them.
               full = "\n\n".join(blocks)

            4. # Apply truncation: drop earliest blocks until fits.
               if self._max_chars is None or len(full) <= self._max_chars:
                   return full
               while len(blocks) > 1 and len("\n\n".join(blocks)) > self._max_chars:
                   blocks.pop(0)
               return "\n\n".join(blocks)

        Implementation notes:

          * **Why JSON-dump the arguments?** The model originally
            emitted JSON; rendering them back as `json.dumps(args)`
            gives the model a consistent format on every turn.
            `repr(args)` would produce Python-dict syntax (single
            quotes) which the model often confuses for "the runtime
            wants Python now."

          * **Why prefix errors with `[error]`?** The model treats
            error observations very differently from successful ones.
            Without the prefix, it often parrots the error string
            back as if it were the answer ("The calculator returned:
            missing required arguments"). With the prefix, instruction-
            tuned models reliably read it as a recovery signal.

          * **Why drop oldest, not newest, on truncation?** The most
            recent observation is the most relevant context for the
            model's next action. Old steps fade into "context the
            model already has internalized" as the loop progresses.
            (A real agent would summarize old steps; we drop them.)

          * **Why the leading "Thought:" of the next turn is NOT
            part of render?** The agent appends `"\\nThought:"` to
            the prompt itself, after the scratchpad, to nudge the
            model into the right format. Letting render add it would
            duplicate the prefix on every step.

        Sanity values:

          * Empty scratchpad → "".

          * One step with Thought + Action + Observation:
                "Thought: use calc\\nAction: calculator\\n"
                "Action Input: {\\"expression\\": \\"2+2\\"}\\n"
                "Observation: 4"

          * Two steps → blocks joined by "\\n\\n".

          * Step with is_error=True observation → "Observation: [error] ..."

          * max_chars=50 with a 200-char scratchpad → only the last
            block(s) appear, totaling ≤50 chars (or the very last
            block alone if it's already over the cap).
        """
        # TODO
        raise NotImplementedError
