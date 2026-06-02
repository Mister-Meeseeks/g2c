# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.agent.memory pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import json  # noqa: F401 (used by render scaffold)
from g2c.agent.base import AgentStep

from g2c.agent.memory import Scratchpad


class _ScratchpadImpl:  # patched onto Scratchpad by apply()
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
        if not self._steps:
            return ""

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

        full = "\n\n".join(blocks)
        if self._max_chars is None or len(full) <= self._max_chars:
            return full

        while len(blocks) > 1 and len("\n\n".join(blocks)) > self._max_chars:
            blocks.pop(0)
        return "\n\n".join(blocks)

