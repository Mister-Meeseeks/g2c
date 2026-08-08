# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.harness.agent pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from g2c.tools import ToolCall


class _HarnessAgentImpl:  # patched onto HarnessAgent by apply()
    def resume(self):
        events = self.log.replay()
        if not any(e.type == "task" for e in events):
            raise ValueError("nothing to resume: the log holds no task event")
        if any(e.type == "final_answer" for e in events):
            raise ValueError(
                "nothing to resume: this run already finished with a "
                "final answer"
            )

        unresolved = [
            e
            for e in events
            if e.type == "tool_call"
            and not any(
                r.type == "tool_result" and r.call_id == e.call_id
                for r in events
            )
        ]
        for e in unresolved:
            self.runner.execute(
                ToolCall(
                    name=e.payload["tool"],
                    arguments=e.payload["arguments"],
                    call_id=e.call_id,
                )
            )
        return self._loop()
