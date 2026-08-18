# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.harness.runner pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from g2c.harness.events import Permission
from g2c.harness.runner import UNKNOWN_OUTCOME_STATUS
from g2c.tools import ToolCall, ToolResult, dispatch_tool_call


class _ToolRunnerImpl:  # patched onto ToolRunner by apply()
    def execute(self, call: ToolCall) -> ToolResult:
        events = self.log.replay()

        # A call id is an idempotency key only if it names one immutable
        # operation. Treat collisions as corrupt history/caller errors.
        issued_events = [
            e
            for e in events
            if e.type == "tool_call" and e.call_id == call.call_id
        ]
        if any(
            e.payload["tool"] != call.name
            or e.payload["arguments"] != call.arguments
            for e in issued_events
        ):
            raise ValueError(
                f"call_id {call.call_id!r} is already bound to a different "
                "tool call"
            )

        # 1. Dedupe: a recorded result is THE result.
        for e in events:
            if e.type == "tool_result" and e.call_id == call.call_id:
                return ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    output=e.payload["output"],
                    is_error=bool(e.payload["is_error"]),
                )

        # 2. Crash window: intent logged, outcome unknown — never
        # blindly re-run.
        if issued_events:
            output = (
                f"call {call.call_id} has an unknown outcome after a crash; "
                "it may or may not have executed. Reconcile workspace state "
                "before deciding whether to retry with a new call id"
            )
            self.log.append(
                "tool_result",
                {
                    "output": output,
                    "is_error": True,
                    "status": UNKNOWN_OUTCOME_STATUS,
                },
                call_id=call.call_id,
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=output,
                is_error=True,
            )

        # 3. Permission policy.
        permission = self.permission_for(call.name)
        refusal: str | None = None
        if permission == Permission.DENY:
            refusal = (
                f"permission denied by harness policy for tool {call.name!r}"
            )
        elif permission == Permission.ASK and not self._approved(call.call_id):
            refusal = (
                f"approval required for tool {call.name!r} and none is on "
                "the log — permission denied"
            )
        if refusal is not None:
            self.log.append(
                "tool_call",
                {"tool": call.name, "arguments": call.arguments},
                call_id=call.call_id,
            )
            self.log.append(
                "tool_result",
                {"output": refusal, "is_error": True, "status": "refused"},
                call_id=call.call_id,
            )
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output=refusal,
                is_error=True,
            )

        # 4. Execute: intent before action, outcome after.
        self.log.append(
            "tool_call",
            {"tool": call.name, "arguments": call.arguments},
            call_id=call.call_id,
        )
        result = dispatch_tool_call(self.registry, call)
        self.log.append(
            "tool_result",
            {
                "output": result.output,
                "is_error": result.is_error,
                "status": "error" if result.is_error else "ok",
            },
            call_id=call.call_id,
        )
        return result
