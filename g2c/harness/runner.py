"""ToolRunner — permissioned tool execution over the log.

A harness that resumes after failures must reason about work whose
outcome was never recorded. The classic crash-consistency window is:

    log tool_call(c3) ──► execute ──► log tool_result(c3)
                     ▲              ▲
                     │              └── crash HERE: the log shows c3
                     │                  issued but unresolved. Blindly
                     │                  re-running would double any
                     │                  side effect that already
                     │                  landed.
                     └── crash HERE: c3 may not have executed, but a
                         restarted process cannot prove that from this
                         log alone.

Call ids dedupe calls whose results were recorded. They do NOT close
the unresolved-outcome window by themselves. This teaching runner
takes the conservative posture: never blindly re-execute an intent
whose outcome is unknown; surface it for reconciliation instead.
True exactly-once effects require support from the tool or storage
layer, such as an idempotency key or a transaction.
"""
from __future__ import annotations

from g2c.tools import (  # noqa: F401 - dispatcher used by scaffold
    ToolCall,
    ToolRegistry,
    ToolResult,
    dispatch_tool_call,
)

from .events import EventLog, Permission

UNKNOWN_OUTCOME_STATUS = "unknown_outcome_after_crash"


class ToolRunner:
    """Executes `ToolCall`s with the log as the arbiter of history.

    Args:
        registry: Module 18's `ToolRegistry`.
        log: the harness's `EventLog`. Every call and every result
            passes through it — the runner has no private memory.
        permissions: per-tool `Permission` overrides. Tools not listed
            get `default`.
        default: permission for unlisted tools. `ALLOW` by default —
            the exercises tighten it.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        log: EventLog,
        *,
        permissions: dict[str, Permission] | None = None,
        default: Permission = Permission.ALLOW,
    ) -> None:
        self.registry = registry
        self.log = log
        self.permissions = dict(permissions or {})
        self.default = default

    def permission_for(self, tool_name: str) -> Permission:
        return self.permissions.get(tool_name, self.default)

    def _approved(self, call_id: str) -> bool:
        """True iff an approval event for `call_id` is on the log."""
        return any(
            e.type == "approval"
            and e.call_id == call_id
            and e.payload.get("approved")
            for e in self.log.replay()
        )

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one call — or refuse, defer, or dedupe it — via the log.

        Returns a `ToolResult` in every case; the runner never raises
        on policy or history grounds (the model is supposed to SEE
        refusals and interruptions as observations).

        The decision ladder, in order:

        1. **Dedupe.** If the log already holds a `tool_result` for
           `call.call_id`, first verify that the id is bound to this exact
           tool and argument object, then reconstruct and return it. A
           mismatched reuse is a caller error, not a cache hit. The function
           must NOT run again — this is what makes retries after a
           crash safe when the result did get recorded.
        2. **Crash window.** If the log holds a `tool_call` for this
           id but NO `tool_result`, a previous process died between
           logging intent and logging outcome. Do NOT re-execute — any
           side effect may already have landed. Log and return an
           error result with `status=UNKNOWN_OUTCOME_STATUS` telling the model
           to verify workspace state before retrying (with a NEW call
           id, if it decides to).
        3. **Permission.** `DENY` → log the call, then log and return
           an error result ("permission denied by harness policy").
           `ASK` without an approval event on the log (`_approved`) →
           same, with "approval required". Note the marker vocabulary:
           "permission denied" is in `DETERMINISTIC_MARKERS`, so
           `classify_failure` will not retry a policy refusal.
        4. **Execute.** Log the `tool_call` event FIRST (intent before
           action — the whole crash-window story depends on this
           order), then `dispatch_tool_call(self.registry, call)`,
           then log the `tool_result`, then return it.

        Event payload shapes (the tests read these):
            tool_call:   {"tool": call.name, "arguments": call.arguments}
            tool_result: {"output": ..., "is_error": ...,
                          "status": "ok" | "error" |
                                    UNKNOWN_OUTCOME_STATUS | "refused"}

        Recipe sketch:
            events = self.log.replay()
            issued = [e for e in events if e.type == "tool_call"
                      and e.call_id == call.call_id]
            if any(e.payload["tool"] != call.name
                   or e.payload["arguments"] != call.arguments
                   for e in issued):
                raise ValueError(...)  # call id collision
            for e in events:  # 1. dedupe
                if e.type == "tool_result" and e.call_id == call.call_id:
                    return ToolResult(call.call_id, call.name,
                                      e.payload["output"], e.payload["is_error"])
            issued = any(e.type == "tool_call" and e.call_id == call.call_id
                         for e in events)
            if issued:        # 2. crash window
                ... log + return the unknown-outcome error result
            permission = self.permission_for(call.name)
            ...               # 3. DENY / un-approved ASK
            self.log.append("tool_call", {...}, call_id=call.call_id)
            result = dispatch_tool_call(self.registry, call)
            self.log.append("tool_result",
                            {"output": result.output,
                             "is_error": result.is_error,
                             "status": "error" if result.is_error else "ok"},
                            call_id=call.call_id)
            return result
        """
        # TODO
        raise NotImplementedError
