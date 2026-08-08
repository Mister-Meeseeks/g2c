"""HarnessAgent — Module 19's reasoning loop, rebuilt over the event log.

The loop's shape is the same ReAct cycle as Module 19's `Agent`. What
changed is where state lives: every turn renders its prompt from the
REPLAYED LOG via `compact_context`, every action passes through the
permissioned `ToolRunner`, failures route through `classify_failure`,
and the stopping rules are explicit `Budgets`. Under this module's
single-process, intact-filesystem assumptions, a fresh process can
rebuild the run from the log and workspace.

The driver (`run`, `_loop`) is provided; `resume` — rebuilding a run
from its log after a crash — is the scaffold.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from g2c.agent.parser import parse_react_step
from g2c.agent.prompts import render_system_prompt
from g2c.inference import Backend
from g2c.tools import ToolCall, ToolRegistry

from .events import Budgets, Event, EventLog, Permission
from .failures import classify_failure
from .runner import ToolRunner


@dataclass
class HarnessRunResult:
    """End-of-run roll-up. The LOG is the real record; this is a view."""

    final_answer: str | None
    stopped_reason: str
    n_steps: int
    events: list[Event]
    metadata: dict[str, Any] = field(default_factory=dict)


class HarnessAgent:
    """A resumable, budgeted, log-backed ReAct agent.

    Args:
        backend: a Module 16 inference `Backend`.
        registry: a Module 18 `ToolRegistry`.
        log: the `EventLog` this run appends to. One log = one task;
            reuse a log only to `resume()` it.
        permissions: per-tool `Permission` map for the runner.
        budgets: the stopping rules.
        context_policy: deterministic renderer from replayed events and
            a token budget to prompt segments. The default preserves
            the task and recent events while compacting older output.
        retry_wait: seconds between transient-failure retries. The
            tests set 0.0; real runs want a small backoff.
        max_new_tokens, temperature: forwarded to the backend.
    """

    def __init__(
        self,
        backend: Backend,
        registry: ToolRegistry,
        log: EventLog,
        *,
        permissions: dict[str, Permission] | None = None,
        budgets: Budgets = Budgets(),
        context_policy: Callable[[list[Event], int], list[str]] | None = None,
        retry_wait: float = 0.0,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.log = log
        self.runner = ToolRunner(registry, log, permissions=permissions)
        self.budgets = budgets
        if context_policy is None:
            # Resolve at construction time so g2c.solutions.apply() can
            # replace the module-level scaffold before an agent is built.
            from . import context as context_module

            context_policy = context_module.compact_context
        self.context_policy = context_policy
        self.retry_wait = retry_wait
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Provided driver
    # ------------------------------------------------------------------

    def run(self, task: str) -> HarnessRunResult:
        """Start a fresh run: log the task, then drive the loop."""
        if not isinstance(task, str) or not task:
            raise ValueError("task must be a non-empty str")
        if self.log.replay():
            raise ValueError(
                "this log already holds a run — use resume(), or point the "
                "agent at a fresh log"
            )
        self.log.append("task", {"content": task})
        return self._loop()

    def _build_prompt(self, events: list[Event]) -> str:
        segments = self.context_policy(events, self.budgets.context_tokens)
        system = render_system_prompt(self.registry.tools)
        return "\n\n".join([system, *segments]) + "\n\nThought:"

    def _next_call_id(self, events: list[Event]) -> str:
        n = sum(1 for e in events if e.type == "tool_call")
        return f"call_{n}"

    def _finish(self, reason: str, final_answer: str | None) -> HarnessRunResult:
        self.log.append("stopped", {"reason": reason})
        events = self.log.replay()
        n_steps = sum(1 for e in events if e.type == "model_turn")
        return HarnessRunResult(
            final_answer=final_answer,
            stopped_reason=reason,
            n_steps=n_steps,
            events=events,
            metadata={"n_events": len(events)},
        )

    def _loop(self) -> HarnessRunResult:
        """The bounded, log-backed ReAct driver. Provided.

        Per iteration: replay → render context → complete → log the
        turn → parse → act. Repeat-detection and budgets are enforced
        here; failure classification decides retries.
        """
        last_action_key: tuple[str, str] | None = None
        repeats = 0

        for _ in range(self.budgets.max_steps):
            events = self.log.replay()
            prompt = self._build_prompt(events)
            inference = self.backend.complete(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            self.log.append("model_turn", {"completion": inference.completion})
            parsed = parse_react_step(inference.completion)

            if parsed.final_answer is not None:
                self.log.append(
                    "final_answer", {"content": parsed.final_answer}
                )
                return self._finish("final_answer", parsed.final_answer)

            if parsed.action is None:
                # Stuck step: the parse failure is already visible in
                # the logged model_turn; let the model recover.
                continue

            action_key = (
                parsed.action.tool,
                repr(sorted(parsed.action.arguments.items())),
            )
            if action_key == last_action_key:
                repeats += 1
                if repeats >= self.budgets.max_repeats:
                    return self._finish("repeat_budget", None)
            else:
                repeats = 0
                last_action_key = action_key

            call = ToolCall(
                name=parsed.action.tool,
                arguments=parsed.action.arguments,
                call_id=self._next_call_id(self.log.replay()),
            )
            result = self.runner.execute(call)

            retries = 0
            while result.is_error and retries < self.budgets.max_retries:
                decision = classify_failure(result)
                if not decision.retry:
                    break
                retries += 1
                if self.retry_wait:
                    time.sleep(self.retry_wait * retries)
                retry_call = ToolCall(
                    name=call.name,
                    arguments=call.arguments,
                    call_id=f"{call.call_id}_retry{retries}",
                )
                result = self.runner.execute(retry_call)

        return self._finish("max_steps", None)

    # ------------------------------------------------------------------
    # The scaffold
    # ------------------------------------------------------------------

    def resume(self) -> HarnessRunResult:
        """Continue a crashed run from its log.

        Under the module's process-crash assumptions, the log plus the
        workspace holds the state needed to resume, so
        resuming is: validate the log, resolve anything a crash left
        half-done, and re-enter the same loop. No other bookkeeping
        exists to restore — that absence is the design working.

        Returns:
            A `HarnessRunResult`, exactly as `run` would have.

        Raises:
            ValueError: if the log holds no `task` event (nothing to
                resume), or already holds a `final_answer` (nothing
                left to do — resuming a finished run is a caller bug
                worth failing loudly on).

        Recipe:
            1. events = self.log.replay()
               if no event has type "task": raise ValueError(...)
               if any event has type "final_answer": raise ValueError(...)
            2. # Resolve the crash window: a tool_call whose call_id
               # has no matching tool_result has an UNKNOWN outcome.
               # Re-submit it through the runner — the
               # runner's own ladder (dedupe / crash-window) decides
               # what actually happens; resume() does NOT re-implement
               # that policy:
               unresolved = [e for e in events if e.type == "tool_call"
                             and not any(r.type == "tool_result"
                                         and r.call_id == e.call_id
                                         for r in events)]
               for e in unresolved:
                   self.runner.execute(ToolCall(
                       name=e.payload["tool"],
                       arguments=e.payload["arguments"],
                       call_id=e.call_id,
                   ))
            3. return self._loop()

        After step 2, the log shows the ambiguity as data (an
        `UNKNOWN_OUTCOME_STATUS` result the model will see in its context) and
        the loop continues as though the process never died.
        """
        # TODO
        raise NotImplementedError
