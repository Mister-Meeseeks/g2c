"""compact_context — the context window as a POLICY over the event log.

With the log as the source of truth, the context stops being an
accumulating buffer and becomes a rendering decision: given everything
that has happened and a token budget, what does the model most need to
see right now?

The invariant discipline:

    always keep (verbatim):          safe to compress:
    ─────────────────────            ─────────────────
    the task statement               old tool output
    (re-derivable from nothing)      (re-derivable from the workspace)

`render_event` (provided) turns one event into a prompt line;
`compact_context` (scaffolded) is the policy.
"""
from __future__ import annotations

from .events import Event, estimate_tokens  # noqa: F401 - used by scaffold

# How many characters of an old tool result survive compaction.
COMPACT_KEEP_CHARS = 80
COMPACT_SUFFIX = " …[compacted]"


def render_event(event: Event) -> str:
    """One event as one prompt line. Provided plumbing.

    Only event types the model should SEE are rendered; bookkeeping
    types (approvals and stopped markers) return an
    empty string and are skipped by the context builder.
    """
    p = event.payload
    if event.type == "task":
        return f"Task: {p['content']}"
    if event.type == "model_turn":
        return f"Thought: {p['completion']}"
    if event.type == "tool_call":
        return f"Action: {p['tool']}  Input: {p['arguments']}"
    if event.type == "tool_result":
        prefix = "[error] " if p.get("is_error") else ""
        return f"Observation: {prefix}{p['output']}"
    if event.type == "final_answer":
        return f"Final Answer: {p['content']}"
    return ""


def compact_context(events: list[Event], budget_tokens: int) -> list[str]:
    """Render the log into prompt segments under a token budget.

    Args:
        events: the full replayed log, oldest first.
        budget_tokens: target ceiling for the rendered context, in
            `estimate_tokens` units.

    Returns:
        A list of prompt segments (strings), oldest first, satisfying
        three guarantees the tests pin:

        1. **The task statement is ALWAYS present, verbatim** — first
           segment, no matter how tight the budget. An agent that
           forgets its instructions doesn't stop; it drifts.
        2. **The most recent events are verbatim.** Recency is what the
           model acts on; compaction eats the middle, not the end.
        3. **The estimate stays within budget when possible** —
           `sum(estimate_tokens(s)) <= budget_tokens`, unless the task
           statement plus the single most recent event alone exceed it
           (then those two survive anyway; guarantee 1 outranks the
           budget).

    Recipe:
        1. lines = [(e, render_event(e)) for e in events], dropping
           empty renders. Split off the task line (guarantee 1).
        2. If everything fits in the budget, return it all verbatim.
        3. Otherwise, walk the NON-task lines oldest → newest —
           EXEMPTING the newest line (guarantee 2) — and truncate each
           `tool_result` render to `COMPACT_KEEP_CHARS` chars +
           `COMPACT_SUFFIX`, re-checking the budget after each
           truncation. Old tool output is the safe fat: the workspace
           still holds the ground truth.
        4. Still over budget? Drop non-task lines entirely, oldest
           first, until it fits or only the task plus the most recent
           line remain.
        5. Return [task line] + surviving lines, in original order.
    """
    # TODO
    raise NotImplementedError
