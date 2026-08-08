# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.harness.context pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from g2c.harness.context import COMPACT_KEEP_CHARS, COMPACT_SUFFIX, render_event
from g2c.harness.events import Event, estimate_tokens


def compact_context(events: list[Event], budget_tokens: int) -> list[str]:
    rendered = [(e, render_event(e)) for e in events]
    rendered = [(e, line) for e, line in rendered if line]

    task_lines = [line for e, line in rendered if e.type == "task"]
    task_line = task_lines[0] if task_lines else None
    others = [(e, line) for e, line in rendered if e.type != "task"]

    def total(lines: list[str]) -> int:
        return sum(estimate_tokens(line) for line in lines)

    def assemble(other_lines: list[str]) -> list[str]:
        return ([task_line] if task_line else []) + other_lines

    lines = [line for _, line in others]
    if total(assemble(lines)) <= budget_tokens:
        return assemble(lines)

    # Pass 1: truncate old tool results, oldest first. The NEWEST line
    # is exempt — recency is what the model acts on.
    for i, (event, line) in enumerate(others[:-1]):
        if event.type != "tool_result":
            continue
        if len(line) > COMPACT_KEEP_CHARS:
            lines[i] = line[:COMPACT_KEEP_CHARS] + COMPACT_SUFFIX
        if total(assemble(lines)) <= budget_tokens:
            return assemble(lines)

    # Pass 2: drop non-task lines entirely, oldest first, keeping at
    # least the most recent line.
    while len(lines) > 1 and total(assemble(lines)) > budget_tokens:
        lines.pop(0)
    return assemble(lines)
