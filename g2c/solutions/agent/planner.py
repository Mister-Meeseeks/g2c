# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.agent.planner pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

import re
from g2c.inference import Backend
from g2c.tools import ToolRegistry
from g2c.agent.base import Plan
from g2c.agent.prompts import render_planning_prompt
_GOAL_RE = re.compile(r"Goal\s*:\s*([^\n]+)", re.IGNORECASE)
_STEP_RE = re.compile(
    r"^[ \t]*(\d+)[ \t]*[\.\)\-][ \t]*(.+?)[ \t]*$",
    re.MULTILINE,
)


def extract_plan(text: str, user_message: str) -> Plan | None:
    """Parse a planning-prompt completion into a `Plan`.

    The expected format (what the planning prompt asks for):

        Goal: <one sentence>
        1. <step>
        2. <step>
        3. <step>

    The contract:

      * If a `Goal:` line is present, use it. Otherwise fall back to
        the user's original message — this preserves "we have at least
        the goal even if the model skipped formatting it."
      * Find numbered list items via the regex `^\\s*(\\d+)[\\.\\)\\-]\\s*(.+)$`.
        Tolerates `1.`, `1)`, `1 -` styles.
      * Steps must be in increasing order — drop any later items that
        repeat or reverse the index. (A model occasionally produces
        "1. ...\\n2. ...\\n2. ..." or restarts numbering.)
      * Empty step bodies are dropped.
      * If no steps were extracted at all, return `None` — the caller
        runs without a plan rather than with a malformed one.
      * Cap at 5 steps: more than 5 is usually the model rambling, and
        the prompt asked for 1-5.

    Args:
        text: the model's completion to the planning prompt.
        user_message: the user's original task. Used as the fallback
            goal when the model doesn't emit a `Goal:` line.

    Returns:
        A `Plan` with the extracted goal + steps, or `None` if no
        usable steps were found (the loop runs without a plan).

    Raises:
        TypeError: if either arg isn't a string.

    Recipe:

        1. # Type-check.
           if not isinstance(text, str):
               raise TypeError(...)
           if not isinstance(user_message, str) or not user_message:
               raise ValueError("user_message must be a non-empty str")

        2. # Extract the goal.
           m = _GOAL_RE.search(text)
           if m:
               goal = m.group(1).strip()
           else:
               goal = user_message.strip()
           if not goal:
               goal = user_message.strip()

        3. # Extract numbered steps.
           steps: list[str] = []
           seen_indices: list[int] = []
           for m in _STEP_RE.finditer(text):
               idx = int(m.group(1))
               body = m.group(2).strip()
               if not body:
                   continue
               # Skip restarts / out-of-order numbers.
               if seen_indices and idx <= seen_indices[-1]:
                   continue
               seen_indices.append(idx)
               steps.append(body)
               if len(steps) >= 5:
                   break

        4. # No steps → return None.
           if not steps:
               return None

        5. return Plan(goal=goal, steps=steps)

    Implementation notes:

      * **Why fall back to user_message for the goal?** The Plan
        dataclass requires a non-empty goal. Models sometimes skip
        the Goal: header entirely and go straight to the numbered
        list. We'd rather have a Plan with a slightly less polished
        goal than no Plan at all.

      * **Why drop on out-of-order indices?** Some models produce
        "1. X\\n2. Y\\n1. Z" — restarting the numbering. The simple
        heuristic "indices must increase" handles this without false
        positives on legitimate "1, 2, 3" plans.

      * **Why cap at 5?** The planning prompt explicitly asks for 1-5
        steps. Beyond that, the model is usually padding with vague
        steps ("Step 7: think about this carefully"). A hard cap
        keeps the plan focused.

      * **Empty step bodies dropped silently.** "1. \\n2. Real step"
        becomes ["Real step"]. Strict mode (raise on empty) would
        punish minor formatting wobble.

    Sanity values:

      * `extract_plan("Goal: solve\\n1. read X\\n2. transform Y", "...")`
        → Plan(goal="solve", steps=["read X", "transform Y"]).

      * `extract_plan("1. step a\\n2. step b", "task")` →
        Plan(goal="task", steps=["step a", "step b"]).

      * `extract_plan("just prose, no numbered list", "task")` → None.

      * `extract_plan("Goal: ...\\n1. ...\\n1. ...\\n2. ...", "x")` →
        Plan with [first step, third step] — second "1." dropped as
        out-of-order.

      * `extract_plan("1. a\\n2. b\\n3. c\\n4. d\\n5. e\\n6. f", "x")` →
        Plan with first 5 steps, "f" dropped.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a str, got {type(text).__name__}")
    if not isinstance(user_message, str):
        raise TypeError(
            f"user_message must be a str, got {type(user_message).__name__}"
        )
    if not user_message:
        raise ValueError("user_message must be a non-empty str")

    goal_match = _GOAL_RE.search(text)
    if goal_match:
        goal = goal_match.group(1).strip()
    else:
        goal = user_message.strip()
    if not goal:
        goal = user_message.strip()

    steps: list[str] = []
    seen_indices: list[int] = []
    for step_match in _STEP_RE.finditer(text):
        idx = int(step_match.group(1))
        body = step_match.group(2).strip()
        if not body:
            continue
        if seen_indices and idx <= seen_indices[-1]:
            continue
        seen_indices.append(idx)
        steps.append(body)
        if len(steps) >= 5:
            break

    if not steps:
        return None
    return Plan(goal=goal, steps=steps)
