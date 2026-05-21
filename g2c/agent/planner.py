"""Planner — optional planning phase before the main loop.

Two functions:

  * `extract_plan(text, user_message) -> Plan | None`. Parses the
    output of a planning prompt (`render_planning_prompt`) into a
    `Plan(goal=..., steps=[...])`. **SCAFFOLDED.** It's a small but
    real lesson on parsing structured-but-wobbly model output —
    different from `parse_react_step`'s shape but the same flavor.

  * `make_plan(backend, user_message, registry, **kwargs) -> Plan | None`.
    Composition: render the planning prompt, call the backend, parse
    the response. Implemented (it's wiring, not the lesson).

Why have a planner at all?

  * **Tasks with structure benefit.** "Read X, transform Y, write Z"
    is structurally a 3-step plan. A model that sees the plan up front
    is less likely to forget step 3 in the middle of step 1's tool
    output.

  * **Tasks without structure don't.** "What's 2+2?" doesn't need a
    plan; planning would add latency and noise.

  * **The plan is a soft prior, not a contract.** The model is free
    to deviate when an Observation suggests a better path. The plan
    just gives it something to start from.

  * **Planning is the one piece the original ReAct paper does NOT do.**
    ReAct interleaves Thought / Action / Observation but doesn't have
    an explicit planning phase. Adding one is closer to the "Plan and
    Solve" paper (Wang et al. 2023) and to LangChain's Plan-and-Execute
    agents. It's a small extension that helps on multi-step tasks.

When to skip planning:

  * Trivial tasks (one tool call suffices).
  * When the model's instruction-following is shaky and a bad plan
    would mislead more than help.
  * When latency matters and one extra backend call is too much.

The agent's `plan=True` / `plan=False` toggle controls this. Defaults
to True; tests exercise both paths.
"""
from __future__ import annotations

import re

from g2c.inference import Backend
from g2c.tools import ToolRegistry

from .base import Plan
from .prompts import render_planning_prompt

# `Goal:` line up to a newline.
_GOAL_RE = re.compile(r"Goal\s*:\s*([^\n]+)", re.IGNORECASE)
# Numbered list items: "1. step text", "2. step text", ... up to end of line.
# Tolerates "1)" / "1 -" / "1." styles. The whitespace classes are
# `[ \t]*` (horizontal only) — `\s*` would let the regex cross
# newlines and slurp the next line's content into a single match.
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
    # TODO
    raise NotImplementedError


def make_plan(
    backend: Backend,
    user_message: str,
    registry: ToolRegistry,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
) -> Plan | None:
    """Run the planning phase: prompt the backend, extract a Plan.

    Args:
        backend: an inference Backend (Module 16).
        user_message: the user's task.
        registry: the tool registry — included in the planning prompt
            so the model proposes tool-grounded steps.
        max_new_tokens: forwarded to backend.complete. 256 is enough
            for a 5-step plan with one-sentence steps.
        temperature: forwarded. 0.2 is near-greedy, which biases the
            planner toward consistency. Plans aren't a place to want
            creativity.

    Returns:
        A `Plan` if the backend produced parseable output, else
        `None`. A `None` return is not an error — the agent runs
        without a plan, which is a valid mode.

    Implemented (not scaffolded). Composition: render → complete →
    extract.
    """
    if not isinstance(user_message, str) or not user_message:
        raise ValueError("user_message must be a non-empty str")
    prompt = render_planning_prompt(user_message, registry.tools)
    inference = backend.complete(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return extract_plan(inference.completion, user_message)
