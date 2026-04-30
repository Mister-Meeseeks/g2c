"""Prompts — the system prompt template and rendering helpers.

The ReAct prompt has a very specific shape that instruction-tuned
models recognize from their training data:

    System: <description of the task and tools>
    Tools:
      - calculator: ...
      - read_file: ...

    Use this format:

    Question: <the user's question>
    Thought: <reasoning about what to do next>
    Action: <one of the tool names>
    Action Input: <JSON object with the tool's arguments>
    Observation: <the tool's output, provided by the runtime>
    ... (Thought/Action/Action Input/Observation can repeat) ...
    Thought: I now know the final answer.
    Final Answer: <answer to the original question>

    Begin.

    Question: <the actual user message>

The model fills in everything from `Thought:` onward. The runtime
parses Thought/Action/Action Input out of the completion, dispatches
the action, formats the result as `Observation: ...`, and asks the
model for the next turn.

This file is fully implemented. The lessons are in `parser.py`,
`memory.py`, `planner.py`, and `agent.py`. The prompt template here
is a recipe.

Why is this format so specific?

  * **It's what the ReAct paper uses.** Yao et al. 2022 named the
    format and demonstrated empirically that this exact phrasing —
    "Thought:", "Action:", "Action Input:", "Observation:", "Final
    Answer:" — works well across HotpotQA, FEVER, ALFWorld, and
    WebShop. Instruction-tuning datasets later picked it up;
    LangChain made it standard.
  * **It's regex-friendly.** Each marker is on its own line, prefixed
    by a fixed string. The parser is a few `re.search` calls.
  * **It separates reasoning from action.** Without "Thought:", the
    model conflates "what to do" with "the action" and tool selection
    becomes noisy. The Thought line gives the model a place to put
    its reasoning where the parser can ignore it.

Other agent formats exist (function-calling structured output, XML
agents, JSON-only protocols). They'd work; ReAct is what this module
teaches because (a) it's the canonical research format, (b) it's
text-only so it works with any backend, and (c) the explicit
separation of reasoning and action is a clean pedagogical lens.
"""
from __future__ import annotations

from collections.abc import Iterable

from g2c.tools import Tool

# The system prompt template. `{tools_block}` is filled in with the
# rendered tools list; the rest is fixed. The Begin / Question /
# Thought sequence is what the ReAct paper prescribes.
DEFAULT_AGENT_SYSTEM = """\
You are a careful agent that solves problems step by step using tools.

You have access to the following tools:
{tools_block}

Use this exact format on every step:

Thought: <one sentence of reasoning about what to do next>
Action: <one of {tool_names}>
Action Input: <a JSON object with the tool's arguments>

After you emit Thought / Action / Action Input, stop. The runtime will
execute the action and append:

Observation: <the tool's output>

You can then continue with another Thought / Action / Action Input,
repeating until you have enough information.

When you are ready to answer, emit:

Thought: I now know the final answer.
Final Answer: <your answer to the user's question>

Rules:
- Use exactly one Action per step. Do not chain multiple Actions.
- Action Input must be a single JSON object on one line.
- Do not invent tool names. Only use tools from the list above.
- If a tool returns an error, read the message and try a different
  approach on the next step.
- When you give the Final Answer, do not emit any Action.
"""

# The planning-phase prompt. Asked once before the main loop.
# The model produces a numbered plan; `extract_plan` parses it.
DEFAULT_PLANNING_PROMPT = """\
You are a planning assistant. Given a user task and a list of available
tools, produce a short numbered plan of how you would solve it.

Tools available:
{tools_block}

User task: {user_message}

Output exactly this format:

Goal: <one sentence restating the task>
1. <first step>
2. <second step>
3. <third step>

Use 1 to 5 numbered steps. Each step should be a single short sentence.
Do not output any other text — no preamble, no explanation, no Thought
or Action lines. Just Goal: and the numbered list.
"""


def _render_tools_block(tools: Iterable[Tool]) -> str:
    """Render a tool list as the "Tools available" body. Internal.

    Used by both the system prompt and the planning prompt. The format
    is "  - name: description" — indented, one per line.

    Empty input renders as "  (no tools registered)" — a valid (if
    pointless) configuration that tests exercise.
    """
    tools_list = list(tools)
    if not tools_list:
        return "  (no tools registered)"
    lines = []
    for t in tools_list:
        lines.append(f"  - {t.name}: {t.description}")
    return "\n".join(lines)


def render_system_prompt(tools: Iterable[Tool]) -> str:
    """Render the agent's system prompt with the tool list spliced in.

    Args:
        tools: an iterable of Tool. May be empty (the prompt still
            renders, but the model will have nothing to call).

    Returns:
        The full system prompt as a string. Includes the rules block,
        the format reminder, and the tool list. Splice this at the
        front of every backend.complete call in the main loop.
    """
    tools_list = list(tools)
    tools_block = _render_tools_block(tools_list)
    if tools_list:
        names = ", ".join(t.name for t in tools_list)
    else:
        names = "(none)"
    return DEFAULT_AGENT_SYSTEM.format(
        tools_block=tools_block,
        tool_names=names,
    )


def render_planning_prompt(user_message: str, tools: Iterable[Tool]) -> str:
    """Render the planning-phase prompt.

    Args:
        user_message: the user's task.
        tools: the tool list — included so the planner suggests
            tool-grounded steps rather than abstract advice.

    Returns:
        The full planning prompt. One backend.complete call against
        this returns a numbered plan that `extract_plan` parses.
    """
    if not isinstance(user_message, str) or not user_message:
        raise ValueError("user_message must be a non-empty str")
    return DEFAULT_PLANNING_PROMPT.format(
        tools_block=_render_tools_block(tools),
        user_message=user_message,
    )


def render_plan_block(goal: str, steps: list[str]) -> str:
    """Render a plan as a text block to splice into the main prompt.

    Output:

        Plan:
        Goal: <goal>
        1. <step 1>
        2. <step 2>
        ...

    The agent appends this between the system prompt and the user
    question so the model sees its plan at every step.

    Implemented (not scaffolded). Trivial template.
    """
    lines = ["Plan:", f"Goal: {goal}"]
    for i, s in enumerate(steps, start=1):
        lines.append(f"{i}. {s}")
    return "\n".join(lines)
