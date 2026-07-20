"""g2c.assistant — Module 20 capstone, the integrated assistant.

This package ties together everything from the previous modules:

  * Module 16 (inference) — the `Backend` substrate.
  * Module 17 (RAG) — optional retrieval over a corpus.
  * Module 18 (tools) — the tool registry.
  * Module 19 (agent) — the ReAct loop.
  * + `Conversation` — multi-turn memory across turns.
  * + `Assistant` — the orchestration layer.
  * + `EvalSuite` — regression gating.
  * + a CLI for actually using the thing.

Most code in this package is wiring. The two scaffolded methods
are `Conversation.format_for_prompt` and `Assistant.chat` — the
multi-turn analog of Module 19's single-turn agent.
"""
from .assistant import Assistant, AssistantTurn
from .cli import CLI_HELP, run_cli
from .config import AssistantConfig, AssistantError
from .conversation import Conversation, Message
from .eval import (
    AssistantEvalReport,
    EvalCase,
    EvalCaseResult,
    run_capability_baseline,
    run_evaluation,
)

__all__ = [
    "Assistant",
    "AssistantConfig",
    "AssistantError",
    "AssistantTurn",
    "CLI_HELP",
    "Conversation",
    "AssistantEvalReport",
    "EvalCase",
    "EvalCaseResult",
    "Message",
    "run_cli",
    "run_capability_baseline",
    "run_evaluation",
]
