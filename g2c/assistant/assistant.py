"""Assistant — the capstone orchestration layer.

`Assistant.chat(user_message)` is the one method that ties together
everything from Modules 16-19:

  * Module 16 (`Backend`) — the inference substrate.
  * Module 17 (`DenseRetriever` / `RAGPipeline`) — optional retrieval.
  * Module 18 (`ToolRegistry`) — the tool palette.
  * Module 19 (`Agent`) — the ReAct loop.

The orchestration:

  1. Append the user's message to the `Conversation` (Module 20's new
     primitive).
  2. If a retriever is configured AND `rag_enabled`, retrieve k chunks
     for the current message.
  3. Compose a contextualized message: history + (optional context
     block) + the current question.
  4. Hand that contextualized message to `Agent.run` — the agent's
     loop handles tools, planning, scratchpad, error recovery.
  5. Append the agent's `final_answer` to the `Conversation`.
  6. Return an `AssistantTurn` summarizing what happened.

Two scaffolds in this package:

  * `Conversation.format_for_prompt` (in `conversation.py`).
  * `Assistant.chat` (here).

Both are about composition — taking the substrate from earlier
modules and threading them into a single call. There's no new
algorithm to learn, just the integration choices: where do the
retrieved chunks go in the prompt, how does the conversation
history thread through, where does the agent's `final_answer`
end up.

The lesson of the capstone is the architecture, not any one piece.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from g2c.agent import Agent, AgentRunResult
from g2c.inference import Backend
from g2c.tools import ToolRegistry

from .config import AssistantConfig, AssistantError
from .conversation import Conversation, Message  # noqa: F401 (used in chat scaffold)


class _Retriever(Protocol):
    """The minimal retriever protocol the assistant accepts.

    Any object with a `retrieve(query, *, k) -> Iterable[...]` method
    works. Module 17's `DenseRetriever` matches this; a hand-rolled
    BM25 retriever or a `RAGPipeline.retriever` does too.

    The returned objects need a `.chunk.text` attribute (i.e., the
    Module 17 `RetrievedChunk` shape). The assistant accesses that
    field when formatting context. No deeper coupling.
    """

    def retrieve(self, query: str, *, k: int = 5) -> Iterable[Any]:
        ...


@dataclass
class AssistantTurn:
    """The result of one `Assistant.chat` call.

    Attributes:
        user_message: the user's input for this turn.
        final_answer: the assistant's response. `None` if the agent
            ran out of steps without producing a Final Answer (in
            which case the conversation history records a synthetic
            "(no answer — stopped: <reason>)" so the user-facing
            log stays coherent).
        agent_run: the full `AgentRunResult` from the underlying
            agent. Has `steps`, `stopped_reason`, `metadata`, etc.
            for debugging.
        retrieved_context: the retrieved-chunks block that was
            spliced into the prompt for this turn, or `""` if RAG
            was disabled or no retriever is configured. Persisted
            so callers can show "the assistant looked at these
            documents" alongside the answer.
        contextualized_message: the full `user_message` that was
            passed to `agent.run` — history + context + current
            question. Useful for debugging "the model saw what?"
        metadata: per-turn extras. The assistant fills in the turn
            number, whether RAG fired, whether a plan was produced.

    Mutable so callers (eval harness, CLI) can attach post-hoc
    analysis. Mirrors the convention from `AgentRunResult` and
    `RAGAnswer`.
    """

    user_message: str
    final_answer: str | None
    agent_run: AgentRunResult
    retrieved_context: str
    contextualized_message: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _format_context_block(retrieved: Iterable[Any]) -> str:
    """Render retrieved chunks as a numbered context block.

    Mirrors the style of `g2c.rag.assemble_rag_prompt` (numbered,
    with source labels) but inlined here so the assistant doesn't
    have to round-trip through the full RAG prompt template (which
    has its own system prompt that would conflict with the agent's).

    Returns `""` for an empty input — the caller can then skip the
    block entirely instead of emitting "Context:" with nothing under
    it (which empirically confuses small models).
    """
    items = list(retrieved)
    if not items:
        return ""
    lines = ["Context from documents:"]
    for i, rc in enumerate(items, start=1):
        # Be defensive: support both `RetrievedChunk` (with .chunk)
        # and bare `Chunk` (with .text directly).
        chunk = getattr(rc, "chunk", rc)
        text = getattr(chunk, "text", str(chunk))
        source = getattr(chunk, "source", None)
        if source:
            lines.append(f"[{i}] (source: {source})\n{text}")
        else:
            lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


class Assistant:
    """The capstone integrated assistant.

    Args:
        backend: an inference `Backend` (Module 16). Required.
        registry: a `ToolRegistry` (Module 18). Required, even if
            empty — the assistant must have somewhere to dispatch
            tool calls if the model emits any.
        config: an `AssistantConfig`. If None, the defaults are used
            (which produces a sensible baseline assistant).
        retriever: a retriever object with `retrieve(query, *, k)`.
            Optional. If absent, RAG context is never spliced in
            regardless of `config.rag_enabled`.
        conversation: an existing `Conversation` to attach to. If
            None, a fresh one is constructed using
            `config.max_history_messages` as the cap. Tests pass an
            existing conversation; the CLI lets the user bring their
            own.
        agent: an existing `Agent` to use. If None (the usual case),
            one is constructed from `backend`, `registry`, and the
            relevant `config` fields. Tests can inject a custom
            agent; the production path uses the auto-built one.

    Raises:
        AssistantError: on misconfigured args (bad backend type, bad
            retriever interface, etc.).

    The assistant is reusable — a single instance can drive many
    chat sessions (across `reset()` calls). It's also reentrant:
    `chat()` is stateless on the assistant itself; the only state
    is in the `Conversation` (and that's where it belongs).
    """

    def __init__(
        self,
        backend: Backend,
        registry: ToolRegistry,
        *,
        config: AssistantConfig | None = None,
        retriever: _Retriever | None = None,
        conversation: Conversation | None = None,
        agent: Agent | None = None,
    ) -> None:
        if not isinstance(backend, Backend):
            raise AssistantError(
                f"backend must be a Backend, got {type(backend).__name__}"
            )
        if not isinstance(registry, ToolRegistry):
            raise AssistantError(
                f"registry must be a ToolRegistry, "
                f"got {type(registry).__name__}"
            )
        if config is None:
            config = AssistantConfig()
        elif not isinstance(config, AssistantConfig):
            raise AssistantError(
                f"config must be an AssistantConfig, "
                f"got {type(config).__name__}"
            )
        if retriever is not None and not hasattr(retriever, "retrieve"):
            raise AssistantError(
                "retriever must have a `retrieve(query, *, k)` method; "
                f"got {type(retriever).__name__} without one"
            )
        if conversation is None:
            conversation = Conversation(
                max_messages=config.max_history_messages
            )
        elif not isinstance(conversation, Conversation):
            raise AssistantError(
                f"conversation must be a Conversation, "
                f"got {type(conversation).__name__}"
            )
        if agent is None:
            agent = Agent(
                backend,
                registry,
                max_steps=config.max_steps,
                plan=config.plan,
                loop_detection=config.loop_detection,
                halt_on_stuck=config.halt_on_stuck,
                scratchpad_max_chars=config.scratchpad_max_chars,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
            )
        elif not isinstance(agent, Agent):
            raise AssistantError(
                f"agent must be an Agent, got {type(agent).__name__}"
            )

        self._backend = backend
        self._registry = registry
        self._config = config
        self._retriever = retriever
        self._conversation = conversation
        self._agent = agent
        self._turns: list[AssistantTurn] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def config(self) -> AssistantConfig:
        return self._config

    @property
    def retriever(self) -> _Retriever | None:
        return self._retriever

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def turns(self) -> list[AssistantTurn]:
        """Snapshot of completed turns since the last `reset()`."""
        return list(self._turns)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the conversation history and turn log.

        The agent itself is reusable across resets — only the
        outer-scope memory (conversation, completed turns) gets
        cleared.
        """
        self._conversation.clear()
        self._turns.clear()

    def __repr__(self) -> str:
        return (
            f"Assistant(name={self._config.name!r}, "
            f"backend={self._backend.info.name}, "
            f"tools={self._registry.names()}, "
            f"rag={'on' if self._retriever else 'off'}, "
            f"turns={len(self._turns)})"
        )

    # ------------------------------------------------------------------
    # The orchestration helpers — implemented; chat is the scaffold.
    # ------------------------------------------------------------------

    def _maybe_retrieve(self, query: str, *, use_rag: bool | None) -> str:
        """Run retrieval if appropriate; return the formatted context
        block (or `""` if RAG didn't fire).

        Implemented (not scaffolded). Decision logic:

          * If `use_rag` is explicitly False, no retrieval.
          * Else if `use_rag` is None, defer to `config.rag_enabled`.
          * If no retriever was passed, no retrieval (regardless).
          * Otherwise call `retriever.retrieve(query, k=config.rag_k)`
            and format the result.

        Errors from the retriever propagate as `AssistantError` —
        unlike tool errors (which the agent handles gracefully),
        retrieval errors are configuration-level and not something
        the model can recover from. They surface to the caller.
        """
        if self._retriever is None:
            return ""
        effective = self._config.rag_enabled if use_rag is None else use_rag
        if not effective:
            return ""
        try:
            retrieved = self._retriever.retrieve(query, k=self._config.rag_k)
        except Exception as e:
            raise AssistantError(f"retriever failed: {e}") from e
        return _format_context_block(retrieved)

    def _build_contextualized_message(
        self,
        user_message: str,
        history_block: str,
        context_block: str,
    ) -> str:
        """Compose the agent's `user_message` from history + RAG +
        the current question.

        Layout:

            [Previous conversation:
            User: ...
            Assistant: ...]

            [Context from documents:
            [1] (source: ...)
            ...]

            <user_message>      ← the actual current question

        Each section is emitted only if non-empty. Sections are joined
        by a blank line for readability. Implemented (not scaffolded).
        """
        parts: list[str] = []
        if history_block:
            parts.append("Previous conversation:\n" + history_block)
        if context_block:
            parts.append(context_block)
        parts.append(user_message)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # The lesson — SCAFFOLDED.
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        *,
        use_rag: bool | None = None,
    ) -> AssistantTurn:
        """Run one chat turn.

        Args:
            user_message: the user's input for this turn. Non-empty str.
            use_rag: per-turn override of `config.rag_enabled`.
                `None` (default) defers to the config; `True` / `False`
                forces retrieval on or off for this turn only.

        Returns:
            `AssistantTurn` summarizing what happened. The
            `final_answer` is also recorded in `self.conversation`
            so subsequent turns see it as history.

        The method NEVER raises on agent / model wobble. Bad parses,
        loop detection trips, max-steps timeouts — all surface as a
        non-clean `stopped_reason` on the returned turn. The only
        ways `chat` raises:

          * `user_message` is not a non-empty str (TypeError /
            AssistantError).
          * The retriever raised (AssistantError, wrapped).

        Recipe:

            1. # Validate.
               if not isinstance(user_message, str) or not user_message:
                   raise AssistantError(
                       "user_message must be a non-empty str"
                   )

            2. # Render the prior conversation history (BEFORE adding
               # the current message — the current message goes in
               # separately as the "current question").
               history_block = self._conversation.format_for_prompt()

            3. # Optional retrieval.
               context_block = self._maybe_retrieve(
                   user_message, use_rag=use_rag
               )

            4. # Compose the agent's input.
               contextualized = self._build_contextualized_message(
                   user_message=user_message,
                   history_block=history_block,
                   context_block=context_block,
               )

            5. # Run the agent. NEVER raises on model wobble — the
               # agent surfaces errors as data on the run result.
               agent_run = self._agent.run(contextualized)

            6. # Determine the user-facing answer. If the agent did
               # NOT produce a final answer (max_steps, duplicate
               # action, no_progress), record a synthetic placeholder
               # in the conversation so the history stays coherent
               # for the next turn — but expose final_answer=None on
               # the AssistantTurn so callers can detect the failure.
               final_answer = agent_run.final_answer
               displayed = final_answer
               if displayed is None:
                   displayed = (
                       f"(no answer — stopped: {agent_run.stopped_reason})"
                   )

            7. # Update conversation memory: user message THEN assistant.
               # Order matters — if you add the user message before
               # rendering history (step 2), the history will already
               # contain the message, and the model will see the
               # current question twice (once as history, once as the
               # current question).
               self._conversation.add_user(user_message)
               self._conversation.add_assistant(displayed)

            8. # Build and stash the turn.
               turn = AssistantTurn(
                   user_message=user_message,
                   final_answer=final_answer,
                   agent_run=agent_run,
                   retrieved_context=context_block,
                   contextualized_message=contextualized,
                   metadata={
                       "turn_index": len(self._turns),
                       "rag_fired": bool(context_block),
                       "had_plan": agent_run.plan is not None,
                       "n_agent_steps": len(agent_run.steps),
                       "stopped_reason": agent_run.stopped_reason,
                       "n_tool_calls": agent_run.metadata.get(
                           "n_tool_calls", 0
                       ),
                   },
               )
               self._turns.append(turn)
               return turn

        Implementation notes:

          * **Why render history BEFORE adding the current user
            message?** Because the rendered history is meant to be
            the prior turns — exchanges that already finished. The
            current message gets a distinct slot ("Current question")
            so the model can tell "this is what you're asked NOW"
            apart from "this is what was asked before."

          * **Why record a synthetic placeholder for failed runs in
            the conversation?** So the next turn's history doesn't
            silently skip the failed exchange. If the user follows up
            with "actually, never mind, do X instead," the model
            should see the prior failed attempt as context (otherwise
            the user's follow-up makes no sense). The placeholder is
            a low-information stand-in that's at least coherent.

          * **Why is `final_answer` left as None on the returned
            turn even though the conversation gets a placeholder?**
            So the eval harness can tell "the run failed" by
            checking `turn.final_answer is None` instead of
            string-matching on placeholder text. The two channels
            (conversation log + machine-readable turn) carry
            different information.

          * **Why doesn't `chat` catch retriever exceptions?** Because
            a broken retriever is a config bug, not a model bug. If
            the embedder is misconfigured or the vector store is
            empty, the user wants to know loudly. Tool errors
            (model-driven) and retrieval errors (dev-driven) get
            different treatment by design.

        Sanity values:

          * Empty conversation, no RAG, agent emits Final Answer
            immediately:
              chat("hi") → AssistantTurn(
                  user_message="hi",
                  final_answer="hello",
                  contextualized_message="hi",   # no history block
                  retrieved_context="",
                  ...
              )
            and self.conversation has [User: hi, Assistant: hello].

          * Second turn with prior history:
              chat("again") → contextualized_message starts with
              "Previous conversation:\\nUser: hi\\nAssistant: hello\\n\\nagain"

          * RAG on, retriever returns 2 chunks:
              chat("Q") → contextualized_message contains
              "Context from documents:\\n[1] ...\\n\\n[2] ..."
              between the history block and the current question.

          * Agent times out (max_steps, no Final Answer):
              chat("hard") → AssistantTurn(final_answer=None, ...).
              self.conversation last assistant message is
              "(no answer — stopped: max_steps)".
        """
        # TODO
        raise NotImplementedError
