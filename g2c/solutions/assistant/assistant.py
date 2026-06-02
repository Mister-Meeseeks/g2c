# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.assistant.assistant pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol
from g2c.agent import Agent, AgentRunResult, NativeAgent
from g2c.inference import Backend, is_thinking_model
from g2c.tools import ToolRegistry
from g2c.assistant.config import AssistantConfig, AssistantError
from g2c.assistant.conversation import Conversation, Message

from g2c.assistant.assistant import Assistant


class _AssistantImpl:  # patched onto Assistant by apply()
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
        if not isinstance(user_message, str) or not user_message:
            raise AssistantError("user_message must be a non-empty str")

        history_block = self._conversation.format_for_prompt()
        context_block = self._maybe_retrieve(user_message, use_rag=use_rag)
        contextualized = self._build_contextualized_message(
            user_message=user_message,
            history_block=history_block,
            context_block=context_block,
        )

        agent_run = self._agent.run(contextualized)

        final_answer = agent_run.final_answer
        displayed_answer = final_answer
        if displayed_answer is None:
            displayed_answer = (
                f"(no answer — stopped: {agent_run.stopped_reason})"
            )

        self._conversation.add_user(user_message)
        self._conversation.add_assistant(displayed_answer)

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
                "n_tool_calls": agent_run.metadata.get("n_tool_calls", 0),
            },
        )
        self._turns.append(turn)
        return turn

