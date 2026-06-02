# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.assistant.conversation pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from g2c.assistant.config import AssistantError
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
_VALID_ROLES = frozenset({USER_ROLE, ASSISTANT_ROLE})

from g2c.assistant.conversation import Conversation


class _ConversationImpl:  # patched onto Conversation by apply()
    def format_for_prompt(self) -> str:
        """Render the conversation history as a prompt-prefix string.

        Output format (one line per message):

            User: <m1 content>
            Assistant: <m2 content>
            User: <m3 content>
            ...

        Multi-line message contents are preserved as-is (newlines
        stay; the role prefix only appears on the first line of each
        message).

        The CURRENT user message — the one being processed by the
        in-flight `Assistant.chat` call — is NOT included by this
        method. The caller (`Assistant.chat`) appends the current
        message separately so it can be presented as "Current
        question" rather than buried in the history block.

        If `max_messages` is set and the conversation has more, drop
        the OLDEST messages until the count fits — the most recent
        exchanges are most relevant to the current turn. (A more
        sophisticated truncation would summarize old turns; we just
        drop them, same as the Scratchpad.)

        Returns:
            The rendered conversation history. Empty string if the
            conversation has no messages.

        Recipe:

            1. # Empty case.
               if not self._messages:
                   return ""

            2. # Apply max_messages truncation. Drop oldest first.
               msgs = list(self._messages)
               if (self._max_messages is not None
                       and len(msgs) > self._max_messages):
                   msgs = msgs[-self._max_messages:]

            3. # Render each message as "Role: content".
               lines: list[str] = []
               for m in msgs:
                   prefix = "User" if m.role == USER_ROLE else "Assistant"
                   lines.append(f"{prefix}: {m.content}")

            4. # One message per "line" (with embedded newlines preserved).
               # Separate messages by a single blank line for readability.
               return "\n".join(lines)

        Implementation notes:

          * **Why "User:" and "Assistant:" instead of chat-template
            tokens (`<|user|>`, `<|im_start|>user`, etc.)?** This
            module is backend-agnostic. Different chat-tuned models
            use different tokens; chat-template wrapping is the
            backend's problem (the Ollama / MLX SDK handles it). At
            the assistant layer, we use plain text labels — every
            instruction-tuned model has seen these and treats them
            as conversational role markers.

          * **Why drop oldest, not newest, on truncation?** The user
            cares about the recent exchange. Old turns fade in
            relevance as the topic shifts. (A real assistant would
            summarize old turns into a single message; we drop them.)

          * **Why is the current user message excluded?** Because
            `Assistant.chat` builds the agent's input as
            `<history>\n\n<rag block>\n\n<current question>`. If
            `format_for_prompt` included the current message, the
            current question would be buried in the history and the
            "Current question:" label would be redundant.

          * **Single newline (not blank line) between messages?**
            This format is dense by design — each message is one
            row in a chat log, not a paragraph. The trailing blank
            line between sections is added by the caller.

        Sanity values:

          * Empty conversation → `""`.

          * One user + one assistant message:
            `"User: hi\nAssistant: hi back"`.

          * `max_messages=2` on a 4-message conversation: only the
            last two messages appear.

          * Multi-line content (`"line1\nline2"`):
            `"User: line1\nline2"` — the prefix appears once.
        """
        if not self._messages:
            return ""

        messages = list(self._messages)
        if (
            self._max_messages is not None
            and len(messages) > self._max_messages
        ):
            messages = messages[-self._max_messages :]

        lines: list[str] = []
        for message in messages:
            prefix = "User" if message.role == USER_ROLE else "Assistant"
            lines.append(f"{prefix}: {message.content}")
        return "\n".join(lines)

