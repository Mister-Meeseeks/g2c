"""Conversation — multi-turn memory across `Assistant.chat` calls.

Module 19's agent was single-turn: one user message → one ReAct loop
→ one final answer. Module 20's assistant is multi-turn: many
exchanges share state (the user's name, the last topic, an earlier
clarifying question). This file is where that state lives.

Two pieces:

  * `Message(role, content)` — frozen value type. One user or assistant
    utterance. Boilerplate.

  * `Conversation` — accumulator + renderer. `add_user`, `add_assistant`,
    `clear`, `__len__` are implemented. **`format_for_prompt` is
    SCAFFOLDED** — it's the lesson, the multi-turn analog of
    Module 19's `Scratchpad.render`.

The relationship between Conversation and Scratchpad:

    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  Scratchpad (Module 19) — INTRA-TURN state                      │
    │     within a single Agent.run, accumulates                      │
    │     (thought, action, observation) records.                     │
    │     Cleared at the end of the run.                              │
    │                                                                 │
    │  Conversation (Module 20) — INTER-TURN state                    │
    │     across Assistant.chat calls, accumulates                    │
    │     (user_msg, final_answer) pairs.                             │
    │     Survives across runs; cleared by the user or `/clear`.      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Why two separate memory primitives instead of one unified store? Because
they have different scopes and different consumers:

  * The scratchpad is the agent's working memory during a tool-using
    multi-step task. The model needs to see its own recent reasoning
    AND tool observations to keep on track. That's high-bandwidth.

  * The conversation is the user-facing chat history. The model needs
    to know "the user said X two turns ago" but not "the agent called
    read_file three times during turn 2." That's low-bandwidth — only
    the final-answer outputs matter, not the intermediate steps.

Mixing them is tempting but a bug magnet: showing the model its own
prior tool calls from a finished agent run pollutes the new turn's
scratchpad with stale, no-longer-relevant action records.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .config import AssistantError

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
_VALID_ROLES = frozenset({USER_ROLE, ASSISTANT_ROLE})


@dataclass(frozen=True)
class Message:
    """One user or assistant utterance.

    Attributes:
        role: `"user"` or `"assistant"`. No `"system"` here — system
            text lives on the assistant's prompt template, not in
            the conversation log.
        content: the message text. Non-empty.

    Frozen because messages are values; we never edit a turn after
    it's recorded. (If you want to retract a message — e.g., a
    `/clear` after a bad answer — drop it from the conversation, not
    edit it in place.)
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise AssistantError(
                f"Message.role must be one of {sorted(_VALID_ROLES)}, "
                f"got {self.role!r}"
            )
        if not isinstance(self.content, str) or not self.content:
            raise AssistantError("Message.content must be a non-empty str")


class Conversation:
    """Accumulator + renderer for the chat history.

    Args:
        messages: optional iterable of `Message` to seed the history.
            Convenience for round-tripping a saved transcript.
        max_messages: cap on the number of messages rendered by
            `format_for_prompt`. The full history is always stored
            (you can iterate via `messages`); this only affects what
            the model sees per turn. `None` (default) means unlimited.

    Use `add_user(content)` / `add_assistant(content)` to grow the
    history. `format_for_prompt()` produces the text block the
    assistant splices into the next prompt. `clear()` resets to empty.
    """

    def __init__(
        self,
        messages: Iterable[Message] | None = None,
        *,
        max_messages: int | None = None,
    ) -> None:
        if max_messages is not None and max_messages <= 0:
            raise AssistantError(
                f"max_messages must be > 0 or None, got {max_messages}"
            )
        self._messages: list[Message] = []
        self._max_messages = max_messages
        if messages is not None:
            for m in messages:
                if not isinstance(m, Message):
                    raise AssistantError(
                        f"Conversation seed contained non-Message: "
                        f"{type(m).__name__}"
                    )
                self._messages.append(m)

    def add_user(self, content: str) -> Message:
        """Append a user message. Returns the appended `Message`."""
        msg = Message(role=USER_ROLE, content=content)
        self._messages.append(msg)
        return msg

    def add_assistant(self, content: str) -> Message:
        """Append an assistant message. Returns the appended `Message`."""
        msg = Message(role=ASSISTANT_ROLE, content=content)
        self._messages.append(msg)
        return msg

    def clear(self) -> None:
        """Drop all stored messages. The Conversation is reusable
        afterward — same instance, fresh history.
        """
        self._messages.clear()

    @property
    def messages(self) -> list[Message]:
        """Snapshot of stored messages. Safe to iterate without
        affecting the conversation.
        """
        return list(self._messages)

    @property
    def max_messages(self) -> int | None:
        return self._max_messages

    def last_user_message(self) -> Message | None:
        """The most recent user message, or `None` if none recorded
        yet. Used by callers that want to find "what did the user
        last ask" without scanning the whole list.
        """
        for m in reversed(self._messages):
            if m.role == USER_ROLE:
                return m
        return None

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def __repr__(self) -> str:
        return (
            f"Conversation(n_messages={len(self._messages)}, "
            f"max={self._max_messages})"
        )

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
