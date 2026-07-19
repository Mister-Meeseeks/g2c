# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.sft.chat_template pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from g2c.sft.chat_template import ChatTemplate
from g2c.sft.data import SFTExample


class _ChatTemplateImpl:  # patched onto ChatTemplate by apply()
    def render(self, messages: list[dict]) -> str:
        """Render `messages` to a single string using the template.

        Args:
            messages: list of dicts, each with keys `"role"` (one of
                `"user"`, `"assistant"`) and `"content"` (a str). At
                least one message is required; a ValueError is raised
                on an empty list. No constraints on alternation — a
                three-message `[user, assistant, user]` list is valid
                and produces a string ending with the second user turn.

        Returns:
            A single string consisting of each turn in order, formatted
            per the ChatML-lite template described in the module
            docstring. The returned string does NOT have a trailing
            newline; it ends with `<|end|>` (if the last turn is
            assistant) or with a trailing `\\n` (if the last turn is
            user).

        Recipe:
            1. Validate: messages must be non-empty.
            2. Walk messages in order. For each:
                 if role == "user":
                     parts.append(f"{self.USER}\\n{content}\\n")
                 elif role == "assistant":
                     parts.append(f"{self.ASSISTANT}\\n{content}{self.END}")
                 else:
                     raise ValueError("unknown role: ...")
            3. Return "".join(parts).

        Examples:
            single turn:
                [{"role": "user",      "content": "Hi"},
                 {"role": "assistant", "content": "Hello"}]
              →  "<|user|>\\nHi\\n<|assistant|>\\nHello<|end|>"

            user-only (e.g. inference-time prompt assembly):
                [{"role": "user", "content": "Hi"}]
              →  "<|user|>\\nHi\\n"
              The trailing newline matters — it's how the BPE
              tokenization of the next role marker stays consistent
              with the training-time tokenization.
        """
        if not messages:
            raise ValueError("messages must be non-empty")

        parts: list[str] = []
        for turn in messages:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                parts.append(f"{self.USER}\n{content}\n")
            elif role == "assistant":
                parts.append(f"{self.ASSISTANT}\n{content}{self.END}")
            else:
                raise ValueError(f"unknown role: {role!r}")

        return "".join(parts)

    def render_with_mask(
        self,
        messages: list[dict],
        tokenizer: BPETokenizer,
        *,
        vocab_size: int | None = None,
    ) -> SFTExample:
        """Render `messages` AND build the assistant-only loss mask.

        Returns an `SFTExample(ids, mask)` where:
          - `ids` is the full tokenized sequence (the same sequence
            you'd get by tokenizing `self.render(messages)`).
          - `mask[t] == 1` if `ids[t]` is part of an assistant turn's
            CONTENT (including the trailing `<|end|>`), else `0`.
            Specifically, the mask is `0` for:
              - every user-turn token (role marker, content, trailing
                newline);
              - the assistant role marker (`<|assistant|>\\n`); and
              - any system or other-role tokens.
            The mask is `1` for:
              - every byte of the assistant content; and
              - every byte of the trailing `<|end|>` string.

        The mask is aligned with `ids` (length `T`). The collator
        (`pad_and_collate`) is responsible for shifting it by one to
        align with `targets = ids[1:]`.

        Args:
            messages: same shape as `render`'s argument.
            tokenizer: a `BPETokenizer` (or any object exposing
                `.encode(str) -> list[int]`).
            vocab_size: optional effective model vocabulary size. Use this
                when the tokenizer artifact has more learned merges than the
                model's embedding/output tables. Special tokens still remain
                atomic; learned merges with IDs outside the model vocab are
                skipped.

        Returns:
            `SFTExample(ids=[...], mask=[...])` with `len(mask) == len(ids)`.

        Recipe:
            1. ids:  list[int] = []
               mask: list[int] = []

            2. Walk messages in order. For each turn:
               `_encode_template_chunk` is pre-implemented plumbing;
               the conceptual work is deciding which chunks get mask 0
               and which chunks get mask 1.

               if role == "user":
                   chunk = f"{self.USER}\\n{content}\\n"
                   chunk_ids = _encode_template_chunk(tokenizer, chunk, vocab_size=vocab_size)
                   ids.extend(chunk_ids)
                   mask.extend([0] * len(chunk_ids))

               elif role == "assistant":
                   # The role marker + newline is part of the prompt
                   # — the model should NOT learn to emit it.
                   prefix = f"{self.ASSISTANT}\\n"
                   prefix_ids = _encode_template_chunk(tokenizer, prefix, vocab_size=vocab_size)
                   ids.extend(prefix_ids)
                   mask.extend([0] * len(prefix_ids))

                   # The content + <|end|> IS the response — the
                   # model should learn to emit every token of this.
                   target = f"{content}{self.END}"
                   target_ids = _encode_template_chunk(tokenizer, target, vocab_size=vocab_size)
                   ids.extend(target_ids)
                   mask.extend([1] * len(target_ids))

               else:
                   raise ValueError("unknown role: ...")

            3. return SFTExample(ids=ids, mask=mask)

        Two correctness pins worth restating:

          * The `<|assistant|>\\n` prefix is NOT part of the response.
            Mask `0`. The student's first response token should be the
            first character of the actual content, not the role marker.
          * The trailing `<|end|>` IS part of the response. Mask `1`.
            Without this, the model never learns to stop and inference
            loops to `max_new_tokens` every time.
        """
        ids: list[int] = []
        mask: list[int] = []

        for turn in messages:
            role = turn["role"]
            content = turn["content"]

            if role == "user":
                chunk = f"{self.USER}\n{content}\n"
                chunk_ids = _encode_template_chunk(
                    tokenizer,
                    chunk,
                    vocab_size=vocab_size,
                )
                ids.extend(chunk_ids)
                mask.extend([0] * len(chunk_ids))

            elif role == "assistant":
                prefix = f"{self.ASSISTANT}\n"
                prefix_ids = _encode_template_chunk(
                    tokenizer,
                    prefix,
                    vocab_size=vocab_size,
                )
                ids.extend(prefix_ids)
                mask.extend([0] * len(prefix_ids))

                target = f"{content}{self.END}"
                target_ids = _encode_template_chunk(
                    tokenizer,
                    target,
                    vocab_size=vocab_size,
                )
                ids.extend(target_ids)
                mask.extend([1] * len(target_ids))

            else:
                raise ValueError(f"unknown role: {role!r}")

        return SFTExample(ids=ids, mask=mask)

