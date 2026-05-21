"""The SFT chat template — the single source of truth for the format.

A chat template is a deterministic encoder from a list of role-tagged
messages to (a) a single string for tokenization, and (b) a label mask
that says which token positions the loss should fire on.

The format we use is `ChatML-lite` — the spirit of OpenAI's ChatML
adapted to the course-native special tokens reserved by Module 04's
tokenizer. The role markers are still ordinary strings at the template
level, but the course tokenizer treats each complete marker as one
atomic token ID:

    <|user|>
    {user_content}
    <|assistant|>
    {assistant_content}<|end|>

Notes on the format:

  * The role marker is on its own line; user content begins on the next
    line. So the byte sequence is `<|user|>\\n{content}\\n` for a user
    turn, and `<|assistant|>\\n{content}<|end|>` for an assistant turn.
  * The user turn ends with a trailing newline (which separates it from
    the next role marker). The assistant turn ends with a literal
    `<|end|>` and NO trailing newline — `<|end|>` is the model's stop
    token.
  * Multi-turn conversations are simply concatenations of these turn
    strings. The next user turn's `<|user|>` is what closes the
    previous assistant turn at inference time, after the model emits
    `<|end|>`.

The `ChatTemplate` is a stateless class — it carries no per-call state
beyond the role marker constants. The reason it's a class instead of
free functions: every part of the rest of the course (Module 14 DPO,
Module 17 RAG, Module 19 agent) imports `ChatTemplate()` and calls
`.render()` on its prompt assembly. A typo in the marker convention
between training and inference is the single most-common bug that
silently reverts SFT'd behavior to base-model behavior; pinning the
format down in one class is the defense.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .data import SFTExample

if TYPE_CHECKING:
    from g2c.tokenizer import BPETokenizer


def _encode_template_chunk(
    tokenizer: "BPETokenizer",
    text: str,
    *,
    vocab_size: int | None,
) -> list[int]:
    """Encode one rendered template chunk, optionally capped to model vocab.

    This is compatibility plumbing, not the Module 13 concept. Some notebooks
    load a model whose embedding/output tables use only a prefix of a larger
    tokenizer artifact. When the tokenizer knows how to cap merges via
    ``encode_with_vocab_size``, use it; otherwise encode normally and fail
    loudly if an ID cannot fit in the loaded model.
    """
    if vocab_size is None:
        return tokenizer.encode(text)
    if hasattr(tokenizer, "encode_with_vocab_size"):
        return tokenizer.encode_with_vocab_size(text, vocab_size)
    ids = tokenizer.encode(text)
    too_large = [token_id for token_id in ids if token_id >= vocab_size]
    if too_large:
        raise ValueError(
            "rendered SFT example contains token IDs outside the model "
            f"vocab: max={max(too_large)}, vocab_size={vocab_size}"
        )
    return ids


class ChatTemplate:
    """The ChatML-lite chat template used throughout the course.

    Constants:
        USER:      the literal byte string opening a user turn.
        ASSISTANT: the literal byte string opening an assistant turn.
        END:       the literal byte string closing an assistant turn.

    These are class-level constants so callers can write
    ``ChatTemplate.USER`` without instantiating. They're also exposed as
    instance attributes so a future variant (e.g. an Alpaca-style
    template) can override them.

    Why these specific markers? They're rare enough in natural prose
    that a base model trained on TinyShakespeare or TinyStories has
    essentially never seen them — so SFT's job of associating them
    with role-switching behavior starts from a clean slate. They are
    also reserved as tokenizer special tokens, so the model sees one
    stable ID for `<|user|>`, one stable ID for `<|assistant|>`, and
    one stable ID for `<|end|>`.
    """

    USER: str = "<|user|>"
    ASSISTANT: str = "<|assistant|>"
    END: str = "<|end|>"

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
        # TODO
        raise NotImplementedError

    def render_with_mask(
        self,
        messages: list[dict],
        tokenizer: "BPETokenizer",
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
        # TODO
        raise NotImplementedError
