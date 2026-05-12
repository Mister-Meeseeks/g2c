"""Prompt assembly — splice retrieved chunks into the model's input.

Once the retriever has handed you the top-k chunks, the question
becomes: how do you stuff them into a prompt the model will actually
read? The prompt template is doing real work:

  * **It TELLS the model that some text is "context"**, not part of
    the user's question. Without that signal, the model can confuse
    "the user said X" with "the retrieved document says X" — a
    subtle but real source of hallucination.

  * **It NUMBERS the chunks** so the model can cite back to them.
    Citations are how a RAG answer becomes verifiable: the user (or a
    downstream parser) can check whether `[2]` actually contains the
    claim the model attributed to it.

  * **It SETS expectations** about what to do when the context is
    insufficient. "If the context doesn't contain the answer, say
    you don't know" cuts hallucination materially. The model isn't
    guaranteed to obey, but the prompt sets the floor.

Different RAG implementations make different template choices.
Anthropic's Claude docs recommend XML-tagged context. LlamaIndex
defaults to a "Context information is below.\\n---..." block. We use
a numbered-list format because it makes citations cleanest.

Default output shape:

```
{system}

Context:
[1] (source: {chunks[0].source})
{chunks[0].text}

[2] (source: {chunks[1].source})
{chunks[1].text}

...

Question: {question}

{instruction}
```

Boilerplate (default templates, the `RAGPrompt` dataclass):
implemented. Scaffolded (`assemble_rag_prompt`): the lesson.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .chunk import Chunk
from .retrieve import RetrievedChunk

DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using "
    "ONLY the context below. Refer to context items by their bracket "
    "numbers, e.g. [1], [2]."
)

DEFAULT_INSTRUCTION = (
    "If the context does not contain the information needed to answer "
    "the question, reply with: \"I don't know based on the provided "
    "context.\" Do not invent facts that are not in the context."
)


@dataclass(frozen=True)
class RAGPrompt:
    """Container for an assembled RAG prompt.

    Attributes:
        text: the full prompt string ready to feed to a `Backend`.
        question: the original question (for round-tripping).
        chunks: the chunks that were spliced in, in the order they
            appear in `text`. Index 0 corresponds to citation `[1]`
            in the prompt.

    Frozen — the prompt is the result of an immutable composition.
    Useful as a return value: callers get the rendered prompt AND
    the chunks that produced it, in case they want to log citations
    alongside the model's response.
    """

    text: str
    question: str
    chunks: tuple[Chunk, ...]


def _coerce_chunks(items: Iterable) -> list[Chunk]:
    """Accept either `list[Chunk]` or `list[RetrievedChunk]`; emit
    `list[Chunk]`.

    The retriever returns `RetrievedChunk`s; a caller assembling a
    prompt by hand might pass `Chunk`s directly. Accepting both keeps
    the API ergonomic without a separate `assemble_from_retrieved`.
    """
    out: list[Chunk] = []
    for item in items:
        if isinstance(item, RetrievedChunk):
            out.append(item.chunk)
        elif isinstance(item, Chunk):
            out.append(item)
        else:
            raise TypeError(
                f"chunks must contain Chunk or RetrievedChunk, "
                f"got {type(item).__name__}"
            )
    return out


def assemble_rag_prompt(
    question: str,
    chunks: Iterable,
    *,
    system: str = DEFAULT_SYSTEM,
    instruction: str = DEFAULT_INSTRUCTION,
) -> RAGPrompt:
    """Splice `question` + `chunks` into a numbered-citation prompt.

    Args:
        question: the user's question. Non-empty string.
        chunks: an iterable of `Chunk` OR `RetrievedChunk`. May be
            empty — in which case the prompt has a "Context:" header
            with no items beneath it. (Empty context is its own kind
            of failure mode; see Exercise 5.)
        system: leading system instructions. Defaults to
            `DEFAULT_SYSTEM`. Pass `""` to omit the leading paragraph.
        instruction: trailing instruction (typically the "say I don't
            know" guard). Defaults to `DEFAULT_INSTRUCTION`. Pass `""`
            to omit.

    Returns:
        `RAGPrompt` with the rendered text + the chunks that were
        spliced in. The chunks order matches their citation numbers
        in the prompt: chunks[0] is `[1]`, chunks[1] is `[2]`, etc.

    Raises:
        TypeError: if `question` is not a str.
        ValueError: if `question` is empty.
        TypeError: if `chunks` contains anything other than `Chunk` or
            `RetrievedChunk`.

    Recipe:
        1. # Validate the question.
           if not isinstance(question, str):
               raise TypeError(...)
           if len(question) == 0:
               raise ValueError("question must be non-empty")

        2. # Coerce iterables of Chunk OR RetrievedChunk into list[Chunk].
           chunk_list = _coerce_chunks(chunks)

        3. # Build the chunk block, one entry per chunk.
           # Note the 1-based indexing: humans cite [1], not [0].
           chunk_block_parts: list[str] = []
           for i, c in enumerate(chunk_list, start=1):
               chunk_block_parts.append(
                   f"[{i}] (source: {c.source})\n{c.text}"
               )
           chunk_block = "\n\n".join(chunk_block_parts)

        4. # Compose the full prompt. Skip empty sections cleanly so
           # callers can pass `system=""` without leaving a leading
           # blank line.
           parts: list[str] = []
           if system:
               parts.append(system)
           parts.append("Context:\n" + chunk_block if chunk_block else "Context:")
           parts.append(f"Question: {question}")
           if instruction:
               parts.append(instruction)
           text = "\n\n".join(parts)

        5. # Return the immutable container.
           return RAGPrompt(
               text=text,
               question=question,
               chunks=tuple(chunk_list),
           )

    Implementation notes:

      * 1-based citation indexing. Humans say "see [1]", not "see [0]".
        Forgetting `start=1` produces a prompt that confuses the model
        AND any downstream citation parser.

      * `(source: {c.source})` is a small but real detail. The model
        often chooses to surface the source name when answering — "as
        described in 16-inference.md, the KV cache..." — which makes
        the citation human-readable. Without the source label, the
        model can only refer to `[1]`, which is less useful for the
        user.

      * Empty `chunks` is allowed but pathological. The resulting
        prompt has `"Context:\\n\\nQuestion: ..."` — the model is told
        to answer using only the context but given no context. A
        well-behaved model should default to "I don't know"; many
        models will hallucinate. Decide upstream whether to
        short-circuit before assembling.

      * The two-newline separator between sections (`"\\n\\n".join`)
        is a markdown-style paragraph break. Most chat-tuned models
        respect it. Some chat templates (ChatML, Llama 3) wrap the
        whole thing in `<|user|>...<|assistant|>` markers; you would
        do that AFTER assembling — the chat template lives outside
        this function.

      * No truncation. If the chunks plus the question exceed the
        model's context window, the prompt is too long and the model
        will (best case) error or (worst case) silently drop the
        leading content. A real RAG pipeline either re-ranks chunks
        AND keeps a token budget, or summarizes long chunks before
        assembly. We don't — that's Exercise 7.

    Sanity values:

      * `assemble_rag_prompt("Q?", [])` → `RAGPrompt` whose `text` is
        roughly:
            "{DEFAULT_SYSTEM}\n\nContext:\n\nQuestion: Q?\n\n
             {DEFAULT_INSTRUCTION}"

      * Single chunk with `source="docs/foo.md"`, `text="Madrid is..."`:
        the prompt contains the line `[1] (source: docs/foo.md)` and
        on the next line, the chunk text.

      * `system=""` and `instruction=""`: the prompt is just
        `"Context:\n[...]\n\nQuestion: ..."` with no leading or
        trailing paragraphs.
    """
    if not isinstance(question, str):
        raise TypeError(
            f"question must be str, got {type(question).__name__}"
        )
    if len(question) == 0:
        raise ValueError("question must be non-empty")

    chunk_list = _coerce_chunks(chunks)

    chunk_parts: list[str] = []
    for index, chunk in enumerate(chunk_list, start=1):
        chunk_parts.append(
            f"[{index}] (source: {chunk.source})\n{chunk.text}"
        )
    chunk_block = "\n\n".join(chunk_parts)

    parts: list[str] = []
    if system:
        parts.append(system)
    parts.append("Context:\n" + chunk_block if chunk_block else "Context:")
    parts.append(f"Question: {question}")
    if instruction:
        parts.append(instruction)

    return RAGPrompt(
        text="\n\n".join(parts),
        question=question,
        chunks=tuple(chunk_list),
    )
