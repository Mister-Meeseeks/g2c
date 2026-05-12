"""Document chunking — the first step of any RAG pipeline.

A `Chunk` is a contiguous slice of a source document, sized to fit
inside the model's eventual prompt budget. The chunker takes a string
plus a `source` label (a filename, URL, or any identifier you'll use
later for citations) and returns a list of `Chunk`s.

Why chunk at all? RAG retrieval works at the chunk level — we embed
chunks, search chunks, paste the top-k into the model's prompt. The
unit of retrieval is therefore the unit of indexing. The size and
overlap of chunks shape retrieval quality:

  * Too small → the answer span gets split across two adjacent chunks.
    To reconstruct the answer the retriever has to fetch BOTH, and the
    combined embedding signal is weaker than a single well-sized chunk.
  * Too big → the embedding becomes a coarse summary of the chunk's
    average content, and the distinctive sentence that actually
    answers the user's question gets smeared by the surrounding noise.

Common defaults (LangChain, LlamaIndex): 200–800 tokens per chunk,
10–25% overlap. This module works in CHARACTERS rather than tokens
to keep the dependency graph clean (no tokenizer required for
chunking). 1500 characters is roughly 300–400 English-text tokens —
well inside any modern model's context window when you stuff 5 of
them in alongside a question.

Boilerplate (the `Chunk` dataclass): implemented.
Scaffolded (the actual chunking algorithm): `chunk_text`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """One retrievable slice of a source document.

    Attributes:
        text: the chunk's content. Non-empty.
        source: an identifier for where this chunk came from — a
            filename, URL, document title, whatever the caller wants
            to surface as a citation later. Free-form string. Empty
            string is allowed but discouraged: anonymous chunks are
            uncitable.
        start: the chunk's start character offset within the source
            document. `0` for the first chunk. The slice
            `source_text[start:end]` is exactly `text`.
        end: the chunk's end character offset (exclusive). Always
            equals `start + len(text)`.
        metadata: free-form per-chunk dict. Useful for tagging chunks
            with the section name, page number, document timestamp,
            language, etc. Not used by the retriever; persisted into
            the citation if the prompt template references it.

    Frozen because chunks are values — once you've indexed them into a
    vector store, mutating one in place would silently desync the
    embedding from the text. The `metadata` field is mutable in-place
    (Python's `frozen=True` freezes attribute assignment, not
    contained dicts), but treat it as read-only after indexing.

    Validation: text must be non-empty; start/end must be >= 0;
    `end == start + len(text)`. The constructor enforces all three.
    """

    text: str
    source: str
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or len(self.text) == 0:
            raise ValueError("Chunk.text must be a non-empty str")
        if not isinstance(self.source, str):
            raise TypeError(
                f"Chunk.source must be str, got {type(self.source).__name__}"
            )
        if self.start < 0:
            raise ValueError(f"Chunk.start must be >= 0, got {self.start}")
        if self.end != self.start + len(self.text):
            raise ValueError(
                f"Chunk.end ({self.end}) must equal start + len(text) "
                f"({self.start} + {len(self.text)} = {self.start + len(self.text)})"
            )


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Slice `text` into overlapping chunks of `chunk_size` characters.

    Args:
        text: the source document, as a single string. May contain
            newlines, unicode, anything. Must be non-empty.
        source: the source identifier, attached to every emitted
            chunk's `Chunk.source`. Free-form string.
        chunk_size: chunk length in characters. Must be > 0.
        chunk_overlap: how many characters each chunk overlaps with
            the previous one. Must be `>= 0` and `< chunk_size`. The
            overlap exists so that an answer-bearing sentence near a
            chunk boundary is fully present in at least one chunk.
        metadata: optional dict copied into every emitted chunk's
            `metadata`. Useful for tagging "this whole document is
            from 2023" or similar. Each chunk gets an INDEPENDENT
            copy of the dict — caller mutations on the original after
            chunking won't leak.

    Returns:
        A list of `Chunk`. Order matches reading order. The first
        chunk starts at offset 0; each subsequent chunk's start is
        `previous_start + (chunk_size - chunk_overlap)` (the "stride").
        The final chunk's `end` equals `len(text)` — the chunker
        always covers the whole document.

    Raises:
        ValueError: on empty text, non-positive chunk_size, negative
            overlap, overlap >= chunk_size.

    Recipe:
        1. # Validate.
           if len(text) == 0:
               raise ValueError("text must be non-empty")
           if chunk_size <= 0:
               raise ValueError("chunk_size must be > 0")
           if chunk_overlap < 0:
               raise ValueError("chunk_overlap must be >= 0")
           if chunk_overlap >= chunk_size:
               raise ValueError("chunk_overlap must be < chunk_size")

        2. # Stride is how far we advance the START of the window
           # between chunks. For chunk_size=1500, overlap=150,
           # stride is 1350.
           stride = chunk_size - chunk_overlap

        3. # If the document is shorter than one chunk, emit a single
           # chunk and return.
           if len(text) <= chunk_size:
               return [Chunk(
                   text=text,
                   source=source,
                   start=0,
                   end=len(text),
                   metadata=dict(metadata) if metadata is not None else {},
               )]

        4. # Otherwise, walk through the text emitting chunks.
           chunks: list[Chunk] = []
           start = 0
           while start < len(text):
               end = min(start + chunk_size, len(text))
               chunks.append(Chunk(
                   text=text[start:end],
                   source=source,
                   start=start,
                   end=end,
                   metadata=dict(metadata) if metadata is not None else {},
               ))
               # Stop when this chunk hit the end of the document.
               if end == len(text):
                   break
               start += stride
           return chunks

    Implementation notes:

      * Each chunk gets its OWN copy of the metadata dict — `dict(...)`
        not `metadata`. If we shared a reference, a caller mutating
        `chunks[0].metadata` would silently mutate every chunk's
        metadata. (Yes, `Chunk` is frozen, but the dict inside isn't.)

      * The `if end == len(text): break` is what makes the last chunk
        well-behaved. Without it, the loop would advance `start` by
        `stride` past the end and emit a degenerate empty-text chunk
        (which would also fail the Chunk constructor's non-empty
        validation). Always exit the loop the moment we've consumed
        the whole document.

      * Character-level chunking ignores word boundaries. A real
        production chunker also tries to break on sentence/paragraph
        boundaries to avoid splitting `"in 1492"` into `"in 14"` and
        `"92"`. We don't — keeping the chunker dependency-free and
        the algorithm one screen long is the lesson. Exercise 6 walks
        through a smarter chunker.

      * The chunker is purely positional. It does not deduplicate
        repeated text, it does not skip whitespace-only regions, it
        does not paragraph-aware its choice of split point. All three
        improvements are real production concerns; none change the
        retrieval shape, so they're left as exercises.

    Sanity values:

      * `chunk_size=10, chunk_overlap=0` on `"abcdefghijklmno"` (15 chars):
        → [Chunk(text="abcdefghij", start=0, end=10),
           Chunk(text="klmno",       start=10, end=15)]

      * `chunk_size=10, chunk_overlap=3` on `"abcdefghijklmno"` (15 chars):
        stride = 7
        → [Chunk(text="abcdefghij", start=0,  end=10),
           Chunk(text="hijklmno",   start=7,  end=15)]

      * `chunk_size=100, chunk_overlap=10` on `"hello"` (5 chars):
        → [Chunk(text="hello", start=0, end=5)]   # whole doc fits

      * `chunk_size=5, chunk_overlap=0` on `"abcde"` (exactly 5 chars):
        → [Chunk(text="abcde", start=0, end=5)]   # one chunk, ends
                                                   # exactly at len(text)
    """
    if len(text) == 0:
        raise ValueError("text must be non-empty")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap must be < chunk_size; got {chunk_overlap} >= {chunk_size}"
        )

    metadata_base = dict(metadata) if metadata is not None else {}
    if len(text) <= chunk_size:
        return [
            Chunk(
                text=text,
                source=source,
                start=0,
                end=len(text),
                metadata=dict(metadata_base),
            )
        ]

    chunks: list[Chunk] = []
    stride = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(
            Chunk(
                text=text[start:end],
                source=source,
                start=start,
                end=end,
                metadata=dict(metadata_base),
            )
        )
        if end == len(text):
            break
        start += stride
    return chunks
