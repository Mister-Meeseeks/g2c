# Hand-maintained solutions mirror — canonical, lives in `main` (see AGENTS.md).
"""Solutions for g2c.rag.chunk pedagogical scaffolds.

Patched onto the scaffold targets by g2c.solutions.apply().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
