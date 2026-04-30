"""Retriever — compose `Embedder` + `NumpyVectorStore` into one object.

A retriever's job is dead simple: take a question, hand back the
top-k chunks. The only logic is "embed the question, then search
the store." Two reasons to put it in its own class:

  1. **Substitutability.** Modules 18 (tools) and 19 (agent loops) take
     a `Retriever` as a dependency. They don't want to know whether
     they're getting a dense retriever, a BM25 retriever, a hybrid
     retriever, or a re-ranking retriever wrapped around any of those.
     One interface, many implementations — same pattern as `Backend`
     in Module 16.

  2. **Future-proofing without abstraction astronomy.** Today,
     `DenseRetriever` IS the retriever. Tomorrow you might want a
     `HybridRetriever` that combines BM25 + dense scores. The
     `retrieve(query, k) -> list[RetrievedChunk]` method is small
     enough that future implementations slot in without forcing an
     ABC migration.

This module ships ONE concrete retriever (`DenseRetriever`) and the
`RetrievedChunk` value type. Both are fully implemented — the
component pieces (`Embedder`, `NumpyVectorStore`) carry the
pedagogical content; this is the wiring.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunk import Chunk
from .embed import Embedder
from .store import NumpyVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """One result of a `retrieve` call.

    Attributes:
        chunk: the matched chunk.
        score: cosine similarity between the query embedding and the
            chunk's embedding. On `[-1, 1]` for general cosine; on
            `[0, 1]` in practice for the embedders in this module
            because they produce non-negative-coordinate vectors
            (HashEmbedder) or non-negative-correlation vectors
            (OllamaEmbedder, on natural-language inputs).
        rank: 1-based position in the result list. The top result has
            rank=1. Useful when results are passed downstream and you
            want to retain "this was the best match" without keeping
            the whole list shape.

    Frozen because retrieved chunks are values, not handles.
    """

    chunk: Chunk
    score: float
    rank: int


class DenseRetriever:
    """Embed the query, then top-k against the store.

    Args:
        embedder: an `Embedder`. Used at `retrieve` time to embed the
            query. If you want to retrieve against a corpus you've
            already embedded with embedder X, you must pass the same
            embedder X here — vectors from different embedders aren't
            comparable.
        store: a `NumpyVectorStore` populated with chunks + their
            embeddings (presumably from the same embedder).

    The constructor does NOT verify that the embedder and store agree
    on `dim` — the store could legally be empty at construction time
    and filled later. The dim mismatch surfaces at search time,
    inside `NumpyVectorStore.search`, with a clear error message.

    Implementation note: this class is fully implemented. The lesson
    is in the components; once `HashEmbedder.embed` and
    `NumpyVectorStore.search` are filled in, this works for free.
    """

    def __init__(self, embedder: Embedder, store: NumpyVectorStore) -> None:
        if embedder is None:
            raise ValueError("DenseRetriever requires a non-None embedder")
        if store is None:
            raise ValueError("DenseRetriever requires a non-None store")
        self._embedder = embedder
        self._store = store

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def store(self) -> NumpyVectorStore:
        return self._store

    def retrieve(self, query: str, *, k: int = 5) -> list[RetrievedChunk]:
        """Return the top-`k` chunks most similar to `query`.

        Args:
            query: the user's question or search string. Non-empty.
            k: number of chunks to return. Must be > 0. The store
                may have fewer than k chunks; in that case all
                of them are returned.

        Returns:
            list[RetrievedChunk] in descending similarity order, with
            `rank=1` on the top result. Length is `min(k, len(store))`.

        Raises:
            ValueError: on empty query, non-positive k, empty store
                (re-raised from `NumpyVectorStore.search`).

        The work:
            1. Embed `[query]` to get a `(1, dim)` array.
            2. Squeeze to `(dim,)`.
            3. Call `store.search(qvec, k=k)` for the (chunk, score)
               pairs.
            4. Wrap each in a `RetrievedChunk` with the right rank.
        """
        if not isinstance(query, str) or len(query) == 0:
            raise ValueError("retrieve: query must be a non-empty str")
        if k <= 0:
            raise ValueError(f"retrieve: k must be > 0, got {k}")

        qvec_2d = self._embedder.embed([query])
        qvec = qvec_2d[0]

        results = self._store.search(qvec, k=k)
        return [
            RetrievedChunk(chunk=c, score=s, rank=i + 1)
            for i, (c, s) in enumerate(results)
        ]

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(embedder={self._embedder!r}, "
            f"store={self._store!r})"
        )
