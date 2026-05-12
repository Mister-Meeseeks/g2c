"""In-memory vector store — keep chunks + their embeddings, search by cosine.

A vector store is the data structure that holds `(chunk, embedding)`
pairs and supports approximate-nearest-neighbour retrieval against a
query embedding. Production stores (Chroma, LanceDB, FAISS, Pinecone,
Weaviate) implement HNSW, IVF-PQ, or other ANN indexes for sub-linear
search across millions of chunks. We don't need any of that.

`NumpyVectorStore` is the simplest possible implementation: a Python
list of `Chunk`s plus a contiguous `(N, dim)` numpy array of vectors.
Search is exact: dot the query against every row and pick the top k.
This is `O(N · dim)` per query, which is fast enough for the course's
working corpus (a few thousand chunks at most). When you outgrow it —
a corpus past 100k chunks, or a latency budget under a millisecond —
swap in a real ANN store. The interface is small and stable enough
that the swap is a one-file change.

Boilerplate (constructor, `add`, `__len__`, `__repr__`):
implemented. Scaffolded (`search`): the lesson.

Cosine similarity convention: every embedder in this module returns
L2-normalized rows. Cosine similarity therefore reduces to the
dot product `q · v` for unit-norm `q` and `v`. The store assumes
this — if you swap in an embedder that doesn't normalize, search
will return nonsense. The contract is explicit on the `Embedder`
side; the store doesn't re-normalize defensively.
"""
from __future__ import annotations

import numpy as np

from .chunk import Chunk


class NumpyVectorStore:
    """A flat in-memory vector store backed by a numpy array.

    Attributes:
        dim: the dimensionality of vectors. Set at construction; every
            subsequent `add` call must pass vectors with this dim.

    Backing storage:
        * A `list[Chunk]` (`self._chunks`).
        * A `(len(chunks), dim)` float32 array (`self._vectors`).
        Both stay in lockstep — `chunks[i]` corresponds to
        `vectors[i]`. Don't mutate either directly; use `add`.

    The store assumes vectors are L2-normalized for the cosine-as-
    dot-product trick. The embedders in `embed.py` enforce this on
    their end. The store does not re-normalize — you'd waste compute
    on every `add`, and `search` would silently produce wrong results
    for any caller who passed un-normalized vectors thinking the store
    would handle it.

    The boilerplate (constructor + `add` + `__len__`) is implemented.
    `search` is scaffolded.
    """

    def __init__(self, *, dim: int) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim}")

        self._dim = int(dim)
        self._chunks: list[Chunk] = []
        # Start with an empty (0, dim) array so `add` can vstack onto it.
        self._vectors: np.ndarray = np.zeros((0, self._dim), dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        """Read-only view of the indexed chunks. Returns the live list —
        treat as read-only; mutating it desyncs from the vectors array."""
        return self._chunks

    @property
    def vectors(self) -> np.ndarray:
        """Read-only view of the indexed vectors. Live array — treat as
        read-only."""
        return self._vectors

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Append `chunks` and their `vectors` to the store.

        Args:
            chunks: list of Chunk. May be empty (no-op).
            vectors: shape `(len(chunks), self.dim)` float array.

        Raises:
            ValueError: on shape mismatch (chunks vs vectors, dim
                vs self.dim).
        """
        if len(chunks) == 0 and vectors.shape[0] == 0:
            return
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"add: chunks and vectors must have matching length; "
                f"got len(chunks)={len(chunks)}, "
                f"vectors.shape[0]={vectors.shape[0]}"
            )
        if vectors.ndim != 2:
            raise ValueError(
                f"add: vectors must be a 2-D array, got shape {vectors.shape}"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"add: vectors have dim {vectors.shape[1]}, expected {self._dim}"
            )

        self._chunks.extend(chunks)
        # Cast to float32 for memory savings — most embedders return
        # float32 anyway. astype(copy=False) avoids the copy when the
        # input is already float32.
        self._vectors = np.vstack(
            [self._vectors, vectors.astype(np.float32, copy=False)]
        )

    def search(
        self,
        query: np.ndarray,
        *,
        k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Top-`k` cosine-similarity matches for `query`.

        Args:
            query: shape `(self.dim,)` 1-D array OR `(1, self.dim)`
                2-D. Either way, must be L2-normalized for cosine
                similarity to reduce to a dot product. The function
                squeezes 2-D `(1, dim)` inputs to 1-D as a convenience.
            k: number of results to return. Must be > 0. If the store
                has fewer than k chunks, returns all of them
                (sorted by similarity).

        Returns:
            A list of `(chunk, similarity)` tuples, ordered by
            similarity DESCENDING (most-similar first). Length is
            `min(k, len(self))`. Similarities are plain Python floats
            (not numpy scalars) — friendlier for JSON dumping.

        Raises:
            ValueError: on empty store, k <= 0, dim mismatch.

        Recipe:
            1. # Validate.
               if len(self) == 0:
                   raise ValueError("cannot search an empty store")
               if k <= 0:
                   raise ValueError(f"k must be > 0, got {k}")

            2. # Normalize input shape: accept (dim,) or (1, dim).
               if query.ndim == 2 and query.shape[0] == 1:
                   query = query[0]
               if query.ndim != 1:
                   raise ValueError(
                       f"query must be 1-D (dim,) or 2-D (1, dim); "
                       f"got shape {query.shape}"
                   )
               if query.shape[0] != self._dim:
                   raise ValueError(
                       f"query dim {query.shape[0]} != store dim {self._dim}"
                   )

            3. # Cosine = dot product (vectors are L2-normalized by
               # contract).
               sims = self._vectors @ query.astype(self._vectors.dtype)
               # sims shape: (len(self),)

            4. # Top-k. argpartition is O(N) for getting the top-k
               # unsorted; we then sort just those k for the final
               # ordering.
               k_eff = min(k, len(self))
               if k_eff == len(self):
                   top_idx = np.argsort(sims)[::-1]
               else:
                   # Partition so the top k are in the last k positions
                   # (no guaranteed order within), then sort those k.
                   part = np.argpartition(sims, -k_eff)[-k_eff:]
                   top_idx = part[np.argsort(sims[part])[::-1]]

            5. # Return as Python tuples for JSON-friendliness.
               return [
                   (self._chunks[int(i)], float(sims[int(i)]))
                   for i in top_idx
               ]

        Implementation notes:

          * The dot-product trick is correct ONLY when both query and
            stored vectors are L2-normalized. If a caller passes a
            raw, un-normalized query vector, the "similarities"
            returned are scaled by `||query||` — which doesn't change
            the argmax order (so the TOP-K is still right!) but does
            distort the actual similarity numbers. That's a subtle
            footgun: results LOOK ranked correctly, but the float
            scores are meaningless.

          * `argpartition` vs `argsort`: argpartition is `O(N)` for
            "get the top k indices unordered." For small k (< ~50)
            this is materially faster than sorting the whole array,
            which is `O(N log N)`. For large k (k ≥ N/2), sort is
            simpler and not slower. The branch in step 4 picks the
            right one.

          * `query.astype(self._vectors.dtype)` is cheap (no copy if
            dtypes already match) and avoids the silent f64↔f32
            promotion that would otherwise upcast `self._vectors`.

          * Returning Python floats (not numpy scalars) matters for
            anyone JSON-dumping results. `json` chokes on numpy
            scalar types; the explicit `float(...)` cast prevents
            "Object of type float32 is not JSON serializable."

        Sanity values:

          * Store with N chunks, k=N+1: returns N results (all of
            them) — does not pad with None / does not raise.

          * Store has the query vector itself (e.g. `add(c, query)`
            then `search(query)`): that chunk is the top result with
            similarity ≈ 1.0 (exactly 1.0 modulo float rounding).

          * Two identical chunks with identical vectors: both appear
            in the top-k with the same similarity; their relative
            order is implementation-defined.

          * k=1: returns the single best match.
        """
        if len(self) == 0:
            raise ValueError("cannot search an empty store")
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.ndim != 1:
            raise ValueError(
                "query must be 1-D (dim,) or 2-D (1, dim); "
                f"got shape {query.shape}"
            )
        if query.shape[0] != self._dim:
            raise ValueError(
                f"query dim {query.shape[0]} != store dim {self._dim}"
            )

        sims = self._vectors @ query.astype(self._vectors.dtype, copy=False)
        k_eff = min(k, len(self))
        if k_eff == len(self):
            top_idx = np.argsort(sims)[::-1]
        else:
            partitioned = np.argpartition(sims, -k_eff)[-k_eff:]
            top_idx = partitioned[np.argsort(sims[partitioned])[::-1]]

        return [
            (self._chunks[int(index)], float(sims[int(index)]))
            for index in top_idx
        ]

    def __repr__(self) -> str:
        return f"NumpyVectorStore(dim={self._dim}, n={len(self)})"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors.

    Returns 0.0 if either vector has zero norm — preferred over
    raising or returning NaN; downstream code can sort by similarity
    without special-casing.

    Implemented (not scaffolded) — the math is one line and is the
    same building block `NumpyVectorStore.search` uses internally
    when normalization isn't promised. Exposed as a public helper
    for diagnostics and one-off comparisons.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"cosine_similarity needs 1-D inputs; got shapes {a.shape}, {b.shape}"
        )
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"cosine_similarity dim mismatch: {a.shape[0]} vs {b.shape[0]}"
        )
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b)) / (norm_a * norm_b)
