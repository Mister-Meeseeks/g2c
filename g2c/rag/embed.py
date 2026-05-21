"""Embedders — turn a list of strings into a `(n, dim)` array of vectors.

The `Embedder` ABC declares one method, `embed(texts) -> ndarray`,
with `dim` exposed as a property. Two concrete embedders live here:

  * `HashEmbedder` — a stdlib-only character-n-gram hashing embedder.
    Deterministic, dependency-free, and good enough to exercise the
    pipeline end-to-end without an embedding service running. Captures
    SOME signal — strings sharing substrings get correlated vectors —
    but it isn't semantic; "Madrid" and "the capital of Spain"
    will look basically uncorrelated. Useful for testing, teaching,
    and as a fallback when no embedding model is available.

  * `OllamaEmbedder` — wraps Ollama's `/api/embeddings` HTTP endpoint.
    Defaults to `nomic-embed-text` (768-dim, ~265 MB), one of the
    better small open embedders. This is the embedder you actually
    want for real retrieval; `HashEmbedder` is for tests.

Both produce L2-NORMALIZED rows so cosine similarity reduces to a
dot product downstream — `NumpyVectorStore.search` assumes this.

Boilerplate (constructors, `dim`, `OllamaEmbedder.embed`):
implemented. Scaffolded (`HashEmbedder.embed`): the lesson.

Why is `OllamaEmbedder.embed` implemented but `OllamaBackend.complete`
(Module 16) was scaffolded? Because the HTTP pattern was the lesson
in Module 16; you've seen it now. Module 17's lesson is the embedding
*math* — what a hash embedder does, what cosine similarity does, how
chunks become coordinates. Repeating the HTTP boilerplate would just
be busywork.
"""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"


class Embedder(ABC):
    """Maps a list of strings to a `(n, dim)` row-normalized matrix.

    Subclasses implement `dim` (a property — should be cheap and
    constant after construction) and `embed`. `embed` MUST return an
    array of shape `(len(texts), dim)` whose rows are L2-normalized
    (zero rows are allowed for degenerate inputs but should be rare).

    The L2-normalization convention is what lets the vector store
    use a plain dot product for cosine similarity — saving a per-
    query division. Keep both ends of the contract honest.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """The dimensionality of vectors produced by `embed`."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed `texts` to a `(len(texts), self.dim)` float32 array."""


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize every row of a 2-D float array IN PLACE-LIKE.

    Returns a new array (never modifies input). Zero rows stay zero
    rather than producing NaNs from a divide-by-zero.

    Used by both embedders to satisfy the Embedder contract; exposed
    here as a private helper so we don't reimplement it twice.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Where the row is all zero, leave it alone. Replace the 0-norm
    # with 1.0 to avoid the divide-by-zero, then the row stays 0.
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return matrix / safe_norms


class HashEmbedder(Embedder):
    """Character-n-gram hashing embedder. Stdlib only.

    For each text:

      1. Lowercase it.
      2. Slide a window of every n in `ngram_range` across the
         characters.
      3. Hash each n-gram (with a salted hash so different `seed`
         values give different vectors) into a bucket in `[0, dim)`.
      4. Increment that bucket by 1.0.
      5. L2-normalize the resulting `(dim,)` vector.

    The math is a degenerate cousin of TF-IDF: every n-gram becomes
    a coordinate, two strings get a high dot product when they share
    n-grams. No true semantics — `"Madrid"` and `"the capital of
    Spain"` share zero non-trivial n-grams (case-insensitive their
    shared n-grams are " " or "th" or other stopword fragments) and
    so will look uncorrelated. But strings that share words or
    prefixes will cluster.

    Args:
        dim: the embedding dimension. Larger reduces hash collisions
            (multiple n-grams landing in the same bucket); smaller
            saves memory. 512 is plenty for character-level hashing
            on short strings; 4096 is overkill but cheap.
        ngram_range: a (min_n, max_n) tuple — inclusive on both ends.
            `(3, 5)` means trigrams, 4-grams, AND 5-grams all get
            hashed. Wider ranges capture more signal at the cost of
            more compute.
        seed: salt for the hash function. Different seeds produce
            different (but equally valid) embeddings of the same
            corpus. Useful for ensembling or for not having two
            embedders accidentally collide if you hash the same n-gram
            in two different parts of a system.

    Boilerplate (constructor + dim) implemented; `embed` scaffolded.
    """

    def __init__(
        self,
        *,
        dim: int = 512,
        ngram_range: tuple[int, int] = (3, 5),
        seed: int = 0,
    ) -> None:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim}")
        n_min, n_max = ngram_range
        if not (1 <= n_min <= n_max):
            raise ValueError(
                f"ngram_range must be a (min, max) with 1 <= min <= max, "
                f"got {ngram_range}"
            )
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")

        self._dim = int(dim)
        self._ngram_range = (int(n_min), int(n_max))
        self._seed = int(seed)
        # Pre-compute the seed bytes once. Each n-gram is hashed as
        # `seed_bytes + ngram_bytes` so different seeds give different
        # bucket assignments.
        self._seed_bytes = self._seed.to_bytes(8, "big", signed=False)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ngram_range(self) -> tuple[int, int]:
        return self._ngram_range

    def _bucket(self, ngram: str) -> int:
        """Hash one n-gram into a bucket index in `[0, self._dim)`.

        Helper exposed for the recipe — uses BLAKE2b (fast, stdlib,
        and stable across Python versions; unlike `hash()` which is
        randomized per-process). Salted with `self._seed_bytes`.
        """
        h = hashlib.blake2b(
            self._seed_bytes + ngram.encode("utf-8"),
            digest_size=8,
        ).digest()
        return int.from_bytes(h, "big") % self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """Hash-embed `texts` to a `(len(texts), self._dim)` float32 array.

        Args:
            texts: list of strings. May be empty. Each text may be
                empty (it embeds to a zero row).

        Returns:
            `(len(texts), self._dim)` float32, L2-normalized rows.

        Recipe:
            1. # Allocate the output.
               out = np.zeros((len(texts), self._dim), dtype=np.float32)

            2. # Tally n-gram bucket counts per text.
               n_min, n_max = self._ngram_range
               for i, text in enumerate(texts):
                   t = text.lower()
                   for n in range(n_min, n_max + 1):
                       if len(t) < n:
                           continue
                       for j in range(len(t) - n + 1):
                           ngram = t[j:j + n]
                           bucket = self._bucket(ngram)
                           out[i, bucket] += 1.0

            3. # L2-normalize each row.
               return _l2_normalize_rows(out)

        Implementation notes:

          * `texts` may legitimately be empty; the early return is a
            zero-row array. Don't raise on empty input — callers like
            `NumpyVectorStore.add` may have nothing to add.

          * Empty individual texts produce zero rows. They stay zero
            after `_l2_normalize_rows` (the helper handles zero norms).
            Cosine similarity against a zero row is 0 — which is the
            right behavior: "this thing has no signal."

          * The lowercase-then-window-then-hash pipeline is order-
            sensitive. Lowercasing AFTER hashing would make `"the"` and
            `"The"` collide differently — defeating the point of the
            normalization.

          * Character n-grams are robust to morphology and minor
            spelling differences ("running" and "runs" share `"run"`).
            Word n-grams are stronger when you have a tokenizer; we
            don't pull one in for this module.

          * BLAKE2b is overkill for a hash embedder (a one-line
            polynomial hash like FNV-1a would also work). The reason
            we use it is determinism: Python's built-in `hash()` is
            salted differently in each process, which breaks
            reproducibility — embeddings produced today wouldn't match
            embeddings produced tomorrow against the same text. Stable
            tests want stable hashes.

        Sanity values:

          * `embed([])` → shape `(0, dim)` float32 array.

          * `embed([""])` → shape `(1, dim)`, the single row is all
            zeros (no n-grams).

          * `embed(["abc", "abc"])` → both rows identical.

          * `embed(["abc"])` row is L2-normalized: `(out**2).sum() ==
            1.0` to within float32 precision.
        """
        # TODO
        raise NotImplementedError

    def __repr__(self) -> str:
        n_min, n_max = self._ngram_range
        return (
            f"HashEmbedder(dim={self._dim}, "
            f"ngram_range=({n_min}, {n_max}), seed={self._seed})"
        )


class OllamaEmbedder(Embedder):
    """Embedder backed by Ollama's `/api/embeddings` endpoint.

    Args:
        model_id: the Ollama embedding model tag. Defaults to
            `"nomic-embed-text"`. Pull it first: `ollama pull
            nomic-embed-text`. Other options: `"mxbai-embed-large"`
            (1024-dim), `"all-minilm"` (384-dim, smaller).
        dim: the dimension you expect this model to produce. Defaults
            to 768 (correct for `nomic-embed-text`). The first call
            verifies the actual response matches; if not, raises.
            Pre-declaring `dim` lets `Embedder.dim` work BEFORE any
            `embed` call has happened.
        base_url: Ollama server URL. Defaults to localhost:11434.
        timeout: per-request HTTP timeout in seconds.
        urlopen: optional `urllib.request.urlopen` override. Tests
            inject a fake; production callers leave it alone. Same
            pattern as `OllamaBackend`.

    The constructor + `embed` are both implemented. The HTTP pattern
    (POST → JSON → parse) is the same as `OllamaBackend.complete`
    from Module 16; the lesson there was the pattern, the lesson
    here is what RAG embeds and how the vectors get used. We don't
    re-scaffold it.

    The Ollama embeddings endpoint takes ONE prompt at a time, not a
    batch. `embed([t1, t2, ...])` issues N HTTP calls in sequence. A
    real production embedder would batch-async or batch via a single
    request — Ollama's API doesn't support batched embeddings as of
    the version this module targets.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_OLLAMA_EMBED_MODEL,
        *,
        dim: int = 768,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 120.0,
        urlopen: Any = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("OllamaEmbedder requires a non-empty model_id")
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim}")
        if not isinstance(base_url, str) or not base_url:
            raise ValueError("OllamaEmbedder requires a non-empty base_url")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")

        self._model_id = model_id
        self._dim = int(dim)
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._urlopen = urlopen if urlopen is not None else urllib.request.urlopen

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def base_url(self) -> str:
        return self._base_url

    def embed(self, texts: list[str]) -> np.ndarray:
        """POST each text to `/api/embeddings` and stack the results.

        Returns:
            `(len(texts), self._dim)` float32 array, L2-normalized.

        Raises:
            OllamaEmbedError: on HTTP failure, JSON parse error,
                missing `embedding` field, or returned vector with
                a different dim than `self._dim`.
        """
        if len(texts) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)

        url = f"{self._base_url}/api/embeddings"
        rows = np.zeros((len(texts), self._dim), dtype=np.float32)

        for i, text in enumerate(texts):
            body = {"model": self._model_id, "prompt": text}
            body_bytes = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with self._urlopen(req, timeout=self._timeout) as resp:
                    resp_bytes = resp.read()
            except urllib.error.HTTPError as e:
                raise OllamaEmbedError(
                    f"Ollama HTTP error {e.code} at {url}: {e.reason}"
                ) from e
            except urllib.error.URLError as e:
                raise OllamaEmbedError(
                    f"Could not reach Ollama at {url}: {e.reason}"
                ) from e

            try:
                data = json.loads(resp_bytes)
            except json.JSONDecodeError as e:
                raise OllamaEmbedError(
                    f"Ollama returned non-JSON response from {url}"
                ) from e
            if "embedding" not in data:
                raise OllamaEmbedError(
                    f"Ollama embeddings response missing 'embedding' field: "
                    f"keys={list(data.keys())}"
                )

            vec = data["embedding"]
            if len(vec) != self._dim:
                raise OllamaEmbedError(
                    f"Ollama returned a {len(vec)}-dim vector for "
                    f"model {self._model_id!r}; OllamaEmbedder was "
                    f"configured with dim={self._dim}. Reconstruct with "
                    f"the correct dim, or check the model tag."
                )
            rows[i] = vec

        return _l2_normalize_rows(rows)

    def __repr__(self) -> str:
        return (
            f"OllamaEmbedder(model_id={self._model_id!r}, "
            f"dim={self._dim}, base_url={self._base_url!r})"
        )


class OllamaEmbedError(RuntimeError):
    """Raised on any HTTP / JSON / shape-mismatch failure from `OllamaEmbedder`.

    Same role as `g2c.inference.OllamaError` for the chat backend:
    a single exception type so callers don't need to catch three
    stdlib types to handle "Ollama isn't running."
    """
