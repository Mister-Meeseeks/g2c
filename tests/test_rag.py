"""Tests for g2c/rag/ — Module 17 (Retrieval-augmented generation).

Suggested order to implement & turn green:

  1. `chunk_text` (g2c/rag/chunk.py).
     The sliding-window-with-overlap algorithm. Easiest place to start
     — pure string slicing, no numpy, no HTTP.
     Turns green: TestChunkText, TestChunkTextEdgeCases.

  2. `HashEmbedder.embed` (g2c/rag/embed.py).
     Hash character n-grams into bucket counts; L2-normalize. Tests
     don't require any external service.
     Turns green: TestHashEmbedderEmbed.

  3. `NumpyVectorStore.search` (g2c/rag/store.py).
     Dot product + top-k via argpartition/argsort.
     Turns green: TestNumpyVectorStoreSearch, TestVectorStoreSearchEdgeCases.

  4. `assemble_rag_prompt` (g2c/rag/prompt.py).
     Splice question + chunks + system + instruction into a numbered-
     citation prompt. Pure string construction.
     Turns green: TestAssembleRAGPrompt, TestAssembleRAGPromptEdgeCases.

The boilerplate tests pass from the start (since the boilerplate is
implemented), serving as a sanity check on the test file itself.
The integration tests (TestDenseRetriever, TestRAGPipeline) turn
green automatically once the four scaffolded methods above are done —
they exercise the wiring around the components.

Headline tests to watch:

  * test_chunk_text_overlap_correct — pins the stride math: each chunk
    starts `chunk_size - chunk_overlap` characters past the previous.
    Off-by-one here breaks every retrieval downstream.

  * test_chunk_text_covers_full_document — the chunker must reach the
    end. A bug that stops one stride short silently loses the last
    paragraph of every document.

  * test_hash_embedder_rows_are_unit_norm — the L2-normalization
    contract. Without it, `NumpyVectorStore.search`'s dot-product
    trick computes the wrong similarity.

  * test_search_returns_descending_order — top-1 is the most similar.
    A reversed sort silently degrades retrieval to anti-retrieval.

  * test_assemble_rag_prompt_uses_one_based_citations — humans cite
    [1], not [0]. Forgetting `start=1` confuses the model AND any
    citation parser.

  * test_assemble_rag_prompt_includes_source_label — the source name
    in the prompt is what makes citations human-readable.

  * test_pipeline_propagates_retrieved_chunks_to_prompt — wire bug
    catcher: the chunks the retriever returned MUST be the chunks the
    prompt cites. A confused implementation could embed the question,
    retrieve, then assemble a prompt against a different chunk list.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from g2c.inference import Backend, BackendInfo, InferenceResult
from g2c.rag import (
    DEFAULT_INSTRUCTION,
    DEFAULT_OLLAMA_EMBED_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_SYSTEM,
    Chunk,
    DenseRetriever,
    Embedder,
    HashEmbedder,
    NumpyVectorStore,
    OllamaEmbedder,
    OllamaEmbedError,
    RAGAnswer,
    RAGPipeline,
    RAGPrompt,
    RetrievedChunk,
    assemble_rag_prompt,
    chunk_text,
    cosine_similarity,
)

# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class _FakeBackendInfo:
    """A BackendInfo-equivalent for the FakeBackend below."""

    name: str = "fake"
    model_id: str = "fake-model"


class _FakeBackend(Backend):
    """A `Backend` whose `complete` returns a scripted answer.

    Used to test `RAGPipeline` without spinning up a real model.
    Records the last prompt it was called with for assertion.
    """

    def __init__(self, scripted_answer: str = "MOCK ANSWER") -> None:
        self._info = BackendInfo(name="fake", model_id="fake-model")
        self.scripted_answer = scripted_answer
        self.last_prompt: str | None = None
        self.last_kwargs: dict[str, Any] | None = None

    @property
    def info(self) -> BackendInfo:
        return self._info

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> InferenceResult:
        self.last_prompt = prompt
        self.last_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
        return InferenceResult(
            prompt=prompt,
            completion=self.scripted_answer,
            prompt_tokens=len(prompt),
            completion_tokens=len(self.scripted_answer),
            latency_ms=1.0,
            backend=self._info,
            metadata={},
        )


class _FakeEmbedder(Embedder):
    """An embedder whose vectors are explicitly registered per-text.

    Lets a test set up "the query embedding looks like X, the chunks
    embed to Y, Z, W" without going through the hash machinery — so
    `NumpyVectorStore.search` can be tested independently of
    `HashEmbedder.embed`.
    """

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self._table: dict[str, np.ndarray] = {}

    def register(self, text: str, vector: list[float]) -> None:
        v = np.asarray(vector, dtype=np.float32)
        if v.shape != (self._dim,):
            raise ValueError(f"register expects shape ({self._dim},)")
        self._table[text] = v

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t not in self._table:
                raise KeyError(f"_FakeEmbedder: unregistered text {t!r}")
            out[i] = self._table[t]
        return out


class _FakeResponse:
    """A minimal HTTP response stand-in for `urlopen`-injected tests."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def _make_fake_urlopen(
    *,
    response_body: dict | None = None,
    raise_exc: BaseException | None = None,
    bad_json: bytes | None = None,
    captured: list | None = None,
):
    """Construct a urlopen-callable for OllamaEmbedder tests.

    Args:
        response_body: dict to JSON-encode and return.
        raise_exc: exception to raise instead of returning a response.
        bad_json: raw bytes to return (for testing JSONDecodeError).
        captured: optional list to append (request, kwargs) tuples to.
    """

    def fake(req, *args, **kwargs):
        if captured is not None:
            captured.append((req, kwargs))
        if raise_exc is not None:
            raise raise_exc
        if bad_json is not None:
            return _FakeResponse(bad_json)
        body = json.dumps(response_body or {}).encode("utf-8")
        return _FakeResponse(body)

    return fake


# =============================================================================
# Chunk dataclass — boilerplate (passes immediately)
# =============================================================================


class TestChunkBoilerplate:
    def test_construct_basic(self) -> None:
        c = Chunk(text="hello", source="doc.md", start=0, end=5)
        assert c.text == "hello"
        assert c.source == "doc.md"
        assert c.start == 0
        assert c.end == 5
        assert c.metadata == {}

    def test_metadata_defaults_to_empty_dict(self) -> None:
        c = Chunk(text="x", source="s", start=0, end=1)
        assert c.metadata == {}

    def test_metadata_is_independent_per_chunk(self) -> None:
        c1 = Chunk(text="x", source="s", start=0, end=1)
        c2 = Chunk(text="y", source="s", start=0, end=1)
        c1.metadata["k"] = 1
        assert "k" not in c2.metadata

    def test_metadata_passes_through(self) -> None:
        c = Chunk(text="x", source="s", start=0, end=1, metadata={"page": 7})
        assert c.metadata == {"page": 7}

    def test_frozen_no_attribute_assignment(self) -> None:
        c = Chunk(text="x", source="s", start=0, end=1)
        with pytest.raises(Exception):
            c.text = "different"  # type: ignore[misc]

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Chunk(text="", source="s", start=0, end=0)

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="start"):
            Chunk(text="x", source="s", start=-1, end=0)

    def test_end_must_match_start_plus_len(self) -> None:
        with pytest.raises(ValueError, match="end"):
            Chunk(text="hello", source="s", start=0, end=3)

    def test_end_correct_with_offset(self) -> None:
        c = Chunk(text="abc", source="s", start=10, end=13)
        assert c.start == 10
        assert c.end == 13

    def test_non_str_source_rejected(self) -> None:
        with pytest.raises(TypeError, match="source"):
            Chunk(text="x", source=123, start=0, end=1)  # type: ignore[arg-type]

    def test_equality(self) -> None:
        c1 = Chunk(text="hi", source="s", start=0, end=2)
        c2 = Chunk(text="hi", source="s", start=0, end=2)
        assert c1 == c2
        c3 = Chunk(text="hi", source="t", start=0, end=2)
        assert c1 != c3


# =============================================================================
# chunk_text — scaffolded
# =============================================================================


class TestChunkText:
    def test_returns_list_of_chunks(self) -> None:
        result = chunk_text("hello world", source="s", chunk_size=100, chunk_overlap=0)
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_short_doc_yields_single_chunk(self) -> None:
        result = chunk_text("hi there", source="s", chunk_size=100, chunk_overlap=0)
        assert len(result) == 1
        assert result[0].text == "hi there"
        assert result[0].start == 0
        assert result[0].end == 8

    def test_doc_exactly_chunk_size_yields_single_chunk(self) -> None:
        text = "abcde"
        result = chunk_text(text, source="s", chunk_size=5, chunk_overlap=0)
        assert len(result) == 1
        assert result[0].text == text
        assert result[0].end == 5

    def test_two_chunks_no_overlap(self) -> None:
        text = "abcdefghijklmno"  # 15 chars
        result = chunk_text(text, source="s", chunk_size=10, chunk_overlap=0)
        assert len(result) == 2
        assert result[0].text == "abcdefghij"
        assert result[0].start == 0
        assert result[0].end == 10
        assert result[1].text == "klmno"
        assert result[1].start == 10
        assert result[1].end == 15

    def test_chunk_text_overlap_correct(self) -> None:
        text = "abcdefghijklmno"  # 15 chars
        result = chunk_text(text, source="s", chunk_size=10, chunk_overlap=3)
        # stride = 10 - 3 = 7
        assert len(result) == 2
        assert result[0].text == "abcdefghij"
        assert result[0].start == 0
        assert result[0].end == 10
        assert result[1].start == 7
        assert result[1].text == "hijklmno"
        assert result[1].end == 15

    def test_chunk_text_covers_full_document(self) -> None:
        text = "x" * 100
        result = chunk_text(text, source="s", chunk_size=30, chunk_overlap=5)
        # First chunk starts at 0; last chunk ends at len(text).
        assert result[0].start == 0
        assert result[-1].end == 100

    def test_chunk_text_propagates_source(self) -> None:
        result = chunk_text("hello", source="docs/foo.md", chunk_size=100, chunk_overlap=0)
        assert all(c.source == "docs/foo.md" for c in result)

    def test_chunk_text_propagates_metadata(self) -> None:
        result = chunk_text(
            "hello", source="s", chunk_size=100, chunk_overlap=0, metadata={"lang": "en"}
        )
        assert all(c.metadata == {"lang": "en"} for c in result)

    def test_chunk_text_metadata_independence(self) -> None:
        result = chunk_text(
            "abc" * 100, source="s", chunk_size=10, chunk_overlap=0, metadata={"k": 1}
        )
        assert len(result) > 1
        # Mutating one chunk's metadata must not affect another's.
        result[0].metadata["k"] = 999
        assert result[1].metadata["k"] == 1

    def test_chunk_text_long_doc(self) -> None:
        text = "abcdefghij" * 10  # 100 chars
        result = chunk_text(text, source="s", chunk_size=20, chunk_overlap=5)
        # stride = 15. Starts: 0, 15, 30, 45, 60, 75, 90 (last ends at 100).
        starts = [c.start for c in result]
        assert starts[:6] == [0, 15, 30, 45, 60, 75]
        # Last chunk's end is exactly len(text).
        assert result[-1].end == len(text)

    def test_chunk_text_starts_strictly_increasing(self) -> None:
        text = "x" * 200
        result = chunk_text(text, source="s", chunk_size=50, chunk_overlap=10)
        starts = [c.start for c in result]
        assert all(b > a for a, b in zip(starts, starts[1:]))


class TestChunkTextValidation:
    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            chunk_text("", source="s", chunk_size=100)

    def test_rejects_zero_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_text("hello", source="s", chunk_size=0)

    def test_rejects_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("hello", source="s", chunk_size=10, chunk_overlap=-1)

    def test_rejects_overlap_geq_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("hello", source="s", chunk_size=10, chunk_overlap=10)


class TestChunkTextEdgeCases:
    def test_unicode_text(self) -> None:
        text = "héllo wörld 🌍"
        result = chunk_text(text, source="s", chunk_size=100, chunk_overlap=0)
        # Single chunk, exact text round-trip.
        assert len(result) == 1
        assert result[0].text == text

    def test_newlines_preserved(self) -> None:
        text = "line one\nline two\nline three"
        result = chunk_text(text, source="s", chunk_size=100, chunk_overlap=0)
        assert result[0].text == text

    def test_chunks_are_a_partition_when_overlap_is_zero(self) -> None:
        text = "x" * 30
        result = chunk_text(text, source="s", chunk_size=10, chunk_overlap=0)
        # Reassembling chunks back-to-back recovers the source.
        assert "".join(c.text for c in result) == text


# =============================================================================
# Embedder ABC — boilerplate
# =============================================================================


class TestEmbedderABC:
    def test_abstract_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Embedder()  # type: ignore[abstract]

    def test_subclass_can_implement(self) -> None:
        emb = _FakeEmbedder(dim=4)
        assert emb.dim == 4


# =============================================================================
# HashEmbedder boilerplate (passes immediately)
# =============================================================================


class TestHashEmbedderBoilerplate:
    def test_default_construction(self) -> None:
        emb = HashEmbedder()
        assert emb.dim == 512
        assert emb.ngram_range == (3, 5)

    def test_construction_with_args(self) -> None:
        emb = HashEmbedder(dim=128, ngram_range=(2, 4), seed=42)
        assert emb.dim == 128
        assert emb.ngram_range == (2, 4)

    def test_zero_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            HashEmbedder(dim=0)

    def test_negative_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            HashEmbedder(dim=-1)

    def test_inverted_ngram_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="ngram_range"):
            HashEmbedder(ngram_range=(5, 3))

    def test_zero_ngram_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="ngram_range"):
            HashEmbedder(ngram_range=(0, 3))

    def test_repr_contains_dim_and_seed(self) -> None:
        r = repr(HashEmbedder(dim=64, seed=7))
        assert "dim=64" in r
        assert "seed=7" in r

    def test_bucket_helper_returns_in_range(self) -> None:
        emb = HashEmbedder(dim=128)
        for ng in ["abc", "the", " of", "xyz", "🌍🌍"]:
            b = emb._bucket(ng)
            assert 0 <= b < 128


# =============================================================================
# HashEmbedder.embed — scaffolded
# =============================================================================


class TestHashEmbedderEmbed:
    def test_empty_input_returns_zero_rows(self) -> None:
        emb = HashEmbedder(dim=64)
        out = emb.embed([])
        assert out.shape == (0, 64)
        assert out.dtype == np.float32

    def test_single_text_shape(self) -> None:
        emb = HashEmbedder(dim=64)
        out = emb.embed(["hello world"])
        assert out.shape == (1, 64)
        assert out.dtype == np.float32

    def test_multiple_texts_shape(self) -> None:
        emb = HashEmbedder(dim=128)
        out = emb.embed(["a quick brown fox", "the lazy dog", "another"])
        assert out.shape == (3, 128)

    def test_hash_embedder_rows_are_unit_norm(self) -> None:
        emb = HashEmbedder(dim=128)
        out = emb.embed(["hello world from a slightly longer string"])
        norm = float(np.linalg.norm(out[0]))
        assert abs(norm - 1.0) < 1e-5

    def test_empty_string_produces_zero_row(self) -> None:
        emb = HashEmbedder(dim=64)
        out = emb.embed([""])
        # No n-grams to hash, so every bucket stays 0; norm is 0.
        assert float(np.linalg.norm(out[0])) == 0.0

    def test_very_short_string_below_min_n_produces_zero_row(self) -> None:
        # Min n=3, string is 2 chars -> no n-grams of any length in range.
        emb = HashEmbedder(dim=64, ngram_range=(3, 3))
        out = emb.embed(["ab"])
        assert float(np.linalg.norm(out[0])) == 0.0

    def test_deterministic_same_seed(self) -> None:
        emb = HashEmbedder(dim=64, seed=12345)
        a = emb.embed(["the quick brown fox"])
        b = emb.embed(["the quick brown fox"])
        assert np.array_equal(a, b)

    def test_deterministic_across_instances(self) -> None:
        a = HashEmbedder(dim=64, seed=42).embed(["sentence"])
        b = HashEmbedder(dim=64, seed=42).embed(["sentence"])
        assert np.array_equal(a, b)

    def test_different_seeds_produce_different_vectors(self) -> None:
        a = HashEmbedder(dim=64, seed=1).embed(["the quick brown fox"])
        b = HashEmbedder(dim=64, seed=2).embed(["the quick brown fox"])
        # Almost certainly different — different salt, different buckets.
        assert not np.array_equal(a, b)

    def test_lowercase_normalization(self) -> None:
        emb = HashEmbedder(dim=64)
        a = emb.embed(["Hello"])
        b = emb.embed(["hello"])
        assert np.allclose(a, b)

    def test_similar_strings_correlate(self) -> None:
        emb = HashEmbedder(dim=512)
        # "running" and "runs" share the trigram "run" — should have
        # at least *some* positive cosine similarity.
        out = emb.embed(["running", "runs", "completely different topic"])
        sim_close = float(out[0] @ out[1])
        sim_far = float(out[0] @ out[2])
        assert sim_close > sim_far

    def test_identical_strings_produce_identical_rows(self) -> None:
        emb = HashEmbedder(dim=64)
        out = emb.embed(["foo", "foo"])
        assert np.array_equal(out[0], out[1])


# =============================================================================
# OllamaEmbedder — implemented (HTTP wiring + the embed call)
# =============================================================================


class TestOllamaEmbedderBoilerplate:
    def test_default_construction(self) -> None:
        emb = OllamaEmbedder()
        assert emb.model_id == DEFAULT_OLLAMA_EMBED_MODEL
        assert emb.dim == 768
        assert emb.base_url == DEFAULT_OLLAMA_URL

    def test_construction_with_args(self) -> None:
        emb = OllamaEmbedder(
            "mxbai-embed-large",
            dim=1024,
            base_url="http://example:8080",
            timeout=30.0,
        )
        assert emb.model_id == "mxbai-embed-large"
        assert emb.dim == 1024
        assert emb.base_url == "http://example:8080"

    def test_strips_trailing_slash_from_base_url(self) -> None:
        emb = OllamaEmbedder(base_url="http://example:8080/")
        assert emb.base_url == "http://example:8080"

    def test_empty_model_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            OllamaEmbedder("")

    def test_zero_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            OllamaEmbedder(dim=0)

    def test_empty_base_url_rejected(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            OllamaEmbedder(base_url="")

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            OllamaEmbedder(timeout=0)

    def test_repr_contains_model_id(self) -> None:
        r = repr(OllamaEmbedder("nomic-embed-text"))
        assert "nomic-embed-text" in r


class TestOllamaEmbedderEmbed:
    def test_empty_input_returns_zero_rows(self) -> None:
        # Empty input is a no-op — does not even need urlopen.
        emb = OllamaEmbedder(dim=8, urlopen=lambda *a, **kw: None)
        out = emb.embed([])
        assert out.shape == (0, 8)

    def test_basic_embed_call(self) -> None:
        captured: list = []
        fake = _make_fake_urlopen(
            response_body={"embedding": [1.0, 0.0, 0.0, 0.0]},
            captured=captured,
        )
        emb = OllamaEmbedder(dim=4, urlopen=fake)
        out = emb.embed(["hello"])
        assert out.shape == (1, 4)
        assert len(captured) == 1
        # The body should contain the prompt and model.
        req, _ = captured[0]
        body = json.loads(req.data.decode("utf-8"))
        assert body["model"] == DEFAULT_OLLAMA_EMBED_MODEL
        assert body["prompt"] == "hello"

    def test_multiple_calls_one_per_text(self) -> None:
        captured: list = []
        fake = _make_fake_urlopen(
            response_body={"embedding": [1.0, 0.0, 0.0, 0.0]},
            captured=captured,
        )
        emb = OllamaEmbedder(dim=4, urlopen=fake)
        emb.embed(["a", "b", "c"])
        # Ollama embed endpoint takes one prompt at a time.
        assert len(captured) == 3

    def test_url_is_correct(self) -> None:
        captured: list = []
        fake = _make_fake_urlopen(
            response_body={"embedding": [1.0, 0.0, 0.0, 0.0]},
            captured=captured,
        )
        emb = OllamaEmbedder(
            base_url="http://example:11434", dim=4, urlopen=fake
        )
        emb.embed(["x"])
        req, _ = captured[0]
        assert req.full_url == "http://example:11434/api/embeddings"

    def test_returns_normalized_rows(self) -> None:
        # An un-normalized response from the server should be normalized
        # by the embedder.
        fake = _make_fake_urlopen(response_body={"embedding": [3.0, 4.0, 0.0, 0.0]})
        emb = OllamaEmbedder(dim=4, urlopen=fake)
        out = emb.embed(["x"])
        assert abs(float(np.linalg.norm(out[0])) - 1.0) < 1e-5

    def test_dim_mismatch_raises(self) -> None:
        fake = _make_fake_urlopen(response_body={"embedding": [1.0, 0.0, 0.0]})
        emb = OllamaEmbedder(dim=4, urlopen=fake)
        with pytest.raises(OllamaEmbedError, match="dim"):
            emb.embed(["x"])

    def test_missing_embedding_field_raises(self) -> None:
        fake = _make_fake_urlopen(response_body={"other": "noise"})
        emb = OllamaEmbedder(dim=4, urlopen=fake)
        with pytest.raises(OllamaEmbedError, match="embedding"):
            emb.embed(["x"])

    def test_bad_json_raises(self) -> None:
        fake = _make_fake_urlopen(bad_json=b"not valid json {{{")
        emb = OllamaEmbedder(dim=4, urlopen=fake)
        with pytest.raises(OllamaEmbedError, match="JSON"):
            emb.embed(["x"])

    def test_url_error_wrapped(self) -> None:
        import urllib.error

        fake = _make_fake_urlopen(
            raise_exc=urllib.error.URLError("connection refused")
        )
        emb = OllamaEmbedder(dim=4, urlopen=fake)
        with pytest.raises(OllamaEmbedError, match="reach"):
            emb.embed(["x"])

    def test_http_error_wrapped(self) -> None:
        import urllib.error

        http_err = urllib.error.HTTPError(
            "http://x", 500, "Internal Server Error", {}, io.BytesIO(b"")
        )
        fake = _make_fake_urlopen(raise_exc=http_err)
        emb = OllamaEmbedder(dim=4, urlopen=fake)
        with pytest.raises(OllamaEmbedError, match="500"):
            emb.embed(["x"])


# =============================================================================
# NumpyVectorStore boilerplate (passes immediately)
# =============================================================================


class TestNumpyVectorStoreBoilerplate:
    def test_default_construction(self) -> None:
        store = NumpyVectorStore(dim=8)
        assert store.dim == 8
        assert len(store) == 0

    def test_zero_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            NumpyVectorStore(dim=0)

    def test_negative_dim_rejected(self) -> None:
        with pytest.raises(ValueError, match="dim"):
            NumpyVectorStore(dim=-1)

    def test_add_basic(self) -> None:
        store = NumpyVectorStore(dim=4)
        chunks = [Chunk(text=f"c{i}", source="s", start=i, end=i + 2) for i in range(3)]
        vectors = np.eye(3, 4, dtype=np.float32)
        store.add(chunks, vectors)
        assert len(store) == 3

    def test_add_empty_is_noop(self) -> None:
        store = NumpyVectorStore(dim=4)
        store.add([], np.zeros((0, 4), dtype=np.float32))
        assert len(store) == 0

    def test_add_dim_mismatch_raises(self) -> None:
        store = NumpyVectorStore(dim=4)
        chunks = [Chunk(text="x", source="s", start=0, end=1)]
        vectors = np.zeros((1, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="dim"):
            store.add(chunks, vectors)

    def test_add_length_mismatch_raises(self) -> None:
        store = NumpyVectorStore(dim=4)
        chunks = [Chunk(text="x", source="s", start=0, end=1)]
        vectors = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="match"):
            store.add(chunks, vectors)

    def test_add_appends(self) -> None:
        store = NumpyVectorStore(dim=4)
        chunks_a = [Chunk(text="a", source="s", start=0, end=1)]
        chunks_b = [Chunk(text="b", source="s", start=0, end=1)]
        store.add(chunks_a, np.array([[1, 0, 0, 0]], dtype=np.float32))
        store.add(chunks_b, np.array([[0, 1, 0, 0]], dtype=np.float32))
        assert len(store) == 2
        assert store.chunks[0].text == "a"
        assert store.chunks[1].text == "b"

    def test_repr_includes_dim_and_n(self) -> None:
        store = NumpyVectorStore(dim=4)
        store.add(
            [Chunk(text="x", source="s", start=0, end=1)],
            np.zeros((1, 4), dtype=np.float32),
        )
        r = repr(store)
        assert "dim=4" in r
        assert "n=1" in r


# =============================================================================
# NumpyVectorStore.search — scaffolded
# =============================================================================


def _build_store_with_eye_vectors(n: int = 4) -> NumpyVectorStore:
    """Helper: a store of n chunks whose vectors are the n×n identity rows
    (extended with zeros if dim > n)."""
    store = NumpyVectorStore(dim=n)
    chunks = [
        Chunk(text=f"chunk-{i}", source="s", start=i * 10, end=i * 10 + 7)
        for i in range(n)
    ]
    vectors = np.eye(n, dtype=np.float32)
    store.add(chunks, vectors)
    return store


class TestNumpyVectorStoreSearch:
    def test_search_returns_descending_order(self) -> None:
        store = NumpyVectorStore(dim=4)
        chunks = [
            Chunk(text="far", source="s", start=0, end=3),
            Chunk(text="near", source="s", start=0, end=4),
            Chunk(text="mid", source="s", start=0, end=3),
        ]
        # Query is [1, 0, 0, 0]. Vectors with descending similarity:
        vectors = np.array(
            [
                [0.1, 0.99, 0.0, 0.0],  # far  — sim ~ 0.1
                [0.99, 0.1, 0.0, 0.0],  # near — sim ~ 0.99
                [0.7, 0.7, 0.0, 0.0],   # mid  — sim ~ 0.7
            ],
            dtype=np.float32,
        )
        # Normalize so cosine = dot.
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        store.add(chunks, vectors)

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, k=3)

        assert len(results) == 3
        assert results[0][0].text == "near"
        assert results[1][0].text == "mid"
        assert results[2][0].text == "far"
        # Scores must be descending.
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_truncates(self) -> None:
        store = _build_store_with_eye_vectors(n=4)
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = store.search(query, k=2)
        assert len(results) == 2

    def test_search_returns_tuples(self) -> None:
        store = _build_store_with_eye_vectors(n=4)
        results = store.search(np.eye(4, dtype=np.float32)[0], k=1)
        c, s = results[0]
        assert isinstance(c, Chunk)
        assert isinstance(s, float)  # not numpy.float32

    def test_search_query_2d_accepted(self) -> None:
        store = _build_store_with_eye_vectors(n=4)
        query_2d = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = store.search(query_2d, k=1)
        assert results[0][0].text == "chunk-0"

    def test_search_k_larger_than_store_returns_all(self) -> None:
        store = _build_store_with_eye_vectors(n=3)
        results = store.search(np.array([1, 0, 0], dtype=np.float32), k=100)
        assert len(results) == 3

    def test_search_top_result_self_match(self) -> None:
        # When the query equals one of the stored vectors, it should be
        # the top result with similarity ≈ 1.0.
        store = _build_store_with_eye_vectors(n=5)
        query = np.eye(5, dtype=np.float32)[2]  # third vector
        results = store.search(query, k=1)
        assert results[0][0].text == "chunk-2"
        assert abs(results[0][1] - 1.0) < 1e-5

    def test_search_descending_scores(self) -> None:
        store = _build_store_with_eye_vectors(n=5)
        query = np.ones(5, dtype=np.float32) / np.sqrt(5)  # normalized
        results = store.search(query, k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestVectorStoreSearchEdgeCases:
    def test_empty_store_raises(self) -> None:
        store = NumpyVectorStore(dim=4)
        with pytest.raises(ValueError, match="empty"):
            store.search(np.zeros(4, dtype=np.float32))

    def test_zero_k_raises(self) -> None:
        store = _build_store_with_eye_vectors(n=2)
        with pytest.raises(ValueError, match="k"):
            store.search(np.zeros(2, dtype=np.float32), k=0)

    def test_query_dim_mismatch_raises(self) -> None:
        store = _build_store_with_eye_vectors(n=4)
        with pytest.raises(ValueError, match="dim"):
            store.search(np.zeros(8, dtype=np.float32), k=1)

    def test_query_3d_rejected(self) -> None:
        store = _build_store_with_eye_vectors(n=4)
        with pytest.raises(ValueError, match="(1-D|2-D|shape)"):
            store.search(np.zeros((1, 1, 4), dtype=np.float32), k=1)


# =============================================================================
# cosine_similarity helper — implemented
# =============================================================================


class TestCosineSimilarityHelper:
    def test_basic_orthogonal(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == 0.0

    def test_basic_parallel(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([2.0, 0.0])
        assert abs(cosine_similarity(a, b) - 1.0) < 1e-9

    def test_basic_antiparallel(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) + 1.0) < 1e-9

    def test_zero_norm_returns_zero(self) -> None:
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 1.0])
        assert cosine_similarity(a, b) == 0.0

    def test_dim_mismatch_raises(self) -> None:
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="(dim|mismatch)"):
            cosine_similarity(a, b)

    def test_2d_input_rejected(self) -> None:
        a = np.array([[1.0, 0.0]])
        b = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity(a, b)


# =============================================================================
# DenseRetriever — implemented (composition test)
# =============================================================================


class TestDenseRetrieverBoilerplate:
    def test_construction(self) -> None:
        emb = HashEmbedder(dim=32)
        store = NumpyVectorStore(dim=32)
        r = DenseRetriever(emb, store)
        assert r.embedder is emb
        assert r.store is store

    def test_none_embedder_rejected(self) -> None:
        with pytest.raises(ValueError, match="embedder"):
            DenseRetriever(None, NumpyVectorStore(dim=4))  # type: ignore[arg-type]

    def test_none_store_rejected(self) -> None:
        with pytest.raises(ValueError, match="store"):
            DenseRetriever(HashEmbedder(dim=4), None)  # type: ignore[arg-type]


class TestDenseRetrieve:
    """End-to-end tests of `retrieve` using a fake embedder + a real store."""

    def _build(self, dim: int = 4) -> tuple[_FakeEmbedder, NumpyVectorStore]:
        emb = _FakeEmbedder(dim=dim)
        store = NumpyVectorStore(dim=dim)
        return emb, store

    def test_retrieve_basic_top_1(self) -> None:
        emb, store = self._build()
        c1 = Chunk(text="The capital of Spain is Madrid.", source="es.md",
                   start=0, end=31)
        c2 = Chunk(text="Bananas are yellow.", source="fruit.md",
                   start=0, end=19)

        store.add(
            [c1, c2],
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
        )

        emb.register("What is the capital of Spain?", [1.0, 0.0, 0.0, 0.0])

        retriever = DenseRetriever(emb, store)
        results = retriever.retrieve("What is the capital of Spain?", k=1)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].chunk.text == c1.text
        assert results[0].rank == 1

    def test_retrieve_ranks_assigned_in_order(self) -> None:
        emb, store = self._build()
        chunks = [
            Chunk(text=f"c{i}", source="s", start=0, end=2)
            for i in range(3)
        ]
        # Vectors: aligned in descending similarity to query.
        vecs = np.array(
            [[1, 0, 0, 0], [0.7, 0.7, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.add(chunks, vecs)

        emb.register("q", [1.0, 0.0, 0.0, 0.0])

        retriever = DenseRetriever(emb, store)
        results = retriever.retrieve("q", k=3)
        assert [r.rank for r in results] == [1, 2, 3]
        assert results[0].chunk.text == "c0"
        assert results[1].chunk.text == "c1"
        assert results[2].chunk.text == "c2"

    def test_retrieve_score_is_descending(self) -> None:
        emb, store = self._build()
        chunks = [Chunk(text=f"c{i}", source="s", start=0, end=2) for i in range(3)]
        vecs = np.array(
            [[1, 0, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 1, 0]], dtype=np.float32
        )
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.add(chunks, vecs)
        emb.register("q", [1.0, 0.0, 0.0, 0.0])

        retriever = DenseRetriever(emb, store)
        results = retriever.retrieve("q", k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_empty_query_rejected(self) -> None:
        emb, store = self._build()
        emb.register("x", [1, 0, 0, 0])
        store.add(
            [Chunk(text="x", source="s", start=0, end=1)],
            np.array([[1, 0, 0, 0]], dtype=np.float32),
        )
        retriever = DenseRetriever(emb, store)
        with pytest.raises(ValueError, match="query"):
            retriever.retrieve("", k=1)

    def test_retrieve_zero_k_rejected(self) -> None:
        emb, store = self._build()
        emb.register("x", [1, 0, 0, 0])
        store.add(
            [Chunk(text="x", source="s", start=0, end=1)],
            np.array([[1, 0, 0, 0]], dtype=np.float32),
        )
        retriever = DenseRetriever(emb, store)
        with pytest.raises(ValueError, match="k"):
            retriever.retrieve("x", k=0)


# =============================================================================
# RetrievedChunk dataclass — boilerplate
# =============================================================================


class TestRetrievedChunkBoilerplate:
    def test_construct(self) -> None:
        c = Chunk(text="x", source="s", start=0, end=1)
        rc = RetrievedChunk(chunk=c, score=0.9, rank=1)
        assert rc.chunk is c
        assert rc.score == 0.9
        assert rc.rank == 1

    def test_frozen(self) -> None:
        c = Chunk(text="x", source="s", start=0, end=1)
        rc = RetrievedChunk(chunk=c, score=0.5, rank=1)
        with pytest.raises(Exception):
            rc.score = 0.7  # type: ignore[misc]


# =============================================================================
# assemble_rag_prompt — scaffolded
# =============================================================================


class TestAssembleRAGPromptBoilerplate:
    def test_rag_prompt_dataclass_construct(self) -> None:
        c = Chunk(text="x", source="s", start=0, end=1)
        p = RAGPrompt(text="hello", question="q", chunks=(c,))
        assert p.text == "hello"
        assert p.question == "q"
        assert p.chunks == (c,)

    def test_rag_prompt_frozen(self) -> None:
        p = RAGPrompt(text="t", question="q", chunks=())
        with pytest.raises(Exception):
            p.text = "new"  # type: ignore[misc]

    def test_default_constants_are_strings(self) -> None:
        assert isinstance(DEFAULT_SYSTEM, str)
        assert isinstance(DEFAULT_INSTRUCTION, str)
        assert len(DEFAULT_SYSTEM) > 0
        assert len(DEFAULT_INSTRUCTION) > 0


class TestAssembleRAGPrompt:
    def _chunks(self, n: int = 2) -> list[Chunk]:
        out: list[Chunk] = []
        for i in range(n):
            text = f"This is chunk number {i}."
            out.append(Chunk(text=text, source=f"doc-{i}.md", start=0, end=len(text)))
        return out

    def test_returns_rag_prompt(self) -> None:
        result = assemble_rag_prompt("Q?", self._chunks(2))
        assert isinstance(result, RAGPrompt)

    def test_text_includes_question(self) -> None:
        result = assemble_rag_prompt("What is the capital of Spain?", self._chunks(1))
        assert "What is the capital of Spain?" in result.text

    def test_assemble_rag_prompt_uses_one_based_citations(self) -> None:
        result = assemble_rag_prompt("Q?", self._chunks(3))
        # Should contain "[1]", "[2]", "[3]" — not "[0]".
        assert "[1]" in result.text
        assert "[2]" in result.text
        assert "[3]" in result.text
        assert "[0]" not in result.text

    def test_assemble_rag_prompt_includes_source_label(self) -> None:
        chunks = [Chunk(text="Hello.", source="docs/foo.md", start=0, end=6)]
        result = assemble_rag_prompt("Q?", chunks)
        assert "docs/foo.md" in result.text

    def test_chunk_text_appears_in_prompt(self) -> None:
        chunks = [
            Chunk(
                text="Madrid is the capital of Spain.",
                source="es.md",
                start=0,
                end=31,
            )
        ]
        result = assemble_rag_prompt("Q?", chunks)
        assert "Madrid is the capital of Spain." in result.text

    def test_chunks_attached_to_returned_prompt(self) -> None:
        chunks = self._chunks(2)
        result = assemble_rag_prompt("Q?", chunks)
        assert tuple(result.chunks) == tuple(chunks)

    def test_uses_default_system(self) -> None:
        result = assemble_rag_prompt("Q?", self._chunks(1))
        assert DEFAULT_SYSTEM in result.text

    def test_uses_default_instruction(self) -> None:
        result = assemble_rag_prompt("Q?", self._chunks(1))
        assert DEFAULT_INSTRUCTION in result.text

    def test_custom_system(self) -> None:
        result = assemble_rag_prompt(
            "Q?", self._chunks(1), system="CUSTOM SYSTEM HEADER"
        )
        assert "CUSTOM SYSTEM HEADER" in result.text
        assert DEFAULT_SYSTEM not in result.text

    def test_custom_instruction(self) -> None:
        result = assemble_rag_prompt(
            "Q?", self._chunks(1), instruction="CUSTOM INSTR FOOTER"
        )
        assert "CUSTOM INSTR FOOTER" in result.text
        assert DEFAULT_INSTRUCTION not in result.text

    def test_empty_system_omits_leading_paragraph(self) -> None:
        result = assemble_rag_prompt(
            "Q?", self._chunks(1), system="", instruction=""
        )
        # Should not have the default system text.
        assert DEFAULT_SYSTEM not in result.text
        # Should not start with a blank line.
        assert not result.text.startswith("\n")

    def test_accepts_retrieved_chunks(self) -> None:
        chunks = self._chunks(2)
        retrieved = [RetrievedChunk(chunk=c, score=0.9, rank=i + 1) for i, c in enumerate(chunks)]
        result = assemble_rag_prompt("Q?", retrieved)
        # Should still cite the underlying chunks.
        assert "[1]" in result.text
        assert "[2]" in result.text

    def test_question_round_trips(self) -> None:
        result = assemble_rag_prompt("My question.", self._chunks(1))
        assert result.question == "My question."

    def test_citations_in_correct_order(self) -> None:
        chunks = self._chunks(2)
        result = assemble_rag_prompt("Q?", chunks)
        # [1] should appear before [2] in the text.
        idx_1 = result.text.find("[1]")
        idx_2 = result.text.find("[2]")
        assert idx_1 != -1 and idx_2 != -1
        assert idx_1 < idx_2


class TestAssembleRAGPromptValidation:
    def test_rejects_non_str_question(self) -> None:
        with pytest.raises(TypeError, match="question"):
            assemble_rag_prompt(123, [])  # type: ignore[arg-type]

    def test_rejects_empty_question(self) -> None:
        with pytest.raises(ValueError, match="question"):
            assemble_rag_prompt("", [])

    def test_rejects_invalid_chunk_type(self) -> None:
        with pytest.raises(TypeError, match="(Chunk|RetrievedChunk)"):
            assemble_rag_prompt("Q?", ["not a chunk"])  # type: ignore[list-item]


class TestAssembleRAGPromptEdgeCases:
    def test_empty_chunk_list_still_produces_prompt(self) -> None:
        result = assemble_rag_prompt("Q?", [])
        assert "Q?" in result.text
        # Question is in there even with no context.
        assert isinstance(result, RAGPrompt)
        assert result.chunks == ()


# =============================================================================
# RAGAnswer + RAGPipeline — implemented (composition tests)
# =============================================================================


class TestRAGAnswerBoilerplate:
    def test_construct(self) -> None:
        c = Chunk(text="x", source="s", start=0, end=1)
        rc = RetrievedChunk(chunk=c, score=0.5, rank=1)
        info = BackendInfo(name="x", model_id="m")
        ir = InferenceResult(
            prompt="p",
            completion="a",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
            backend=info,
        )
        prompt = RAGPrompt(text="t", question="q", chunks=(c,))
        ans = RAGAnswer(
            question="q", answer="a", retrieved=[rc], prompt=prompt, inference=ir
        )
        assert ans.question == "q"
        assert ans.answer == "a"
        assert ans.metadata == {}


class TestRAGPipelineBoilerplate:
    def test_construction(self) -> None:
        emb = HashEmbedder(dim=32)
        store = NumpyVectorStore(dim=32)
        retriever = DenseRetriever(emb, store)
        backend = _FakeBackend()
        pipeline = RAGPipeline(retriever, backend)
        assert pipeline.retriever is retriever
        assert pipeline.backend is backend

    def test_none_retriever_rejected(self) -> None:
        with pytest.raises(ValueError, match="retriever"):
            RAGPipeline(None, _FakeBackend())  # type: ignore[arg-type]

    def test_none_backend_rejected(self) -> None:
        emb = HashEmbedder(dim=32)
        store = NumpyVectorStore(dim=32)
        with pytest.raises(ValueError, match="backend"):
            RAGPipeline(DenseRetriever(emb, store), None)  # type: ignore[arg-type]


class TestRAGPipelineAnswer:
    """Integration-style tests: pipeline composes retriever + backend."""

    def _build(self) -> tuple[RAGPipeline, _FakeBackend]:
        emb = _FakeEmbedder(dim=4)
        store = NumpyVectorStore(dim=4)
        chunks = [
            Chunk(
                text=f"This is fact {i}.",
                source=f"doc-{i}.md",
                start=0,
                end=15,
            )
            for i in range(3)
        ]
        vecs = np.eye(3, 4, dtype=np.float32)  # one-hot
        store.add(chunks, vecs)
        emb.register("q", [1.0, 0.0, 0.0, 0.0])
        retriever = DenseRetriever(emb, store)
        backend = _FakeBackend(scripted_answer="The fact is X. [1]")
        return RAGPipeline(retriever, backend), backend

    def test_returns_rag_answer(self) -> None:
        pipeline, _ = self._build()
        ans = pipeline.answer("q", k=1)
        assert isinstance(ans, RAGAnswer)
        assert ans.question == "q"
        assert ans.answer == "The fact is X. [1]"

    def test_retrieved_attached(self) -> None:
        pipeline, _ = self._build()
        ans = pipeline.answer("q", k=2)
        assert len(ans.retrieved) == 2
        assert ans.retrieved[0].rank == 1

    def test_pipeline_propagates_retrieved_chunks_to_prompt(self) -> None:
        pipeline, backend = self._build()
        ans = pipeline.answer("q", k=2)
        # The chunks attached to the RAGPrompt must match the retrieved chunks.
        prompt_chunks = list(ans.prompt.chunks)
        retrieved_chunks = [r.chunk for r in ans.retrieved]
        assert prompt_chunks == retrieved_chunks
        # And the chunks must appear in the text the backend was called with.
        assert backend.last_prompt is not None
        for c in prompt_chunks:
            assert c.text in backend.last_prompt

    def test_pipeline_metadata_records_k(self) -> None:
        pipeline, _ = self._build()
        ans = pipeline.answer("q", k=2)
        assert ans.metadata["k"] == 2
        assert ans.metadata["n_retrieved"] == 2
        assert "backend_name" in ans.metadata

    def test_pipeline_forwards_sampling_args(self) -> None:
        pipeline, backend = self._build()
        pipeline.answer(
            "q", k=1, max_new_tokens=42, temperature=0.5, top_k=10, top_p=0.9
        )
        assert backend.last_kwargs == {
            "max_new_tokens": 42,
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.9,
        }

    def test_empty_question_rejected(self) -> None:
        pipeline, _ = self._build()
        with pytest.raises(ValueError, match="question"):
            pipeline.answer("", k=1)

    def test_zero_k_rejected(self) -> None:
        pipeline, _ = self._build()
        with pytest.raises(ValueError, match="k"):
            pipeline.answer("q", k=0)


# =============================================================================
# Integration smoke — chunker + HashEmbedder + store + retriever + pipeline
# =============================================================================


class TestIntegrationSmoke:
    """End-to-end test using only HashEmbedder (no external deps).

    Verifies the whole pipeline composes once the four scaffolded
    methods (chunk_text, HashEmbedder.embed, NumpyVectorStore.search,
    assemble_rag_prompt) are in.
    """

    def test_full_pipeline_with_hash_embedder(self) -> None:
        # A tiny document with two distinguishable topics.
        text = (
            "Madrid is the capital of Spain. The country has a population of "
            "about forty seven million people. Madrid is also the largest "
            "city in Spain. It is located in the center of the country.\n\n"
            "Bananas are a yellow fruit. They grow in tropical climates and "
            "are rich in potassium. Bananas turn brown when overripe."
        )
        chunks = chunk_text(text, source="facts.md", chunk_size=120, chunk_overlap=20)
        assert len(chunks) >= 2

        emb = HashEmbedder(dim=512, ngram_range=(3, 5), seed=0)
        vectors = emb.embed([c.text for c in chunks])

        store = NumpyVectorStore(dim=512)
        store.add(chunks, vectors)

        retriever = DenseRetriever(emb, store)
        backend = _FakeBackend(scripted_answer="grounded answer")
        pipeline = RAGPipeline(retriever, backend)

        ans = pipeline.answer(
            "What is the capital of Spain?", k=1, max_new_tokens=16
        )
        assert isinstance(ans, RAGAnswer)
        # The retrieved chunk must mention Madrid (the capital topic) — the
        # banana chunk shares almost no n-grams with the Spain question.
        assert "Madrid" in ans.retrieved[0].chunk.text

    def test_full_pipeline_returns_normalized_vectors(self) -> None:
        text = "Hello world. This is a sentence."
        chunks = chunk_text(text, source="t", chunk_size=200)
        emb = HashEmbedder(dim=128)
        vectors = emb.embed([c.text for c in chunks])
        # Every row L2-norm ≈ 1.
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


# =============================================================================
# Module exports
# =============================================================================


class TestModuleExports:
    def test_public_api_imports(self) -> None:
        # Re-importing here makes the test independent of test order.
        from g2c import rag

        for name in [
            "Chunk",
            "DEFAULT_INSTRUCTION",
            "DEFAULT_OLLAMA_EMBED_MODEL",
            "DEFAULT_OLLAMA_URL",
            "DEFAULT_SYSTEM",
            "DenseRetriever",
            "Embedder",
            "HashEmbedder",
            "NumpyVectorStore",
            "OllamaEmbedError",
            "OllamaEmbedder",
            "RAGAnswer",
            "RAGPipeline",
            "RAGPrompt",
            "RetrievedChunk",
            "assemble_rag_prompt",
            "chunk_text",
            "cosine_similarity",
        ]:
            assert hasattr(rag, name), f"g2c.rag is missing {name}"

    def test_chunk_dataclass_fields(self) -> None:
        assert "text" in Chunk.__dataclass_fields__
        assert "source" in Chunk.__dataclass_fields__
        assert "start" in Chunk.__dataclass_fields__
        assert "end" in Chunk.__dataclass_fields__

    def test_rag_answer_dataclass_fields(self) -> None:
        assert "question" in RAGAnswer.__dataclass_fields__
        assert "answer" in RAGAnswer.__dataclass_fields__
        assert "retrieved" in RAGAnswer.__dataclass_fields__
