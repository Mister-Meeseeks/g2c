"""Tests for `g2c.inference` — Module 16.

Suggested order to implement & turn green:

  1. **`LocalTransformerBackend.complete`** in `g2c/inference/local.py`.
     Encode prompt → time → call `generate` → slice off prompt tokens
     → decode tail → build `InferenceResult`. Tests use a monkeypatched
     `generate` that returns a deterministic id sequence, so the local
     backend can be exercised even before Module 11's `generate` is
     filled in. Turns green:
       * `TestLocalTransformerBackendComplete`
       * `TestLocalTransformerBackendValidation`

  2. **`OllamaBackend.complete`** in `g2c/inference/ollama.py`.
     Build JSON body → POST → parse JSON response → wrap stdlib errors
     in `OllamaError`. Tests inject a fake `urlopen`, so no real Ollama
     server is needed. Turns green:
       * `TestOllamaBackendComplete`
       * `TestOllamaBackendOptionsHandling`
       * `TestOllamaBackendErrorPaths`
       * `TestOllamaBackendValidation`

  3. **`benchmark`** in `g2c/inference/benchmark.py`. Iterate `complete`
     over prompts, aggregate latencies and token counts, percentile +
     mean. Tests use a `_FakeBackend` that returns scripted results.
     Turns green:
       * `TestBenchmark`
       * `TestBenchmarkEdgeCases`

The boilerplate tests pass from the start as a sanity check on the
test file itself: dataclass construction, validation, the
`tokens_per_second` derived property, and the `BackendInfo` /
`BenchmarkResult` containers.

The end-to-end transformer test (`test_local_backend_real_transformer_smoke`)
pulls in `g2c.transformer.TransformerLM` AND `g2c.sampling.generate`
— if either of those modules is unfilled, this single test fails on
a prerequisite. Same convention as the eval / DPO / SFT
end-to-end tests.
"""
from __future__ import annotations

import json
import urllib.error
from dataclasses import FrozenInstanceError

import pytest
import torch

from g2c.inference import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_PRODLM_MODEL_ID,
    PRODLM_KIND,
    PRODLM_NAME,
    ArtifactBackend,
    Backend,
    BackendInfo,
    BenchmarkResult,
    InferenceResult,
    LocalTransformerBackend,
    OllamaBackend,
    OllamaError,
    benchmark,
    load_default_backend,
    load_prodlm_backend,
    prodlm_manifest_exists,
    write_prodlm_manifest,
)
from g2c.artifacts import LoadedModelArtifact

# -----------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------


class _CharTokenizer:
    """Encode = ord(c) for each c; decode = ''.join(chr(i) for i in ids).

    Mirrors the fixture used in `test_eval.py`. Vocab size is implicit
    (we just use whatever ord values appear); the model's vocab_size
    must be at least max(ord) for forward to make sense.
    """

    def encode(self, s: str) -> list[int]:
        return [ord(c) for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class _VocabAwareTokenizer(_CharTokenizer):
    special_to_id = {"<|end|>": 4}

    def __init__(self) -> None:
        self.encode_with_vocab_size_calls: list[tuple[str, int]] = []

    def encode_with_vocab_size(self, s: str, vocab_size: int) -> list[int]:
        self.encode_with_vocab_size_calls.append((s, vocab_size))
        return self.encode(s)


class _TinyModel(torch.nn.Module):
    """A model with `parameters()` yielding a real device-bearing
    tensor and a `max_seq_len` attribute.

    The `forward` method is unused by the LocalTransformerBackend
    tests (they monkeypatch `generate`), but it has to exist to
    satisfy `nn.Module` machinery. Returns zeros — never actually
    invoked in these tests.
    """

    def __init__(self, vocab_size: int = 256, max_seq_len: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        b, t = x.shape
        return torch.zeros(b, t, self.vocab_size)


class _FakeBackend(Backend):
    """A `Backend` that returns scripted results without real work.

    Used by the benchmark tests. The constructor takes a list of
    `(completion, latency_ms, prompt_tokens, completion_tokens)`
    tuples; each call to `complete` consumes the next one. Calls
    beyond the script's length raise.
    """

    def __init__(
        self,
        scripted: list[tuple[str, float, int | None, int | None]],
        *,
        info: BackendInfo | None = None,
    ) -> None:
        self._scripted = list(scripted)
        self._idx = 0
        self._info = info or BackendInfo(name="fake", model_id="fake-1")
        self.calls: list[dict] = []

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
        self.calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
        )
        if self._idx >= len(self._scripted):
            raise RuntimeError("FakeBackend script exhausted")
        completion, latency_ms, p_tok, c_tok = self._scripted[self._idx]
        self._idx += 1
        return InferenceResult(
            prompt=prompt,
            completion=completion,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            latency_ms=latency_ms,
            backend=self._info,
        )


class _FakeResponse:
    """Stdlib-style HTTP response: read() returns body bytes; supports
    `with ... as resp:` context-manager protocol."""

    def __init__(self, body_bytes: bytes) -> None:
        self._body = body_bytes

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args) -> bool:
        return False


def _make_fake_urlopen(
    *,
    response: dict | None = None,
    raise_exc: Exception | None = None,
    bad_json: bytes | None = None,
    captured: list | None = None,
):
    """Construct a fake `urlopen` callable for OllamaBackend tests.

    If `raise_exc` is set, every call raises it. Otherwise returns a
    `_FakeResponse` wrapping `json.dumps(response).encode('utf-8')`,
    or raw `bad_json` bytes if provided.

    `captured` is an optional list that the fake appends to every call
    with `(url, body_bytes, headers_dict, timeout)`. Lets the test
    assert what was sent.
    """

    def _urlopen(req, timeout=None):
        if captured is not None:
            captured.append(
                {
                    "url": req.full_url,
                    "body": req.data,
                    "headers": dict(req.headers),
                    "timeout": timeout,
                    "method": req.get_method(),
                }
            )
        if raise_exc is not None:
            raise raise_exc
        if bad_json is not None:
            return _FakeResponse(bad_json)
        return _FakeResponse(json.dumps(response or {}).encode("utf-8"))

    return _urlopen


# -----------------------------------------------------------------------
# Boilerplate: BackendInfo
# -----------------------------------------------------------------------


class TestBackendInfoBoilerplate:
    def test_construction_minimal(self) -> None:
        info = BackendInfo(name="local", model_id="g2c-tiny")
        assert info.name == "local"
        assert info.model_id == "g2c-tiny"
        assert info.extra == {}

    def test_construction_with_extra(self) -> None:
        info = BackendInfo(
            name="ollama",
            model_id="llama3.2:3b",
            extra={"base_url": "http://localhost:11434", "quant": "Q4_K_M"},
        )
        assert info.extra["base_url"] == "http://localhost:11434"
        assert info.extra["quant"] == "Q4_K_M"

    def test_default_extra_is_independent_dict(self) -> None:
        a = BackendInfo(name="a", model_id="x")
        b = BackendInfo(name="b", model_id="y")
        a.extra["key"] = "value"
        assert "key" not in b.extra

    def test_frozen_blocks_field_reassignment(self) -> None:
        info = BackendInfo(name="local", model_id="x")
        with pytest.raises(FrozenInstanceError):
            info.name = "rebranded"  # type: ignore[misc]

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            BackendInfo(name="", model_id="x")

    def test_empty_model_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            BackendInfo(name="local", model_id="")

    def test_non_str_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="name"):
            BackendInfo(name=42, model_id="x")  # type: ignore[arg-type]

    def test_equality_by_value(self) -> None:
        a = BackendInfo(name="local", model_id="x")
        b = BackendInfo(name="local", model_id="x")
        assert a == b

    def test_equality_includes_extra(self) -> None:
        a = BackendInfo(name="local", model_id="x", extra={"k": "v"})
        b = BackendInfo(name="local", model_id="x", extra={"k": "w"})
        assert a != b


# -----------------------------------------------------------------------
# Boilerplate: InferenceResult
# -----------------------------------------------------------------------


class TestInferenceResultBoilerplate:
    def _info(self) -> BackendInfo:
        return BackendInfo(name="local", model_id="x")

    def test_construction_full(self) -> None:
        r = InferenceResult(
            prompt="hi",
            completion=" world",
            prompt_tokens=2,
            completion_tokens=6,
            latency_ms=42.0,
            backend=self._info(),
        )
        assert r.prompt == "hi"
        assert r.completion == " world"
        assert r.prompt_tokens == 2
        assert r.completion_tokens == 6
        assert r.latency_ms == 42.0
        assert r.metadata == {}

    def test_metadata_default_independent(self) -> None:
        a = InferenceResult(
            prompt="a", completion="b", prompt_tokens=1,
            completion_tokens=1, latency_ms=1.0, backend=self._info(),
        )
        b = InferenceResult(
            prompt="a", completion="b", prompt_tokens=1,
            completion_tokens=1, latency_ms=1.0, backend=self._info(),
        )
        a.metadata["k"] = "v"
        assert "k" not in b.metadata

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValueError, match="latency_ms"):
            InferenceResult(
                prompt="a", completion="b", prompt_tokens=1,
                completion_tokens=1, latency_ms=-0.001, backend=self._info(),
            )

    def test_negative_prompt_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="prompt_tokens"):
            InferenceResult(
                prompt="a", completion="b", prompt_tokens=-1,
                completion_tokens=1, latency_ms=1.0, backend=self._info(),
            )

    def test_negative_completion_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="completion_tokens"):
            InferenceResult(
                prompt="a", completion="b", prompt_tokens=1,
                completion_tokens=-1, latency_ms=1.0, backend=self._info(),
            )

    def test_none_token_counts_allowed(self) -> None:
        r = InferenceResult(
            prompt="a", completion="b", prompt_tokens=None,
            completion_tokens=None, latency_ms=1.0, backend=self._info(),
        )
        assert r.prompt_tokens is None
        assert r.completion_tokens is None


class TestInferenceResultTokensPerSecond:
    def _info(self) -> BackendInfo:
        return BackendInfo(name="local", model_id="x")

    def test_computed_correctly(self) -> None:
        # 100 tokens in 500 ms -> 200 tok/s
        r = InferenceResult(
            prompt="p", completion="c", prompt_tokens=10,
            completion_tokens=100, latency_ms=500.0, backend=self._info(),
        )
        assert r.tokens_per_second == pytest.approx(200.0)

    def test_returns_none_when_completion_tokens_none(self) -> None:
        r = InferenceResult(
            prompt="p", completion="c", prompt_tokens=10,
            completion_tokens=None, latency_ms=500.0, backend=self._info(),
        )
        assert r.tokens_per_second is None

    def test_returns_none_when_completion_tokens_zero(self) -> None:
        r = InferenceResult(
            prompt="p", completion="", prompt_tokens=10,
            completion_tokens=0, latency_ms=500.0, backend=self._info(),
        )
        assert r.tokens_per_second is None

    def test_returns_none_when_latency_zero(self) -> None:
        r = InferenceResult(
            prompt="p", completion="c", prompt_tokens=10,
            completion_tokens=100, latency_ms=0.0, backend=self._info(),
        )
        assert r.tokens_per_second is None


# -----------------------------------------------------------------------
# Boilerplate: Backend ABC
# -----------------------------------------------------------------------


class TestBackendABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Backend()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        backend = _FakeBackend([("hi", 1.0, 1, 1)])
        assert isinstance(backend, Backend)
        assert backend.info.name == "fake"


# -----------------------------------------------------------------------
# Boilerplate: ProdLM helpers
# -----------------------------------------------------------------------


class TestProdLMHelpers:
    def test_constants(self) -> None:
        assert PRODLM_NAME == "ProdLM"
        assert PRODLM_KIND == "ollama_backend"
        assert DEFAULT_PRODLM_MODEL_ID == "llama3.2:3b"

    def test_write_manifest(self, tmp_path) -> None:
        root = write_prodlm_manifest(
            model_id="qwen2.5:7b-instruct",
            base_url="http://localhost:11434",
            timeout=45.0,
            repo_root=tmp_path,
            notes="test manifest",
        )

        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        assert config["kind"] == PRODLM_KIND
        assert config["backend"] == "ollama"
        assert config["model_id"] == "qwen2.5:7b-instruct"
        assert config["timeout"] == pytest.approx(45.0)
        assert manifest["name"] == PRODLM_NAME
        assert manifest["role"] == "ProdLM"
        assert prodlm_manifest_exists(repo_root=tmp_path)

    def test_load_prodlm_backend_reads_manifest(self, tmp_path) -> None:
        write_prodlm_manifest(
            model_id="mistral:7b-instruct",
            base_url="http://test-host:11434",
            timeout=60.0,
            repo_root=tmp_path,
        )

        backend = load_prodlm_backend(repo_root=tmp_path)
        assert isinstance(backend, OllamaBackend)
        assert backend.info.name == "prodlm"
        assert backend.info.model_id == "mistral:7b-instruct"
        assert backend.base_url == "http://test-host:11434"
        assert backend.info.extra["configured_name"] == PRODLM_NAME

    def test_load_default_backend_prefers_prodlm_manifest(self, tmp_path) -> None:
        write_prodlm_manifest(model_id="llama3.2:1b", repo_root=tmp_path)
        backend = load_default_backend(repo_root=tmp_path)
        assert isinstance(backend, OllamaBackend)
        assert backend.info.name == "prodlm"
        assert backend.info.model_id == "llama3.2:1b"

    def test_load_prodlm_backend_required_without_manifest(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_prodlm_backend(repo_root=tmp_path, required=True)


# -----------------------------------------------------------------------
# Boilerplate: ArtifactBackend
# -----------------------------------------------------------------------


class TestArtifactBackend:
    def _artifact(self, tmp_path, tokenizer) -> LoadedModelArtifact:
        return LoadedModelArtifact(
            name="TinyLLM-DPO",
            canonical_name="TinyLLM-30M",
            display_name="TinyLLM 30M DPO",
            rank=40,
            artifact_dir=tmp_path,
            model=_TinyModel(vocab_size=256),
            tokenizer=tokenizer,
            tokenizer_artifact=None,
            manifest={"source": "test artifact", "kind": "course_transformer"},
            training_config={},
        )

    def test_wraps_artifact_metadata_and_eos(self, tmp_path, monkeypatch) -> None:
        tokenizer = _VocabAwareTokenizer()
        artifact = self._artifact(tmp_path, tokenizer)
        backend = ArtifactBackend(artifact)

        def _fake_generate(model, prompt_ids, **kwargs):
            assert kwargs["eos_id"] == 4
            return torch.tensor([ord("h"), ord("i"), ord("!")], dtype=torch.long)

        monkeypatch.setattr("g2c.inference.local.generate", _fake_generate)
        result = backend.complete("hi", max_new_tokens=1)

        assert result.completion == "!"
        assert result.backend.name == "artifact"
        assert result.backend.model_id == "TinyLLM-DPO"
        assert result.backend.extra["canonical_name"] == "TinyLLM-30M"
        assert result.backend.extra["source"] == "test artifact"
        assert result.metadata["eos_id"] == 4
        assert tokenizer.encode_with_vocab_size_calls == [("hi", 256)]
        assert backend.artifact is artifact
        assert backend.raw_tokenizer is tokenizer


# -----------------------------------------------------------------------
# Boilerplate: LocalTransformerBackend (constructor + info)
# -----------------------------------------------------------------------


class TestLocalTransformerBackendBoilerplate:
    def test_construction_defaults(self) -> None:
        model = _TinyModel()
        tok = _CharTokenizer()
        backend = LocalTransformerBackend(model, tok)
        assert backend.info.name == "local"
        assert backend.info.model_id == "g2c-local"
        assert backend.info.extra == {}
        assert backend.model is model
        assert backend.tokenizer is tok

    def test_construction_with_overrides(self) -> None:
        model = _TinyModel()
        tok = _CharTokenizer()
        backend = LocalTransformerBackend(
            model, tok,
            model_id="my-checkpoint-20m",
            eos_id=255,
            extra={"dtype": "fp32", "device": "mps"},
        )
        assert backend.info.model_id == "my-checkpoint-20m"
        assert backend.info.extra["dtype"] == "fp32"
        assert backend.info.extra["device"] == "mps"

    def test_none_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="model"):
            LocalTransformerBackend(None, _CharTokenizer())

    def test_none_tokenizer_rejected(self) -> None:
        with pytest.raises(ValueError, match="tokenizer"):
            LocalTransformerBackend(_TinyModel(), None)

    def test_tokenizer_without_encode_rejected(self) -> None:
        class _Bad:
            def decode(self, ids):
                return ""

        with pytest.raises(TypeError, match="encode"):
            LocalTransformerBackend(_TinyModel(), _Bad())

    def test_non_int_eos_id_rejected(self) -> None:
        with pytest.raises(TypeError, match="eos_id"):
            LocalTransformerBackend(
                _TinyModel(), _CharTokenizer(), eos_id="end"  # type: ignore[arg-type]
            )

    def test_repr(self) -> None:
        backend = LocalTransformerBackend(
            _TinyModel(), _CharTokenizer(), model_id="abc"
        )
        s = repr(backend)
        assert "LocalTransformerBackend" in s
        assert "abc" in s


# -----------------------------------------------------------------------
# LocalTransformerBackend.complete (SCAFFOLDED)
# -----------------------------------------------------------------------


class TestLocalTransformerBackendComplete:
    def _make(self, *, eos_id: int | None = None) -> LocalTransformerBackend:
        return LocalTransformerBackend(
            _TinyModel(), _CharTokenizer(), eos_id=eos_id, model_id="t"
        )

    def _patch_generate(self, monkeypatch, scripted_full: list[int]):
        """Replace `g2c.inference.local.generate` with a fake that
        ignores its inputs and returns `scripted_full` as a 1-D
        LongTensor."""
        called = {}

        def _fake_generate(model, prompt_ids, **kwargs):
            called["model"] = model
            called["prompt_ids"] = prompt_ids
            called["kwargs"] = kwargs
            return torch.tensor(scripted_full, dtype=torch.long)

        monkeypatch.setattr("g2c.inference.local.generate", _fake_generate)
        return called

    def test_returns_inference_result(self, monkeypatch) -> None:
        # prompt encodes to [104, 105] (h,i); generate appends [33] (!)
        full = [104, 105, 33]
        self._patch_generate(monkeypatch, full)
        backend = self._make()
        result = backend.complete("hi", max_new_tokens=1)
        assert isinstance(result, InferenceResult)

    def test_completion_excludes_prompt(self, monkeypatch) -> None:
        # prompt "ab" -> [97, 98]; generate returns [97, 98, 99, 100, 101]
        # So completion should be [99, 100, 101] -> "cde"
        self._patch_generate(monkeypatch, [97, 98, 99, 100, 101])
        backend = self._make()
        result = backend.complete("ab", max_new_tokens=3)
        assert result.completion == "cde"
        assert result.prompt == "ab"

    def test_token_counts(self, monkeypatch) -> None:
        # prompt "hello" -> 5 tokens; full has 8 -> 3 new tokens
        self._patch_generate(
            monkeypatch, [ord("h"), ord("e"), ord("l"), ord("l"), ord("o"), 33, 34, 35]
        )
        backend = self._make()
        result = backend.complete("hello", max_new_tokens=3)
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 3

    def test_latency_is_positive(self, monkeypatch) -> None:
        self._patch_generate(monkeypatch, [104, 105, 33])
        backend = self._make()
        result = backend.complete("hi", max_new_tokens=1)
        assert result.latency_ms >= 0.0
        # perf_counter resolution on macOS is sub-microsecond, so a real
        # elapsed call should register > 0. But a too-strict assertion
        # here is brittle on fast machines — accept >= 0 as the
        # contract.

    def test_backend_field_carries_info(self, monkeypatch) -> None:
        self._patch_generate(monkeypatch, [104, 105, 33])
        backend = LocalTransformerBackend(
            _TinyModel(), _CharTokenizer(), model_id="my-tag"
        )
        result = backend.complete("hi", max_new_tokens=1)
        assert result.backend.model_id == "my-tag"
        assert result.backend.name == "local"

    def test_metadata_contains_sampling(self, monkeypatch) -> None:
        self._patch_generate(monkeypatch, [104, 105, 33])
        backend = self._make()
        result = backend.complete(
            "hi", max_new_tokens=4, temperature=0.7, top_k=40, top_p=0.95
        )
        s = result.metadata["sampling"]
        assert s["max_new_tokens"] == 4
        assert s["temperature"] == 0.7
        assert s["top_k"] == 40
        assert s["top_p"] == 0.95

    def test_metadata_records_eos_id(self, monkeypatch) -> None:
        self._patch_generate(monkeypatch, [104, 105, 33])
        backend = self._make(eos_id=255)
        result = backend.complete("hi", max_new_tokens=4)
        assert result.metadata["eos_id"] == 255

    def test_generate_receives_sampling_args(self, monkeypatch) -> None:
        called = self._patch_generate(monkeypatch, [104, 105, 33])
        backend = self._make(eos_id=42)
        backend.complete(
            "hi", max_new_tokens=8, temperature=0.5, top_k=20, top_p=0.9
        )
        kwargs = called["kwargs"]
        assert kwargs["max_new_tokens"] == 8
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_k"] == 20
        assert kwargs["top_p"] == 0.9
        assert kwargs["eos_id"] == 42

    def test_prompt_tensor_is_one_dim(self, monkeypatch) -> None:
        called = self._patch_generate(monkeypatch, [104, 105, 33])
        backend = self._make()
        backend.complete("hi", max_new_tokens=1)
        assert called["prompt_ids"].dim() == 1
        assert called["prompt_ids"].dtype == torch.long


class TestLocalTransformerBackendValidation:
    def _make(self) -> LocalTransformerBackend:
        return LocalTransformerBackend(
            _TinyModel(), _CharTokenizer(), model_id="t"
        )

    def test_empty_prompt_rejected(self, monkeypatch) -> None:
        # `_CharTokenizer.encode("")` returns [], which is invalid.
        # Implementation should raise before reaching generate.
        monkeypatch.setattr(
            "g2c.inference.local.generate",
            lambda *a, **k: pytest.fail(
                "generate should not be called for empty prompt"
            ),
        )
        backend = self._make()
        with pytest.raises(ValueError):
            backend.complete("", max_new_tokens=1)

    def test_zero_max_new_tokens_rejected(self) -> None:
        backend = self._make()
        with pytest.raises(ValueError):
            backend.complete("hi", max_new_tokens=0)

    def test_negative_max_new_tokens_rejected(self) -> None:
        backend = self._make()
        with pytest.raises(ValueError):
            backend.complete("hi", max_new_tokens=-5)

    def test_non_str_prompt_rejected(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "g2c.inference.local.generate",
            lambda *a, **k: pytest.fail("generate should not be called"),
        )
        backend = self._make()
        with pytest.raises(TypeError):
            backend.complete(42, max_new_tokens=1)  # type: ignore[arg-type]


# -----------------------------------------------------------------------
# OllamaBackend boilerplate
# -----------------------------------------------------------------------


class TestOllamaBackendBoilerplate:
    def test_construction_defaults(self) -> None:
        backend = OllamaBackend()
        assert backend.info.name == "ollama"
        assert backend.info.model_id == "llama3.2:3b"
        assert backend.base_url == DEFAULT_OLLAMA_URL
        assert backend.info.extra["base_url"] == DEFAULT_OLLAMA_URL

    def test_construction_explicit(self) -> None:
        backend = OllamaBackend(
            "qwen2.5:7b", base_url="http://example.com:8080/", timeout=30.0
        )
        # Trailing slash should be stripped
        assert backend.base_url == "http://example.com:8080"
        assert backend.info.model_id == "qwen2.5:7b"

    def test_extra_merged(self) -> None:
        backend = OllamaBackend(
            "x:1", extra={"quant": "Q4_K_M"}
        )
        # base_url is auto-injected; quant comes from caller
        assert backend.info.extra["base_url"] == DEFAULT_OLLAMA_URL
        assert backend.info.extra["quant"] == "Q4_K_M"

    def test_empty_model_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            OllamaBackend("")

    def test_empty_base_url_rejected(self) -> None:
        with pytest.raises(ValueError):
            OllamaBackend("x:1", base_url="")

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError):
            OllamaBackend("x:1", timeout=0.0)

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValueError):
            OllamaBackend("x:1", timeout=-1.0)

    def test_repr(self) -> None:
        backend = OllamaBackend("foo:bar")
        s = repr(backend)
        assert "OllamaBackend" in s
        assert "foo:bar" in s

    def test_default_constant_value(self) -> None:
        assert DEFAULT_OLLAMA_URL == "http://localhost:11434"


# -----------------------------------------------------------------------
# OllamaBackend.complete (SCAFFOLDED)
# -----------------------------------------------------------------------


def _ok_response(
    *,
    response: str = "Hello!",
    prompt_eval_count: int = 4,
    eval_count: int = 7,
    total_duration: int = 1_500_000_000,  # 1.5 s in ns
    eval_duration: int = 1_300_000_000,
) -> dict:
    return {
        "model": "x:1",
        "response": response,
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "total_duration": total_duration,
        "eval_duration": eval_duration,
    }


class TestOllamaBackendComplete:
    def _make(self, fake_urlopen) -> OllamaBackend:
        return OllamaBackend("x:1", urlopen=fake_urlopen)

    def test_returns_inference_result(self) -> None:
        fake = _make_fake_urlopen(response=_ok_response())
        backend = self._make(fake)
        r = backend.complete("hello world")
        assert isinstance(r, InferenceResult)

    def test_completion_text(self) -> None:
        fake = _make_fake_urlopen(response=_ok_response(response="42"))
        backend = self._make(fake)
        r = backend.complete("hi", max_new_tokens=4)
        assert r.completion == "42"
        assert r.prompt == "hi"

    def test_token_counts(self) -> None:
        fake = _make_fake_urlopen(
            response=_ok_response(prompt_eval_count=12, eval_count=24)
        )
        backend = self._make(fake)
        r = backend.complete("hi", max_new_tokens=24)
        assert r.prompt_tokens == 12
        assert r.completion_tokens == 24

    def test_token_counts_missing_yield_none(self) -> None:
        # Ollama may omit prompt_eval_count / eval_count in some cases
        # (e.g., when a request is satisfied entirely from cache). The
        # backend must handle absence gracefully.
        body = {"response": "ok", "done": True}  # no counts
        fake = _make_fake_urlopen(response=body)
        backend = self._make(fake)
        r = backend.complete("hi")
        assert r.prompt_tokens is None
        assert r.completion_tokens is None

    def test_latency_recorded(self) -> None:
        fake = _make_fake_urlopen(response=_ok_response())
        backend = self._make(fake)
        r = backend.complete("hi")
        assert r.latency_ms >= 0.0

    def test_backend_field_carries_info(self) -> None:
        fake = _make_fake_urlopen(response=_ok_response())
        backend = OllamaBackend("foo:bar", urlopen=fake)
        r = backend.complete("hi")
        assert r.backend.name == "ollama"
        assert r.backend.model_id == "foo:bar"

    def test_request_url_is_api_generate(self) -> None:
        captured: list[dict] = []
        fake = _make_fake_urlopen(response=_ok_response(), captured=captured)
        backend = OllamaBackend(
            "x:1", base_url="http://test:99", urlopen=fake
        )
        backend.complete("hi")
        assert len(captured) == 1
        assert captured[0]["url"] == "http://test:99/api/generate"

    def test_request_method_is_post(self) -> None:
        captured: list[dict] = []
        fake = _make_fake_urlopen(response=_ok_response(), captured=captured)
        backend = self._make(fake)
        backend.complete("hi")
        assert captured[0]["method"] == "POST"

    def test_request_body_includes_required_fields(self) -> None:
        captured: list[dict] = []
        fake = _make_fake_urlopen(response=_ok_response(), captured=captured)
        backend = OllamaBackend("foo:1", urlopen=fake)
        backend.complete("the prompt", max_new_tokens=42, temperature=0.5)

        body = json.loads(captured[0]["body"])
        assert body["model"] == "foo:1"
        assert body["prompt"] == "the prompt"
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 42
        assert body["options"]["temperature"] == 0.5

    def test_request_content_type_header(self) -> None:
        captured: list[dict] = []
        fake = _make_fake_urlopen(response=_ok_response(), captured=captured)
        backend = self._make(fake)
        backend.complete("hi")
        # urllib normalizes header capitalization
        headers = {k.lower(): v for k, v in captured[0]["headers"].items()}
        assert headers["content-type"] == "application/json"

    def test_metadata_includes_server_durations(self) -> None:
        fake = _make_fake_urlopen(
            response=_ok_response(
                total_duration=2_000_000_000,  # 2000 ms
                eval_duration=1_500_000_000,  # 1500 ms
            )
        )
        backend = self._make(fake)
        r = backend.complete("hi")
        assert r.metadata["server_total_duration_ms"] == pytest.approx(2000.0)
        assert r.metadata["server_eval_duration_ms"] == pytest.approx(1500.0)

    def test_metadata_includes_sampling(self) -> None:
        fake = _make_fake_urlopen(response=_ok_response())
        backend = self._make(fake)
        r = backend.complete(
            "hi", max_new_tokens=20, temperature=0.3, top_k=10, top_p=0.95
        )
        s = r.metadata["sampling"]
        assert s["max_new_tokens"] == 20
        assert s["temperature"] == 0.3
        assert s["top_k"] == 10
        assert s["top_p"] == 0.95

    def test_timeout_is_passed_through(self) -> None:
        captured: list[dict] = []
        fake = _make_fake_urlopen(response=_ok_response(), captured=captured)
        backend = OllamaBackend("x:1", timeout=37.5, urlopen=fake)
        backend.complete("hi")
        assert captured[0]["timeout"] == 37.5


class TestOllamaBackendOptionsHandling:
    def test_top_k_omitted_when_none(self) -> None:
        captured: list[dict] = []
        fake = _make_fake_urlopen(response=_ok_response(), captured=captured)
        backend = OllamaBackend("x:1", urlopen=fake)
        backend.complete("hi", top_k=None, top_p=None)
        body = json.loads(captured[0]["body"])
        assert "top_k" not in body["options"]
        assert "top_p" not in body["options"]

    def test_top_k_included_when_set(self) -> None:
        captured: list[dict] = []
        fake = _make_fake_urlopen(response=_ok_response(), captured=captured)
        backend = OllamaBackend("x:1", urlopen=fake)
        backend.complete("hi", top_k=20, top_p=0.9)
        body = json.loads(captured[0]["body"])
        assert body["options"]["top_k"] == 20
        assert body["options"]["top_p"] == pytest.approx(0.9)


class TestOllamaBackendValidation:
    def _make(self) -> OllamaBackend:
        return OllamaBackend("x:1", urlopen=_make_fake_urlopen(response=_ok_response()))

    def test_empty_prompt_rejected(self) -> None:
        backend = self._make()
        with pytest.raises(ValueError):
            backend.complete("")

    def test_zero_max_new_tokens_rejected(self) -> None:
        backend = self._make()
        with pytest.raises(ValueError):
            backend.complete("hi", max_new_tokens=0)


class TestOllamaBackendErrorPaths:
    def test_url_error_wrapped(self) -> None:
        fake = _make_fake_urlopen(
            raise_exc=urllib.error.URLError("Connection refused")
        )
        backend = OllamaBackend("x:1", urlopen=fake)
        with pytest.raises(OllamaError) as exc_info:
            backend.complete("hi")
        assert "Could not reach Ollama" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, urllib.error.URLError)

    def test_http_error_wrapped(self) -> None:
        http_err = urllib.error.HTTPError(
            url="http://localhost:11434/api/generate",
            code=500,
            msg="Internal Server Error",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        fake = _make_fake_urlopen(raise_exc=http_err)
        backend = OllamaBackend("x:1", urlopen=fake)
        with pytest.raises(OllamaError) as exc_info:
            backend.complete("hi")
        assert "500" in str(exc_info.value)

    def test_bad_json_wrapped(self) -> None:
        fake = _make_fake_urlopen(bad_json=b"not valid json {{{")
        backend = OllamaBackend("x:1", urlopen=fake)
        with pytest.raises(OllamaError):
            backend.complete("hi")

    def test_missing_response_field_wrapped(self) -> None:
        fake = _make_fake_urlopen(response={"done": True})
        backend = OllamaBackend("x:1", urlopen=fake)
        with pytest.raises(OllamaError, match="response"):
            backend.complete("hi")


# -----------------------------------------------------------------------
# Boilerplate: BenchmarkResult
# -----------------------------------------------------------------------


class TestBenchmarkResultBoilerplate:
    def test_construction(self) -> None:
        info = BackendInfo(name="x", model_id="y")
        br = BenchmarkResult(
            backend=info,
            n=2,
            latency_ms_total=100.0,
            latency_ms_mean=50.0,
            latency_ms_p50=50.0,
            latency_ms_p90=60.0,
            completion_tokens_total=20,
            tokens_per_second_overall=200.0,
            per_request_latency_ms=[40.0, 60.0],
            per_request_tokens_per_second=[250.0, 166.7],
            results=[],
        )
        assert br.n == 2
        assert br.metadata == {}

    def test_repr_compact(self) -> None:
        info = BackendInfo(name="x", model_id="y")
        br = BenchmarkResult(
            backend=info,
            n=1,
            latency_ms_total=10.0,
            latency_ms_mean=10.0,
            latency_ms_p50=10.0,
            latency_ms_p90=10.0,
            completion_tokens_total=5,
            tokens_per_second_overall=500.0,
            per_request_latency_ms=[10.0],
            per_request_tokens_per_second=[500.0],
            results=[],
        )
        s = repr(br)
        assert "BenchmarkResult" in s
        assert "n=1" in s
        assert "500.00" in s

    def test_repr_handles_none_throughput(self) -> None:
        info = BackendInfo(name="x", model_id="y")
        br = BenchmarkResult(
            backend=info,
            n=1,
            latency_ms_total=10.0,
            latency_ms_mean=10.0,
            latency_ms_p50=10.0,
            latency_ms_p90=10.0,
            completion_tokens_total=None,
            tokens_per_second_overall=None,
            per_request_latency_ms=[10.0],
            per_request_tokens_per_second=[None],
            results=[],
        )
        s = repr(br)
        # The em-dash placeholder for None
        assert "—" in s


# -----------------------------------------------------------------------
# benchmark (SCAFFOLDED)
# -----------------------------------------------------------------------


class TestBenchmark:
    def test_returns_benchmark_result(self) -> None:
        backend = _FakeBackend([("x", 10.0, 1, 5)])
        out = benchmark(backend, ["hi"])
        assert isinstance(out, BenchmarkResult)

    def test_n_matches_prompt_count(self) -> None:
        backend = _FakeBackend(
            [("a", 1.0, 1, 1), ("b", 1.0, 1, 1), ("c", 1.0, 1, 1)]
        )
        out = benchmark(backend, ["x", "y", "z"])
        assert out.n == 3
        assert len(out.results) == 3

    def test_iterates_in_order(self) -> None:
        backend = _FakeBackend(
            [("a", 1.0, 1, 1), ("b", 1.0, 1, 1), ("c", 1.0, 1, 1)]
        )
        benchmark(backend, ["p1", "p2", "p3"])
        assert [c["prompt"] for c in backend.calls] == ["p1", "p2", "p3"]

    def test_forwards_sampling_kwargs(self) -> None:
        backend = _FakeBackend([("a", 1.0, 1, 1)])
        benchmark(
            backend, ["hi"],
            max_new_tokens=64, temperature=0.7, top_k=40, top_p=0.95,
        )
        c = backend.calls[0]
        assert c["max_new_tokens"] == 64
        assert c["temperature"] == 0.7
        assert c["top_k"] == 40
        assert c["top_p"] == 0.95

    def test_latency_total_is_sum(self) -> None:
        backend = _FakeBackend(
            [("a", 10.0, 1, 1), ("b", 20.0, 1, 1), ("c", 30.0, 1, 1)]
        )
        out = benchmark(backend, ["x", "y", "z"])
        assert out.latency_ms_total == pytest.approx(60.0)

    def test_latency_mean(self) -> None:
        backend = _FakeBackend(
            [("a", 10.0, 1, 1), ("b", 20.0, 1, 1), ("c", 30.0, 1, 1)]
        )
        out = benchmark(backend, ["x", "y", "z"])
        assert out.latency_ms_mean == pytest.approx(20.0)

    def test_per_request_latency_in_order(self) -> None:
        backend = _FakeBackend(
            [("a", 1.0, 1, 1), ("b", 2.0, 1, 1), ("c", 3.0, 1, 1)]
        )
        out = benchmark(backend, ["x", "y", "z"])
        assert out.per_request_latency_ms == [1.0, 2.0, 3.0]

    def test_completion_tokens_total(self) -> None:
        backend = _FakeBackend(
            [("a", 10.0, 1, 5), ("b", 10.0, 1, 7), ("c", 10.0, 1, 13)]
        )
        out = benchmark(backend, ["x", "y", "z"])
        assert out.completion_tokens_total == 25

    def test_tokens_per_second_overall(self) -> None:
        # 30 tokens in 30 ms total = 1000 tok/s
        backend = _FakeBackend(
            [("a", 10.0, 1, 10), ("b", 10.0, 1, 10), ("c", 10.0, 1, 10)]
        )
        out = benchmark(backend, ["x", "y", "z"])
        assert out.tokens_per_second_overall == pytest.approx(1000.0)

    def test_completion_tokens_total_none_when_any_missing(self) -> None:
        backend = _FakeBackend(
            [("a", 10.0, 1, 5), ("b", 10.0, 1, None), ("c", 10.0, 1, 7)]
        )
        out = benchmark(backend, ["x", "y", "z"])
        assert out.completion_tokens_total is None
        assert out.tokens_per_second_overall is None

    def test_per_request_tokens_per_second(self) -> None:
        # request 0: 5 tokens / 10 ms = 500 tok/s
        # request 1: 10 tokens / 10 ms = 1000 tok/s
        backend = _FakeBackend(
            [("a", 10.0, 1, 5), ("b", 10.0, 1, 10)]
        )
        out = benchmark(backend, ["x", "y"])
        assert out.per_request_tokens_per_second[0] == pytest.approx(500.0)
        assert out.per_request_tokens_per_second[1] == pytest.approx(1000.0)

    def test_backend_field_propagates(self) -> None:
        info = BackendInfo(name="my_kind", model_id="my_id")
        backend = _FakeBackend([("a", 1.0, 1, 1)], info=info)
        out = benchmark(backend, ["hi"])
        assert out.backend.name == "my_kind"
        assert out.backend.model_id == "my_id"

    def test_metadata_copied_not_referenced(self) -> None:
        backend = _FakeBackend([("a", 1.0, 1, 1)])
        meta = {"suite": "tinyqa-v1"}
        out = benchmark(backend, ["hi"], metadata=meta)
        meta["suite"] = "MUTATED"
        assert out.metadata["suite"] == "tinyqa-v1"


class TestBenchmarkEdgeCases:
    def test_empty_prompts_rejected(self) -> None:
        backend = _FakeBackend([])
        with pytest.raises(ValueError):
            benchmark(backend, [])

    def test_single_prompt_percentiles_collapse(self) -> None:
        backend = _FakeBackend([("a", 42.0, 1, 5)])
        out = benchmark(backend, ["hi"])
        assert out.latency_ms_mean == pytest.approx(42.0)
        assert out.latency_ms_p50 == pytest.approx(42.0)
        assert out.latency_ms_p90 == pytest.approx(42.0)

    def test_p90_above_p50_for_skewed_latencies(self) -> None:
        # 9 fast calls + 1 slow call. p50 ~ 1, p90 should reflect the
        # slow tail.
        backend = _FakeBackend(
            [("c", 1.0, 1, 1)] * 9 + [("c", 100.0, 1, 1)]
        )
        out = benchmark(backend, ["x"] * 10)
        assert out.latency_ms_p50 < out.latency_ms_p90


# -----------------------------------------------------------------------
# End-to-end smoke test (requires Module 09 + Module 11)
# -----------------------------------------------------------------------


class TestLocalTransformerBackendRealSmoke:
    """End-to-end check: a real `TransformerLM` + a real BPE tokenizer
    + the real `generate` produce an `InferenceResult`. Depends on
    Module 09 and Module 11. If either is unfilled, this single test
    fails on `NotImplementedError`."""

    def test_real_pipeline_smoke(self) -> None:
        try:
            from g2c.tokenizer.bpe import BPETokenizer
            from g2c.transformer.transformer_lm import TransformerLM
        except Exception as e:  # pragma: no cover
            pytest.skip(f"prerequisite import failed: {e}")

        torch.manual_seed(0)

        # Train a microscopic BPE on a tiny corpus.
        try:
            tok = BPETokenizer()
            tok.train("aabbccdd " * 50, vocab_size=300)
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"BPE not implemented: {e}")

        try:
            model = TransformerLM(
                vocab_size=max(300, 256),
                embedding_dim=8,
                num_layers=1,
                num_heads=2,
                max_seq_len=32,
            )
            model.eval()
        except (NotImplementedError, TypeError) as e:
            pytest.skip(f"TransformerLM not implemented: {e}")

        backend = LocalTransformerBackend(model, tok, model_id="smoke-test")
        try:
            result = backend.complete(
                "ab", max_new_tokens=4, temperature=0.0
            )
        except NotImplementedError as e:
            pytest.skip(f"Prerequisite NotImplementedError: {e}")

        assert isinstance(result, InferenceResult)
        assert result.prompt == "ab"
        assert result.prompt_tokens is not None and result.prompt_tokens >= 1
        assert result.completion_tokens is not None
        assert 0 <= result.completion_tokens <= 4
        assert result.latency_ms >= 0.0


# -----------------------------------------------------------------------
# Cross-cutting: module exports
# -----------------------------------------------------------------------


class TestModuleExports:
    def test_public_api_imports(self) -> None:
        from g2c.inference import (
            DEFAULT_OLLAMA_URL,
            DEFAULT_PRODLM_MODEL_ID,
            PRODLM_NAME,
            ArtifactBackend,
            Backend,
            BackendInfo,
            BenchmarkResult,
            InferenceResult,
            LocalTransformerBackend,
            OllamaBackend,
            OllamaError,
            benchmark,
            load_artifact_backend,
            load_default_backend,
            load_prodlm_backend,
            write_prodlm_manifest,
        )

        # Each is a usable name, not just a string lookup.
        assert callable(ArtifactBackend)
        assert callable(LocalTransformerBackend)
        assert callable(OllamaBackend)
        assert callable(benchmark)
        assert callable(load_artifact_backend)
        assert callable(load_default_backend)
        assert callable(load_prodlm_backend)
        assert callable(write_prodlm_manifest)
        assert isinstance(DEFAULT_OLLAMA_URL, str)
        assert isinstance(DEFAULT_PRODLM_MODEL_ID, str)
        assert PRODLM_NAME == "ProdLM"
        assert issubclass(OllamaError, Exception)
        # `name` etc. are dataclass fields — registered on
        # __dataclass_fields__, not on the class as attributes.
        assert "name" in BackendInfo.__dataclass_fields__
        assert "n" in BenchmarkResult.__dataclass_fields__
        assert hasattr(InferenceResult, "tokens_per_second")  # property
        assert Backend.__abstractmethods__ == {"complete", "info"}
