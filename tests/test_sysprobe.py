"""Unit tests for scripts/sysprobe.py.

Covers the pure-logic pieces: GGUF binary parsing, parameter-count
inference from metadata, KV cache sizing, and throughput tiering.
The training/inference/HTTP probes are integration paths and aren't
exercised here.
"""
from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SYSPROBE_PATH = REPO_ROOT / "scripts" / "sysprobe.py"


def _load_sysprobe():
    spec = importlib.util.spec_from_file_location("sysprobe", SYSPROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sysprobe"] = mod
    spec.loader.exec_module(mod)
    return mod


sp = _load_sysprobe()


# ---------------------------------------------------------------------------
# Minimal GGUF builder for testing the parser without a real model file.
# ---------------------------------------------------------------------------

def _pack_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_scalar_kv(key: str, type_id: int, fmt: str, value) -> bytes:
    return _pack_string(key) + struct.pack("<I", type_id) + struct.pack(fmt, value)


def _pack_string_kv(key: str, value: str) -> bytes:
    return (
        _pack_string(key)
        + struct.pack("<I", sp.GGUF_TYPE_STRING)
        + _pack_string(value)
    )


def _pack_string_array_kv(key: str, values: list[str]) -> bytes:
    body = (
        struct.pack("<I", sp.GGUF_TYPE_STRING)
        + struct.pack("<Q", len(values))
        + b"".join(_pack_string(v) for v in values)
    )
    return _pack_string(key) + struct.pack("<I", sp.GGUF_TYPE_ARRAY) + body


def _build_gguf(kv_pairs: list[bytes]) -> bytes:
    """Assemble a fake GGUF blob with the given metadata KV pairs."""
    header = (
        b"GGUF"
        + struct.pack("<I", 3)            # version
        + struct.pack("<Q", 0)            # tensor count (we don't write tensors)
        + struct.pack("<Q", len(kv_pairs))
    )
    return header + b"".join(kv_pairs)


# ---------------------------------------------------------------------------
# GGUF parser
# ---------------------------------------------------------------------------

def test_gguf_reader_parses_string_and_scalar_metadata():
    buf = _build_gguf([
        _pack_string_kv("general.architecture", "llama"),
        _pack_scalar_kv("general.file_type", sp.GGUF_TYPE_UINT32, "<I", 15),
        _pack_scalar_kv("llama.block_count", sp.GGUF_TYPE_UINT32, "<I", 32),
    ])
    r = sp._GGUFReader(buf)
    assert r.read(4) == b"GGUF"
    assert r.u32() == 3
    assert r.u64() == 0          # tensor count
    assert r.u64() == 3          # kv count

    keys: dict[str, object] = {}
    for _ in range(3):
        k = r.string()
        t = r.u32()
        keys[k] = r.value(t)

    assert keys["general.architecture"] == "llama"
    assert keys["general.file_type"] == 15
    assert keys["llama.block_count"] == 32


def test_gguf_reader_skips_large_string_arrays():
    """Tokenizer arrays sit in metadata; the parser must not materialize
    >1024-element arrays or large models would balloon parse cost."""
    big_array = [f"tok_{i}" for i in range(2000)]
    buf = _build_gguf([
        _pack_string_array_kv("tokenizer.ggml.tokens", big_array),
        _pack_string_kv("general.architecture", "llama"),
    ])
    md = _parse_full(buf)
    assert isinstance(md["tokenizer.ggml.tokens"], str)
    assert "skipped" in md["tokenizer.ggml.tokens"]
    # The keys after the big array must still be reachable.
    assert md["general.architecture"] == "llama"


def _parse_full(buf: bytes) -> dict:
    """Mimic fetch_gguf_metadata's parsing loop on an in-memory buffer."""
    r = sp._GGUFReader(buf)
    assert r.read(4) == b"GGUF"
    r.u32()        # version
    r.u64()        # tensor count
    kv_count = r.u64()
    out: dict = {}
    for _ in range(kv_count):
        k = r.string()
        t = r.u32()
        out[k] = r.value(t)
    return out


# ---------------------------------------------------------------------------
# Parameter-count inference
# ---------------------------------------------------------------------------

def test_gguf_param_count_prefers_explicit_field():
    md = {"general.parameter_count": 1_234_567_890}
    assert sp.gguf_param_count(md) == 1_234_567_890


def test_gguf_param_count_parses_size_label():
    assert sp.gguf_param_count({"general.size_label": "1B"}) == 1_000_000_000
    assert sp.gguf_param_count({"general.size_label": "8B"}) == 8_000_000_000
    assert sp.gguf_param_count({"general.size_label": "1.5B"}) == 1_500_000_000
    assert sp.gguf_param_count({"general.size_label": "70M"}) == 70_000_000


def test_gguf_param_count_size_label_takes_precedence_over_arch():
    md = {
        "general.size_label": "1B",
        "general.architecture": "llama",
        "llama.vocab_size": 1, "llama.embedding_length": 1,
        "llama.block_count": 1, "llama.feed_forward_length": 1,
        "llama.attention.head_count": 1,
    }
    # size_label wins over arch math
    assert sp.gguf_param_count(md) == 1_000_000_000


def test_gguf_param_count_computes_from_architecture_keys():
    # Llama 3.2 1B-ish shape: V=128K, D=2048, N=16, H=8192,
    # head_count=32, head_count_kv=8, head/value length=64.
    md = {
        "general.architecture": "llama",
        "llama.vocab_size": 128_256,
        "llama.embedding_length": 2048,
        "llama.block_count": 16,
        "llama.feed_forward_length": 8192,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 64,
        "llama.attention.value_length": 64,
    }
    n = sp.gguf_param_count(md)
    # Real Llama 3.2 1B is ~1.24B. Our computation excludes a few small
    # tensors (norms, biases) and isn't exact, but should land within
    # ~30% of the true count.
    assert 0.7e9 < n < 1.6e9


def test_gguf_param_count_returns_none_on_empty_metadata():
    assert sp.gguf_param_count({}) is None


# ---------------------------------------------------------------------------
# Quantization bpw lookup
# ---------------------------------------------------------------------------

def test_gguf_quant_bpw_known_types():
    assert sp.gguf_quant_bpw({"general.file_type": 0}) == 32.0    # F32
    assert sp.gguf_quant_bpw({"general.file_type": 1}) == 16.0    # F16
    assert sp.gguf_quant_bpw({"general.file_type": 15}) == 4.8    # Q4_K_M
    assert sp.gguf_quant_bpw({"general.file_type": 18}) == 6.6    # Q6_K


def test_gguf_quant_bpw_returns_none_when_absent():
    assert sp.gguf_quant_bpw({}) is None
    # Unknown file type → None (not a guess)
    assert sp.gguf_quant_bpw({"general.file_type": 9999}) is None


# ---------------------------------------------------------------------------
# KV cache sizing
# ---------------------------------------------------------------------------

def test_kv_cache_scales_with_context_length():
    cand = sp.ProdLMCandidate(
        display_name="x", ollama_tag="x", hf_repo="x", gguf_filename="x",
        n_layers=32, n_kv_heads=8, head_dim=128, nominal_params_b=8.0,
    )
    kv_1k = sp.estimate_kv_cache_gb(cand, 1024)
    kv_4k = sp.estimate_kv_cache_gb(cand, 4096)
    # Linear in context length
    assert abs(kv_4k - 4 * kv_1k) < 1e-9
    # Sanity-magnitude check: 32L × 2(K,V) × 8KV × 128 × fp16 × 4096 tokens
    # = 0.5 GB. Matches real Llama 3.1 8B at 4K context.
    assert 0.4 < kv_4k < 0.6


def test_kv_cache_scales_with_layers_and_kv_heads():
    base = sp.ProdLMCandidate(
        display_name="x", ollama_tag="x", hf_repo="x", gguf_filename="x",
        n_layers=16, n_kv_heads=4, head_dim=128, nominal_params_b=1.0,
    )
    doubled_layers = sp.ProdLMCandidate(
        display_name="x", ollama_tag="x", hf_repo="x", gguf_filename="x",
        n_layers=32, n_kv_heads=4, head_dim=128, nominal_params_b=1.0,
    )
    assert abs(sp.estimate_kv_cache_gb(doubled_layers, 2048)
               - 2 * sp.estimate_kv_cache_gb(base, 2048)) < 1e-9


def test_decoder_kv_cache_matches_prodlm_formula():
    cand = sp.ProdLMCandidate(
        display_name="x", ollama_tag="x", hf_repo="x", gguf_filename="x",
        n_layers=28, n_kv_heads=8, head_dim=128, nominal_params_b=0.6,
    )
    generic = sp.estimate_decoder_kv_cache_gb(
        num_hidden_layers=28,
        num_key_value_heads=8,
        head_dim=128,
        context_length=4096,
        bytes_per_element=2,
    )
    assert abs(generic - sp.estimate_kv_cache_gb(cand, 4096)) < 1e-12


def test_baselm_candidate_param_counts_match_size_tiers():
    by_name = {c.display_name: c for c in sp.BASELM_INFERENCE_CANDIDATES}
    assert 0.12 < by_name["SmolLM 135M"].param_count() / 1e9 < 0.16
    assert 0.32 < by_name["SmolLM 360M"].param_count() / 1e9 < 0.40
    assert 0.55 < by_name["Qwen3 0.6B Base"].param_count() / 1e9 < 0.65


def test_estimate_baselm_inference_clamps_context_and_reports_fp32():
    sysinfo = sp.SystemInfo(
        platform="Darwin",
        chip="Apple M2 Max",
        physical_memory_gb=32.0,
        bandwidth_gbps=400.0,
        mps_available=True,
        python_version="3.11",
        torch_version="2.x",
    )
    smollm = next(
        c for c in sp.BASELM_INFERENCE_CANDIDATES
        if c.display_name == "SmolLM 135M"
    )

    result = sp.estimate_baselm_inference(smollm, sysinfo, context_length=4096)

    assert result.status == "ok"
    assert result.context_length == 2048
    assert "context clamped" in result.note
    assert result.notebook_memory_fp32_gb > result.notebook_memory_fp16_gb
    assert result.notebook_memory_fp16_gb > result.model_memory_fp16_gb
    assert result.practical_memory_gb == 28.0


def test_estimate_baselm_posttraining_uses_course_shapes():
    sysinfo = sp.SystemInfo(
        platform="Darwin",
        chip="Apple M2 Max",
        physical_memory_gb=32.0,
        bandwidth_gbps=400.0,
        mps_available=True,
        python_version="3.11",
        torch_version="2.x",
    )
    qwen = next(
        c for c in sp.BASELM_INFERENCE_CANDIDATES
        if c.display_name == "Qwen3 0.6B Base"
    )

    sft = sp.estimate_baselm_posttraining(qwen, sysinfo, stage="SFT")
    dpo = sp.estimate_baselm_posttraining(qwen, sysinfo, stage="DPO")

    assert sft.stage == "SFT"
    assert sft.batch_size == sp.BASELM_SFT_BATCH_SIZE
    assert sft.context_length == sp.BASELM_POSTTRAIN_CONTEXT_LENGTH
    assert dpo.stage == "DPO"
    assert dpo.batch_size == sp.BASELM_DPO_BATCH_SIZE
    assert dpo.context_length == sp.BASELM_POSTTRAIN_CONTEXT_LENGTH
    assert sft.notebook_memory_fp32_gb > sft.notebook_memory_fp16_gb
    assert dpo.notebook_memory_fp32_gb > dpo.notebook_memory_fp16_gb
    assert sft.practical_memory_gb == 28.0
    assert dpo.practical_memory_gb == 28.0


def test_estimate_baselm_posttraining_rejects_unknown_stage():
    sysinfo = sp.SystemInfo(
        platform="Darwin",
        chip="Apple M2 Max",
        physical_memory_gb=32.0,
        bandwidth_gbps=400.0,
        mps_available=True,
        python_version="3.11",
        torch_version="2.x",
    )
    cand = sp.BASELM_INFERENCE_CANDIDATES[0]

    with pytest.raises(ValueError, match="unknown BaseLM post-training stage"):
        sp.estimate_baselm_posttraining(cand, sysinfo, stage="RLHF")


# ---------------------------------------------------------------------------
# Throughput tier classification
# ---------------------------------------------------------------------------

def test_throughput_tier_thresholds():
    assert sp.throughput_tier(100.0) == "interactive"
    assert sp.throughput_tier(30.0) == "interactive"
    assert sp.throughput_tier(29.9) == "usable"
    assert sp.throughput_tier(10.0) == "usable"
    assert sp.throughput_tier(9.9) == "slow"
    assert sp.throughput_tier(3.0) == "slow"
    assert sp.throughput_tier(2.9) == "painful"
    assert sp.throughput_tier(0.5) == "painful"


# ---------------------------------------------------------------------------
# Canonical shapes look sane
# ---------------------------------------------------------------------------

def test_canonical_shapes_match_named_size_tiers():
    by_name = {s.name: s for s in sp.CANONICAL_SHAPES}
    # Names like StoryLM-1M / TinyLLM-100M should produce a param count
    # whose magnitude matches the size in the name.
    for name, shape in by_name.items():
        size_label = name.rsplit("-", 1)[1]   # "1M", "5M", "30M", "100M"
        unit = size_label[-1]
        n = int(size_label[:-1])
        nominal = n * (1e6 if unit == "M" else 1e9)
        actual = shape.param_count()
        ratio = actual / nominal
        # The naming convention buckets to size tiers; allow loose match.
        assert 0.5 < ratio < 2.5, f"{name}: actual={actual} nominal={nominal}"


def test_find_shape_returns_canonical_shape_by_name():
    shape = sp.find_shape("TinyLLM-30M")
    assert shape.name == "TinyLLM-30M"
    assert shape.batch_size == 16
    with pytest.raises(KeyError, match="unknown canonical shape"):
        sp.find_shape("MissingLM")


# ---------------------------------------------------------------------------
# Batch sweep helpers
# ---------------------------------------------------------------------------

def test_parse_batch_list_preserves_order_and_deduplicates():
    assert sp.parse_batch_list("8,16,16,32") == [8, 16, 32]
    assert sp.parse_batch_list(" 4, 8 , 32 ") == [4, 8, 32]


def test_parse_batch_list_rejects_empty_and_non_positive_values():
    with pytest.raises(ValueError, match="at least one"):
        sp.parse_batch_list(" , ")
    with pytest.raises(ValueError, match="positive"):
        sp.parse_batch_list("8,0,16")
    with pytest.raises(ValueError, match="invalid"):
        sp.parse_batch_list("8,big,16")


def test_classify_training_headroom_marks_ok_tight_and_fail():
    sysinfo = sp.SystemInfo(
        platform="Darwin",
        chip="Apple M2 Max",
        physical_memory_gb=64.0,
        bandwidth_gbps=400.0,
        mps_available=True,
        python_version="3.11",
        torch_version="2.x",
    )

    def result_with_peak(peak_gb: float) -> sp.ProbeResult:
        return sp.ProbeResult(
            shape_name="TinyLLM-100M",
            param_count_m=91.2,
            status="ok",
            peak_memory_gb=peak_gb,
            tokens_per_sec=1000.0,
            batch_size=8,
            context_length=512,
            steps_run=20,
        )

    ok = sp.classify_training_headroom(result_with_peak(10.0), sysinfo)
    tight = sp.classify_training_headroom(result_with_peak(35.0), sysinfo)
    fail = sp.classify_training_headroom(result_with_peak(50.0), sysinfo)

    # 64 GB unified memory - 4 GB OS reserve = 60 GB practical budget.
    # The notebook estimate is 3x probe memory.
    assert ok.status == "ok"
    assert ok.estimated_notebook_memory_gb == 30.0
    assert ok.practical_memory_gb == 60.0

    assert tight.status == "tight"
    assert tight.estimated_notebook_memory_gb == 105.0
    assert "exceeds practical memory" in tight.note

    assert fail.status == "fail"
    assert fail.estimated_notebook_memory_gb == 150.0
    assert "exceeds 2x practical memory" in fail.note


def test_classify_training_headroom_populates_estimates_for_ok_result():
    sysinfo = sp.SystemInfo(
        platform="Darwin",
        chip="Apple M2 Max",
        physical_memory_gb=32.0,
        bandwidth_gbps=400.0,
        mps_available=True,
        python_version="3.11",
        torch_version="2.x",
    )
    result = sp.ProbeResult(
        shape_name="TinyLLM-30M",
        param_count_m=29.4,
        status="ok",
        peak_memory_gb=6.5,
        tokens_per_sec=1000.0,
        batch_size=16,
        context_length=512,
        steps_run=20,
    )

    classified = sp.classify_training_headroom(result, sysinfo)

    assert classified.status == "ok"
    assert classified.estimated_notebook_memory_gb == 19.5
    assert classified.practical_memory_gb == 28.0


def test_probe_batch_sweep_stops_after_failed_batch(monkeypatch):
    shape = sp.find_shape("TinyLLM-100M")
    sysinfo = sp.SystemInfo(
        platform="Darwin",
        chip="Apple M2 Max",
        physical_memory_gb=64.0,
        bandwidth_gbps=400.0,
        mps_available=True,
        python_version="3.11",
        torch_version="2.x",
    )
    batches_seen: list[int] = []

    def fake_probe_training(swept_shape, device, n_steps=100, progress_cb=None):
        batches_seen.append(swept_shape.batch_size)
        status = "fail" if swept_shape.batch_size == 16 else "ok"
        return sp.ProbeResult(
            shape_name=swept_shape.name,
            param_count_m=swept_shape.param_count() / 1e6,
            status=status,
            peak_memory_gb=None if status == "fail" else 1.0,
            tokens_per_sec=None if status == "fail" else 1000.0,
            batch_size=swept_shape.batch_size,
            context_length=swept_shape.context_length,
            steps_run=n_steps,
        )

    monkeypatch.setattr(sp, "probe_training", fake_probe_training)
    results = sp.probe_batch_sweep(
        shape,
        [8, 16, 32],
        "mps",
        sysinfo,
        n_steps=3,
    )

    assert batches_seen == [8, 16]
    assert [r.batch_size for r in results] == [8, 16, 32]
    assert [r.status for r in results] == ["ok", "fail", "skip"]
    assert results[2].note == "skipped after B=16 failed"
