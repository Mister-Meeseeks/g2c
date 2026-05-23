"""System-capability probe for the g2c course.

Reports what your machine can handle across three dimensions:

  1. **Training** — can you train each canonical size (1M, 5M, 30M, 100M)?
  2. **Inference** — what's the from-scratch generation speed at each size?
  3. **BaseLM** — what would Hugging Face/PyTorch inference likely require for
     the course fallback base models?
  4. **ProdLM** — for a curated short-list of Ollama instruct models, will the
     model fit in unified memory, and what throughput should you expect?

The training/inference probes build size-matched transformers using
`torch.nn` primitives (this is a diagnostic, not a pedagogical exercise) and
run a few forward/backward steps on dummy data. They report peak memory and
tokens/sec.

The ProdLM probe is pre-download: it range-fetches the GGUF header of each
candidate model from Hugging Face (a few KB per model) to read param count
and quantization, then estimates fit + throughput against the detected
unified memory and memory bandwidth.

Run with:

    .venv/bin/python scripts/sysprobe.py
    .venv/bin/python scripts/sysprobe.py --json out.json
    .venv/bin/python scripts/sysprobe.py --skip-prodlm   # no network
    .venv/bin/python scripts/sysprobe.py --skip-baselm   # hide HF estimate
    .venv/bin/python scripts/sysprobe.py --skip-training # fast run
    .venv/bin/python scripts/sysprobe.py --baselm-model HuggingFaceTB/SmolLM-360M
    .venv/bin/python scripts/sysprobe.py --prodlm-model bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf
    .venv/bin/python scripts/sysprobe.py --skip-training --skip-prodlm --batch-sweep TinyLLM-30M --sweep-batches 8,16,32
    .venv/bin/python scripts/sysprobe.py --skip-training --skip-prodlm --batch-sweep TinyLLM-100M --sweep-batches 8,16,32
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from typing import Any


# ---------------------------------------------------------------------------
# Canonical training sizes pulled from notebooks/clean/10-tinyllm.ipynb.
# These mirror what students actually train, so the probe reports the real
# training/inference picture rather than synthetic numbers.
# ---------------------------------------------------------------------------

@dataclass
class ModelShape:
    name: str
    vocab_size: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    hidden_dim: int
    batch_size: int       # training batch
    context_length: int   # training context len

    def param_count(self) -> int:
        V, D, N, H = self.vocab_size, self.embedding_dim, self.num_layers, self.hidden_dim
        # Tied embeddings (V*D once) + per-block (4*D*D for attn QKVO + 2*D*H for FFN)
        # + LN biases + head bias. Approximate, but close to actuals.
        embed = V * D
        per_block = 4 * D * D + 2 * D * H
        return embed + N * per_block + V


CANONICAL_SHAPES: list[ModelShape] = [
    ModelShape(
        name="StoryLM-1M",
        vocab_size=4096,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        max_seq_len=128,
        hidden_dim=512,
        batch_size=32,
        context_length=128,
    ),
    ModelShape(
        name="StoryLM-5M",
        vocab_size=4096,
        embedding_dim=256,
        num_layers=6,
        num_heads=8,
        max_seq_len=256,
        hidden_dim=1024,
        batch_size=16,
        context_length=256,
    ),
    ModelShape(
        name="StoryLM-30M",
        vocab_size=4096,
        embedding_dim=512,
        num_layers=9,
        num_heads=8,
        max_seq_len=256,
        hidden_dim=2048,
        batch_size=8,
        context_length=256,
    ),
    ModelShape(
        # The TinyLLM-30M config runs at 4× more tokens/step than StoryLM-30M
        # (B=16, T=512 vs B=8, T=256), so its activation memory is the
        # dominant signal in the 30M tier.
        name="TinyLLM-30M",
        vocab_size=8192,
        embedding_dim=512,
        num_layers=8,
        num_heads=8,
        max_seq_len=512,
        hidden_dim=2048,
        batch_size=16,
        context_length=512,
    ),
    ModelShape(
        name="TinyLLM-100M",   # stretch track
        vocab_size=8192,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_len=512,
        hidden_dim=3072,
        batch_size=8,
        context_length=512,
    ),
]


# ---------------------------------------------------------------------------
# ProdLM candidates: (display_name, ollama_tag, hf_repo, gguf_filename,
# n_layers, n_kv_heads, head_dim).
#
# The arch fields (n_layers, n_kv_heads, head_dim) are used to compute
# KV-cache memory at a given context length. They're hardcoded because
# they're stable model-architecture facts; pulling them from the GGUF
# header is possible but adds parser complexity for a fixed list.
#
# Param count and quantization come from the GGUF header at runtime.
# ---------------------------------------------------------------------------

@dataclass
class ProdLMCandidate:
    display_name: str
    ollama_tag: str
    hf_repo: str
    gguf_filename: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    nominal_params_b: float   # fallback if GGUF metadata can't be fetched


@dataclass
class BaseLMInferenceCandidate:
    display_name: str
    model_id: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_context: int

    def param_count(self) -> int:
        """Approximate tied-embedding decoder parameter count.

        These are HF/Llama-style causal LM configs. Biases and norm weights are
        small enough that they do not materially change memory estimates.
        """
        embed = self.vocab_size * self.hidden_size
        q_proj = self.hidden_size * (self.num_attention_heads * self.head_dim)
        k_proj = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        v_proj = self.hidden_size * (self.num_key_value_heads * self.head_dim)
        o_proj = (self.num_attention_heads * self.head_dim) * self.hidden_size
        ffn = 3 * self.hidden_size * self.intermediate_size
        per_block = q_proj + k_proj + v_proj + o_proj + ffn
        return embed + self.num_hidden_layers * per_block


PRODLM_CANDIDATES: list[ProdLMCandidate] = [
    ProdLMCandidate(
        display_name="Llama 3.2 1B Instruct (Q4_K_M)",
        ollama_tag="llama3.2:1b",
        hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
        gguf_filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        n_layers=16, n_kv_heads=8, head_dim=64,
        nominal_params_b=1.24,
    ),
    ProdLMCandidate(
        display_name="Llama 3.2 3B Instruct (Q4_K_M)  [course default]",
        ollama_tag="llama3.2:3b",
        hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        gguf_filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        n_layers=28, n_kv_heads=8, head_dim=128,
        nominal_params_b=3.21,
    ),
    ProdLMCandidate(
        display_name="Qwen 2.5 3B Instruct (Q4_K_M)",
        ollama_tag="qwen2.5:3b",
        hf_repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
        gguf_filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        n_layers=36, n_kv_heads=2, head_dim=128,
        nominal_params_b=3.09,
    ),
    ProdLMCandidate(
        display_name="Qwen 2.5 7B Instruct (Q4_K_M)",
        ollama_tag="qwen2.5:7b",
        hf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        gguf_filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        n_layers=28, n_kv_heads=4, head_dim=128,
        nominal_params_b=7.62,
    ),
    ProdLMCandidate(
        display_name="Llama 3.1 8B Instruct (Q4_K_M)",
        ollama_tag="llama3.1:8b",
        hf_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        gguf_filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        n_layers=32, n_kv_heads=8, head_dim=128,
        nominal_params_b=8.03,
    ),
    ProdLMCandidate(
        display_name="Gemma 2 9B Instruct (Q4_K_M)",
        ollama_tag="gemma2:9b",
        hf_repo="bartowski/gemma-2-9b-it-GGUF",
        gguf_filename="gemma-2-9b-it-Q4_K_M.gguf",
        n_layers=42, n_kv_heads=8, head_dim=256,
        nominal_params_b=9.24,
    ),
    ProdLMCandidate(
        display_name="Qwen 2.5 14B Instruct (Q4_K_M)",
        ollama_tag="qwen2.5:14b",
        hf_repo="bartowski/Qwen2.5-14B-Instruct-GGUF",
        gguf_filename="Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        n_layers=48, n_kv_heads=8, head_dim=128,
        nominal_params_b=14.77,
    ),
]


BASELM_INFERENCE_CANDIDATES: list[BaseLMInferenceCandidate] = [
    BaseLMInferenceCandidate(
        display_name="SmolLM 135M",
        model_id="HuggingFaceTB/SmolLM-135M",
        vocab_size=49_152,
        hidden_size=576,
        intermediate_size=1536,
        num_hidden_layers=30,
        num_attention_heads=9,
        num_key_value_heads=3,
        head_dim=64,
        max_context=2048,
    ),
    BaseLMInferenceCandidate(
        display_name="SmolLM 360M",
        model_id="HuggingFaceTB/SmolLM-360M",
        vocab_size=49_152,
        hidden_size=960,
        intermediate_size=2560,
        num_hidden_layers=32,
        num_attention_heads=15,
        num_key_value_heads=5,
        head_dim=64,
        max_context=2048,
    ),
    BaseLMInferenceCandidate(
        display_name="Qwen3 0.6B Base",
        model_id="Qwen/Qwen3-0.6B-Base",
        vocab_size=151_936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_context=32_768,
    ),
]


# ---------------------------------------------------------------------------
# M-series memory bandwidth (GB/s). Inference on Apple Silicon is bandwidth-
# bound for any model larger than a few hundred MB, so peak tok/s scales
# tightly with this number. Source: Apple chip spec sheets.
# ---------------------------------------------------------------------------

CHIP_BANDWIDTH_GBPS: dict[str, float] = {
    "Apple M1":         68.0,
    "Apple M1 Pro":    200.0,
    "Apple M1 Max":    400.0,
    "Apple M1 Ultra":  800.0,
    "Apple M2":        100.0,
    "Apple M2 Pro":    200.0,
    "Apple M2 Max":    400.0,
    "Apple M2 Ultra":  800.0,
    "Apple M3":        100.0,
    "Apple M3 Pro":    150.0,
    "Apple M3 Max":    400.0,
    "Apple M4":        120.0,
    "Apple M4 Pro":    273.0,
    "Apple M4 Max":    546.0,
}


# The raw training probe measures the MPS allocator for a minimal synthetic
# training step. A real notebook run also carries Jupyter state, tokenizer
# artifacts, token buffers, plots, history, and allocator cache. Treat this as
# a deliberately conservative translation from probe memory to practical
# notebook memory.
TRAINING_OS_OVERHEAD_GB = 4.0
TRAINING_NOTEBOOK_MEMORY_FACTOR = 3.0
TRAINING_LIKELY_NOT_POSSIBLE_MULTIPLE = 2.0

# HF/PyTorch inference has a large fixed overhead compared with the raw model
# tensors: Python, Transformers, tokenizer state, Jupyter, MPS allocator cache,
# and temporary buffers. We report both raw model memory and this conservative
# in-process notebook estimate.
BASELM_INFERENCE_OS_OVERHEAD_GB = 4.0
BASELM_INFERENCE_RUNTIME_OVERHEAD_GB = 4.0
BASELM_INFERENCE_MEMORY_FACTOR = 2.0
BASELM_INFERENCE_TIGHT_MULTIPLE = 2.0

# Modules 13/14 currently do full-model SFT/DPO, not LoRA. The estimate below
# reflects the notebook defaults for HF BaseLM artifacts.
BASELM_SFT_BATCH_SIZE = 4
BASELM_DPO_BATCH_SIZE = 1
BASELM_POSTTRAIN_CONTEXT_LENGTH = 128
BASELM_POSTTRAIN_OS_OVERHEAD_GB = 4.0
BASELM_POSTTRAIN_RUNTIME_OVERHEAD_GB = 4.0
BASELM_POSTTRAIN_MEMORY_FACTOR = 1.3
BASELM_POSTTRAIN_TIGHT_MULTIPLE = 2.0
BASELM_ACTIVATION_SAVED_TENSOR_FACTOR = 8.0
BASELM_LOGITS_TEMP_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"


def color_for_verdict(v: str) -> str:
    return {
        "ok":       GREEN,
        "tight":    YELLOW,
        "fail":     RED,
        "skip":     DIM,
    }.get(v, "")


def header(text: str) -> None:
    print()
    print(f"{BOLD}{BLUE}━━━ {text} ━━━{RESET}")


# ---------------------------------------------------------------------------
# System detection
# ---------------------------------------------------------------------------

@dataclass
class SystemInfo:
    platform: str
    chip: str
    physical_memory_gb: float
    bandwidth_gbps: float | None
    mps_available: bool
    python_version: str
    torch_version: str | None
    gpu_cores: int | None = None


def _detect_apple_gpu_cores() -> int | None:
    """Parse `Total Number of Cores:` from system_profiler's GPU section."""
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Total Number of Cores:"):
            try:
                return int(stripped.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def detect_system() -> SystemInfo:
    sys_platform = platform.system()
    chip = "unknown"
    mem_gb = 0.0
    bandwidth = None
    gpu_cores = None

    if sys_platform == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass
        try:
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip())
            mem_gb = mem_bytes / (1024 ** 3)
        except Exception:
            pass
        bandwidth = CHIP_BANDWIDTH_GBPS.get(chip)
        gpu_cores = _detect_apple_gpu_cores()

    try:
        import torch
        torch_v = torch.__version__
        mps = torch.backends.mps.is_available()
    except ImportError:
        torch_v = None
        mps = False

    return SystemInfo(
        platform=sys_platform,
        chip=chip,
        physical_memory_gb=mem_gb,
        bandwidth_gbps=bandwidth,
        mps_available=mps,
        python_version=sys.version.split()[0],
        torch_version=torch_v,
        gpu_cores=gpu_cores,
    )


# ---------------------------------------------------------------------------
# Training/inference probe — torch.nn primitives
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    shape_name: str
    param_count_m: float
    status: str           # "ok", "tight", "fail", "skip"
    peak_memory_gb: float | None
    tokens_per_sec: float | None
    estimated_notebook_memory_gb: float | None = None
    practical_memory_gb: float | None = None
    batch_size: int = 0
    context_length: int = 0
    steps_run: int = 0
    note: str = ""


def build_torch_transformer(shape: ModelShape, device: str):
    """Build a decoder-only transformer with shape-matched dimensions
    using torch.nn primitives. Diagnostic-only; not pedagogical."""
    import torch
    import torch.nn as nn

    class TestTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(shape.vocab_size, shape.embedding_dim)
            self.pos = nn.Embedding(shape.max_seq_len, shape.embedding_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=shape.embedding_dim,
                nhead=shape.num_heads,
                dim_feedforward=shape.hidden_dim,
                batch_first=True,
                norm_first=True,
                activation="gelu",
                dropout=0.0,
            )
            self.blocks = nn.TransformerEncoder(
                layer, num_layers=shape.num_layers, enable_nested_tensor=False
            )
            self.ln_f = nn.LayerNorm(shape.embedding_dim)
            self.head_bias = nn.Parameter(torch.zeros(shape.vocab_size))

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            B, T = ids.shape
            pos_ids = torch.arange(T, device=ids.device)
            x = self.embed(ids) + self.pos(pos_ids)
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
            x = self.blocks(x, mask=mask, is_causal=True)
            x = self.ln_f(x)
            return x @ self.embed.weight.T + self.head_bias

    return TestTransformer().to(device)


def _mps_allocated_gb() -> float | None:
    import torch
    for attr in ("driver_allocated_memory", "current_allocated_memory"):
        fn = getattr(torch.mps, attr, None)
        if callable(fn):
            return fn() / (1024 ** 3)
    return None


def probe_training(shape: ModelShape, device: str, n_steps: int = 25,
                   progress_cb=None) -> ProbeResult:
    """Run training steps and report observed peak memory + tail throughput.

    Memory: we sample `driver_allocated_memory` after every step and keep
    the running max. This is still not a true within-step peak (transient
    backward spikes can be missed), but the repeated steps usually expose the
    allocator footprint that matters for fit. Use `--steps 100` when you want
    a slower, more stable high-water measurement.

    Throughput: timed over the last 20% of steps (steady state) rather
    than from step 1, so allocator warmup and lazy init don't depress
    the number.
    """
    import torch

    params_m = shape.param_count() / 1e6
    B, T = shape.batch_size, shape.context_length

    try:
        if device == "mps":
            torch.mps.empty_cache()

        model = build_torch_transformer(shape, device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Single warmup pass so optimizer state is allocated before we
        # start sampling memory and timing.
        ids = torch.randint(0, shape.vocab_size, (B, T), device=device)
        targets = torch.randint(0, shape.vocab_size, (B, T), device=device)
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, shape.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        opt.step()
        opt.zero_grad()
        if device == "mps":
            torch.mps.synchronize()

        peak_gb = _mps_allocated_gb() if device == "mps" else None
        timing_starts_at = max(1, int(n_steps * 0.8))
        timed_elapsed = 0.0
        timed_steps = 0

        for step in range(n_steps):
            # Re-randomize each step so the allocator sees varied input
            # patterns, matching a real DataLoader more closely than a
            # single reused tensor would.
            ids = torch.randint(0, shape.vocab_size, (B, T), device=device)
            targets = torch.randint(0, shape.vocab_size, (B, T), device=device)

            if step == timing_starts_at and device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter() if step >= timing_starts_at else None

            logits = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, shape.vocab_size), targets.reshape(-1)
            )
            loss.backward()
            opt.step()
            opt.zero_grad()

            if step >= timing_starts_at:
                if device == "mps":
                    torch.mps.synchronize()
                timed_elapsed += time.perf_counter() - t0
                timed_steps += 1

            if device == "mps":
                sample = _mps_allocated_gb()
                if sample is not None and (peak_gb is None or sample > peak_gb):
                    peak_gb = sample

            if progress_cb is not None:
                progress_cb(step + 1, n_steps)

        tok_per_sec = None
        if timed_steps > 0 and timed_elapsed > 0:
            tok_per_sec = (timed_steps * B * T) / timed_elapsed

        return ProbeResult(
            shape_name=shape.name,
            param_count_m=params_m,
            status="ok",
            peak_memory_gb=peak_gb,
            tokens_per_sec=tok_per_sec,
            batch_size=B,
            context_length=T,
            steps_run=n_steps,
        )

    except (RuntimeError, MemoryError) as e:
        msg = str(e)
        is_oom = "out of memory" in msg.lower() or "MPS backend out of memory" in msg
        return ProbeResult(
            shape_name=shape.name,
            param_count_m=params_m,
            status="fail",
            peak_memory_gb=None,
            tokens_per_sec=None,
            batch_size=B,
            context_length=T,
            note="OOM" if is_oom else msg[:120],
        )


def find_shape(name: str) -> ModelShape:
    """Return a canonical model shape by name."""
    for shape in CANONICAL_SHAPES:
        if shape.name == name:
            return shape
    raise KeyError(f"unknown canonical shape: {name}")


def parse_batch_list(text: str) -> list[int]:
    """Parse a comma-separated positive-int batch list.

    Duplicates are removed while preserving order so ``8,16,16,32`` is
    treated as ``8,16,32``.
    """
    batches: list[int] = []
    seen: set[int] = set()
    for raw in text.split(","):
        part = raw.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid batch size {part!r}") from exc
        if value <= 0:
            raise ValueError(f"batch sizes must be positive, got {value}")
        if value not in seen:
            batches.append(value)
            seen.add(value)
    if not batches:
        raise ValueError("at least one batch size is required")
    return batches


def classify_training_headroom(result: ProbeResult, sysinfo: SystemInfo) -> ProbeResult:
    """Mark training feasibility using notebook-adjusted memory estimates.

    The raw probe is MPS allocator memory for a minimal synthetic training
    step. Practical notebook runs tend to use much more unified memory because
    Jupyter, token buffers, Python objects, plots, histories, and allocator
    caches stay resident. We therefore:

      1. reserve 4 GB for macOS and background processes,
      2. multiply the probe memory by 3x,
      3. mark yellow/tight once the estimate exceeds practical memory,
      4. mark red/fail once it exceeds 2x practical memory.
    """
    if result.status != "ok" or result.peak_memory_gb is None:
        return result
    practical_gb = max(0.0, sysinfo.physical_memory_gb - TRAINING_OS_OVERHEAD_GB)
    estimated_gb = result.peak_memory_gb * TRAINING_NOTEBOOK_MEMORY_FACTOR
    if practical_gb <= 0:
        return replace(
            result,
            status="fail",
            estimated_notebook_memory_gb=estimated_gb,
            practical_memory_gb=practical_gb,
            note=_append_note(result.note, "no practical memory after OS reserve"),
        )
    if estimated_gb <= practical_gb:
        return replace(
            result,
            estimated_notebook_memory_gb=estimated_gb,
            practical_memory_gb=practical_gb,
        )
    fail_cutoff_gb = practical_gb * TRAINING_LIKELY_NOT_POSSIBLE_MULTIPLE
    if estimated_gb <= fail_cutoff_gb:
        return replace(
            result,
            status="tight",
            estimated_notebook_memory_gb=estimated_gb,
            practical_memory_gb=practical_gb,
            note=_append_note(
                result.note,
                "3x notebook estimate exceeds practical memory",
            ),
        )
    return replace(
        result,
        status="fail",
        estimated_notebook_memory_gb=estimated_gb,
        practical_memory_gb=practical_gb,
        note=_append_note(
            result.note,
            "3x notebook estimate exceeds 2x practical memory",
        ),
    )


def _append_note(note: str, extra: str) -> str:
    return extra if not note else f"{note}; {extra}"


def probe_batch_sweep(
    shape: ModelShape,
    batches: list[int],
    device: str,
    sysinfo: SystemInfo,
    *,
    n_steps: int = 20,
    stop_after_fail: bool = True,
    progress_cb_factory=None,
    result_cb=None,
) -> list[ProbeResult]:
    """Sweep batch sizes for one shape.

    This reuses the training probe but with fewer steps by default. A sweep is
    meant to locate the fit/throughput frontier; it does not need the same
    allocator high-water confidence as a longer benchmark run.
    """
    results: list[ProbeResult] = []
    stopped_after: int | None = None
    for batch_size in batches:
        if stopped_after is not None:
            result = ProbeResult(
                shape_name=shape.name,
                param_count_m=shape.param_count() / 1e6,
                status="skip",
                peak_memory_gb=None,
                tokens_per_sec=None,
                batch_size=batch_size,
                context_length=shape.context_length,
                steps_run=0,
                note=f"skipped after B={stopped_after} failed",
            )
            results.append(result)
            if result_cb is not None:
                result_cb(result)
            continue

        progress_cb = None
        if progress_cb_factory is not None:
            progress_cb = progress_cb_factory(batch_size)

        swept_shape = replace(shape, batch_size=batch_size)
        result = probe_training(
            swept_shape,
            device,
            n_steps=n_steps,
            progress_cb=progress_cb,
        )
        result = classify_training_headroom(result, sysinfo)
        results.append(result)
        if result_cb is not None:
            result_cb(result)
        if stop_after_fail and result.status == "fail":
            stopped_after = batch_size
    return results


def probe_inference(shape: ModelShape, device: str, progress_cb=None) -> ProbeResult:
    """Forward-only autoregressive generation timing.

    Not a true KV-cached generation loop — uses re-prefill to keep the probe
    in pure torch.nn without a custom cache. That's pessimistic but
    sufficient for relative comparisons across model sizes.
    """
    import torch

    params_m = shape.param_count() / 1e6
    try:
        if device == "mps":
            torch.mps.empty_cache()

        model = build_torch_transformer(shape, device).eval()

        prompt_len = 32
        new_tokens = 32
        max_new_tokens = max(0, min(new_tokens, shape.max_seq_len - prompt_len))
        ids = torch.randint(0, shape.vocab_size, (1, prompt_len), device=device)

        with torch.no_grad():
            _ = model(ids)  # warmup
            if device == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            progress_elapsed = 0.0
            generated = 0
            for _ in range(max_new_tokens):
                logits = model(ids)
                next_id = logits[:, -1:, :].argmax(dim=-1)
                ids = torch.cat([ids, next_id], dim=1)
                generated += 1
                if progress_cb is not None:
                    progress_t0 = time.perf_counter()
                    progress_cb(generated, max_new_tokens)
                    progress_elapsed += time.perf_counter() - progress_t0
                if ids.shape[1] >= shape.max_seq_len:
                    break
            if device == "mps":
                torch.mps.synchronize()
            elapsed = max(0.0, time.perf_counter() - t0 - progress_elapsed)

        tok_per_sec = generated / elapsed if generated > 0 and elapsed > 0 else None
        return ProbeResult(
            shape_name=shape.name,
            param_count_m=params_m,
            status="ok" if tok_per_sec is not None else "skip",
            peak_memory_gb=None,
            tokens_per_sec=tok_per_sec,
            steps_run=generated,
            note="" if tok_per_sec is not None else "no room for generated tokens",
        )

    except (RuntimeError, MemoryError) as e:
        msg = str(e)
        is_oom = "out of memory" in msg.lower()
        return ProbeResult(
            shape_name=shape.name,
            param_count_m=params_m,
            status="fail",
            peak_memory_gb=None,
            tokens_per_sec=None,
            note="OOM" if is_oom else msg[:120],
        )


# ---------------------------------------------------------------------------
# GGUF header parsing — pre-download model metadata
#
# Spec: github.com/ggerganov/ggml/blob/master/docs/gguf.md
#
# We only read the fixed header + KV metadata block. Tensor info and
# weights live further into the file; we never touch them.
# ---------------------------------------------------------------------------

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

# struct formats for scalar GGUF types
_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8:   ("<B", 1),
    GGUF_TYPE_INT8:    ("<b", 1),
    GGUF_TYPE_UINT16:  ("<H", 2),
    GGUF_TYPE_INT16:   ("<h", 2),
    GGUF_TYPE_UINT32:  ("<I", 4),
    GGUF_TYPE_INT32:   ("<i", 4),
    GGUF_TYPE_FLOAT32: ("<f", 4),
    GGUF_TYPE_BOOL:    ("<?", 1),
    GGUF_TYPE_UINT64:  ("<Q", 8),
    GGUF_TYPE_INT64:   ("<q", 8),
    GGUF_TYPE_FLOAT64: ("<d", 8),
}


class _GGUFReader:
    """Tiny GGUF metadata-block parser. Consumes a bytes buffer sequentially."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def read(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise EOFError(f"GGUF header truncated at byte {self.pos}+{n}")
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")

    def scalar(self, type_id: int) -> Any:
        fmt, size = _SCALAR_FORMATS[type_id]
        return struct.unpack(fmt, self.read(size))[0]

    def skip_string(self) -> None:
        n = self.u64()
        self.pos += n
        if self.pos > len(self.buf):
            raise EOFError("array string crosses truncation boundary")

    def value(self, type_id: int, _skip_large: bool = True) -> Any:
        if type_id == GGUF_TYPE_STRING:
            return self.string()
        if type_id == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            # Tokenizer arrays (e.g. `tokenizer.ggml.tokens`, ~128K strings)
            # sit in metadata and would balloon parse cost / memory. Skip
            # over the bytes instead of materializing the list.
            if _skip_large and count > 1024:
                if elem_type == GGUF_TYPE_STRING:
                    for _ in range(count):
                        self.skip_string()
                    return f"<skipped string array len={count}>"
                if elem_type in _SCALAR_FORMATS:
                    _, size = _SCALAR_FORMATS[elem_type]
                    self.pos += size * count
                    if self.pos > len(self.buf):
                        raise EOFError("scalar array crosses truncation boundary")
                    return f"<skipped scalar array len={count}>"
            return [self.value(elem_type, _skip_large=_skip_large)
                    for _ in range(count)]
        return self.scalar(type_id)


def fetch_gguf_metadata(repo: str, filename: str, max_bytes: int = 524288,
                        timeout: float = 8.0) -> dict[str, Any]:
    """Range-fetch the GGUF header from Hugging Face and return its KV table.

    `max_bytes` (default 512 KB) is enough for the metadata block in every
    common model — the tensor info that follows is bigger but we don't need it.
    """
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={
        "Range": f"bytes=0-{max_bytes - 1}",
        "User-Agent": "g2c-sysprobe/0.1",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = resp.read()

    r = _GGUFReader(buf)
    magic = r.read(4)
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file (magic={magic!r})")
    version = r.u32()
    _tensor_count = r.u64()
    kv_count = r.u64()

    metadata: dict[str, Any] = {"_gguf_version": version}
    for _ in range(kv_count):
        try:
            key = r.string()
            type_id = r.u32()
            metadata[key] = r.value(type_id)
        except EOFError:
            # Hit our truncation boundary before exhausting the KV block.
            # Partial metadata is still useful.
            metadata["_truncated"] = True
            break
    return metadata


def gguf_quant_bpw(metadata: dict[str, Any]) -> float | None:
    """Bits per weight for the GGUF file's main tensor type.

    Picks the dominant quantization scheme from `general.file_type`. Numbers
    are llama.cpp-spec averages, not exact for every tensor in the file.
    """
    # See ggml's ftype enum.
    ftype = metadata.get("general.file_type")
    if ftype is None:
        return None
    table = {
        0: 32.0,    # ALL_F32
        1: 16.0,    # MOSTLY_F16
        2: 4.5,     # MOSTLY_Q4_0
        3: 4.5,     # MOSTLY_Q4_1
        7: 8.5,     # MOSTLY_Q8_0
        8: 5.5,     # MOSTLY_Q5_0
        9: 5.5,     # MOSTLY_Q5_1
        10: 2.6,    # MOSTLY_Q2_K
        11: 3.4,    # MOSTLY_Q3_K_S
        12: 3.9,    # MOSTLY_Q3_K_M
        13: 4.5,    # MOSTLY_Q3_K_L
        14: 4.6,    # MOSTLY_Q4_K_S
        15: 4.8,    # MOSTLY_Q4_K_M
        16: 5.5,    # MOSTLY_Q5_K_S
        17: 5.7,    # MOSTLY_Q5_K_M
        18: 6.6,    # MOSTLY_Q6_K
        30: 5.0,    # MOSTLY_IQ4_NL
    }
    return table.get(ftype)


def gguf_param_count(metadata: dict[str, Any]) -> int | None:
    """Best-effort parameter count from GGUF metadata.

    Tries, in order:
      1. `general.parameter_count` — direct, when present.
      2. `general.size_label` — short string like "1B", "8B", "14B".
      3. Computed from architecture keys: embedding * vocab + per-block
         (attn + GQA-aware FFN, SwiGLU = 3 matrices).

    Returns `None` if none of these are available.
    """
    val = metadata.get("general.parameter_count")
    if isinstance(val, int):
        return val

    label = metadata.get("general.size_label")
    if isinstance(label, str):
        s = label.strip().upper()
        try:
            if s.endswith("B"):
                return int(float(s[:-1]) * 1e9)
            if s.endswith("M"):
                return int(float(s[:-1]) * 1e6)
        except ValueError:
            pass

    arch = metadata.get("general.architecture")
    if isinstance(arch, str):
        prefix = arch
        try:
            V    = metadata[f"{prefix}.vocab_size"]
            D    = metadata[f"{prefix}.embedding_length"]
            N    = metadata[f"{prefix}.block_count"]
            H    = metadata[f"{prefix}.feed_forward_length"]
            n_h  = metadata[f"{prefix}.attention.head_count"]
            n_kv = metadata.get(f"{prefix}.attention.head_count_kv", n_h)
            kd   = metadata.get(f"{prefix}.attention.key_length", D // n_h)
            vd   = metadata.get(f"{prefix}.attention.value_length", D // n_h)
        except KeyError:
            return None
        # Tied embeddings (typical for modern Llama/Qwen small models); we
        # don't add a separate LM head. Off by V*D in the worst case.
        embed = V * D
        attn  = D * (n_h * kd) + D * (n_kv * kd) + D * (n_kv * vd) + (n_h * kd) * D
        ffn   = 3 * D * H   # SwiGLU
        per_block = attn + ffn
        return embed + N * per_block
    return None


# ---------------------------------------------------------------------------
# User-supplied candidate builders — resolve a single HF model id or
# GGUF spec at runtime so users can probe models outside the curated lists.
# ---------------------------------------------------------------------------

def fetch_hf_config(model_id: str, timeout: float = 8.0) -> dict[str, Any]:
    """Fetch a Hugging Face repo's `config.json` as a parsed dict."""
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={
        "User-Agent": "g2c-sysprobe/0.1",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def baselm_candidate_from_hf(model_id: str) -> BaseLMInferenceCandidate:
    """Build a BaseLMInferenceCandidate from a HF model id."""
    cfg = fetch_hf_config(model_id)
    try:
        vocab_size = int(cfg["vocab_size"])
        hidden_size = int(cfg["hidden_size"])
        num_hidden_layers = int(cfg["num_hidden_layers"])
        num_attention_heads = int(cfg["num_attention_heads"])
    except KeyError as exc:
        raise ValueError(
            f"{model_id} config.json missing required field {exc.args[0]!r}"
        ) from exc

    num_key_value_heads = int(cfg.get("num_key_value_heads", num_attention_heads))
    intermediate_size = int(cfg.get("intermediate_size", 4 * hidden_size))
    head_dim = int(cfg.get("head_dim", hidden_size // num_attention_heads))
    max_context = int(cfg.get("max_position_embeddings", 2048))

    return BaseLMInferenceCandidate(
        display_name=model_id.rsplit("/", 1)[-1],
        model_id=model_id,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_context=max_context,
    )


def prodlm_arch_from_gguf(metadata: dict[str, Any]) -> dict[str, int]:
    """Pull `n_layers`, `n_kv_heads`, `head_dim` from a parsed GGUF header."""
    arch = metadata.get("general.architecture")
    if not isinstance(arch, str):
        raise ValueError("GGUF header missing general.architecture")
    try:
        n_layers = int(metadata[f"{arch}.block_count"])
        n_h = int(metadata[f"{arch}.attention.head_count"])
        D = int(metadata[f"{arch}.embedding_length"])
    except KeyError as exc:
        raise ValueError(
            f"GGUF header missing required key {exc.args[0]!r}"
        ) from exc
    n_kv = int(metadata.get(f"{arch}.attention.head_count_kv", n_h))
    head_dim = int(metadata.get(f"{arch}.attention.key_length", D // n_h))
    return {"n_layers": n_layers, "n_kv_heads": n_kv, "head_dim": head_dim}


def prodlm_candidate_from_spec(spec: str) -> ProdLMCandidate:
    """Build a ProdLMCandidate from either a curated Ollama tag or an
    `HF_REPO:GGUF_FILENAME` spec.

    Curated tags (e.g. `qwen2.5:7b`) are matched against the built-in list and
    reused as-is. Anything containing a `/` is treated as an HF repo; arch
    fields and a nominal parameter count are then read from the GGUF header.
    """
    for cand in PRODLM_CANDIDATES:
        if cand.ollama_tag == spec:
            return cand

    repo, sep, filename = spec.rpartition(":")
    if not sep or not repo or not filename or "/" not in repo:
        curated = ", ".join(c.ollama_tag for c in PRODLM_CANDIDATES)
        raise ValueError(
            f"{spec!r} is not a curated Ollama tag and does not look like "
            f"HF_REPO:GGUF_FILENAME (the repo part must contain a `/`, e.g. "
            f"bartowski/Llama-3.2-1B-Instruct-GGUF:"
            f"Llama-3.2-1B-Instruct-Q4_K_M.gguf). Curated tags: {curated}"
        )
    try:
        metadata = fetch_gguf_metadata(repo, filename)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError) as exc:
        raise ValueError(
            f"could not fetch GGUF header for {repo}/{filename}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    arch = prodlm_arch_from_gguf(metadata)
    params = gguf_param_count(metadata)
    short_tag = filename.removesuffix(".gguf") if filename.endswith(".gguf") else filename
    return ProdLMCandidate(
        display_name=f"{filename} [user]",
        ollama_tag=short_tag,
        hf_repo=repo,
        gguf_filename=filename,
        n_layers=arch["n_layers"],
        n_kv_heads=arch["n_kv_heads"],
        head_dim=arch["head_dim"],
        nominal_params_b=(params / 1e9) if params else 0.0,
    )


# ---------------------------------------------------------------------------
# ProdLM fit + throughput estimate
# ---------------------------------------------------------------------------

@dataclass
class ProdLMResult:
    display_name: str
    ollama_tag: str
    status: str           # "ok", "tight", "fail", "skip"
    params_b: float | None
    weights_gb: float | None
    kv_gb_at_ctx: float | None
    total_gb: float | None
    expected_tok_per_sec: float | None
    throughput_tier: str
    note: str = ""


@dataclass
class BaseLMInferenceResult:
    display_name: str
    model_id: str
    status: str
    params_b: float
    context_length: int
    model_memory_fp16_gb: float
    notebook_memory_fp16_gb: float
    notebook_memory_fp32_gb: float
    practical_memory_gb: float
    note: str = ""


@dataclass
class BaseLMPostTrainingResult:
    display_name: str
    model_id: str
    stage: str
    status: str
    params_b: float
    batch_size: int
    context_length: int
    notebook_memory_fp32_gb: float
    notebook_memory_fp16_gb: float
    practical_memory_gb: float
    note: str = ""


def estimate_kv_cache_gb(cand: ProdLMCandidate, context_length: int,
                         bytes_per_element: int = 2) -> float:
    """KV cache memory at a given context. K and V each are
    (n_layers, n_kv_heads, head_dim) per token. fp16 = 2 bytes."""
    per_token = 2 * cand.n_layers * cand.n_kv_heads * cand.head_dim * bytes_per_element
    return (per_token * context_length) / (1024 ** 3)


def estimate_decoder_kv_cache_gb(
    *,
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    context_length: int,
    bytes_per_element: int = 2,
) -> float:
    """KV cache memory for a decoder-only model at one batch element."""
    per_token = (
        2
        * num_hidden_layers
        * num_key_value_heads
        * head_dim
        * bytes_per_element
    )
    return (per_token * context_length) / (1024 ** 3)


def _memory_status(
    estimated_gb: float,
    practical_gb: float,
    *,
    tight_multiple: float,
) -> str:
    if practical_gb <= 0:
        return "fail"
    if estimated_gb <= practical_gb:
        return "ok"
    if estimated_gb <= practical_gb * tight_multiple:
        return "tight"
    return "fail"


def throughput_tier(tok_per_sec: float) -> str:
    if tok_per_sec >= 30:
        return "interactive"
    if tok_per_sec >= 10:
        return "usable"
    if tok_per_sec >= 3:
        return "slow"
    return "painful"


def estimate_baselm_inference(
    cand: BaseLMInferenceCandidate,
    sysinfo: SystemInfo,
    context_length: int,
) -> BaseLMInferenceResult:
    """Estimate in-process HF/PyTorch inference memory before download."""
    effective_context = min(context_length, cand.max_context)
    params = cand.param_count()

    weights_fp16_gb = params * 2 / (1024 ** 3)
    kv_fp16_gb = estimate_decoder_kv_cache_gb(
        num_hidden_layers=cand.num_hidden_layers,
        num_key_value_heads=cand.num_key_value_heads,
        head_dim=cand.head_dim,
        context_length=effective_context,
        bytes_per_element=2,
    )
    model_fp16_gb = weights_fp16_gb + kv_fp16_gb
    notebook_fp16_gb = (
        model_fp16_gb * BASELM_INFERENCE_MEMORY_FACTOR
        + BASELM_INFERENCE_RUNTIME_OVERHEAD_GB
    )

    weights_fp32_gb = params * 4 / (1024 ** 3)
    kv_fp32_gb = estimate_decoder_kv_cache_gb(
        num_hidden_layers=cand.num_hidden_layers,
        num_key_value_heads=cand.num_key_value_heads,
        head_dim=cand.head_dim,
        context_length=effective_context,
        bytes_per_element=4,
    )
    model_fp32_gb = weights_fp32_gb + kv_fp32_gb
    notebook_fp32_gb = (
        model_fp32_gb * BASELM_INFERENCE_MEMORY_FACTOR
        + BASELM_INFERENCE_RUNTIME_OVERHEAD_GB
    )

    practical_gb = max(0.0, sysinfo.physical_memory_gb - BASELM_INFERENCE_OS_OVERHEAD_GB)
    if practical_gb <= 0:
        status = "fail"
        note = "no practical memory after OS reserve"
    elif notebook_fp16_gb <= practical_gb:
        status = "ok"
        note = ""
    elif notebook_fp16_gb <= practical_gb * BASELM_INFERENCE_TIGHT_MULTIPLE:
        status = "tight"
        note = "fp16 notebook estimate exceeds practical memory"
    else:
        status = "fail"
        note = "fp16 notebook estimate exceeds 2x practical memory"

    if effective_context != context_length:
        note = _append_note(
            note,
            f"context clamped from {context_length:,} to model max {effective_context:,}",
        )

    return BaseLMInferenceResult(
        display_name=cand.display_name,
        model_id=cand.model_id,
        status=status,
        params_b=params / 1e9,
        context_length=effective_context,
        model_memory_fp16_gb=model_fp16_gb,
        notebook_memory_fp16_gb=notebook_fp16_gb,
        notebook_memory_fp32_gb=notebook_fp32_gb,
        practical_memory_gb=practical_gb,
        note=note,
    )


def estimate_baselm_posttraining(
    cand: BaseLMInferenceCandidate,
    sysinfo: SystemInfo,
    *,
    stage: str,
) -> BaseLMPostTrainingResult:
    """Estimate full-model SFT/DPO memory for the course notebooks.

    Assumes the Module 13/14 path: full trainable policy, custom AdamW with
    two optimizer-state tensors, and one extra frozen model copy (base
    comparison model for SFT, reference model for DPO).
    """
    normalized_stage = stage.upper()
    if normalized_stage == "SFT":
        batch_size = BASELM_SFT_BATCH_SIZE
        retained_policy_graphs = 1
    elif normalized_stage == "DPO":
        batch_size = BASELM_DPO_BATCH_SIZE
        # DPO retains chosen and rejected policy graphs at once; reference
        # forwards run under no_grad and mostly add temporary logits.
        retained_policy_graphs = 2
    else:
        raise ValueError(f"unknown BaseLM post-training stage: {stage!r}")

    context_length = min(BASELM_POSTTRAIN_CONTEXT_LENGTH, cand.max_context)
    fp32_estimate = _estimate_baselm_posttraining_memory_gb(
        cand,
        batch_size=batch_size,
        context_length=context_length,
        retained_policy_graphs=retained_policy_graphs,
        bytes_per_element=4,
    )
    fp16_estimate = _estimate_baselm_posttraining_memory_gb(
        cand,
        batch_size=batch_size,
        context_length=context_length,
        retained_policy_graphs=retained_policy_graphs,
        bytes_per_element=2,
    )
    practical_gb = max(0.0, sysinfo.physical_memory_gb - BASELM_POSTTRAIN_OS_OVERHEAD_GB)
    status = _memory_status(
        fp32_estimate,
        practical_gb,
        tight_multiple=BASELM_POSTTRAIN_TIGHT_MULTIPLE,
    )
    note = ""
    if practical_gb <= 0:
        note = "no practical memory after OS reserve"
    elif status != "ok" and fp16_estimate <= practical_gb:
        note = "fp16 estimate fits; default BaseLM post-training is fp32"
    elif status == "tight":
        note = "fp32 estimate exceeds practical memory"
    elif status == "fail":
        note = "fp32 estimate exceeds 2x practical memory"

    return BaseLMPostTrainingResult(
        display_name=cand.display_name,
        model_id=cand.model_id,
        stage=normalized_stage,
        status=status,
        params_b=cand.param_count() / 1e9,
        batch_size=batch_size,
        context_length=context_length,
        notebook_memory_fp32_gb=fp32_estimate,
        notebook_memory_fp16_gb=fp16_estimate,
        practical_memory_gb=practical_gb,
        note=note,
    )


def _estimate_baselm_posttraining_memory_gb(
    cand: BaseLMInferenceCandidate,
    *,
    batch_size: int,
    context_length: int,
    retained_policy_graphs: int,
    bytes_per_element: int,
) -> float:
    params = cand.param_count()
    # trainable policy: weights + gradients + AdamW m + AdamW v = 4x.
    # extra frozen model: base comparison copy for SFT or ref model for DPO.
    parameter_state_gb = params * bytes_per_element * 5 / (1024 ** 3)

    hidden_activation_gb = (
        cand.num_hidden_layers
        * batch_size
        * context_length
        * cand.hidden_size
        * bytes_per_element
        * retained_policy_graphs
        * BASELM_ACTIVATION_SAVED_TENSOR_FACTOR
        / (1024 ** 3)
    )
    policy_logits_gb = (
        batch_size
        * context_length
        * cand.vocab_size
        * bytes_per_element
        * retained_policy_graphs
        * BASELM_LOGITS_TEMP_FACTOR
        / (1024 ** 3)
    )
    # The no-grad base/reference forward still creates temporary logits while
    # the policy graph is resident.
    frozen_temp_logits_gb = (
        batch_size
        * context_length
        * cand.vocab_size
        * bytes_per_element
        / (1024 ** 3)
    )
    raw_training_gb = (
        parameter_state_gb
        + hidden_activation_gb
        + policy_logits_gb
        + frozen_temp_logits_gb
    )
    return (
        raw_training_gb * BASELM_POSTTRAIN_MEMORY_FACTOR
        + BASELM_POSTTRAIN_RUNTIME_OVERHEAD_GB
    )


def probe_prodlm(cand: ProdLMCandidate, sysinfo: SystemInfo,
                 context_length: int = 4096) -> ProdLMResult:
    fallback_note = ""
    metadata: dict[str, Any] | None = None
    try:
        metadata = fetch_gguf_metadata(cand.hf_repo, cand.gguf_filename)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError) as e:
        fallback_note = f"GGUF fetch failed ({type(e).__name__}); using nominal sizing"

    params: int | None = None
    bpw: float | None = None
    if metadata is not None:
        params = gguf_param_count(metadata)
        bpw = gguf_quant_bpw(metadata)

    if params is None:
        params = int(cand.nominal_params_b * 1e9)
        if not fallback_note:
            fallback_note = "param count missing from GGUF; using nominal"
    if bpw is None:
        bpw = 4.8   # Q4_K_M default
        if not fallback_note:
            fallback_note = "quantization missing from GGUF; assuming Q4_K_M"

    weights_gb = (params * bpw) / 8 / (1024 ** 3)
    kv_gb = estimate_kv_cache_gb(cand, context_length)
    overhead_gb = 1.0  # runtime, activations, OS slack
    total_gb = weights_gb + kv_gb + overhead_gb

    # 70% of physical memory is the practical envelope on macOS once the OS,
    # browser, IDE, and Python interpreter are accounted for.
    available_gb = sysinfo.physical_memory_gb * 0.70

    if total_gb <= available_gb * 0.7:
        fit_status = "ok"
    elif total_gb <= available_gb:
        fit_status = "tight"
    else:
        fit_status = "fail"

    expected_tps: float | None = None
    tier = "unknown"
    if sysinfo.bandwidth_gbps is not None and fit_status != "fail":
        # Bandwidth-bound estimate: each new token reads the whole weight
        # block plus the growing KV cache. We use weights only here as the
        # dominant term; the KV reads are a smaller correction.
        # Apply 0.65 efficiency factor (llama.cpp on MPS hits roughly that
        # of theoretical peak in practice).
        expected_tps = (sysinfo.bandwidth_gbps * 0.65) / weights_gb
        tier = throughput_tier(expected_tps)

    return ProdLMResult(
        display_name=cand.display_name,
        ollama_tag=cand.ollama_tag,
        status=fit_status,
        params_b=params / 1e9,
        weights_gb=weights_gb,
        kv_gb_at_ctx=kv_gb,
        total_gb=total_gb,
        expected_tok_per_sec=expected_tps,
        throughput_tier=tier,
        note=fallback_note,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_gb(x: float | None) -> str:
    return f"{x:5.2f} GB" if x is not None else "    n/a"


def fmt_tps(x: float | None) -> str:
    return f"{x:7.1f} tok/s" if x is not None else "       n/a"


def fmt_params_b(x: float | None) -> str:
    return f"{x:6.2f}B" if x is not None else "    n/a"


def fmt_status(s: str) -> str:
    c = color_for_verdict(s)
    return f"{c}{s:^7}{RESET}"


def clear_status_line() -> None:
    print("\033[2K\r", end="", flush=True)


def print_status_line(text: str) -> None:
    print(f"\033[2K\r  {DIM}{text}{RESET}", end="", flush=True)


def print_system(sysinfo: SystemInfo) -> None:
    header("System")
    print(f"  platform        {sysinfo.platform}")
    print(f"  chip            {sysinfo.chip}")
    if sysinfo.gpu_cores is not None:
        print(f"  gpu cores       {sysinfo.gpu_cores}")
    print(f"  unified memory  {sysinfo.physical_memory_gb:.1f} GB")
    print(f"  training budget {max(0.0, sysinfo.physical_memory_gb - TRAINING_OS_OVERHEAD_GB):.1f} GB  "
          f"{DIM}(unified memory - {TRAINING_OS_OVERHEAD_GB:.0f} GB OS reserve){RESET}")
    if sysinfo.bandwidth_gbps is not None:
        print(f"  bandwidth       {sysinfo.bandwidth_gbps:.0f} GB/s  "
              f"{DIM}(specced){RESET}")
    else:
        print(f"  bandwidth       {DIM}unknown chip — throughput estimates "
              f"will be skipped{RESET}")
    print(f"  python          {sysinfo.python_version}")
    print(f"  torch           {sysinfo.torch_version or 'NOT INSTALLED'}")
    print(f"  MPS             {'available' if sysinfo.mps_available else 'unavailable'}")


def print_training_header() -> None:
    header("Training probe (forward + backward + AdamW, on MPS)")
    print(f"  {'model':<14} {'params':>8} {'shape':>11}  {'status':^9} "
          f"{'probe mem':>9}  {'3x est':>9}  {'budget':>9}  {'throughput':>14}  notes")
    print(f"  {'-'*14} {'-'*8} {'-'*11:>11}  {'-'*7:^9} {'-'*9:>9}  "
          f"{'-'*9:>9}  {'-'*9:>9}  {'-'*14:>14}  {'-'*40}")


def print_training_row(r: ProbeResult) -> None:
    shape_str = f"B={r.batch_size},T={r.context_length}"
    print(f"  {r.shape_name:<14} {r.param_count_m:>6.1f}M  "
          f"{shape_str:>11}  {fmt_status(r.status)} "
          f"{fmt_gb(r.peak_memory_gb):>9}  "
          f"{fmt_gb(r.estimated_notebook_memory_gb):>9}  "
          f"{fmt_gb(r.practical_memory_gb):>9}  "
          f"{fmt_tps(r.tokens_per_sec):>14}  {r.note}")


def print_training_footer(results: list[ProbeResult]) -> None:
    print(f"\n  {DIM}peak mem is the max of `mps.driver_allocated_memory` "
          f"sampled after each of {results[0].steps_run if results else 0} "
          f"training steps. 3x est = peak mem × "
          f"{TRAINING_NOTEBOOK_MEMORY_FACTOR:.0f}, a conservative notebook "
          f"estimate for Jupyter state, token buffers, plots, histories, and "
          f"MPS cache. Budget = unified memory - "
          f"{TRAINING_OS_OVERHEAD_GB:.0f} GB OS reserve; yellow means the "
          f"estimate exceeds that budget, red means it exceeds "
          f"{TRAINING_LIKELY_NOT_POSSIBLE_MULTIPLE:.0f}× that budget.{RESET}")


def print_training(results: list[ProbeResult]) -> None:
    print_training_header()
    for r in results:
        print_training_row(r)
    print_training_footer(results)


def print_inference_header() -> None:
    header("Inference probe (forward-only, no KV cache)")
    print(f"  {'model':<14} {'params':>8} {'status':^9} {'throughput':>14}  notes")
    print(f"  {'-'*14} {'-'*8} {'-'*7:^9} {'-'*14:>14}  {'-'*40}")


def print_inference_row(r: ProbeResult) -> None:
    print(f"  {r.shape_name:<14} {r.param_count_m:>6.1f}M  "
          f"{fmt_status(r.status)} {fmt_tps(r.tokens_per_sec):>14}  {r.note}")


def print_inference(results: list[ProbeResult]) -> None:
    print_inference_header()
    for r in results:
        print_inference_row(r)


def print_baselm_inference_header() -> None:
    header("BaseLM inference estimate (Hugging Face/PyTorch, no download)")
    print(f"  {'tag':<28} {'params':>8} {'ctx':>7} {'status':^9} "
          f"{'model fp16':>10}  {'notebook fp16':>13}  "
          f"{'notebook fp32':>13}  {'budget':>9}  notes")
    print(f"  {'-'*28} {'-'*8} {'-'*7:>7} {'-'*7:^9} "
          f"{'-'*10:>10}  {'-'*13:>13}  {'-'*13:>13}  "
          f"{'-'*9:>9}  {'-'*40}")


def print_baselm_inference_row(r: BaseLMInferenceResult) -> None:
    print(f"  {r.model_id:<28} {r.params_b:>6.2f}B  "
          f"{r.context_length:>7,} {fmt_status(r.status)} "
          f"{fmt_gb(r.model_memory_fp16_gb):>10}  "
          f"{fmt_gb(r.notebook_memory_fp16_gb):>13}  "
          f"{fmt_gb(r.notebook_memory_fp32_gb):>13}  "
          f"{fmt_gb(r.practical_memory_gb):>9}  {r.note}")


def print_baselm_inference_footer() -> None:
    print(f"\n  {DIM}model fp16 = weights + KV cache at the shown context. "
          f"notebook fp16/fp32 = model memory × "
          f"{BASELM_INFERENCE_MEMORY_FACTOR:.0f} + "
          f"{BASELM_INFERENCE_RUNTIME_OVERHEAD_GB:.0f} GB for "
          f"Transformers, tokenizer state, Jupyter, temporary tensors, and "
          f"MPS cache. Budget = unified memory - "
          f"{BASELM_INFERENCE_OS_OVERHEAD_GB:.0f} GB OS reserve; status is "
          f"based on the fp16 notebook estimate.{RESET}")


def print_baselm_inference(results: list[BaseLMInferenceResult]) -> None:
    print_baselm_inference_header()
    for r in results:
        print_baselm_inference_row(r)
    print_baselm_inference_footer()


def print_baselm_posttraining_header() -> None:
    header("BaseLM post-training estimate (full SFT/DPO, no download)")
    print(f"  {'tag':<28} {'stage':<4} {'params':>8} {'shape':>9} "
          f"{'status':^9} {'fp32 est':>9}  {'fp16 est':>9}  "
          f"{'budget':>9}  notes")
    print(f"  {'-'*28} {'-'*4:<4} {'-'*8} {'-'*9:>9} "
          f"{'-'*7:^9} {'-'*9:>9}  {'-'*9:>9}  "
          f"{'-'*9:>9}  {'-'*40}")


def print_baselm_posttraining_row(r: BaseLMPostTrainingResult) -> None:
    shape = f"B={r.batch_size},T={r.context_length}"
    print(f"  {r.model_id:<28} {r.stage:<4} {r.params_b:>6.2f}B  "
          f"{shape:>9} {fmt_status(r.status)} "
          f"{fmt_gb(r.notebook_memory_fp32_gb):>9}  "
          f"{fmt_gb(r.notebook_memory_fp16_gb):>9}  "
          f"{fmt_gb(r.practical_memory_gb):>9}  {r.note}")


def print_baselm_posttraining_footer() -> None:
    print(f"\n  {DIM}Estimates match the Module 13/14 full fine-tuning path: "
          f"SFT uses B={BASELM_SFT_BATCH_SIZE},T={BASELM_POSTTRAIN_CONTEXT_LENGTH}; "
          f"DPO uses B={BASELM_DPO_BATCH_SIZE},T={BASELM_POSTTRAIN_CONTEXT_LENGTH} "
          f"and keeps policy + frozen reference models resident. fp32 is the "
          f"default BaseLM artifact dtype. Memory includes trainable weights, "
          f"grads, AdamW m/v state, one frozen model copy, retained activations, "
          f"logit temporaries, and ~{BASELM_POSTTRAIN_RUNTIME_OVERHEAD_GB:.0f} GB "
          f"runtime/Jupyter overhead. Status is based on fp32.{RESET}")


def print_baselm_posttraining(results: list[BaseLMPostTrainingResult]) -> None:
    print_baselm_posttraining_header()
    for r in results:
        print_baselm_posttraining_row(r)
    print_baselm_posttraining_footer()


def print_batch_sweep(results: list[ProbeResult]) -> None:
    if not results:
        return
    model_name = results[0].shape_name
    print_batch_sweep_header(model_name)
    for r in results:
        print_batch_sweep_row(r)
    print_batch_sweep_footer(results)


def print_batch_sweep_header(model_name: str) -> None:
    header(f"Batch sweep ({model_name}, forward + backward + AdamW)")
    print(f"  {'B':>5} {'T':>5} {'tokens/step':>12} {'status':^9} "
          f"{'probe mem':>9}  {'3x est':>9}  {'throughput':>14}  notes")
    print(f"  {'-'*5:>5} {'-'*5:>5} {'-'*12:>12} {'-'*7:^9} "
          f"{'-'*9:>9}  {'-'*9:>9}  {'-'*14:>14}  {'-'*40}")


def print_batch_sweep_row(r: ProbeResult) -> None:
    tokens_per_step = r.batch_size * r.context_length
    print(f"  {r.batch_size:>5} {r.context_length:>5} "
          f"{tokens_per_step:>12,} {fmt_status(r.status)} "
          f"{fmt_gb(r.peak_memory_gb):>9}  "
          f"{fmt_gb(r.estimated_notebook_memory_gb):>9}  "
          f"{fmt_tps(r.tokens_per_sec):>14}  {r.note}")


def print_batch_sweep_footer(results: list[ProbeResult]) -> None:
    viable = [
        r for r in results
        if r.status in ("ok", "tight") and r.tokens_per_sec is not None
    ]
    if viable:
        fastest = max(viable, key=lambda r: r.tokens_per_sec or 0.0)
        b16 = next((r for r in viable if r.batch_size == 16), None)
        print()
        print(f"  fastest measured viable batch: "
              f"{GREEN if fastest.status == 'ok' else YELLOW}"
              f"B={fastest.batch_size}{RESET} "
              f"({fmt_tps(fastest.tokens_per_sec).strip()}, "
              f"{fmt_gb(fastest.peak_memory_gb).strip()})")
        if b16 is not None:
            print(f"  B=16 completed at {16 * b16.context_length:,} "
                  f"tokens/step; compare token budget, not just step count.")
        elif any(r.batch_size == 16 and r.status == "fail" for r in results):
            print(f"  {YELLOW}B=16 failed or exceeded the conservative memory "
                  f"envelope; use the largest viable smaller batch or reduce "
                  f"context before retrying.{RESET}")
    else:
        print(f"\n  {RED}No swept batch completed successfully.{RESET}")

    print(f"\n  {DIM}sweep steps are intentionally shorter than the main probe: "
          f"enough to allocate optimizer state and compare the batch frontier, "
          f"not a final benchmark-quality high-water mark.{RESET}")


def print_prodlm_header() -> None:
    header("ProdLM probe (Ollama candidates, pre-download estimate)")
    print(f"  {'tag':<28} {'params':>8} {'status':^9} {'total':>9}  "
          f"{'est tok/s':>11}  {'tier':<12}")
    print(f"  {'-'*28} {'-'*8} {'-'*7:^9} {'-'*9:>9}  "
          f"{'-'*11:>11}  {'-'*12:<12}")


def print_prodlm_row(r: ProdLMResult) -> None:
    tps = fmt_tps(r.expected_tok_per_sec).strip().replace(" tok/s", "")
    tier_color = {
        "interactive": GREEN,
        "usable":      GREEN,
        "slow":        YELLOW,
        "painful":     RED,
        "unknown":     DIM,
    }.get(r.throughput_tier, "")
    print(f"  {r.ollama_tag:<28} {fmt_params_b(r.params_b):>8} "
          f"{fmt_status(r.status)} "
          f"{fmt_gb(r.total_gb):>9}  {tps:>11}  "
          f"{tier_color}{r.throughput_tier:<12}{RESET}")
    if r.note:
        print(f"    {DIM}↳ {r.note}{RESET}")


def print_prodlm(results: list[ProdLMResult]) -> None:
    print_prodlm_header()
    for r in results:
        print_prodlm_row(r)


def print_recommendation(training: list[ProbeResult],
                         prodlm: list[ProdLMResult]) -> None:
    header("Recommendation")

    if training:
        by_name = {r.shape_name: r for r in training}
        tracks = [
            ("Tiny track",     ["StoryLM-1M", "StoryLM-5M"]),
            ("Standard track", ["StoryLM-30M", "TinyLLM-30M"]),
            ("Stretch track",  ["TinyLLM-100M"]),
        ]
        for track_name, sizes in tracks:
            statuses = [by_name[s].status for s in sizes if s in by_name]
            if not statuses or all(s == "skip" for s in statuses):
                verdict = f"{DIM}skipped{RESET}"
            elif all(s == "ok" for s in statuses):
                verdict = f"{GREEN}available{RESET}"
            elif any(s == "ok" for s in statuses):
                verdict = f"{YELLOW}partial — largest size at edge{RESET}"
            else:
                verdict = f"{RED}not feasible at canonical configs{RESET}"
            print(f"  {track_name:<16}  {verdict}")
    else:
        print(f"  Training tracks   {DIM}skipped{RESET}")

    print()
    if prodlm:
        interactive = [r for r in prodlm if r.throughput_tier == "interactive"
                       and r.status in ("ok", "tight")]
        usable = [r for r in prodlm if r.throughput_tier == "usable"
                  and r.status in ("ok", "tight")]
        if interactive:
            biggest = max(interactive, key=lambda r: r.params_b or 0)
            print(f"  Recommended ProdLM:  {GREEN}{biggest.ollama_tag}{RESET}  "
                  f"({biggest.display_name}, "
                  f"~{biggest.expected_tok_per_sec:.0f} tok/s)")
        elif usable:
            biggest = max(usable, key=lambda r: r.params_b or 0)
            print(f"  Recommended ProdLM:  {YELLOW}{biggest.ollama_tag}{RESET}  "
                  f"(usable, ~{biggest.expected_tok_per_sec:.0f} tok/s)")
        else:
            print(f"  Recommended ProdLM:  {RED}none in interactive/usable "
                  f"tier{RESET} — start with the smallest candidate and "
                  f"accept slow output")
    else:
        print(f"  Recommended ProdLM:  {DIM}skipped{RESET}")

    print()
    print(f"  {DIM}Estimates are conservative; real-world throughput from "
          f"llama.cpp/Ollama on M-series typically lands within ~20% of "
          f"these numbers.{RESET}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    shape_names = ", ".join(shape.name for shape in CANONICAL_SHAPES)
    parser.add_argument("--skip-training", action="store_true",
                        help="skip standard training/inference probes")
    parser.add_argument("--skip-prodlm", action="store_true",
                        help="skip ProdLM probe (no network needed)")
    parser.add_argument("--skip-baselm", action="store_true",
                        help="skip BaseLM Hugging Face inference memory estimates")
    parser.add_argument("--baselm-model", metavar="HF_MODEL_ID",
                        help="probe only this BaseLM model instead of the "
                             "curated list (e.g. HuggingFaceTB/SmolLM-135M); "
                             "the model's config.json is fetched from "
                             "Hugging Face")
    parser.add_argument("--prodlm-model", metavar="TAG_OR_HF_SPEC",
                        help="probe only this ProdLM model instead of the "
                             "curated list. Accepts a curated Ollama tag "
                             "(e.g. qwen2.5:7b) or an HF_REPO:GGUF_FILENAME "
                             "spec (e.g. bartowski/Llama-3.2-1B-Instruct-GGUF:"
                             "Llama-3.2-1B-Instruct-Q4_K_M.gguf); for the "
                             "latter, arch fields and quantization are read "
                             "from the GGUF header")
    parser.add_argument("--steps", type=int, default=25,
                        help="training steps per size for the training probe "
                             "(default 25; use 100 for a more stable memory "
                             "high-water mark and throughput estimate)")
    parser.add_argument("--batch-sweep", metavar="MODEL",
                        choices=[shape.name for shape in CANONICAL_SHAPES],
                        help="run an opt-in batch frontier sweep for one "
                             f"canonical model: {shape_names}")
    parser.add_argument("--sweep-batches", default="8,16,32",
                        help="comma-separated batches for --batch-sweep "
                             "(default: 8,16,32)")
    parser.add_argument("--sweep-steps", type=int, default=20,
                        help="training steps per batch in the sweep "
                             "(default 20; shorter than --steps because this "
                             "is a frontier probe)")
    parser.add_argument("--no-sweep-stop-on-fail", action="store_true",
                        help="continue sweeping larger batches even after a "
                             "batch fails")
    parser.add_argument("--context-length", type=int, default=4096,
                        help="ProdLM context length for KV-cache estimate (default 4096)")
    parser.add_argument("--json", metavar="PATH",
                        help="also write full results as JSON to this path")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.sweep_steps <= 0:
        parser.error("--sweep-steps must be positive")
    if args.baselm_model and args.skip_baselm:
        parser.error("--baselm-model and --skip-baselm are mutually exclusive")
    if args.prodlm_model and args.skip_prodlm:
        parser.error("--prodlm-model and --skip-prodlm are mutually exclusive")
    # Targeting one model implicitly hides the unrelated sections so the
    # output stays focused on the requested model.
    if args.baselm_model and not args.prodlm_model:
        args.skip_prodlm = True
        args.skip_training = True
    if args.prodlm_model and not args.baselm_model:
        args.skip_baselm = True
        args.skip_training = True
    try:
        sweep_batches = parse_batch_list(args.sweep_batches)
    except ValueError as exc:
        parser.error(str(exc))

    if args.baselm_model:
        try:
            baselm_candidates = [baselm_candidate_from_hf(args.baselm_model)]
        except (ValueError, urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, OSError) as exc:
            parser.error(f"--baselm-model: {type(exc).__name__}: {exc}")
    else:
        baselm_candidates = list(BASELM_INFERENCE_CANDIDATES)

    if args.prodlm_model:
        try:
            prodlm_candidates = [prodlm_candidate_from_spec(args.prodlm_model)]
        except ValueError as exc:
            parser.error(f"--prodlm-model: {exc}")
    else:
        prodlm_candidates = list(PRODLM_CANDIDATES)

    sysinfo = detect_system()
    print_system(sysinfo)

    training_results: list[ProbeResult] = []
    inference_results: list[ProbeResult] = []
    baselm_results: list[BaseLMInferenceResult] = []
    baselm_posttrain_results: list[BaseLMPostTrainingResult] = []
    batch_sweep_results: list[ProbeResult] = []
    prodlm_results: list[ProdLMResult] = []

    training_device: str | None = None
    if not args.skip_training or args.batch_sweep is not None:
        if sysinfo.torch_version is None:
            print(f"\n{RED}torch not installed — run ./setup.sh first{RESET}")
            return 1
        training_device = "mps" if sysinfo.mps_available else "cpu"
        if training_device == "cpu":
            print(f"\n{YELLOW}MPS unavailable — running on CPU. Numbers will not "
                  f"reflect production behavior.{RESET}")

    if not args.skip_training:
        assert training_device is not None
        print_training_header()
        training_status_suffix = "this may take a few minutes"
        print_status_line(f"starting training probe...   {training_status_suffix}")

        def _show_progress(name: str):
            def cb(done: int, total: int) -> None:
                print_status_line(
                    f"probing {name}  step {done}/{total}...   "
                    f"{training_status_suffix}"
                )
            return cb

        for shape in CANONICAL_SHAPES:
            print_status_line(
                f"probing {shape.name}...   {training_status_suffix}"
            )
            r = probe_training(shape, training_device, n_steps=args.steps,
                               progress_cb=_show_progress(shape.name))
            r = classify_training_headroom(r, sysinfo)
            training_results.append(r)
            clear_status_line()
            print_training_row(r)
        clear_status_line()
        print_training_footer(training_results)

        print_inference_header()

        def _show_inference_progress(name: str):
            def cb(done: int, total: int) -> None:
                print_status_line(f"probing {name}  token {done}/{total}")
            return cb

        for shape in CANONICAL_SHAPES:
            print_status_line(f"probing {shape.name}...")
            r = probe_inference(
                shape,
                training_device,
                progress_cb=_show_inference_progress(shape.name),
            )
            inference_results.append(r)
            clear_status_line()
            print_inference_row(r)
        clear_status_line()

    if not args.skip_baselm:
        print_baselm_inference_header()
        baselm_status = (
            "estimating Hugging Face/PyTorch inference memory; "
            f"requested context={args.context_length:,}"
        )
        print_status_line(baselm_status)
        for idx, cand in enumerate(baselm_candidates, start=1):
            print_status_line(
                f"{baselm_status} | {idx}/{len(baselm_candidates)} "
                f"{cand.display_name}"
            )
            r = estimate_baselm_inference(
                cand,
                sysinfo,
                context_length=args.context_length,
            )
            baselm_results.append(r)
            clear_status_line()
            print_baselm_inference_row(r)
        clear_status_line()
        print_baselm_inference_footer()

        print_baselm_posttraining_header()
        baselm_posttrain_status = "estimating BaseLM SFT/DPO full fine-tuning memory"
        print_status_line(baselm_posttrain_status)
        total_rows = len(baselm_candidates) * 2
        row_index = 0
        for cand in baselm_candidates:
            for stage in ("SFT", "DPO"):
                row_index += 1
                print_status_line(
                    f"{baselm_posttrain_status} | {row_index}/{total_rows} "
                    f"{cand.display_name} {stage}"
                )
                r = estimate_baselm_posttraining(cand, sysinfo, stage=stage)
                baselm_posttrain_results.append(r)
                clear_status_line()
                print_baselm_posttraining_row(r)
        clear_status_line()
        print_baselm_posttraining_footer()

    if args.batch_sweep is not None:
        assert training_device is not None
        shape = find_shape(args.batch_sweep)
        print_batch_sweep_header(shape.name)
        sweep_status = f"running {args.sweep_steps} training steps per batch"
        print_status_line(sweep_status)

        def _show_sweep_progress(batch_size: int):
            print_status_line(f"{sweep_status} | probing B={batch_size}...")

            def cb(done: int, total: int) -> None:
                print_status_line(
                    f"{sweep_status} | probing B={batch_size}  "
                    f"step {done}/{total}"
                )
            return cb

        def _show_sweep_result(result: ProbeResult) -> None:
            clear_status_line()
            print_batch_sweep_row(result)

        batch_sweep_results = probe_batch_sweep(
            shape,
            sweep_batches,
            training_device,
            sysinfo,
            n_steps=args.sweep_steps,
            stop_after_fail=not args.no_sweep_stop_on_fail,
            progress_cb_factory=_show_sweep_progress,
            result_cb=_show_sweep_result,
        )
        clear_status_line()
        print_batch_sweep_footer(batch_sweep_results)

    if not args.skip_prodlm:
        print_prodlm_header()
        prodlm_status = f"fetching GGUF headers only; context={args.context_length}"
        print_status_line(prodlm_status)
        for idx, cand in enumerate(prodlm_candidates, start=1):
            print_status_line(
                f"{prodlm_status} | probing ProdLM {idx}/{len(prodlm_candidates)}: "
                f"{cand.display_name}..."
            )
            r = probe_prodlm(cand, sysinfo, context_length=args.context_length)
            prodlm_results.append(r)
            clear_status_line()
            print_prodlm_row(r)
        clear_status_line()

    if training_results or prodlm_results:
        print_recommendation(training_results, prodlm_results)

    if args.json:
        payload = {
            "system":     asdict(sysinfo),
            "training":   [asdict(r) for r in training_results],
            "inference":  [asdict(r) for r in inference_results],
            "baselm_inference": [asdict(r) for r in baselm_results],
            "baselm_posttraining": [asdict(r) for r in baselm_posttrain_results],
            "batch_sweep": [asdict(r) for r in batch_sweep_results],
            "prodlm":     [asdict(r) for r in prodlm_results],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n{DIM}wrote {args.json}{RESET}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
