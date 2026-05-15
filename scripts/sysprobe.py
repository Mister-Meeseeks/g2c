"""System-capability probe for the g2c course.

Reports what your machine can handle across three dimensions:

  1. **Training** — can you train each canonical size (1M, 5M, 30M, 100M)?
  2. **Inference** — what's the from-scratch generation speed at each size?
  3. **ProdLM** — for a curated short-list of Ollama instruct models, will the
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
    .venv/bin/python scripts/sysprobe.py --skip-training # fast run
"""
from __future__ import annotations

import argparse
import json
import platform
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
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


def detect_system() -> SystemInfo:
    sys_platform = platform.system()
    chip = "unknown"
    mem_gb = 0.0
    bandwidth = None

    if sys_platform == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        except Exception:
            pass
        try:
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
            ).strip())
            mem_gb = mem_bytes / (1024 ** 3)
        except Exception:
            pass
        bandwidth = CHIP_BANDWIDTH_GBPS.get(chip)

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


def probe_training(shape: ModelShape, device: str, n_steps: int = 100,
                   progress_cb=None) -> ProbeResult:
    """Run training steps and report observed peak memory + tail throughput.

    Memory: we sample `driver_allocated_memory` after every step and keep
    the running max. This is still not a true within-step peak (transient
    backward spikes can be missed), but over `n_steps` it converges to a
    realistic high-water mark for the allocator, which is what predicts
    long-run training memory.

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

            if progress_cb is not None and (step + 1) % 10 == 0:
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


def probe_inference(shape: ModelShape, device: str) -> ProbeResult:
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
        ids = torch.randint(0, shape.vocab_size, (1, prompt_len), device=device)

        with torch.no_grad():
            _ = model(ids)  # warmup
            if device == "mps":
                torch.mps.synchronize()

            t0 = time.perf_counter()
            for _ in range(new_tokens):
                logits = model(ids)
                next_id = logits[:, -1:, :].argmax(dim=-1)
                ids = torch.cat([ids, next_id], dim=1)
                if ids.shape[1] >= shape.max_seq_len:
                    break
            if device == "mps":
                torch.mps.synchronize()
            elapsed = time.perf_counter() - t0

        tok_per_sec = new_tokens / elapsed
        return ProbeResult(
            shape_name=shape.name,
            param_count_m=params_m,
            status="ok",
            peak_memory_gb=None,
            tokens_per_sec=tok_per_sec,
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


def estimate_kv_cache_gb(cand: ProdLMCandidate, context_length: int,
                         bytes_per_element: int = 2) -> float:
    """KV cache memory at a given context. K and V each are
    (n_layers, n_kv_heads, head_dim) per token. fp16 = 2 bytes."""
    per_token = 2 * cand.n_layers * cand.n_kv_heads * cand.head_dim * bytes_per_element
    return (per_token * context_length) / (1024 ** 3)


def throughput_tier(tok_per_sec: float) -> str:
    if tok_per_sec >= 30:
        return "interactive"
    if tok_per_sec >= 10:
        return "usable"
    if tok_per_sec >= 3:
        return "slow"
    return "painful"


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


def fmt_status(s: str) -> str:
    c = color_for_verdict(s)
    return f"{c}{s:^7}{RESET}"


def print_system(sysinfo: SystemInfo) -> None:
    header("System")
    print(f"  platform        {sysinfo.platform}")
    print(f"  chip            {sysinfo.chip}")
    print(f"  unified memory  {sysinfo.physical_memory_gb:.1f} GB")
    if sysinfo.bandwidth_gbps is not None:
        print(f"  bandwidth       {sysinfo.bandwidth_gbps:.0f} GB/s  "
              f"{DIM}(specced){RESET}")
    else:
        print(f"  bandwidth       {DIM}unknown chip — throughput estimates "
              f"will be skipped{RESET}")
    print(f"  python          {sysinfo.python_version}")
    print(f"  torch           {sysinfo.torch_version or 'NOT INSTALLED'}")
    print(f"  MPS             {'available' if sysinfo.mps_available else 'unavailable'}")


def print_training(results: list[ProbeResult]) -> None:
    header("Training probe (forward + backward + AdamW, on MPS)")
    print(f"  {'model':<14} {'params':>8} {'shape':>11}  {'status':^9} "
          f"{'peak mem':>9}  {'throughput':>14}  notes")
    print(f"  {'-'*14} {'-'*8} {'-'*11:>11}  {'-'*7:^9} {'-'*9:>9}  "
          f"{'-'*14:>14}  {'-'*40}")
    for r in results:
        shape_str = f"B={r.batch_size},T={r.context_length}"
        print(f"  {r.shape_name:<14} {r.param_count_m:>6.1f}M  "
              f"{shape_str:>11}  {fmt_status(r.status)} "
              f"{fmt_gb(r.peak_memory_gb):>9}  "
              f"{fmt_tps(r.tokens_per_sec):>14}  {r.note}")
    print(f"\n  {DIM}peak mem is the max of `mps.driver_allocated_memory` "
          f"sampled after each of {results[0].steps_run if results else 0} "
          f"training steps. Add ~0.5-2 GB on top for a real Trainer's "
          f"tokenized corpus, DataLoader buffers, and logging state.{RESET}")


def print_inference(results: list[ProbeResult]) -> None:
    header("Inference probe (forward-only, no KV cache)")
    print(f"  {'model':<14} {'params':>8} {'status':^9} {'throughput':>14}  notes")
    print(f"  {'-'*14} {'-'*8} {'-'*7:^9} {'-'*14:>14}  {'-'*40}")
    for r in results:
        print(f"  {r.shape_name:<14} {r.param_count_m:>6.1f}M  "
              f"{fmt_status(r.status)} {fmt_tps(r.tokens_per_sec):>14}  {r.note}")


def print_prodlm(results: list[ProdLMResult]) -> None:
    header("ProdLM probe (Ollama candidates, pre-download estimate)")
    print(f"  {'model':<48} {'status':^9} {'total':>9}  "
          f"{'est tok/s':>11}  {'tier':<12}")
    print(f"  {'-'*48} {'-'*7:^9} {'-'*9:>9}  {'-'*11:>11}  {'-'*12:<12}")
    for r in results:
        tps = fmt_tps(r.expected_tok_per_sec).strip().replace(" tok/s", "")
        tier_color = {
            "interactive": GREEN,
            "usable":      GREEN,
            "slow":        YELLOW,
            "painful":     RED,
            "unknown":     DIM,
        }.get(r.throughput_tier, "")
        print(f"  {r.display_name:<48} {fmt_status(r.status)} "
              f"{fmt_gb(r.total_gb):>9}  {tps:>11}  "
              f"{tier_color}{r.throughput_tier:<12}{RESET}")
        if r.note:
            print(f"    {DIM}↳ {r.note}{RESET}")


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
    parser.add_argument("--skip-training", action="store_true",
                        help="skip training/inference probes")
    parser.add_argument("--skip-prodlm", action="store_true",
                        help="skip ProdLM probe (no network needed)")
    parser.add_argument("--steps", type=int, default=100,
                        help="training steps per size for the training probe "
                             "(default 100; larger = more accurate memory "
                             "high-water mark, but takes longer)")
    parser.add_argument("--context-length", type=int, default=4096,
                        help="ProdLM context length for KV-cache estimate (default 4096)")
    parser.add_argument("--json", metavar="PATH",
                        help="also write full results as JSON to this path")
    args = parser.parse_args()

    sysinfo = detect_system()
    print_system(sysinfo)

    training_results: list[ProbeResult] = []
    inference_results: list[ProbeResult] = []
    prodlm_results: list[ProdLMResult] = []

    if not args.skip_training:
        if sysinfo.torch_version is None:
            print(f"\n{RED}torch not installed — run ./setup.sh first{RESET}")
            return 1
        device = "mps" if sysinfo.mps_available else "cpu"
        if device == "cpu":
            print(f"\n{YELLOW}MPS unavailable — running on CPU. Numbers will not "
                  f"reflect production behavior.{RESET}")

        header(f"Running training probes ({args.steps} steps each — "
               f"this takes several minutes for the larger sizes)")

        def _show_progress(name: str):
            def cb(done: int, total: int) -> None:
                print(f"\033[2K\r  {DIM}probing {name}  "
                      f"step {done}/{total}{RESET}", end="", flush=True)
            return cb

        for shape in CANONICAL_SHAPES:
            print(f"\033[2K\r  {DIM}probing {shape.name}...{RESET}",
                  end="", flush=True)
            r = probe_training(shape, device, n_steps=args.steps,
                               progress_cb=_show_progress(shape.name))
            training_results.append(r)
        print("\033[2K\r", end="")  # clear in-place status line
        print_training(training_results)

        header("Running inference probes")
        for shape in CANONICAL_SHAPES:
            print(f"\033[2K\r  {DIM}probing {shape.name}...{RESET}", end="", flush=True)
            r = probe_inference(shape, device)
            inference_results.append(r)
        print(" " * 60, end="\r")
        print_inference(inference_results)

    if not args.skip_prodlm:
        header(f"Running ProdLM probes (fetching GGUF headers, "
               f"context={args.context_length})")
        for cand in PRODLM_CANDIDATES:
            print(f"\033[2K\r  {DIM}probing {cand.display_name}...{RESET}",
                  end="", flush=True)
            r = probe_prodlm(cand, sysinfo, context_length=args.context_length)
            prodlm_results.append(r)
        print("\033[2K\r", end="")
        print_prodlm(prodlm_results)

    if training_results or prodlm_results:
        print_recommendation(training_results, prodlm_results)

    if args.json:
        payload = {
            "system":     asdict(sysinfo),
            "training":   [asdict(r) for r in training_results],
            "inference":  [asdict(r) for r in inference_results],
            "prodlm":     [asdict(r) for r in prodlm_results],
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n{DIM}wrote {args.json}{RESET}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
