# Module 16 — Inference backends and production models

> **Question this module answers:** *How do we get from "I built it" to "I can use it"?*

![Module 16 summary diagram showing a tiny course model, a configured production model, and a shared Backend interface used by evaluation, RAG, tools, and agents.](16-inference/Module16-Hero.png)
*The whole module on one page. After 15 modules of "build it from scratch," Module 16 pivots to "use models through a stable interface." StudentLM artifacts, BaseLM experiments, and a capable local ProdLM all speak the same `Backend.complete(...)` API, so Module 17's RAG, Module 18's tools, and Module 19's agent loop do not have to care which model is underneath.*

---
## Before you start

* *Finish* `g2c/sampling` from [[11-sampling]] — `LocalTransformerBackend` drives the from-scratch model through `g2c.sampling.generate`
* *Finish* at least one saved model artifact from Module 10, 13, or 14, or run `./baselm.sh` — `ArtifactBackend` lets you use course-trained and BaseLM artifacts through the inference API
* *Configure* ProdLM with `./prodlm.sh --model-id llama3.2:3b` (or another size from the suggested list below) — by default this pulls the Ollama model and writes a lightweight manifest the later modules can discover

---
## Prerequisites

Module 16 assumes you can already produce completions from at least one model and you understand the evaluation adapter pattern from Module 15. It does **not** assume you already know quantization formats, KV caches, speculative decoding, Ollama internals, or MLX.

### Math

Only basic ratios and memory arithmetic:

- Comfort reading numbers like "tokens per second" and "milliseconds per token."
- Comfort multiplying parameter counts by bytes per parameter.
- Comfort interpreting p50/p90 latency as summary statistics.

### Computer science

- Module 11's autoregressive generation loop: prompt tokens in, sampled continuation out.
- The saved-artifact workflow from Modules 10, 13, and 14, or the BaseLM setup from `./baselm.sh`.
- Module 15's `generate_fn(prompt) -> str` adapter idea.
- The general idea that one process can hold a model while another process calls it through an API.

### Programming

- Basic `dataclass` and abstract-base-class syntax.
- Basic `json.dumps` / `json.loads`.
- Basic `urllib.request` or equivalent HTTP-client experience.
- Basic timing with `time.perf_counter`.
- Basic list aggregation for benchmark summaries.

### What you can skip

- Prior knowledge of GGUF, MLX, quantization, KV caches, or speculative decoding. Those are introduced here.
- Manual model quantization. Download a pre-quantized local model instead.
- Streaming, async I/O, and MLX serving. They are useful extensions, not required for the module deliverable.

## Why we start here

Modules 1–15 built a model. The training works, the architecture works, the loss goes down. But:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  THE PROBLEM AT THE END OF MODULE 15                                  │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Your 20M-param TransformerLM, post-DPO:                             │
   │                                                                       │
   │     ✓  trains end-to-end on a MacBook                                 │
   │     ✓  follows the chat template                                      │
   │     ✓  improves with SFT and DPO                                      │
   │     ✓  has a calibrated, eval-able failure profile                    │
   │                                                                       │
   │     ✗  doesn't know facts                                             │
   │     ✗  can't add 13 + 28                                              │
   │     ✗  hallucinates fluently                                          │
   │     ✗  can't follow a multi-step task                                 │
   │                                                                       │
   │   For RAG, tools, and agents (Modules 17–19) to be more than          │
   │   "watch the toy model fail in slightly different ways," we need      │
   │   a base model that's actually capable. That means a pretrained       │
   │   open model — Llama 3, Qwen 2.5, Mistral. Quantized to fit on a      │
   │   laptop. Run by a real inference server.                             │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The model names in this phase are roles, not one fixed vendor choice:

- **StudentLM** is whatever you trained yourself: ShakespeareLM, StoryLM, TinyLLM, SFT, or DPO variants.
- **BaseLM** is a small Hugging Face base model you can load in PyTorch for trainable SFT/DPO exercises when your own model is too weak.
- **ProdLM** is the capable local instruction model used for RAG, tools, and agents. The default implementation is an Ollama-served quantized model, but downstream code only sees the `Backend` interface.

This module has three jobs:

- Introduce the model formats and runtime patterns you will actually encounter: GGUF, safetensors, MLX, Ollama, and local artifacts.
- Explain the inference-time optimizations that make laptop-scale model use practical: quantization, KV cache, and speculative decoding.
- Build a small Python interface so RAG and agent code can call `backend.complete(prompt, ...)` without knowing whether the model is an in-process artifact, a Hugging Face BaseLM wrapper, or an out-of-process ProdLM server.

The only required code you write is the interface itself.

The "right" ProdLM for this course is usually an instruction-tuned model in the 3B–8B range, quantized to 4 bits. Concretely, one of:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   Suggested ProdLM models for the rest of the course                  │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     llama3.2:3b           ~1.9 GB Q4_K_M     16GB OK    fast          │
   │     llama3.1:8b           ~4.7 GB Q4_K_M     16GB OK    medium        │
   │     qwen2.5:7b-instruct   ~4.4 GB Q4_K_M     16GB OK    medium        │
   │     mistral:7b-instruct   ~4.1 GB Q4_K_M     16GB OK    medium        │
   │                                                                       │
   │   Configure with:  ./prodlm.sh --model-id <model:tag>                 │
   │   Inspect with:    ollama list                                        │
   │   Override in code: load_prodlm_backend(model_id="<model:tag>")       │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

But the rest of the course shouldn't have to care which one. Module 17's RAG should work the same against your saved TinyLLM artifact, a BaseLM checkpoint, or a configured `llama3.2:3b` ProdLM. The eval harness from Module 15 should be re-runnable across a generation gap. The agent loop in Module 19 should be portable. That's what the unified `Backend` interface enables — one method, `backend.complete(prompt, ...) -> InferenceResult`, that abstracts over all of them.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   THE INTERFACE                                                       │
   └──────────────────────────────────────────────────────────────────────┘

      ┌───────────────────────┐       ┌───────────────────────────┐
      │  ArtifactBackend       │       │  ProdLM / OllamaBackend   │
      │                       │       │                           │
      │  wraps saved artifacts:│       │  wraps an HTTP server:    │
      │   • StudentLM          │       │   • POSTs JSON to         │
      │   • BaseLM             │       │     /api/generate         │
      │   • SFT/DPO variants   │       │   • parses the response   │
      │                       │       │   • times the round-trip   │
      │  in-process            │       │                           │
      │  (PyTorch on MPS/CPU)  │       │  out-of-process           │
      │                       │       │  (llama.cpp via Ollama)    │
      └─────────┬─────────────┘       └────────┬─────────────────┘
                │                                │
                ▼                                ▼
                  ┌────────────────────────────────┐
                  │  Backend.complete(...)          │
                  │     returns InferenceResult     │
                  │                                 │
                  │  prompt, completion,            │
                  │  prompt_tokens, completion_tokens,│
                  │  latency_ms, backend, metadata  │
                  └─────────────┬───────────────────┘
                                │
                                ▼
                          everywhere downstream
                          (RAG, tools, agent, eval)
```

The `complete` method is the entire interface. No streaming, no async, no batched generation. The point is the **lowest common denominator** — and the LCD is enough for everything Modules 17–20 need.

## The big idea

### Quantization: how a 7B model fits in 4 GB

![Quantization ladder — more model, less memory. Lower precision means fewer bits per weight, smaller checkpoint, more model fits on a laptop. A vertical "ladder" shows precision steps from FP32 (32 bits/param, 28 GB for 7B, highest fidelity) → FP16/BF16 (16 bits, 14 GB, near-lossless for inference) → INT8 (8 bits, 7 GB, mild accuracy hit) → INT4 (4 bits, ~3.5 GB, larger but still tractable accuracy hit). A right-side panel pins the speed lever: inference is memory-bandwidth-bound, so halving weight bytes roughly halves the time to fetch them per matmul — the headline 2-4× speedup at int4 comes from bandwidth, not faster math. A "GGUF Q-K variants" panel decodes the cryptic naming: `Q4_K_S` (most weights at 4 bits, smallest), `Q4_K_M` (recommended default — most weights at 4 bits, more important ones at 5–6 bits), `Q5_K_M` (slightly bigger, slightly better quality). A "memory budget example" computes 16 GB Mac − 4 GB OS − 2 GB KV cache = ~10 GB usable for weights, which fits a 7B Q4_K_M comfortably. The takeaway pinned at the bottom: quantization buys headroom, not magic — Q4_K_M is the default for the rest of the course.](16-inference/Module16-Quant.png)

*The lookup table for choosing a quantization level. The deliverable post-mortem (exercise 9) asks you to commit to one (model, quant) pair as your default backend; this image is the chart you'll consult to pick. For 16 GB Macs, Q4_K_M at 7B is the sweet spot the rest of Phase V assumes.*

A model's weights are an array of floats. The array's size is `n_parameters × bytes_per_parameter`. Reducing `bytes_per_parameter` is the lever:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   QUANTIZATION ARITHMETIC                                             │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │     Precision   Bits/param   7B model size    Speedup vs fp32        │
   │     ─────────   ──────────   ────────────    ────────────────       │
   │     fp32        32           28.0 GB          1.0×                    │
   │     fp16/bf16   16           14.0 GB          1.5–2× (memory bw)     │
   │     int8        8             7.0 GB          1.5–2.5× (often)       │
   │     int4        4             3.5 GB          2–4× (memory bw)       │
   │                                                                       │
   │   GGUF "K-quant" mixes precisions per-tensor:                         │
   │     Q4_K_M  ≈  4.5 effective bits/param ≈ 4.0 GB for 7B              │
   │     Q5_K_M  ≈  5.5 effective bits/param ≈ 4.7 GB for 7B              │
   │     Q8_0    ≈  8 bits/param            ≈ 7.0 GB for 7B               │
   │                                                                       │
   │   Quantization replaces:                                              │
   │     w  = w_fp16                                                       │
   │   with:                                                               │
   │     w  ≈ scale * round((w_fp16 - bias) / scale)                       │
   │   where (scale, bias) are computed PER-BLOCK (typically 32–128        │
   │   weights per block). The block's worth of weights fits in            │
   │   block_size * bits_per_param bits PLUS one fp16 scale.               │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The accuracy hit from quantization is real but small for instruction-tuned models in the int8 → int4 range. Q4_K_M typically loses 1–3% on benchmarks compared to fp16. For a course project, this is well below the noise of toy-model variance.

The reason quantization speeds inference up isn't the math — int4 multiplications aren't faster than fp16 multiplications on most hardware. It's that **inference is memory-bandwidth-bound**. The time to multiply a weight by an activation is dominated by the time to *fetch the weight from RAM*. Halving the weight size halves the fetch time, even if the math itself is the same speed.

### KV cache: from O(T²) per token to O(T) per token

![KV cache — remember once, reuse forever. Two side-by-side flow diagrams. WITHOUT KV CACHE: each generation step recomputes attention over the whole sequence. Step 1 computes K/V for tokens 0..2; step 2 recomputes K/V for tokens 0..3 (re-doing the work for 0..2); step 3 redoes 0..4; cost per step grows linearly with sequence length, total work is O(T²). WITH KV CACHE: step 1 computes K/V for tokens 0..2, stores them; step 2 only computes K/V for the new token (one row), appends to the cache, attends from the new query against the entire cached K/V; per-step cost is O(T), total work is O(T·T) but with a much smaller constant. A "what is stored" panel pins the layout: `K_cache` and `V_cache` per layer, shape `(max_seq_len, n_heads, head_dim)`, dtype fp16. A memory-budget example for a 7B model (Llama 3.1 8B): 32 layers × 32 heads × 128 head_dim × 2 (K and V) × 2 bytes (fp16) × 2048 context = ~1 GB; doubles linearly with context length. A "scaling with context length" panel shows weights stay fixed while the cache grows — long-context inference is memory-bound on the cache, not the weights. A "the takeaway" panel: KV cache is mandatory at scale; production servers like llama.cpp build one in; the optional course-model cache makes the mechanism inspectable.](16-inference/Module16-KVCache.png)

*The picture for understanding why Ollama's 30 tok/s on a 7B model is even possible. The KV cache turns autoregressive decoding from O(T²) to O(T) per step — a 5–20× speedup at production context lengths. The required path relies on Ollama's production cache; the optional extension builds a small version for your own `TransformerLM` so the mechanism is inspectable.*

Module 11's generation loop recomputes attention over the entire sequence every step:

```
   Step 1:   forward([P1, P2, P3])              →  predict S1
   Step 2:   forward([P1, P2, P3, S1])          →  predict S2
   Step 3:   forward([P1, P2, P3, S1, S2])      →  predict S3
   Step 4:   forward([P1, P2, P3, S1, S2, S3])  →  predict S4
   ...
```

At step `t`, the model recomputes K and V for tokens 0..t. That's `O(T²)` total work for a sequence of length T. The `K_t` and `V_t` projections at step `t` are *the same* as they were at step `t-1` for all positions before `t-1` — there's no need to recompute them.

The KV cache stores `K_0, K_1, ..., K_t` and `V_0, V_1, ..., V_t` as the model generates, so step `t+1` only needs to:

  1. Project the *new* token's K and V (one row's worth of work).
  2. Append to the cache.
  3. Compute attention from the new query against the entire cached K/V (one matmul against a length-`T+1` matrix).

Total: `O(T)` per step → `O(T²)` for the whole generation, but with a ~10× smaller constant. In practice, KV-cached inference is 5–20× faster than the naive loop for sequences past 128 tokens.

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   KV CACHE — STATE THAT CARRIES FORWARD                               │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Cached state, per layer:                                            │
   │     K_cache:  (max_seq_len, n_heads, head_dim)                        │
   │     V_cache:  (max_seq_len, n_heads, head_dim)                        │
   │                                                                       │
   │   Per generation step:                                                │
   │     1. project new token →  K_new, V_new   (one row each)            │
   │     2. write into K_cache[t], V_cache[t]                              │
   │     3. attention =  softmax(Q_new · K_cache[:t+1]ᵀ / √d) · V_cache[:t+1]│
   │                                                                       │
   │   Memory cost:                                                        │
   │     KV cache size  =  2 · max_seq_len · n_layers · n_heads · head_dim  │
   │                       · bytes_per_dtype                                │
   │                                                                       │
   │   For 7B at fp16, 32 layers, 32 heads × 128 dim, T=2048:               │
   │     = 2 · 2048 · 32 · 32 · 128 · 2  bytes                              │
   │     = 1.07 GB                                                          │
   │                                                                       │
   │   This is why 7B at 4-bit + 16k context can blow past 16 GB —          │
   │   the cache scales linearly with context length while weights stay     │
   │   fixed.                                                               │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

The required backend path does not need a cache in your tiny `TransformerLM` — Ollama's underlying llama.cpp is heavily KV-cached, and that is the cache you depend on for ProdLM. The optional extension in this module adds a small, readable cache to the course model anyway, so you can see the state that carries forward from one generated token to the next.

### Speculative decoding — skim only

Speculative decoding (Leviathan et al. 2023, Chen et al. 2023) runs a small "draft" model (e.g., a 1B-param sibling of the 70B target) ahead of the main model, generating several candidate tokens at once. The main model then verifies all of them in a single forward pass — comparing its softmax over each position against what the draft chose. Whichever prefix the main model agrees with is accepted; the rest are discarded; the next round starts.

When the draft is right (the common case for "easy" tokens — function words, punctuation, completions of common phrases), one forward pass through the main model produces several output tokens. When the draft is wrong, no harm — the main model's logits at the divergence point determine the next token, same as without speculation.

In practice: 2–3× throughput improvement on long generations. We don't build it. We acknowledge it because every production serving stack uses it.

### The inference server pattern

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   LOCAL INFERENCE SERVERS — THE LAY OF THE LAND                       │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   llama.cpp     C++ library + CLI. Loads GGUF, runs on CPU+GPU.       │
   │                  The grandparent of every other entry. Open source,    │
   │                  Apache 2.0, ~50k lines of optimized C++.              │
   │                                                                       │
   │   Ollama        Daemon + HTTP API + model registry on top of           │
   │                  llama.cpp. `ollama pull / list / run / serve`.        │
   │                  HTTP API at localhost:11434 by default. The           │
   │                  one-command-install path on macOS.                    │
   │                                                                       │
   │   MLX           Apple's tensor library + python bindings. In-process   │
   │                  (no HTTP). Faster than llama.cpp on M-series for      │
   │                  many ops; format-incompatible with GGUF (uses         │
   │                  safetensors-style sharded checkpoints).               │
   │                                                                       │
   │   vLLM          Server-class. CUDA-only or ROCm-only. Not relevant     │
   │                  for MacBook. Mentioned for completeness.              │
   │                                                                       │
   │   text-gen-webui Higher-level UI on top of llama.cpp + others.        │
   │                  Useful as an exploratory chat client; we don't        │
   │                  use it programmatically.                              │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

Ollama is the right default for this course: the install is one shell command, the API is small, and the model registry is curated (easy to find a working `:tag` for any popular open model). MLX is the alternative when you want maximum throughput on Apple Silicon and are willing to manage the conversion pipeline.

### The unified Backend interface — what we actually build

![Backend abstraction — one interface, many models. A central `backend.complete(prompt, *, max_new_tokens, temperature, top_k, top_p) -> InferenceResult` method is the entire public surface. Above it: a row of consumers (eval harness from Module 15, RAG pipeline from Module 17, tool calls from Module 18, the agent loop from Module 19, the capstone app from Module 20) — none of them know which backend is underneath. Below it: concrete implementations for saved artifacts, Ollama-backed ProdLM, and an optional MLX backend. The same prompt routed through any of them produces an InferenceResult with the same fields. A "design pattern" panel pins the name: dependency injection — same idea as `generate_fn` in the Module 15 eval harness, applied at a coarser granularity. A "the takeaway" panel: the unified interface comes BEFORE the strong model. Build the seam first; the strong model plugs in behind it.](16-inference/Module16-Backend.png)

*The architecture diagram for the rest of the course. Module 17 onward calls `backend.complete(...)` and never branches on whether the backend is a saved artifact, BaseLM, or ProdLM. Exercise 9 (the deliverable post-mortem) commits you to one default backend; the abstraction is what lets you swap in a different model later without touching downstream code.*

Core classes plus two artifact helpers:

```
   ┌─────────────────────────────────────────────────────────────────────┐
   │   g2c/inference/  PUBLIC API                                          │
   ├─────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   Backend                  abstract                                  │
   │     .info       → BackendInfo                                        │
   │     .complete(prompt, *, max_new_tokens, temperature,                │
   │               top_k, top_p) → InferenceResult                        │
   │                                                                       │
   │   LocalTransformerBackend(model, tokenizer, *, ...)                   │
   │     wraps:  a TransformerLM-like model + tokenizer + generate         │
   │                                                                       │
   │   ArtifactBackend(artifact, *, ...)                                   │
   │     wraps:  saved StudentLM / BaseLM / SFT / DPO artifacts            │
   │                                                                       │
   │   OllamaBackend(model_id, *, base_url, timeout, ...)                  │
   │     wraps:  HTTP POST to /api/generate                               │
   │                                                                       │
   │   load_prodlm_backend(...)                                            │
   │     loads:  configured ProdLM manifest, then returns OllamaBackend    │
   │                                                                       │
   │   load_default_backend(...)                                           │
   │     loads:  ProdLM if configured, otherwise the strongest artifact    │
   │                                                                       │
   │   InferenceResult                                                     │
   │     prompt, completion, prompt_tokens, completion_tokens,             │
   │     latency_ms, backend, metadata                                     │
   │     .tokens_per_second  (derived)                                     │
   │                                                                       │
   │   benchmark(backend, prompts, *, ...) → BenchmarkResult               │
   │     iterates complete() over prompts; aggregates timing.              │
   │                                                                       │
   └─────────────────────────────────────────────────────────────────────┘
```

Total scaffolded code: ~70 lines spread across three method bodies (`LocalTransformerBackend.complete`, `OllamaBackend.complete`, `benchmark`). `ArtifactBackend` and the ProdLM manifest helpers are implemented plumbing so notebooks can stay focused on the model comparison. The math is trivial; the lesson is the interface and the wiring.

## Concepts to internalize

- **The unified interface comes BEFORE the strong model.** Modules 17–20 work because every backend speaks the same `complete(prompt, ...)` API. A version of this course that hard-coded `g2c.sampling.generate` everywhere couldn't pivot from StudentLM to ProdLM without a refactor of every downstream module. Build the seam first; the strong model plugs in behind it.
- **Quantization buys you headroom, not magic.** A 4-bit 7B model fits in 4 GB and runs at usable speeds on M2; an fp16 7B does not fit on a 16 GB Mac at all (even before KV cache). The accuracy cost is 1–3% on benchmarks. The headroom is what actually unblocks Phase V on a laptop.
- **The KV cache is mandatory at scale.** Naive `O(T²)` decoding is fine at 32-token contexts (your tiny model); at 2k contexts it's not. Ollama already has one for the real model. The optional toy cache exists to make the mechanism concrete, not because it changes the rest of Phase V.
- **Wall clock vs server-reported latency are different things.** Ollama tells you `total_duration` (the GPU-side compute time). Your wrapper records wall-clock latency (compute + IPC + JSON encoding). Both are valid; they answer different questions. Track both.
- **Tokens-per-second is the headline throughput metric.** But it depends critically on what you count: prompt processing tokens, generation tokens, both, neither. The `InferenceResult.tokens_per_second` property uses *generation tokens / wall-clock seconds* — the user-facing definition. When comparing to published numbers, check the convention.
- **The first request is always slower.** Cold cache, model load, MPS / Metal kernel JIT compilation. A `benchmark` of 1 request reports the cold-start number; a benchmark of 5 requests reports the steady-state. Always warm up before measuring.
- **Backends are values, not globals.** The wrapper classes are stateful (they hold a model or a base URL), but they're not singletons. RAG, tool, and agent code take a `Backend` parameter. This is dependency injection — the same pattern that decoupled `generate_fn` from the eval harness in Module 15, applied at the model level.
- **The generation-API design space is small.** `complete(prompt, ...) -> result` covers the whole course. Streaming, batching, async, and tool-aware completions are all useful production extensions but don't unlock anything pedagogically new — they're optimizations of the same abstraction. Phase V uses the synchronous completion exclusively.

## Scaffolding and how to run the tests

This module ships seven files in `g2c/inference/`:

- **`backend.py`** — the `Backend` ABC plus `BackendInfo` and `InferenceResult` dataclasses. **Boilerplate, fully implemented.**
- **`local.py`** — `LocalTransformerBackend`. Constructor + `info` are implemented. `complete` is **scaffolded**.
- **`ollama.py`** — `OllamaBackend` plus `OllamaError`. Constructor + `info` are implemented. `complete` is **scaffolded**.
- **`benchmark.py`** — `BenchmarkResult` (boilerplate) + `benchmark()` (**scaffolded**).
- **`artifact.py`** — `ArtifactBackend` and `load_artifact_backend`. **Boilerplate, fully implemented.**
- **`prodlm.py`** — ProdLM manifest helpers and `load_default_backend`. **Boilerplate, fully implemented.**
- **`__init__.py`** — public exports.

The optional KV-cache extension lives next to the code it extends:

- **`g2c/transformer/kv_cache.py`** — `LayerKVCache` and `KVCache` containers. **Boilerplate, fully implemented.**
- **`MultiHeadAttention.forward_cached`** — project only the newest token's Q/K/V, append K/V to the layer cache, and attend the newest query against cached K/V. **Scaffolded conceptually, tested against `forward`.**
- **`Block.forward_cached`** and **`TransformerLM.forward_cached`** — thread one cache object per layer through the transformer. **Mostly plumbing.**
- **`g2c/sampling/generate_cached.py`** — a cache-aware sibling of Module 11's `generate`. **Boilerplate around the cached forward path.**

Tests live in `tests/test_inference.py`. The dataclass boilerplate, validation paths, ABC behavior, `BackendInfo` invariants, and helper exports should pass before you write the three pedagogical bodies. The implementation tests fail with `NotImplementedError` until you fill in `LocalTransformerBackend.complete`, `OllamaBackend.complete`, and `benchmark`. The real-transformer smoke test auto-skips when `BPETokenizer.train`, `TransformerLM.forward`, or `generate` aren't implemented yet.

```bash
pytest tests/test_inference.py                          # all module-16 tests
pytest tests/test_inference.py -x                       # stop at first failure
pytest tests/test_inference.py -k Local                 # local-backend tests
pytest tests/test_inference.py -k Ollama                # ollama-backend tests
pytest tests/test_inference.py -k Benchmark             # benchmark tests
pytest tests/test_inference.py -k boilerplate           # the boilerplate sanity tests
pytest tests/test_multi_head_attention.py -k cached     # optional KV-cache attention tests
pytest tests/test_transformer.py -k cached              # optional KV-cache model tests
pytest tests/test_sampling.py -k cached                 # optional KV-cache generation tests
pytest tests/test_inference.py -v                       # verbose
```

Implementation order — three independent steps:

  1. **`LocalTransformerBackend.complete`**. Encode prompt → time → call `generate` → slice off prompt tokens → decode tail → build `InferenceResult`. Tests use a monkeypatched `generate` so this works even before Module 11 is filled in. Turns green: `TestLocalTransformerBackendComplete`, `TestLocalTransformerBackendValidation`.

  2. **`OllamaBackend.complete`**. Build JSON body → POST → parse JSON response → wrap stdlib errors in `OllamaError`. Tests inject a fake `urlopen`, so no real Ollama daemon is required for the test suite. Turns green: `TestOllamaBackendComplete`, `TestOllamaBackendOptionsHandling`, `TestOllamaBackendErrorPaths`, `TestOllamaBackendValidation`.

  3. **`benchmark`**. Iterate `complete` over prompts, aggregate latencies (mean / p50 / p90) and tokens (sum, overall throughput). Tests use a `_FakeBackend` that returns scripted results. Turns green: `TestBenchmark`, `TestBenchmarkEdgeCases`.

Steps 1–3 are independent — work in any order. The boilerplate tests pass from the start as a sanity check.

The end-to-end smoke test (`test_local_backend_real_transformer_smoke`) wires up a real `BPETokenizer` + a real tiny `TransformerLM` + the real `generate` and runs `LocalTransformerBackend.complete` against them. It auto-skips when any prerequisite is unimplemented; once Modules 04, 09, and 11 are done, it runs and verifies end-to-end composition. Same convention as Module 15's `test_run_multiple_choice_eval_real_transformer_smoke`.

Headline tests to watch:

- **`test_completion_excludes_prompt`** — pins the slicing convention: `complete()` returns only the new tokens, not the prompt + new tokens. The most common bug here is forgetting to slice and returning the full sequence as the "completion."
- **`test_token_counts`** — pins that `prompt_tokens` is the encoded-prompt length and `completion_tokens` is the new-tokens length, NOT the full sequence.
- **`test_request_body_includes_required_fields`** — pins the Ollama wire format. `model`, `prompt`, `stream: false`, `options.num_predict`, `options.temperature` — get any of these wrong and the server rejects.
- **`test_top_k_omitted_when_none`** — when sampling parameters are `None`, the request body must omit them entirely (so Ollama uses its defaults). Including a literal `null` would override the server default with "no value," which Ollama doesn't handle gracefully on every endpoint.
- **`test_url_error_wrapped`** / **`test_http_error_wrapped`** — every stdlib HTTP exception path must surface as `OllamaError`. Callers should have one type to catch, not three.
- **`test_completion_tokens_total_none_when_any_missing`** — if any individual request returned `None` for `completion_tokens`, the aggregate throughput is `None`. Don't silently sum-with-zero — that produces a misleading "high tokens/sec" number.
- **`test_p90_above_p50_for_skewed_latencies`** — pins that p90 > p50 on a right-skewed distribution. Catches off-by-one bugs in the percentile lookup.

## What you'll build

Package: `g2c/inference/`

```python
# backend.py
@dataclass(frozen=True)
class BackendInfo:
    name: str
    model_id: str
    extra: dict[str, Any] = field(default_factory=dict)        # implemented

@dataclass
class InferenceResult:
    prompt: str
    completion: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    backend: BackendInfo
    metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def tokens_per_second(self) -> float | None: ...           # implemented

class Backend(ABC):
    @property
    @abstractmethod
    def info(self) -> BackendInfo: ...
    @abstractmethod
    def complete(
        self, prompt: str, *, max_new_tokens: int = 128,
        temperature: float = 1.0, top_k: int | None = None,
        top_p: float | None = None,
    ) -> InferenceResult: ...


# local.py
class LocalTransformerBackend(Backend):
    def __init__(self, model, tokenizer, *,                    # implemented
                 model_id: str = "g2c-local",
                 eos_id: int | None = None,
                 name: str = "local",
                 extra: dict[str, Any] | None = None) -> None: ...
    @property
    def info(self) -> BackendInfo: ...                         # implemented
    @property
    def model(self): ...                                       # implemented
    @property
    def tokenizer(self): ...                                   # implemented
    def complete(self, prompt, *, max_new_tokens=128,
                 temperature=1.0, top_k=None, top_p=None,
                 ) -> InferenceResult:                         # SCAFFOLDED
        ...


# ollama.py
class OllamaError(RuntimeError): ...                           # implemented

class OllamaBackend(Backend):
    def __init__(self, model_id: str = "llama3.2:3b", *,        # implemented
                 base_url: str = DEFAULT_OLLAMA_URL,
                 timeout: float = 120.0,
                 urlopen=None,
                 name: str = "ollama",
                 extra: dict[str, Any] | None = None) -> None: ...
    @property
    def info(self) -> BackendInfo: ...                         # implemented
    @property
    def base_url(self) -> str: ...                             # implemented
    def complete(self, prompt, *, max_new_tokens=128,
                 temperature=1.0, top_k=None, top_p=None,
                 ) -> InferenceResult:                         # SCAFFOLDED
        ...


# artifact.py
class ArtifactBackend(LocalTransformerBackend):                  # implemented
    """Backend wrapper for saved StudentLM, BaseLM, SFT, and DPO artifacts."""

def load_artifact_backend(
    artifact_name: str | None = None, *,
    device: str | None = "auto",
    torch_dtype: str | None = None,
    required: bool = True,
) -> ArtifactBackend | None: ...


# prodlm.py
def write_prodlm_manifest(...): ...                              # implemented
def load_prodlm_backend(...): ...                                # implemented
def load_default_backend(kind: str = "auto", ...): ...           # implemented


# benchmark.py
@dataclass
class BenchmarkResult:                                         # implemented
    backend: BackendInfo
    n: int
    latency_ms_total: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    completion_tokens_total: int | None
    tokens_per_second_overall: float | None
    per_request_latency_ms: list[float]
    per_request_tokens_per_second: list[float | None]
    results: list[InferenceResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

def benchmark(
    backend: Backend, prompts: list[str], *,
    max_new_tokens: int = 64, temperature: float = 1.0,
    top_k: int | None = None, top_p: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> BenchmarkResult:                                          # SCAFFOLDED
    ...
```

Total scaffolded code: roughly 70 lines across three function bodies. The pedagogical content is the wiring — taking a string in, a string out, and recording what happened in between.

## Exercises

These exercises have two paths. The artifact path works with whatever saved StudentLM/BaseLM artifact you have. The ProdLM path requires a running Ollama installation and a configured manifest from `./prodlm.sh --model-id llama3.2:3b` (or your model of choice). The course works with a 16 GB or larger MacBook; 8 GB Macs can run smaller ProdLMs (`llama3.2:1b`, `qwen2.5:0.5b`) but the Phase V experience is degraded.

1. **Smoke-test artifact and ProdLM backends.** Load the strongest saved artifact automatically, load your configured ProdLM, and run a 4-prompt comparison:

   ```python
   prompts = [
       "<|user|>\nWhat is the largest city in Spain?\n<|assistant|>\n",
       "<|user|>\nWhat is 13 + 28?\n<|assistant|>\n",
       "<|user|>\nIn one sentence, who wrote Hamlet?\n<|assistant|>\n",
       "<|user|>\nList three programming languages.\n<|assistant|>\n",
   ]

   from g2c.inference import load_artifact_backend, load_prodlm_backend

   artifact = load_artifact_backend(required=False)
   prodlm = load_prodlm_backend(required=True)
   backends = [backend for backend in (artifact, prodlm) if backend is not None]

   for backend in backends:
       for p in prompts:
           r = backend.complete(p, max_new_tokens=64, temperature=0.0)
           print(f"[{backend.info.name}] {r.completion!r}  ({r.tokens_per_second:.1f} tok/s)")
   ```

   Compare the outputs side by side. Note where the artifact can answer, where it hallucinates, and where ProdLM behaves differently. Save the comparison.

2. **Benchmark ProdLM on its own.** Use `benchmark(prodlm, prompts, max_new_tokens=128)` over a 20-prompt suite of factual / arithmetic / instruction-following questions (re-use your Module 15 eval set). Report:

   - Mean / p50 / p90 latency (ms).
   - Overall tokens/sec.
   - Per-request tokens/sec (notice variance).
   - The cold-start latency of the first request vs the steady-state mean.

   Expected on M2 16GB with `llama3.2:3b`: 25–40 tok/s steady-state, first-request latency 2–5× higher than steady state. On 64 GB with `llama3.1:8b`: 15–25 tok/s. On 8 GB with `llama3.2:1b`: 50–80 tok/s.

3. **Run the same prompts via MLX.** Install `mlx-lm` (`pip install mlx-lm`), download an MLX-converted model (`mlx-community/Llama-3.2-3B-Instruct-4bit` from Hugging Face works), and write a tiny `MLXBackend` subclass of `Backend`. The exercise here is the subclass — same pattern as `OllamaBackend`, but calling `mlx_lm.generate` in-process. Run the same `benchmark` suite. Compare MLX throughput to ProdLM on the same Mac, same model family, same prompts. Expected: MLX is faster (1.5–2× on M2) for the steady state but model loading is slower.

4. **Re-run Module 15's evaluation on ProdLM.** Take `run_generation_eval` from Module 15 and pass it a `generate_fn` that wraps `prodlm.complete` (note: Module 15's harness expects a `Callable[[str], str]`, so write a small adapter `def gen_fn(prompt): return prodlm.complete(prompt).completion`). Run on your hand-built generation eval set. Compare accuracy to your strongest artifact. Expected: much higher accuracy on factual prompts; same hallucination risk on impossible prompts; better but still imperfect arithmetic.

5. **Quantify the quantization tax.** Configure ProdLM for a default Q4 model, then configure it again for an 8-bit or higher-precision variant if your machine can fit it. Run a 50-question MC eval (Module 15) on both. Report accuracy and ECE for each. Expected: higher precision may be 1–3% more accurate on aggregate; ECE is similar; throughput is lower because more memory bandwidth is required.

6. **Build a "router" Backend.** Subclass `Backend` to dispatch between two concrete backends based on prompt length: `< 32 tokens` goes to the artifact backend (fast, local, good enough for trivial completions); `≥ 32 tokens` goes to ProdLM (slower but more capable). Run the benchmark. Expected: throughput is somewhere between artifact-only and ProdLM-only, depending on prompt mix. The exercise is the dispatching pattern — it's the same shape as a routing decision in Module 19's agent loop.

7. **(Optional) Implement the toy KV cache.** Fill in the cached forward path for the course model:

   - `MultiHeadAttention.forward_cached`
   - `Block.forward_cached`
   - `TransformerLM.forward_cached`
   - `generate_cached`

   The cache containers are already implemented. The tests compare cached outputs against the regular full-sequence path, so the target is not "make it faster at all costs"; the target is "same math, less repeated projection work." Run:

   ```bash
   pytest tests/test_multi_head_attention.py -k cached
   pytest tests/test_transformer.py -k cached
   pytest tests/test_sampling.py -k cached
   ```

   Expected: greedy `generate_cached` returns exactly the same token IDs as `generate` until the cache reaches `max_seq_len`. This toy cache does not implement rolling windows, paged attention, batching, or preallocated buffers; production servers do.

8. **(Optional) Streaming.** Subclass `OllamaBackend` and add a `complete_stream(prompt, ...) -> Iterator[str]` method that yields tokens as they arrive. Hint: `urlopen` returns a stream; iterate `resp` line-by-line, parse each line as JSON (the streaming format), yield the `"response"` field. Demo: print tokens to stdout as the model generates. The UX difference between streaming and non-streaming for a 3B-class model is striking.

9. **Inference-stack post-mortem (the deliverable).** Write 3–4 paragraphs in `docs/inference-postmortem.md` covering:

   - **What you ran.** Models, quantization levels, throughput numbers, latency percentiles.
   - **Where the gaps were.** Which prompts your artifact can attempt at all, which only ProdLM handles, which neither handles.
   - **The cost-quality frontier.** A table or plot of `(model_size, quantization) → (memory, latency, eval_accuracy)`. Even three points (your strongest artifact, llama3.2:1b Q4, llama3.2:3b Q4) is enough to draw the shape.
   - **What you'd use for the rest of the course.** Pick one ProdLM model + quant level as your "default backend" for Modules 17–20. Justify the choice in 2–3 sentences. (Spoiler: probably `llama3.2:3b` Q4_K_M unless you have ≥ 32 GB, in which case `llama3.1:8b` Q4_K_M is a better choice for the agent module.)

   This is the deliverable. Not the interface code — the *characterization* of which backend you're going to depend on for the rest of the course, and why.

## Pitfalls to expect

- **Forgetting to slice off the prompt in `LocalTransformerBackend.complete`.** `g2c.sampling.generate` returns `prompt + new_tokens`; the wrapper must slice with `full[len(prompt_ids):]` to get just the new tokens. A wrapper that returns the full sequence in `completion` will report a "completion" that starts with the input prompt — surprising and wrong.

- **Off-by-one in the slice.** `full[len(prompt_ids):]` is correct (start after the last prompt token); `full[len(prompt_ids) - 1:]` includes the last prompt token; `full[len(prompt_ids) + 1:]` skips the first generated token. Pin this in tests by checking that `result.completion_tokens == len(full) - len(prompt_ids)`.

- **Ollama not running.** Easy to forget. The first call fails with `OllamaError("Could not reach Ollama at ...")`. Quickest check: `curl http://localhost:11434/api/tags` in another terminal. Empty list ⇒ Ollama is up but no models pulled. Connection refused ⇒ Ollama daemon isn't running. Recovery: `ollama serve` in a separate terminal (or restart the Ollama macOS app).

- **Configured the wrong tag.** `./prodlm.sh --model-id llama3.2:3b` and `./prodlm.sh --model-id llama3.2:3b-instruct-fp16` point at different files. The default suffix (no `-q...` part) is usually Q4_K_M, but check `ollama list` to confirm. A manifest that points at a tag you have not pulled produces `404` errors at request time.

- **Greedy decoding produces "the same answer" between backends.** When `temperature=0`, both `LocalTransformerBackend` and `OllamaBackend` should be deterministic. If they're not, your Ollama call is hitting a non-zero temperature default — pass `temperature=0.0` explicitly.

- **Wall-clock latency includes JSON encoding.** For 64-token outputs, JSON encoding/decoding is microseconds — negligible. For 4096-token outputs, it can be milliseconds — still small but visible. The `latency_ms` field is wall-clock; `metadata["server_total_duration_ms"]` is what Ollama reports. Compare both if a number looks off.

- **`urllib.request.urlopen` is blocking.** Calling `complete` in a tight loop blocks the entire process per call. For interactive UIs, wrap in a thread or `asyncio.to_thread`. Don't try to make `urlopen` async — it isn't, and the workarounds are subtle.

- **Time math in the wrong unit.** Ollama returns durations in **nanoseconds**. Wall-clock `time.perf_counter()` returns **seconds**. `InferenceResult.latency_ms` is **milliseconds**. Three different units in three different places. Conversions: `ns → ms` is `/1e6`; `s → ms` is `*1000`. Get one wrong and your `tok/s` is off by 1000× — which is sometimes plausible-looking ("the model does 30,000 tok/s!") and sometimes obviously wrong ("the model does 0.003 tok/s").

- **The `Backend` ABC enforces implementation, not behavior.** A subclass that returns `latency_ms=42.0` regardless of actual time technically satisfies the interface; it just lies about timing. There's no contract beyond "returns an `InferenceResult`." Don't rely on subclasses you didn't write to be honest about their timings — wrap them in your own timer if you're benchmarking.

- **`test_real_pipeline_smoke` skips silently.** When Modules 04 / 09 / 11 aren't implemented, the smoke test auto-skips. If you expect it to run and it doesn't, check `pytest -v` to see the skip reason.

- **Forgetting to pass `eos_id` to `LocalTransformerBackend`.** Without an EOS, generation runs to `max_new_tokens` every time — even when the model would have emitted the chat-template `<|end|>` token. `ArtifactBackend` handles the course end tokens automatically; raw `LocalTransformerBackend` does not. For chat-template prompts, you almost always want `eos_id=tokenizer.encode("<|end|>")[0]` (or whatever your end token resolves to).

- **`urllib.request` doesn't follow non-default redirects.** If you're proxying Ollama through a load balancer that issues 3xx redirects, the default `urlopen` handles them. If you're proxying through one that issues 308s (permanent redirects, common in Cloudflare-style setups), the handler chain is different. For local-only Ollama, this never comes up — but it bites once you start running Ollama on a remote machine.

- **Comparing throughput across machines is meaningless.** A 30 tok/s number on M2 16GB and a 60 tok/s number on M3 Max 64GB tell you about the machines, not the models. Within a single machine, different model/quant combinations are comparable; across machines they aren't. For a write-up, always note the machine.

- **Streaming and non-streaming have different request bodies.** When you set `stream: true` (exercise 7), the response is NDJSON, not a single JSON object. `json.loads(resp.read())` fails on the first chunk. Iterate `resp` line by line and `json.loads` each line.

## Reading

Primary:

- **The Ollama API documentation** (https://github.com/ollama/ollama/blob/main/docs/api.md). The reference for every field in the request/response. Read the `POST /api/generate` section in full — that's what you implement against.
- **The llama.cpp README** (https://github.com/ggerganov/llama.cpp/blob/master/README.md). Background context for what's actually running underneath Ollama. Read the "Quantization" section (the GGUF format and Q4_K_M / Q5_K_M / Q8_0 explanations).
- **Dettmers, Lewis, Belkada, Zettlemoyer, "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022).** The canonical paper on int8 quantization for LLMs. Read §3 (the method) and §5 (the results table). The "outlier features" detail is what makes int8 work without large accuracy loss; the K_M / K_S quantization names in GGUF descend from this lineage.

Secondary:

- **The MLX examples repo** (https://github.com/ml-explore/mlx-examples). Skim the `lm` subfolder. It's the Apple-native counterpart to llama.cpp; the patterns are similar.
- **Pope, Douglas, Chowdhery et al., "Efficiently Scaling Transformer Inference" (2022).** The Google paper that popularized KV cache, attention sharding, speculative decoding. Read §3 (the throughput-vs-latency analysis) and §4 (KV cache layout). Production-grade; the abstractions exceed what we build here, but the framing is durable.
- **Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (2023).** The original speculative-decoding paper. Skim §2 (the algorithm) and §4 (the empirical speedup). Don't implement.

Optional:

- **Frantar, Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022).** The other major lineage of LLM quantization (alongside LLM.int8). GPTQ is what GGUF Q4_K_M is morally a descendant of — second-order error correction during quantization. Skim §3.
- **Lin, Tang, Tang et al., "AWQ: Activation-aware Weight Quantization for LLM Compression" (2023).** Argues that activations matter more than weights for which dimensions to keep at higher precision. Influences modern quantization recipes.
- **Hugging Face's "GGUF" reference** (https://huggingface.co/docs/hub/gguf). The GGUF format spec. Useful when you want to introspect what's actually in a `.gguf` file.
- **Chen, Borgeaud, Irving et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).** Independent invention of the speculative-decoding idea, slightly different framing.

## Deliverable checklist

- [ ] All tests in `tests/test_inference.py` pass. The real-transformer end-to-end test may skip if Modules 04 / 09 / 11 are still scaffolded.
- [ ] Ollama installed and ProdLM configured with `./prodlm.sh --model-id <model:tag>`. `ollama list` shows the model. `curl http://localhost:11434/api/tags` returns a non-empty `models` array.
- [ ] Notebook: `notebooks/16-inference.ipynb`. Runs Exercises 1, 2, 4, 5 against your strongest artifact and configured ProdLM. Plots latency / throughput / accuracy comparisons. Commit with outputs visible.
- [ ] **Inference-stack post-mortem** (Exercise 9) in `docs/inference-postmortem.md`. 3–4 paragraphs. The actual deliverable. Cover: what ran, where the gaps were, the cost-quality frontier, and which backend you're committing to for the rest of the course.
- [ ] You can explain — out loud, without notes — the rough memory cost of running a 7B model at fp32 vs fp16 vs int8 vs int4, and which fits on a 16 GB Mac.
- [ ] You can explain — out loud, without notes — what a KV cache is, why it's necessary at scale, and why the required course path relies on ProdLM's production cache instead of your toy cache.
- [ ] You can explain — out loud, without notes — the difference between wall-clock latency (what `InferenceResult.latency_ms` records) and server-reported latency (what Ollama's `total_duration` reports), and what each is good for.
- [ ] You can explain — out loud, without notes — why the unified `Backend` interface exists, and what would have to change in Module 17's RAG code if it weren't there.

## M-series notes

This is the most memory-sensitive module of the course. Choose the model size + quant level to match your hardware:

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │   MEMORY BUDGET BY MAC CONFIGURATION                                  │
   ├──────────────────────────────────────────────────────────────────────┤
   │                                                                       │
   │   8GB Mac:    1B–3B Q4 only.  llama3.2:1b, qwen2.5:0.5b, qwen2.5:1.5b. │
   │                Tight headroom; close other apps before running.        │
   │                                                                       │
   │   16GB Mac:   Comfortable up to 7B Q4.  llama3.2:3b is best default.   │
   │                llama3.1:8b Q4 works but leaves ~3 GB for everything    │
   │                else (browser, IDE, OS) — tight at long contexts.       │
   │                                                                       │
   │   32GB Mac:   Comfortable with 8B Q4 + headroom; 13B Q4 fits.          │
   │                qwen2.5:14b Q4 works but slow.                          │
   │                                                                       │
   │   64GB Mac:   13B Q4 comfortable; 30B Q4 fits but slow;                │
   │                70B Q4 fits at the edge (40+ GB), 1–5 tok/s.            │
   │                                                                       │
   │   128GB Mac:  70B Q4–Q6 comfortable; 70B Q8 fits.                      │
   │                                                                       │
   └──────────────────────────────────────────────────────────────────────┘
```

For the rest of this course, the recommended ProdLM default is `llama3.2:3b` Q4_K_M, configured with `./prodlm.sh --model-id llama3.2:3b`:

- Fits in 4 GB on every reasonable Mac config.
- Fast enough for interactive use (25–40 tok/s on M2 16GB).
- Strong enough for RAG and agent tasks (instruction-tuned, decent factual recall).
- Same family as the larger Llama 3 models, so what you learn here scales up if you later have access to a bigger machine.

If you have ≥ 32 GB, switching to `llama3.1:8b` Q4_K_M is worth it for Module 19 (agent loops) — the 8B handles tool-calling significantly better than the 3B.

Other practical notes:

- **First-call latency.** Cold-start on Ollama (model load from disk) takes 5–30 seconds depending on model size and disk speed. Subsequent calls are warm. Always do one warmup call before benchmarking.
- **Throughput vs latency.** A bigger model has higher latency *per request* but doesn't necessarily have lower throughput per token — the bigger model is doing more compute per token, but the time per token doesn't grow linearly with parameter count. M-series GPUs are memory-bandwidth-bound; throughput tracks memory bandwidth more than compute.
- **MPS vs Metal vs MLX vs CPU.** Your `LocalTransformerBackend` runs on MPS (PyTorch). Ollama runs on Metal (via llama.cpp's Metal backend). MLX runs on Metal directly (Apple-native API). All three target the same Apple GPU; they differ in driver overhead and kernel optimization. For a fixed model + machine, MLX is typically fastest, then Ollama (Metal), then PyTorch MPS.
- **Heat and throttling.** Sustained inference (5+ minutes of continuous calls) makes the M-series heat up; the GPU clock down-throttles to maintain temperature. Steady-state throughput after thermal throttling can be 70–80% of the cold-start steady state. For benchmarks, either run short suites (under a minute) or measure *both* the early and late steady states.
- **Disk space for models.** 4 GB per Q4 7B model. `ollama pull` to a local laptop with a 256 GB SSD eats real space. `ollama rm <model>` cleans up. `ollama list` shows what's there.
- **Network.** Ollama's first `pull` is gigabytes over HTTPS. Plan for the download time. Once pulled, all inference is local — no network during use.
- **Activity monitor.** During inference, you should see GPU usage spike on the GPU page. If GPU stays low and CPU spikes, your Ollama install is somehow running CPU-only (rare, but possible if Metal init failed). `ollama serve --verbose` in a terminal shows backend choice on startup.
