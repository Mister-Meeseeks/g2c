# Beyond — Hardware-aware model architecture

> **Question this module answers:** *Why can a model with fewer FLOPs still run slower—and how does hardware feed back into architecture?*

<!-- TODO(hero pipeline): asset not yet generated -->
![Training, prefill, and decoding workloads pass through arithmetic, memory bandwidth, parallelism, kernels, and communication before becoming wall-clock performance.](hardware/BeyondHardware-Hero.png)

Model architecture is often taught as algebra floating above a machine. Real models run where matrix shapes, memory movement, kernel launches, cache capacity, and available parallelism can matter as much as operation counts. This module builds a durable resource model on an M-series Mac and uses it to read architecture choices—including the hybrid stack in GLM-5.3-Flash—without turning the course into a GPU-programming course.

> **Draft status.** This is a proposed Beyond module specification. Its package, tests, notebook, rubric, and hero asset do not exist yet; the API and exercise counts below are design targets rather than a runnable assignment.

> **This is a Beyond module.** Beyond modules sit outside the numbered course: nothing in Modules 00–20 depends on them, and they are not part of finishing the course.

---
## Before you start

* *Review* [02-tensors](../modules/02-tensors.md) for shape, stride, and matrix multiplication
* *Review* [09-transformer-block](../modules/09-transformer-block.md) for the operations whose execution you will account for
* *Review* [16-inference](../modules/16-inference.md) for prefill, decoding, and the KV cache
* *Review* [The efficient-attention map](attention-efficiency.md) for architecture-level attention tradeoffs
* Use an M-series Mac with PyTorch MPS available for the benchmark notebook; the pure accounting tests remain CPU-only

---
## Where this fits in

The numbered course deliberately chooses readable tensor code over peak performance. That keeps the mechanisms inspectable, but it leaves a model-card question unanswered: why do production architectures contain grouped queries, latent caches, recurrent layers, sparse indices, fewer deeper layers, experts, quantization, and speculative heads in combinations that cannot be ranked by parameter count or FLOPs alone?

This module supplies the missing vocabulary. It does not teach one vendor's hardware catalog. It teaches a resource model—work, bytes, capacity, parallelism, and communication—that transfers from an M-series laptop to accelerators described in current releases. Dated [G2C Briefs](../briefs/index.md) carry the changing product landscape.

## The big idea

Wall-clock time is constrained by more than arithmetic:

```text
architecture + shapes + execution phase
                 ↓
      operations, bytes, live state,
      parallel work, synchronization
                 ↓
       kernels + compiler + hardware
                 ↓
     latency, throughput, energy, capacity
```

A useful first lower bound is the roofline model. If a workload performs `F` floating-point operations, moves `B` bytes, the machine sustains peak compute `P`, and memory bandwidth `W`, then:

```text
arithmetic intensity = F / B

time ≥ max(F / P, B / W)
```

Low-intensity work is bandwidth-bound; high-intensity work can become compute-bound. The bound is not a latency prediction—launches, synchronization, occupancy, layout, and unsupported operations create additional gaps—but it tells you which resource an optimization must change.

### Training, prefill, and decode are different workloads

The same model exposes different shapes and reuse:

| Phase | Characteristic work | Common pressure |
|---|---|---|
| **Training** | Many tokens, forward + backward, saved activations | compute, activation capacity, optimizer state |
| **Prefill** | Many prompt tokens processed together | parallel matrix math, dense attention work, temporary memory |
| **Decode** | One or a few new tokens per sequence | weight/KV movement, small matrix shapes, launch overhead |

A large matrix multiplication can reuse each loaded weight across many tokens. Single-sequence decode looks more like repeated matrix-vector work: the same weights may be streamed again for each token. Batching increases reuse and throughput but can raise latency and live KV capacity. “Faster” is incomplete until the phase, batch, context, and metric are named.

### Shapes matter even when FLOPs match

Two multiplications with equal operation counts can expose different parallelism, tiling, and memory access. Contiguous and strided views can trigger different copies or kernels. Many tiny operations can lose to one larger operation because each dispatch has fixed overhead.

This is where **kernels** belong. A kernel is the concrete device program executing an operation or fused group of operations. Tiling chooses chunks that fit fast local memory; fusion keeps intermediate values near the compute units instead of materializing them; layout determines whether neighboring threads read useful neighboring bytes. FlashAttention is the canonical case: it preserves dense softmax attention while reorganizing the computation to reduce memory traffic.

The required module measures kernels through observable behavior and accounting. Writing a custom Metal kernel would introduce a language, toolchain, and optimization problem larger than the architecture lesson. It is a good later lab, not a prerequisite for understanding why kernels matter.

### Architecture chooses resource bills

| Architecture choice | Bill it can reduce | Bill or trade it retains |
|---|---|---|
| **Fewer, wider layers** | launches and sequential depth | wider matrices and activations |
| **MoE** | active arithmetic per total parameter | expert weight movement, routing, capacity |
| **MQA/GQA/MLA** | KV-cache width and decode bandwidth | token-addressable cache still grows |
| **Linear/recurrent attention** | history-dependent state and per-token scan | compressed memory, recurrent-state kernels |
| **Sparse attention / pooled index** | selected attention work or index size | payload, index, and selection costs remain distinct |
| **Quantization** | weight/cache bytes and often bandwidth | conversion, kernel support, possible quality loss |
| **Speculative decoding** | serial target-model steps | verification work and acceptance dependence |
| **Residual mixing such as mHC** | optimization stability at depth | additional routing computation and state |

No row guarantees a speedup. An architecture only creates an opportunity; shapes, kernels, and the target hardware decide whether the opportunity is realized.

### GLM-5.3-Flash as a worked reading

Z.ai reports a text stack of `[KDA, KDA, KDA, DSA] × 11 + KDA`, with fewer, wider layers; recurrent KDA state; MLA-compressed sparse-attention payloads; a top-k index; four-to-one IndexPool; routed experts; mHC residual mixing; multi-token prediction; and a ReplaySSM draft model. These mechanisms touch different resources:

```text
KDA                  bounds history state in 34 layers
DSA + MLA            retains selected token access with narrower payloads
IndexPool            shrinks a separate sparse index
fewer, wider layers  changes launch count and matrix shapes
MoE                   separates total capacity from active compute
MTP + ReplaySSM       trades draft work for fewer serial target steps
```

The configuration facts and performance claims belong to the dated [GLM-5.3-Flash Brief](../briefs/glm-5.3-flash.md). The **G2C interpretation** here is durable: the model is not “one efficient mechanism.” It is a resource budget assembled across architecture and serving, and vendor ratios describe that complete measured system—not universal constants for KDA, sparsity, or a 3:1 layer pattern.

### Communication is the next boundary

On one M-series package, unified memory removes some explicit CPU–GPU copies but does not make access free; bandwidth and capacity remain finite. Large deployments add communication between devices: tensor, pipeline, data, and expert parallelism exchange different tensors and create synchronization points. This module accounts for communication conceptually but does not build a distributed system. That boundary keeps the required work local and makes clear which conclusions come from single-device measurements.

## Concepts to internalize

- **FLOPs are one resource, not a performance verdict.** Bytes, capacity, launch overhead, parallelism, and synchronization can dominate.
- **Arithmetic intensity predicts the first bottleneck to investigate.** It is a lower-bound model, not a benchmark substitute.
- **Training, prefill, and decode need separate measurements.** Their shapes and reuse differ.
- **Kernels are where algorithms meet hardware.** Tiling, fusion, and layout can change traffic without changing model math.
- **Architecture optimizations compose—and keep separate bills.** Payload, index, selection, recurrent state, expert weights, and draft verification should not be collapsed into one cache or compute number.
- **Hardware-aware does not mean hardware-specific.** Measure one machine; explain the transferable resource model and state where results will not transfer.

### What we don't cover

- **Custom Metal kernels.** The module observes kernel effects through controlled PyTorch workloads; a fused-kernel implementation is a possible later lab.
- **Distributed serving.** Communication is included in the model but the build remains on one Mac.
- **Vendor ranking.** Product availability and benchmark leadership age quickly and belong in dated Briefs.
- **Energy measurement.** Reliable package-level power instrumentation would add platform-specific plumbing beyond the core experiment.
- **Full-model performance reproduction.** GLM-5.3-Flash is a reading case, not a checkpoint, API, or cloud requirement.

---
## What you'll build

Planned package: `g2c/hardware/`

```python
@dataclass(frozen=True)
class Workload:                            # boilerplate provided
    flops: int
    bytes_read: int
    bytes_written: int

    @property
    def arithmetic_intensity(self) -> float: ...       # scaffolded

def roofline_lower_bound(workload, *, peak_flops, bandwidth): ...  # scaffolded
def dense_attention_workload(*, batch, heads, tokens, head_dim, dtype): ...
def kv_decode_workload(*, layers, kv_heads, history, head_dim, dtype): ...
def recurrent_decode_workload(*, layers, heads, head_dim, dtype): ...
```

The notebook will use non-pedagogical timing helpers in `g2c/notebook_extras/hardware.py` to synchronize MPS, warm kernels, and report distributions rather than a single run.

## How to run the tests

Planned commands:

```bash
source .venv/bin/activate

pytest tests/test_hardware.py
pytest tests/test_hardware.py -x
pytest tests/test_hardware.py -k roofline
pytest tests/test_hardware.py -k attention
```

Tests will use exact hand-computable workloads and remain CPU-only. Benchmarks are notebook evidence, never pass/fail tests tied to one Mac generation.

## Exercises

Open the planned working notebook with `./notebook.sh hardware` (or `./notebook.sh hardware --fresh`), write answers in the `Question:` / `Answer:` cells, and ask a coding agent for hints or grading. Partial submissions are fine because blank answers are skipped.

1. **Calibrate the machine.** Estimate sustained memory bandwidth and matrix throughput over several sizes; distinguish the measured ceiling from a product-sheet peak.
2. **Hold FLOPs approximately fixed, change shape.** Benchmark tall, square, and skinny matrix products and explain the spread using reuse and parallelism.
3. **Make layout visible.** Compare contiguous, transposed, and explicitly materialized inputs while recording both timing and bytes.
4. **Separate prefill from decode.** Sweep batch and context for dense cached attention, then compare its state accounting with recurrent attention.
5. **Read an architecture as a budget.** Annotate the GLM-5.3-Flash stack by work, bytes, live state, sequential depth, and speculative verification; do not predict a vendor speedup from course measurements.

## Pitfalls to expect

- **Timing asynchronous MPS work without synchronization.** The host timer can finish before the device does.
- **Benchmarking the first call.** Compilation and allocation contaminate cold-start timing; report cold and warmed paths separately when both matter.
- **Using one sample.** Background work and thermal state create noise; report a distribution and the complete shape/configuration.
- **Counting FLOPs but not bytes.** Decode often moves far more data than its arithmetic count suggests.
- **Treating unified memory as infinite bandwidth.** Shared addressability changes copying semantics, not physical capacity or traffic.
- **Generalizing one Mac's crossover.** The resource explanation transfers; the measured threshold usually does not.

## M-series notes

- Use conservative allocations and release large benchmark tensors between sweeps; unified-memory pressure affects the entire machine.
- The notebook must guard MPS use with `torch.backends.mps.is_available()` and keep an informative CPU fallback.
- Benchmark helpers synchronize before and after timed regions and record dtype, shapes, warmup count, repetitions, OS, and device name.

---
## Reading

- **Williams, Waterman, Patterson, [“Roofline: An Insightful Visual Performance Model for Multicore Architectures”](https://escholarship.org/uc/item/3qf383m0) (2009).** The arithmetic-intensity model this module turns into code.
- **Dao et al., [“FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”](https://arxiv.org/abs/2205.14135) (2022).** The clearest architecture-adjacent example of reducing memory traffic through tiling.
- **Shazeer, [“Fast Transformer Decoding: One Write-Head is All You Need”](https://arxiv.org/abs/1911.02150) (2019).** Multi-query attention motivated from the decode memory-bandwidth bottleneck.
- **DeepSeek-AI, [“DeepSeek-V2”](https://arxiv.org/abs/2405.04434) (2024).** MLA and MoE as interacting capacity, cache, and compute choices.
- **Z.ai, [“GLM-5.3-Flash”](https://z.ai/blog/glm-5.3-flash) (2026).** The current case study; use the dated Brief to separate reported claims from course interpretation.

## Deliverable checklist

- [ ] Planned accounting tests pass without MPS or network access.
- [ ] Notebook reports warmed timing distributions with synchronized MPS measurements.
- [ ] Equal-FLOP/different-shape and layout experiments are explained using work, bytes, and parallelism.
- [ ] Prefill and decode are measured and discussed separately.
- [ ] The GLM stack is annotated as multiple resource trades without converting vendor claims into universal laws.
