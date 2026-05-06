# Module 08 — Multi-head attention

> **Question this module answers:** *How does attention specialize?*

![Multi-head attention end-to-end: input X (B, T, D); three big linear projections produce Q, K, V each of shape (B, T, D); a reshape + transpose splits the channel dim into H heads, each of size d_h = D/H, so every tensor becomes (B, H, T, d_h); H independent attention computations run in parallel (each with its own scaling factor √d_h); concatenation merges the heads back to (B, T, D); a final output projection W_O lands the result. A side panel emphasizes the headline fact: same total parameter count as a single big attention, but H different "ways of looking" at the same sequence.](08-multi-head-attention/Module08-Hero.png)

This week is short in terms of content. Almost everything is already in place from last lesson. Review the scaled dot-product and softmax machinery from the previous module. The conceptual move is "split D into H slots and run H copies of attention in parallel"; the engineering move is "do that with one matmul, not H of them."

--- 
## Before you start

---
## Where this fits

Module 07's single-head attention works, but it has a structural limitation: the Q/K/V projections compress everything one query wants to ask about into a single `D`-dimensional vector. If the model wants to attend differently for different reasons — e.g., one query direction for syntactic dependencies, another for coreference, another for adjacency — a single head must overload all of those onto the same `D` channels.

We address this limitation by introducing multiple heads into a single attention model. The headline empirical result, due to Vaswani et al. and substantially deepened by Anthropic's transformer-circuits work, is that *different heads spontaneously specialize* during training. Some heads learn to attend to the previous token. Some learn to attend to syntactic dependencies. Some implement *induction* — copying a previous occurrence of the current token's predecessor. The mechanism doesn't prescribe specialization. Specialization falls out of the training dynamics when you give the model multiple independent attention slots.

## The big idea

The whole module is one structural change — splitting `D` into `H` slots — and one matching detail change — `√head_dim` instead of `√D`.

Multi-head attention gives the model `H` parallel attention channels, each operating in its own `head_dim = D/H` subspace. Each head computes its own Q, K, V, scores, softmax, and weighted value mix independently. The H per-head outputs are concatenated and passed through a final output projection that mixes the heads' findings back into a coherent `D`-dim representation.

```
  Single-head (Module 07)              Multi-head (Module 08)

      x  ─► Wq ─► Q (T, D)                 x  ─► Wq ─► Q (T, D) ─► view+T ─► (H, T, d_h)
      x  ─► Wk ─► K (T, D)                 x  ─► Wk ─► K (T, D) ─► view+T ─► (H, T, d_h)
      x  ─► Wv ─► V (T, D)                 x  ─► Wv ─► V (T, D) ─► view+T ─► (H, T, d_h)

      scores = QK^T/√D  (T, T)            scores = QK^T/√d_h        (H, T, T)
      mask + softmax                      mask + softmax            (H, T, T)
      mixed = wV         (T, D)           mixed = wV                (H, T, d_h)
                                          concat = T+view           (T, D)
      out = Wo(mixed)    (T, D)           out = Wo(concat)          (T, D)
```

Same operations on both sides. Multi-head just adds an `H` axis to everything in the middle. Crucially, the parameter shapes of the projections — `Wq`, `Wk`, `Wv`, `Wo` — are all `(D, D)` in both versions. The split into heads is structural, not parametric.

### The reshape, in detail

The single most important line in this module is:

```python
q.view(B, T, H, head_dim).transpose(1, 2)   # (B, T, D) → (B, H, T, head_dim)
```

`view(B, T, H, head_dim)` reinterprets the last dim `D` as two dims `(H, head_dim)`. After this view, position `t`'s embedding has been sliced into `H` consecutive chunks of size `head_dim`:

```
  embedding_dim = D = 8, num_heads = H = 4, head_dim = 2

  Before view:    [a0, a1, a2, a3, a4, a5, a6, a7]      (D=8)
  After view:     [[a0, a1], [a2, a3], [a4, a5], [a6, a7]]   (H=4, d_h=2)
                    head 0    head 1    head 2    head 3
```

The `transpose(1, 2)` then swaps `T` and `H` so heads are the leading batch-like dim:

```
  Before transpose: (B, T, H, head_dim)
  After  transpose: (B, H, T, head_dim)
```

Why this order? Because the next operation is `q @ k.transpose(-2, -1)`, and PyTorch's batched matmul treats all dims except the last two as batch dims. With `H` in the leading batch slot, the matmul produces one `(T, T)` score matrix per head independently. With `H` *not* in the leading slot, the matmul would mix queries from different heads together — which is the wrong thing.

![A worked reshape: starting from Q of shape (B=2, T=5, D=8), `view(B, T, H=4, d_h=2)` reinterprets the channel dim as (H, d_h), then `transpose(1, 2)` swaps T and H to land at (B, H, T, d_h). Concrete numbers fill the cells so you can trace exactly which D-channels become which head's d_h slots; a "why this order?" panel explains that PyTorch's batched matmul treats every dim before the last two as batch dims, so heads MUST be in a leading batch-like position before scoring.](08-multi-head-attention/Module08-Reshape.png)
*The two-line `view + transpose` is doing all the work that a Python loop over heads would otherwise do. Internalizing which dim ends up where, and why the matmul that follows treats H as just another batch dim, is the entire engineering content of multi-head attention.*

### One projection plus reshape ≠ H independent projections

Naively, you might implement multi-head attention as `H` independent `(D, head_dim)` linear layers per role:

```python
# WRONG (or rather, *expensive and equivalent for inference,
# strictly less expressive for training*):
q_per_head = [Linear(D, head_dim) for _ in range(H)]
qs = [q_per_head[h](x) for h in range(H)]   # H independent matmuls
```

The implementation in this module uses one `Linear(D, D)` followed by a reshape:

```python
# RIGHT (and standard):
q = self.q_proj(x)                          # one matmul, (B, T, D)
q = q.view(B, T, H, head_dim).transpose(1, 2)
```

These are *not* equivalent. The `(D, D)` projection has a full set of cross-head parameters — entries `W[i, j]` where the row `i` belongs to input dim `i` and the column `j` lands in some head's `d_h` slot. The H-independent version only has `H × (d_h × d_h)` parameters arranged on a block-diagonal; the cross-block parameters are zero.

Practically speaking the single `(D, D)` projection is *strictly more expressive* (it can learn the block-diagonal structure if it wants to) and *strictly cheaper* (one big matmul beats H small ones on modern hardware). It's the better choice on both axes, which is why the literature has converged on it. The "multi-head" part is in how the output of the projection is interpreted, not in how the projection itself is parameterized.

### The √ scaling changes: √head_dim, not √D

In Module 07, the scaling factor was `1/sqrt(D)` because each dot product was over a `D`-dimensional vector pair. In multi-head attention, each per-head dot product is over a `head_dim`-dimensional pair, so the scaling factor is `1/sqrt(head_dim)`.

This is the single most common multi-head bug. Symptom: training is slower than expected, attention weights are flatter than expected, gradients are sluggish — but nothing crashes. The scores are under-scaled by a factor of `sqrt(H)`, so the softmax stays in a high-temperature regime where attention is nearly uniform.

### The output projection becomes load-bearing

In Module 07's single-head version, the output projection `Wo` was mostly ceremonial — it could be folded into the value projection without changing the model's expressiveness.

In multi-head, `Wo` is genuinely necessary. After concatenating the H per-head outputs, you have a `(B, T, D)` tensor in which the first `head_dim` channels came from head 0, the next from head 1, and so on. Without a learned mixing step, those channels would propagate forward as siloed "head 0 said this, head 1 said that" without ever interacting. `Wo` is the parameter matrix that lets the heads contribute to one another's outputs.

You can think of the full multi-head attention as: "compute H independent attention patterns in parallel, then learn how to combine their outputs." `Wo` is the "combine" step.

### Heads specialize during training

This is the empirical surprise: when you train a transformer with multi-head attention, individual heads often learn to specialize on specific kinds of patterns. The phenomenon is studied carefully by Elhage et al. (Anthropic transformer circuits) and Olsson et al. (induction heads). A few canonical specializations:

- **Previous-token heads.** A head that puts almost all weight on position `t-1`. Useful for copying immediate context.
- **Induction heads.** A head that, given the pattern `... A B ... A`, puts weight on the position right after the *previous* `A` — predicting `B`. This is the mechanism behind much of in-context learning.
- **Syntactic heads.** Heads that attend along grammatical dependencies (subject ↔ verb, modifier ↔ noun).
- **Copy heads.** Heads that pass values through almost unchanged — a "do nothing here, let the residual stream carry me" head.

You won't see these in your tiny model from Module 10 with high fidelity — you need many more parameters and training data — but the mechanism is set up here. The visualization exercise (exercise 4) will let you LOOK at attention patterns after training and see whether any heads have learned anything interpretable.

![Four canonical head-specialization patterns visualized as attention heatmaps over a sample sentence: (1) a previous-token head — all mass on the position immediately before the query, a clean sub-diagonal stripe; (2) a syntactic head — sparse off-diagonal links between grammatically related tokens (subject↔verb, modifier↔noun); (3) an induction head — for the pattern "...A B...A", weight concentrates on the token after the previous A; (4) a global / broadcast head — diffuse weight covering most of the sequence. A side caption emphasizes the headline: nothing in the architecture forces these patterns; they emerge during training.](08-multi-head-attention/Module08-DiffHeads.png)
*Multi-head attention's empirical payoff isn't visible from the math — it shows up only when you train and look. With H heads available, the optimizer is free to use one for previous-token copying, another for syntactic dependencies, another for induction, etc., and it tends to do so. Single-head attention has to commit one set of weights to ALL of these jobs simultaneously; multi-head attention can specialize and recombine via W_O. 

## Concepts to internalize

- **Multi-head = H copies of single-head, run in parallel in disjoint subspaces.** Math is identical; only an `H` axis is added.
- **The split is structural, not parametric.** All four projections remain `(D, D)`. The split happens in `view`/`transpose`.
- **Reshape, then transpose, then matmul.** The order matters: heads must end up in a leading batch-like dim before scoring, then move back to position-adjacent before the final concat.
- **Scale by `√head_dim`, not `√D`.** The most common multi-head bug.
- **The output projection is now load-bearing.** It mixes the H per-head outputs back into a coherent representation.
- **Heads specialize empirically.** Different heads learn different attention patterns — the structural reason multi-head outperforms single-head with the same parameter budget.
- **Total parameter count is independent of `H`.** Splitting D into 8 heads instead of 1 doesn't change the parameter count one bit; it changes only the structure of the computation.

### What we didn't cover

- Multi-query / grouped-query attention (one K/V head shared across many Q heads — the optimization that makes long-context inference cheap). Out of scope.
- FlashAttention's IO-aware tiling. The math is the same; only the memory access pattern changes.
- Cross-attention (Q from one source, K/V from another — used in encoder-decoder transformers). The course is decoder-only.

---
## What you'll build

Package: `g2c/attention/`

```python
class MultiHeadAttention(Module):
    embedding_dim: int       # D
    num_heads: int           # H
    head_dim: int            # D // H
    causal: bool
    q_proj: Linear           # (D, D)
    k_proj: Linear           # (D, D)
    v_proj: Linear           # (D, D)
    out_proj: Linear         # (D, D)

    def __init__(self, embedding_dim: int, num_heads: int, *, causal: bool = True): ...   # implemented
    def parameters(self) -> Iterable[torch.Tensor]: ...                                    # implemented

    @staticmethod
    def causal_mask(seq_len: int, device=None) -> torch.Tensor: ...                        # implemented

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...                                # SCAFFOLDED
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor: ...                      # SCAFFOLDED
```

Roughly 20 lines of real code split across the two scaffolded methods. The conceptual delta from Module 07 is small — most of the lesson is in the reshape and the `√head_dim` change.

## How to run the tests

Tests live in `tests/test_multi_head_attention.py`. Initial state: 11 passed (construction + `causal_mask` + parameter counts), 16 failed.

```bash
source .venv/bin/activate

pytest tests/test_multi_head_attention.py             # run all module-08 tests
pytest tests/test_multi_head_attention.py -x          # stop at first failure (recommended)
pytest tests/test_multi_head_attention.py -k forward  # only the forward tests
pytest tests/test_multi_head_attention.py -v          # verbose
```

## Exercises

1. **Verify the reshape is mathematically correct.** Construct `MultiHeadAttention(embedding_dim=4, num_heads=2, causal=False)`. Force the Q and K projections to the identity. Feed `x = torch.tensor([[[1., 0., 0., 1.], [0., 1., 1., 0.]]])` (shape `(1, 2, 4)`). Compute `attn.attention_weights(x)` and compare against the per-head softmax you compute by hand:

   - Head 0 sees `x_h0 = [[1, 0], [0, 1]]` (the first 2 dims of each position).
   - Head 1 sees `x_h1 = [[0, 1], [1, 0]]` (the last 2 dims).
   - For each head, expected weights are `softmax(x_h @ x_h.T / sqrt(2))`.

   The test `test_attention_weights_use_sqrt_head_dim_scaling` automates a randomized version of this check — exercise 1 is to convince yourself you understand it on paper.

2. **Confirm reshape order matters.** Replace `q.view(B, T, H, head_dim).transpose(1, 2)` with `q.view(B, T, head_dim, H).transpose(1, 2)` (note the swapped last two dims of `view`). Run the test suite; observe that the heads now "see" interleaved slices of the embedding instead of contiguous ones. The shapes are still right, the tests *might* still pass for degenerate inputs, but the operation is fundamentally different from canonical multi-head attention. Revert and articulate (in a notebook cell or a comment) what the difference is.

3. **Train a tiny LM at H = 1, 4, 8.** Following the pattern from exercise 3 of Module 07: build a tiny attention-only LM with a `TokenEmbedding`, an additive positional encoding, one `MultiHeadAttention`, and an output `Linear` to vocab logits. Train three configurations on a short tokenized corpus, with the same fixed `embedding_dim = 64` and the same training budget:

   - `MultiHeadAttention(embedding_dim=64, num_heads=1)`  (head_dim = 64)
   - `MultiHeadAttention(embedding_dim=64, num_heads=4)`  (head_dim = 16)
   - `MultiHeadAttention(embedding_dim=64, num_heads=8)`  (head_dim = 8)

   Plot the validation loss curves on the same axes. The parameter counts are identical (verified by `test_parameters_independent_of_head_count`); any difference in loss is purely due to the structural difference. The expected pattern: more heads ≥ fewer heads at small scale, with diminishing returns. At very small `head_dim` (e.g., `head_dim = 1`) you should see degradation — heads need enough subspace to be useful.

4. **Visualize per-head attention patterns.** Take any of the trained models from exercise 3, run a forward pass on a chosen sentence, and plot `attn.attention_weights(x)` as an `H × 1` grid of heatmaps (one per head) using `matplotlib.imshow`. With token labels on the axes. Look for patterns:

   - Does any head look like a "previous-token" head (most weight just below the diagonal)?
   - Does any head look near-uniform (a "no-op" / copy head)?
   - Does any head spike on specific token pairs?

   At Module 08 + tiny training, you may not see anything truly interpretable. That's fine — the mechanism is what matters here. The actual interpretable patterns appear after Module 10's pretraining at scale.

5. **Parameter counts at varying H.** Verify analytically that `MultiHeadAttention(embedding_dim=D, num_heads=H)` has parameter count `4 * (D*D + D)`, independent of `H`. Compute it programmatically for a few `(D, H)` pairs by summing `p.numel()` over `attn.parameters()`. Convince yourself that "more heads" is a free structural change at fixed `D`.

## Pitfalls to avoid

- **Scaling by √D instead of √head_dim.** The single most common multi-head bug. Training sluggish; attention weights are too flat; gradients propagate weakly. Crashes nothing, fails subtly. 

- **Reshape order: `(B, T, head_dim, H)` instead of `(B, T, H, head_dim)`.** The shape is the same after view, but heads end up "seeing" interleaved slices of the embedding (positions 0, H, 2H, ... instead of 0..head_dim-1). Will silently produce a different model.

- **Forgetting `.transpose(1, 2)` after the reshape.** Without it, `T` is in the leading batch-like position and `H` is in the position-adjacent position, so the next matmul will mix queries from different heads.

- **Forgetting `.contiguous()` before the final `view` to concatenate heads.** PyTorch will raise an exception. Insert `.contiguous()` or use `.reshape(...)`.

- **Mask polarity backwards (same as Module 07).** `causal_mask` returns True ABOVE the diagonal — the positions to BLOCK. The `(T, T)` mask broadcasts naturally over `(B, H, T, T)` scores.

- **Implementing per-head with `H` independent linear layers.** Works mathematically, but is `H × `slower (H matmuls instead of 1) and strictly less expressive (block-diagonal projections instead of full ones). The standard idiom is one `(D, D)` projection plus reshape.

- **Returning `attention_weights` of shape `(B, T, T)` instead of `(B, H, T, T)`.** The "averaged over heads" version drops the per-head visualization that motivates exposing this method at all. Keep the H dim. 

- **`embedding_dim` not divisible by `num_heads`.** The constructor raises `ValueError` for this, but if you bypass it (or compute `head_dim` with integer division and silently lose dims), shapes will mismatch downstream. Don't silence the check.

## M-series notes

This module is light on compute 

- Exercise 3's training comparison (3 runs at fixed `D = 64`) is a few hundred steps each on a small corpus; under a couple minutes total on CPU.
- Exercise 4's per-head visualization is a single forward pass on one sentence — milliseconds.
- The clean notebook uses `experiment_device = "auto"` for the training comparison. The plotted attention weights are moved back to CPU before Matplotlib sees them, because Matplotlib cannot plot MPS tensors directly.

---
## Reading

Primary:

- **Vaswani et al., "Attention Is All You Need" (2017), §3.2.2.** Multi-head attention is defined in two equations and one paragraph. This is the section to read. The illustration in figure 2 (right panel) shows the structure.
- **Karpathy, "Let's build GPT: from scratch, in code, spelled out"** (YouTube). The multi-head section walks through this same construction in PyTorch — same reshape idiom, same √head_dim scaling.

Secondary:

- **Elhage et al., "A Mathematical Framework for Transformer Circuits" (Anthropic, 2021).** The introductory and "what do heads compute" sections reframe multi-head attention as `H` independent read-write operations on the residual stream. The framework is illuminating even if you skim the late sections on circuits.
- **Olsson et al., "In-context Learning and Induction Heads" (Anthropic, 2022).** The empirical study of how induction heads emerge during training. Pairs nicely with the visualization exercise — gives a sense of what a "real" specialized head looks like.

Optional:

- **Kim & Vaswani, "Multi-Query Attention"** and follow-ups on grouped-query attention. The optimization that makes long-context inference cheap by sharing K and V across multiple query heads. Out of scope for this course but worth knowing exists.

## Deliverable checklist

- [ ] All tests in `tests/test_multi_head_attention.py` pass.
- [ ] Notebook: `notebooks/clean/08-multi-head-attention.ipynb`. Train tiny LMs at `num_heads = 1, 4, 8` with fixed `embedding_dim = 64` and matched training budgets; plot validation loss curves on the same axes.
- [ ] Notebook: per-head attention visualization on a chosen sentence using one of the trained models — `H` heatmaps in a grid.
- [ ] You can explain — out loud, without notes — why the scaling factor is `√head_dim` rather than `√D`, and what specifically goes wrong if you use `√D`.
- [ ] You can explain — out loud, without notes — why the reshape `view(B, T, H, head_dim).transpose(1, 2)` is the right operation, and what would go wrong with `view(B, T, head_dim, H)`.
